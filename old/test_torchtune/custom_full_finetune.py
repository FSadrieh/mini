# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn
import debugpy

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils, generation
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.embedding_utils import resize_token_embeddings
from torchtune.modules.loss import SFTLoss
from torchtune.modules.optim import OptimizerInBackward
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr

from tqdm import tqdm
from torchmetrics import Accuracy
from torchmetrics.text import ROUGEScore, BLEUScore, SQuAD, Perplexity
from src.data_loading import CustomDataLoader
import datetime
import wandb
import datetime


class FullFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``enable_activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources. To recover some runtime cost,
            we've added an option to enable offloading on a different stream to permit overlapping with
            the computation. This option is currently only available on PyTorch 2.5 or later and will
            be enabled by default if an acceptable torch version is found. Activation offloading can be
            used in conjunction with activation checkpointing.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
            footprint associated with gradients. This can be especially helpful when you are memory
            constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
            These features currently do not work together. For more details on optimizer in backward, please
            see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
            since they help reduce the memory footprint associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training.

            Resuming training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

        - Gradient Clipping. Gradient clipping is supported using the ``clip_grad_norm`` flag. By default,
            ``clip_grad_norm`` is set to ``None``. If you only want to log the grad norm, you can set
            ``clip_grad_norm='inf'``.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``gradient_accumulation_steps > 1`` and ``optimizer_in_bwd`` is `True`.
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # Disable for fp16, as we haven't validated "full" fp16 with this recipe, nor
        # enabled necessary features such as gradient scaling.
        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cfg.checkpointer.output_dir = cfg.checkpointer.output_dir + f"_{timestamp}"
            cfg.output_dir = cfg.output_dir + f"_{timestamp}"
        except AttributeError:
            pass

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        if self._log_peak_memory_stats and self._device.type == "cpu":
            self._logger.info(
                "log_peak_memory_stats was set to True, however, training uses cpu. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self.optimizer_in_bwd = cfg.optimizer_in_bwd

        if self.optimizer_in_bwd and self._clip_grad_norm is not None:
            raise RuntimeError(
                "Gradient clipping is not supported with optimizer_in_bwd: True"
            )
        if self.optimizer_in_bwd and self._gradient_accumulation_steps > 1:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer_in_bwd: True"
            )

        self._checkpoint_client = CheckpointClient(cfg)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0

        ### My code ###
        self._run_val_every_n_steps = cfg.get("run_val_every_n_steps", None)
        self._eval_batches = cfg.get("eval_batches", 0)
        self.sample_table = wandb.Table(
            columns=["timestep", "prompt", "pred", "label", "em", "f1"]
        )
        self.acc = None
        self.squad = SQuAD(
            dist_sync_on_step=True,
        ).to(self._device)
        self.individual_squad = SQuAD(
            dist_sync_on_step=True,
        ).to(self._device)
        self.rouge = ROUGEScore(
            dist_sync_on_step=True,
        )
        self.bleu = BLEUScore(
            dist_sync_on_step=True,
        ).to(self._device)
        self.ppl = Perplexity(
            ignore_index=-100, dist_sync_on_step=False, sync_on_compute=False
        ).to(self._device)

        self.custom_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        self.acc_per_prompt = {}

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly. This includes setting recipe attributes based
        on the ``resume_from_checkpoint`` flag.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        ckpt_dict = self._checkpoint_client.load_base_checkpoint()

        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._compile = cfg.compile
        if cfg.device == "npu" and cfg.compile:
            raise ValueError(
                "NPU does not support model compilation. Please set `compile: False` in the config."
            )
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            compile_model=self._compile,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._logger.info("Tokenizer is initialized from file.")

        if cfg.get("resize_token_embeddings", False):
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        # _setup_optimizer should take in ckpt_dict only if training is resumed from
        # checkpoint. Transforming the opt state dict is handled by this method
        self.optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                ckpt_dict[training.OPT_KEY] if training.OPT_KEY in ckpt_dict else None
            ),
        )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    ckpt_dict = self._checkpoint_client.load_distributed_checkpoint(
                        self._model,
                        self.optimizer,
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(ckpt_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile:
            training.compile_loss(self._loss_fn)

        self._logger.info("Loss is initialized.")

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
        self._tokenizer.pad_id = self._tokenizer.eos_id
        tokenizer_name = cfg.tokenizer._component_.split(".")[-1]
        data_config = {
            "tokenizer": self._tokenizer,
            "tokenizer_name": tokenizer_name,
            "batch_size": cfg.batch_size,
            "firstn_datasets": cfg.dataset.get("firstn_datasets", None),
            "seed": cfg.get("seed", self.seed),
            "data_location": cfg.dataset.get("data_location", "data"),
            "preprocessing_workers": cfg.dataset.get("preprocessing_workers", 1),
        }

        data_config["val_batch_size"] = cfg.get("batch_size_val", cfg.batch_size)

        data_loader = CustomDataLoader(**data_config)
        self._dataloader = self._setup_data(data_loader, "train")
        self._val_dataloader = self._setup_data(data_loader, "validation")
        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        #
        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader, the max_steps_per_epoch param set by the user and the
        # gradient_accumulation_steps param. This value is used for logging and tracking
        # training state. The computation should happen after the dataloader has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        # Setup lr scheduler if a component is included in the config
        lr_scheduler_cfg = cfg.get("lr_scheduler", None)
        if lr_scheduler_cfg is not None:
            self.lr_scheduler = self._setup_lr_scheduler(
                cfg_lr_scheduler=lr_scheduler_cfg,
                num_training_steps=self.total_epochs * self._steps_per_epoch,
                last_epoch=self.global_step - 1,
            )
        else:
            self.lr_scheduler = None

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # TODO: Is num classes correct
        self.acc = Accuracy(
            task="multiclass",
            num_classes=self._model.tok_embeddings.weight.shape[0],
            ignore_index=-100,
            dist_sync_on_step=True,
        ).to(self._device)

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """

        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        self._logger.info(f" Profiler config after instantiation: {profiler_cfg}")

        self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
        if profiler_cfg["enabled"]:
            self.profiler_wait_steps = profiler_cfg["wait_steps"]
            self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
            self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        compile_model: bool,
        model_state_dict: dict[str, Any],
    ) -> nn.Module:
        """
        Set up the model including enabling activation checkpointing.
        """
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        if compile_model:
            training.compile_model(model)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )

        # Enable activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        self._logger.info(f"Model is initialized with precision {self._dtype}.")

        if self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optimizer:
        if self.optimizer_in_bwd:
            optimizer_cls = _get_component_from_path(cfg_optimizer.pop("_component_"))
            optimizer = OptimizerInBackward(
                params=self._model.parameters(),
                optimizer_cls=optimizer_cls,
                **cfg_optimizer,
            )
        else:
            optimizer = config.instantiate(
                cfg_optimizer, params=self._model.parameters()
            )
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)
        self._logger.info("Optimizer is initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> LambdaLR:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self.optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        self._logger.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        data_loader: CustomDataLoader,
        split: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
        dataset = data_loader.load_dataset(split=split)
        collate = data_loader.get_collator()
        dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=collate,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )

        return dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. The recipe save_checkpoint method is responsible for
        correctly creating the checkpoint dict and passing to the checkpointer.
        """
        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=self.optimizer,
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=self.epochs_run,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
            ),
            epoch=epoch,
            single_device=True,
        )

    def _loss_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
        # Shape [b, s], needed for the loss not the model
        labels = batch["labels"]

        with self.activations_handling_ctx:
            outputs = self._model(tokens=batch["tokens"])

        # post process for third party loss functions
        if not isinstance(self._loss_fn, modules.loss.SFTLoss):
            labels = labels.reshape(-1)
            outputs = outputs.reshape(-1, outputs.size(-1))
            if isinstance(outputs, DTensor):
                outputs = outputs.full_tensor()

        # Compute loss
        loss = self._loss_fn(outputs, labels)

        # Compute accuracy and custom losses
        logits = self._model.output(outputs)
        metrics = self.calculate_loss(logits, labels, batch["sample_count"].tolist())
        metrics["loss"] = loss

        del outputs
        return metrics, logits

    def validate(self) -> dict[str, float]:
        """
        Run validation loop and return average validation loss.
        """

        for m in (self.acc, self.bleu, self.squad, self.rouge, self.ppl):
            m.reset()

        self._model.eval()
        running_metrics = {}
        self.acc_per_prompt = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self._val_dataloader):
                if batch_idx >= self._eval_batches:
                    continue
                templates = batch.pop("templates", None)
                utils.batch_to_device(batch, self._device)
                self.generate(
                    batch,
                    batch_idx,
                    templates,
                )
                metrics, logits = self._loss_step(batch)

                # Count tokens excluding padding
                metrics["current_num_tokens"] = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                self.ppl.update(
                    logits,
                    batch["labels"],
                )
                predictions = torch.argmax(logits, axis=-1).view(-1)
                mask = batch["labels"].view(-1).ne(-100)
                self.acc.update(
                    preds=predictions[mask],
                    target=batch["labels"].view(-1)[mask],
                )

                running_metrics = self.log_metrics_over_epoch(metrics, running_metrics)

        metrics = self.calculate_total_metrics(running_metrics)
        squad_metrics = self.squad.compute()
        log_dict = {
            "val/accuracy": self.acc.compute().item(),
            "val/loss": metrics["loss"],
            "val/mean_loss": metrics["mean"],
            "val/sum_loss": metrics["sum"],
            "val/distance_loss": metrics["distance"],
            "val/var_loss": metrics["variance"],
            "val/ppl": self.ppl.compute().item(),
            "val/exact_match": squad_metrics["exact_match"].item(),
            "val/f1": squad_metrics["f1"].item(),
            "val/bleu": self.bleu.compute().item(),
            **{f"val/{k}": float(v) for k, v in self.rouge.compute().items()},
        }

        for prompt, acc in self.acc_per_prompt.items():
            acc = torch.tensor(acc)
            num_samples = acc.shape[0]
            log_dict[f"val/acc_{prompt}"] = (
                (acc.sum() / num_samples) if num_samples > 0 else 0.0
            )

        self._logger.info(f"Validation loss: {metrics['loss']:.4f}")
        self._metric_logger.log_dict(
            log_dict,
            step=self.global_step,
        )

        self._model.train()
        return log_dict

    def train(self) -> None:
        self.optimizer.zero_grad()
        t0 = time.perf_counter()
        running_metrics = {}
        self._profiler.start()

        if (
            self._run_val_every_n_steps is not None
            and self.global_step % self._run_val_every_n_steps == 0
        ):
            self.validate()

        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch)

            for idx, batch in enumerate(self._dataloader):
                # Optionally start memory profiling
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                metrics, __ = self._loss_step(batch)
                metrics["current_num_tokens"] = (
                    batch["labels"] != self._loss_fn.ignore_index
                ).sum()

                # Mean loss for in-bwd optimizers, else multiply by token count
                # We need to treat the loss different as we will call backward
                loss_factor = (
                    metrics["current_num_tokens"] if not self.optimizer_in_bwd else 1.0
                )
                current_loss = metrics["loss"] * loss_factor

                current_loss.backward()
                running_metrics = self.log_metrics_over_epoch(
                    metrics,
                    running_metrics,
                    loss_factor=None if not self.optimizer_in_bwd else 1.0,
                )

                # Take a normal optimizer step
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    grad_norm = None
                    if not self.optimizer_in_bwd:
                        training.scale_grads(
                            self._model, 1.0 / running_metrics["current_num_tokens"]
                        )
                        if self._clip_grad_norm:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(), float(self._clip_grad_norm)
                            )

                    # This will be a no-op for optim in bwd, but prevents a warning w/ LR Scheduler
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.global_step += 1
                    total_metrics = self.calculate_total_metrics(
                        running_metrics,
                        loss_factor=None if not self.optimizer_in_bwd else 1.0,
                    )
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {total_metrics['loss']:.4f}"
                    )

                    if self.global_step % self._log_every_n_steps == 0:
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "train/loss": total_metrics["loss"],
                            "train/mean_loss": total_metrics["mean"],
                            "train/sum_loss": total_metrics["sum"],
                            "train/distance_loss": total_metrics["distance"],
                            "train/var_loss": total_metrics["variance"],
                            "lr": get_lr(self.optimizer),
                            "tokens_per_second_per_gpu": (
                                total_metrics["current_num_tokens"] / time_per_step
                            ),
                        }
                        if self._device.type != "cpu" and self._log_peak_memory_stats:
                            log_dict.update(training.get_memory_stats(self._device))
                        if grad_norm is not None:
                            log_dict["grad_norm"] = grad_norm
                        self._metric_logger.log_dict(log_dict, step=self.global_step)

                    running_metrics = {}
                    t0 = time.perf_counter()

                # Optionally stop memory profiling
                if (
                    curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                self._profiler.step()

                if (
                    self._run_val_every_n_steps is not None
                    and self.global_step % self._run_val_every_n_steps == 0
                ):
                    pbar.refresh()
                    self.validate()
                    self.ppl.reset()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            self.save_checkpoint(epoch=curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        self._metric_logger.log_dict(
            {"sample_table": self.sample_table}, step=self.global_step
        )
        self._metric_logger.close()

    def calculate_loss(self, logits, labels, sample_count):
        per_token_loss = self.custom_loss(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        total_mean_loss = per_token_loss.sum() / labels.ne(-100).sum().float()
        per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))
        per_example_loss = (
            per_token_loss.sum(dim=1) / labels.ne(-100).sum(dim=1).float()
        )
        per_group_loss = torch.split(per_example_loss, sample_count)
        return {
            "distance": sum(
                torch.abs(max(losses) - min(losses)) for losses in per_group_loss
            )
            / len(per_group_loss),
            "variance": sum(torch.var(losses) for losses in per_group_loss)
            / len(per_group_loss),
            "sum": sum(losses.sum() for losses in per_group_loss) / len(per_group_loss),
            "mean": total_mean_loss,
        }

    def log_metrics_over_epoch(
        self,
        new_metrics: dict[str, float],
        running_metrics: dict[str, float],
        loss_factor=None,
    ) -> dict[str, float]:
        loss_factor = loss_factor if loss_factor else new_metrics["current_num_tokens"]
        for key, value in new_metrics.items():
            if key not in running_metrics:
                running_metrics[key] = torch.tensor(0.0, device=self._device)
            if key == "current_num_tokens":
                running_metrics[key] += value
            else:
                running_metrics[key] += value.detach().float() * loss_factor

        return running_metrics

    def calculate_total_metrics(
        self, metrics: dict[str, float], loss_factor=None
    ) -> dict[str, float]:
        loss_factor = loss_factor if loss_factor else metrics["current_num_tokens"]
        total_metrics = {}
        for key, value in metrics.items():
            if key == "current_num_tokens":
                total_metrics[key] = value
                continue
            total_metrics[key] = (
                value / loss_factor if loss_factor > 0 else float("inf")
            )
        return total_metrics

    def generate(self, batch, batch_idx, templates):
        """
        Generate while training to log EM and F1 scores.
        """
        # TODO: Do we need to handle prompts seperatly?
        # predictions = []
        # for prompt, label in zip(prompts, labels):
        #     prompt = prompt[prompt != 0]
        #     label = label[label != -100]
        #     # with self._device:
        #     #     self._model.setup_caches(
        #     #         batch_size=1,
        #     #         dtype=self._dtype,
        #     #         decoder_max_seq_len=prompt.numel() + label.shape[0],
        #     #     )
        #     predictions.append(
        #         generate(
        #             prompt=prompt,
        #             model=self._model,
        #             max_generated_tokens=label.shape[0],
        #             pad_id=0,
        #             temperature=0.6,
        #             top_k=300,
        #             stop_tokens=self._tokenizer.stop_tokens,
        #         )[0][:, prompt.shape[-1] :]
        #     )
        # pred_text = [self._tokenizer.decode(prediction.tolist()[0]) for prediction in predictions]

        # TODO: ValueError: KV-caches for self-attention layers are setup for inference mode, causal masks must be provided! Use the `mask` arg to provide a causal mask.
        # with self._device:
        #     self._model.setup_caches(
        #         batch_size=prompts.shape[0],
        #         dtype=self._dtype,
        #         decoder_max_seq_len=prompts.shape[1] + labels.shape[1],
        #     )

        prompts = batch["prompt"]
        labels = batch["generation_label"]

        predictions = generation.generate(
            prompt=prompts,
            model=self._model,
            max_generated_tokens=labels.shape[1],
            pad_id=self._tokenizer.pad_id,
            temperature=0.6,
            top_k=300,
            stop_tokens=self._tokenizer.stop_tokens,
        )[0]
        predictions = predictions[:, prompts.shape[-1] :]
        self.per_prompt_acc(predictions, labels, templates)
        # Decode labels and predictions
        pred_text = [
            self._tokenizer.decode(prediction, truncate_at_eos=True)
            for prediction in predictions.tolist()
        ]
        label_text = [
            self._tokenizer.decode(label, truncate_at_eos=True)
            for label in labels.tolist()
        ]
        # Get it into the squad format
        squad_label = [
            {"id": str(i), "answers": {"text": [text], "answer_start": [0]}}
            for i, text in enumerate(label_text)
        ]
        squad_pred = [
            {"id": str(i), "prediction_text": text} for i, text in enumerate(pred_text)
        ]
        self.squad.update(
            preds=squad_pred,
            target=squad_label,
        )
        self.rouge.update(
            preds=pred_text,
            target=label_text,
        )
        self.bleu.update(
            preds=pred_text,
            target=label_text,
        )

        if batch_idx == 0:
            prompts = [
                self._tokenizer.decode(prompt, truncate_at_eos=True)
                for prompt in prompts.tolist()
            ]
            self._logger.info(
                f"Generated example:\nPrompt:\n{prompts[0]}\nPred:{pred_text[0]}\nLabel:{label_text[0]}"
            )
            for i in range(batch["sample_count"][0].item()):
                self.individual_squad.reset()
                self.individual_squad.update(
                    preds=[squad_pred[i]], target=[squad_label[i]]
                )
                individual_metrics = self.individual_squad.compute()
                self.sample_table.add_data(
                    self.global_step,
                    prompts[i],
                    pred_text[i],
                    label_text[i],
                    individual_metrics["exact_match"],
                    individual_metrics["f1"],
                )

    def per_prompt_acc(self, predictions, labels, templates):
        per_sample_acc = predictions.eq(labels).float().sum(dim=1) / labels.ne(
            -100
        ).float().sum(dim=1)

        for prompt_id in templates:
            if prompt_id not in self.acc_per_prompt:
                self.acc_per_prompt[prompt_id] = []
            self.acc_per_prompt[prompt_id].append(per_sample_acc.mean().item())


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = FullFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    port = 56789
    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()
    sys.exit(recipe_main())
