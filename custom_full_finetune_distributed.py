# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import debugpy

from functools import partial
from typing import Any, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.optim import Optimizer
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtune import config, modules, training, utils
from torchtune.modules.embedding_utils import resize_token_embeddings
from torchtune.modules.loss import SFTLoss
from torchtune.modules.moe import utils as moe_utils
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import (
    DummyProfiler,
    PROFILER_KEY,
    VALID_BACKENDS_FOR_MEMORY_STATS,
)
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.training.checkpointing._checkpoint_client import (
    CheckpointClient,
    TrainingProgress,
)
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.quantization import (
    convert_to_float8_training,
    is_fp8_tensorwise_scaling,
)

from tqdm import tqdm
import wandb
import datetime
from torchmetrics import Accuracy
from torchmetrics.text import ROUGEScore, BLEUScore, SQuAD, Perplexity

from data_loading import CustomDataLoader

from generate import InferenceRecipe


class FullFinetuneRecipeDistributed(FTRecipeInterface):
    """
    Full finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

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

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

            For example: with batch_size=1, nproc_per_node=2 and gradient_accumulation_steps=32 we get a
            total batch size of 64.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Optimizer state and recipe state (seed, total_epochs, number of epochs run etc) are
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
        RuntimeError: If ``left_pad_sequence`` is set as the data collator.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        device_type = cfg.device
        self._device = utils.get_device(device=device_type)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self._enable_async_checkpointing = cfg.get("enable_async_checkpointing", False)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            device_type,
            offload_ops_to_cpu=self.fsdp_cpu_offload
            or self._enable_async_checkpointing,
        )
        init_process_group(self.distributed_backend)

        # Initialize distributed variables
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self.tp_plan = cfg.get("tensor_parallel_plan", None)
        self.tp_degree = cfg.get("tensor_parallel_dim", 1)
        if self.tp_degree > 1 and self.tp_plan is None:
            raise ValueError(
                "Tensor Parallel plan needs to be provided when tensor parallel is enabled."
            )
        if self.tp_degree > 1:
            # DTensor does not support grouped_mm yet
            moe_utils.use_grouped_mm = False
        self.cp_degree = cfg.get("context_parallel_dim", 1)
        data_shard = cfg.get("data_parallel_shard_dim", -1)  # -1 means to infer
        data_replicate = cfg.get("data_parallel_replicate_dim", 1)

        # Set up n-d device mesh
        self.parallel_dims = training.ParallelDims(
            dp_replicate=data_replicate,
            dp_shard=data_shard,
            tp=self.tp_degree,
            cp=self.cp_degree,
            world_size=self.world_size,
        )
        self.world_mesh = self.parallel_dims.build_mesh(device_type=device_type)
        if self.parallel_dims.dp_enabled:
            dp_mesh = self.world_mesh["dp"]
            self.dp_degree, self.dp_rank = (
                dp_mesh.size(),
                dp_mesh.get_local_rank(),
            )
        else:
            self.dp_degree, self.dp_rank = 1, 0

        # Add timestamp to checkpoint
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cfg.checkpointer.output_dir = cfg.checkpointer.output_dir + f"_{timestamp}"
            cfg.output_dir = cfg.output_dir + f"_{timestamp}"
        except AttributeError:
            pass

        # Logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)
        if (
            self._log_peak_memory_stats
            and self._device.type not in VALID_BACKENDS_FOR_MEMORY_STATS
        ):
            self._logger.info(
                f"log_peak_memory_stats was set to True; however, training device is not in {VALID_BACKENDS_FOR_MEMORY_STATS}."
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._optimizer_in_bwd = cfg.get("optimizer_in_bwd", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._checkpoint_client = CheckpointClient(cfg)
        self._enable_fp8_training = cfg.get("enable_fp8_training", False)
        self._fp8_recipe_name = cfg.get("fp8_recipe_name", None)

        # Optimizer in backward is not compatible with gradient accumulation or gradient clipping
        if self._optimizer_in_bwd:
            if self._clip_grad_norm is not None:
                raise RuntimeError(
                    "Gradient clipping is not supported with optimizer in bwd."
                    "Please set clip_grad_norm=None, or optimizer_in_bwd=False."
                )
            if self._gradient_accumulation_steps > 1:
                raise RuntimeError(
                    "Gradient accumulation is not supported with optimizer in bwd."
                    "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
                )

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        self._activation_offloading_use_streams = cfg.get(
            "activation_offloading_use_streams", True
        )
        if (
            self._enable_activation_offloading
            and self._activation_offloading_use_streams
            and self.parallel_dims.tp_enabled
        ):
            warn(
                message=(
                    "Using activation offloading with streams is not advised in tensor parallel, and may "
                    "cause unstable training. It is advised to set activation_offloading_use_streams: False"
                )
            )
        if self._enable_activation_offloading:
            if device_type != "cuda":
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
        self.acc = None
        self.squad = SQuAD(
            dist_sync_on_step=True,
        ).to(self._device)
        self.individual_squad = SQuAD(
            dist_sync_on_step=True,
        ).to(self._device)
        self.rouge = ROUGEScore(
            dist_sync_on_step=True,
        ).to(self._device)
        self.bleu = BLEUScore(
            dist_sync_on_step=True,
        ).to(self._device)
        self.ppl = Perplexity(
            ignore_index=-100, dist_sync_on_step=False, sync_on_compute=False
        ).to(self._device)

        self.custom_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.generator = InferenceRecipe(
            cfg=cfg,
            device=self._device,
            dtype=self._dtype,
            seed=self.seed,
        )
        self.current_checkpoint_dir = cfg.checkpointer.checkpoint_dir

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
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        if self.fsdp_cpu_offload:
            # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
            # speed up when benchmarking fused AdamW on CPU
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            # log config with parameter override
            self._metric_logger.log_config(cfg)

        # Load the base model
        checkpoint_dict = self._checkpoint_client.load_base_checkpoint()

        compile = cfg.get("compile")
        compile_bool = bool(compile)
        self._compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

        self._compile_model = compile_bool
        self._compile_loss = compile_bool
        self._compile_optimizer_step = compile_bool
        self._compile_scale_grads = compile_bool
        if isinstance(compile, DictConfig):
            self._compile_model = compile.get("model", True)
            self._compile_loss = compile.get("loss", True)
            self._compile_optimizer_step = compile.get("optimizer_step", False)
            self._compile_scale_grads = compile.get("scale_grads", True)
        if self._compile_model:
            # Capture scalar outputs is required to compile MoE
            torch._dynamo.config.capture_scalar_outputs = True

        # This indirection is needed to apply torch.compile to scale_grads step.
        self._grad_scaler = training.scale_grads_
        if self._compile_scale_grads:
            self._grad_scaler = torch.compile(
                self._grad_scaler, backend=self._compile_backend
            )

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            activation_offloading_use_streams=self._activation_offloading_use_streams,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

        if cfg.get("resize_token_embeddings", False):
            resize_token_embeddings(self._model, self._tokenizer.vocab_size)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            optimizer_in_bwd=self._optimizer_in_bwd,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if training.OPT_KEY in checkpoint_dict
                else None
            ),
        )
        if self._compile_optimizer_step:
            if self._optimizer_in_bwd:
                raise ValueError(
                    "optimizer_in_bwd not supported with compiling the optimizer step"
                )
            self._optimizer.step = torch.compile(
                self._optimizer.step,
                backend=self._compile_backend,
            )

        if self._resume_from_checkpoint:
            # If async checkpointing is enabled, intermediate checkpoints are saved asynchronously
            # using the DistributedCheckpointer.
            # Therefore the recipe needs to load the distributed checkpoint to restore the training
            # progress.
            if self._enable_async_checkpointing:
                try:
                    checkpoint_dict = (
                        self._checkpoint_client.load_distributed_checkpoint(
                            self._model,
                            (
                                self._optim_ckpt_wrapper
                                if self._optimizer_in_bwd
                                else self._optimizer
                            ),
                        )
                    )
                except Exception as e:
                    self._logger.warning(
                        f"Failed to load distributed checkpoint: {e}. Training will start from the base checkpoint."
                    )

            # Update the recipe state from the checkpoint state dict.
            self._update_recipe_state(checkpoint_dict)

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if isinstance(self._loss_fn, SFTLoss):
            self._loss_fn.set_model_output(self._model)

        if self._compile_loss:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        utils.log_rank_zero(self._logger, "Loss is initialized.")

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

        # Setup validation dataloader if validation dataset is provided
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

        # Setup lr scheduler
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        self.acc = Accuracy(
            task="multiclass",
            num_classes=self._model.tok_embeddings.weight.shape[0],
            ignore_index=-100,
            dist_sync_on_step=True,
        ).to(self._device)

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        if self._optimizer_in_bwd:
            # Use the first optimizer from the wrapper to represent the learning rate
            optimizer = next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            # Standard case: use the single optimizer
            optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._optimizer_in_bwd:
            # Modify the scheduler for optimizer_in_bwd case
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

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

        utils.log_rank_zero(
            self._logger, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
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
        activation_offloading_use_streams: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: dict[str, Any],
        custom_sharded_layers: Optional[list[str]] = None,
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        utils.log_rank_zero(
            self._logger,
            "Distributed training is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if self._compile_model:
            training.compile_model(model, verbose=self._is_rank_zero)

        if self._enable_fp8_training:
            # Requires https://github.com/pytorch/pytorch/pull/148922
            if torch.__version__ < "2.8.0.dev20250318":
                raise RuntimeError(
                    "Float8 fine-tuning requires PyTorch 2.8.0.dev20250318 or later."
                )
            if self.tp_plan is not None:
                raise ValueError(
                    "FP8 training does not support tensor parallelism yet. "
                    "This will be enabled in the near future."
                )
            if self.cp_degree > 1:
                raise ValueError(
                    "Context Parallel for fp8 training is not currently supported"
                )
            model = convert_to_float8_training(model, self._fp8_recipe_name)

        # Apply tensor parallelism to the model
        if self.parallel_dims.tp_enabled:
            if not self.parallel_dims.dp_enabled and self.fsdp_cpu_offload:
                raise ValueError(
                    "Tensor parallelism is not supported with FSDP CPU offloading when data parallelism is disabled."
                )
            # Use the local number (num_heads, num_kv_heads, embed_dim) to account for tensor parallel
            model = training.prepare_mha_for_tp(model, self.world_mesh["tp"])
            if self.tp_plan is not None:
                self.tp_plan = config.instantiate(
                    self.tp_plan,
                    model=model,
                )
            parallelize_module(
                model,
                self.world_mesh["tp"],
                parallelize_plan=self.tp_plan,
            )

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # Apply Fully Sharded Data Parallelism to the model
        if self.parallel_dims.dp_shard_enabled:
            fsdp_shard_conditions = [
                partial(
                    training.get_shard_conditions,
                    names_to_match=custom_sharded_layers,
                )
            ]

            if self.parallel_dims.dp_replicate_enabled:
                dp_mesh_dim_names = ("dp_replicate", "dp_shard")
            else:
                dp_mesh_dim_names = ("dp_shard",)

            training.shard_model(
                model=model,
                shard_conditions=fsdp_shard_conditions,
                cpu_offload=fsdp_cpu_offload,
                reshard_after_forward=reshard_after_forward,
                dp_mesh=self.world_mesh[dp_mesh_dim_names],
            )

        # Define context manager for context parallelism
        self.context_parallel_manager = training.get_context_parallel_manager(
            enabled=self.cp_degree > 1,
            world_mesh=self.world_mesh,
            model=model,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading, activation_offloading_use_streams
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero and self._device.type != "cpu":
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        if self._device.type != "cpu":
            torch.distributed.barrier(device_ids=[self._device.index])

        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        if optimizer_in_bwd:
            # Maintain a dict of optims for every parameter.
            optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }

            # Register optimizer step hooks on the model to run optimizer in backward.
            training.register_optim_in_bwd_hooks(
                model=self._model, optim_dict=optim_dict
            )
            # Create a wrapper for checkpoint save/load of optimizer states when running in backward.
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=optim_dict
            )
            # Load optimizer states for each param. If optimizer states are being restored in an optimizer in
            # backward run, these need to have been saved with the same setting. Cannot restore from runs that
            # did not use optimizer in backward.
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.optim_map[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
            utils.log_rank_zero(self._logger, "In-backward optimizers are set up.")
            return None
        else:
            optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    optimizer,
                    opt_state_dict,
                    self._device,
                )

            utils.log_rank_zero(self._logger, "Optimizer is initialized.")
            return optimizer

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
        self._model.eval()
        for m in (self.acc, self.bleu, self.squad, self.rouge, self.ppl):
            m.reset()
        running_metrics = {}
        self.em_per_prompt = {}

        self.generator.setup(
            self.current_checkpoint_dir, self._tokenizer.pad_id
        )

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self._val_dataloader),
                disable=not self._is_rank_zero,
                desc="Validating",
                total=self._eval_batches if self._eval_batches > 0 else None,
            ):
                if batch_idx >= self._eval_batches and self._eval_batches > 0:
                    break

                templates = batch.pop("templates", None)
                utils.batch_to_device(batch, self._device)

                # Compute loss

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

                self.generate(batch, templates, batch_idx)

        # Aggregate validation metrics across all ranks
        metrics = self.calculate_total_metrics(running_metrics)
        squad_metrics = self.squad.compute()
        # Validation metrics
        log_dict = {
            "val/accuracy": self.acc.compute().item(),
            "val/loss": metrics["loss"],
            "val/mean_loss": metrics["mean"],
            "val/sum_loss": metrics["sum"],
            "val/distance_loss": metrics["distance"],
            "val/var_loss": metrics["variance"],
            "val/ppl": self.ppl.compute().item(),
        }

        # Generation metrics
        log_dict.update(
            {
                "val/exact_match": squad_metrics["exact_match"].item(),
                "val/f1": squad_metrics["f1"].item(),
                "val/bleu": self.bleu.compute().item(),
                "val/rougeL": self.rouge.compute()["rougeL_fmeasure"].item(),
            }
        )

        for prompt, em in self.em_per_prompt.items():
            em = torch.tensor(em)
            num_samples = em.shape[0]
            log_dict[f"em/{prompt}"] = (
                (em.sum() / num_samples) if num_samples > 0 else 0.0
            )

        self._logger.info(f"Validation loss: {metrics['loss']:.4f}")
        if self._is_rank_zero:
            self._logger.info(log_dict)
            self._metric_logger.log_dict(
                log_dict,
                step=self.global_step,
            )

        self._model.train()
        return log_dict

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        if not self._optimizer_in_bwd:
            self._optimizer.zero_grad()
        else:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        t0 = time.perf_counter()
        running_metrics = {}

        self._profiler.start()
        # Validate before training starts
        if (
            self._run_val_every_n_steps is not None
            and self.global_step % self._run_val_every_n_steps == 0
        ):
            self.validate()

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                utils.batch_to_device(batch, self._device)

                # Loss is normalized by default so we multiply by the number of tokens
                # This way we can normalize by the total number of tokens if we're accumulating gradients
                with self.context_parallel_manager(list(batch.values())):
                    metrics, __ = self._loss_step(batch)

                    # Calculate the number of unmasked tokens in the current batch
                    # and increment the total number of tokens seen in the step
                    metrics["current_num_tokens"] = (
                        batch["labels"] != self._loss_fn.ignore_index
                    ).sum()
                    current_loss = metrics["loss"] * metrics["current_num_tokens"]

                    running_metrics = self.log_metrics_over_epoch(
                        metrics,
                        running_metrics,
                    )

                    # For optimizer in backward, we need to normalize before calling backward
                    # This case and gradient accumulation are mutually exclusive
                    if self._optimizer_in_bwd:
                        torch.distributed.all_reduce(
                            running_metrics["current_num_tokens"]
                        )
                        torch.distributed.all_reduce(running_metrics["loss"])
                        current_loss = current_loss * (
                            self.dp_degree / metrics["current_num_tokens"]
                        )
                    current_loss.backward()

                # Optimizer step (if not fused in backward call)
                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    if not self._optimizer_in_bwd:
                        # Get total number of tokens across all ranks to normalize gradients
                        torch.distributed.all_reduce(
                            running_metrics["current_num_tokens"]
                        )
                        # This will ensure that the logged loss matches what we're optimizing
                        torch.distributed.all_reduce(running_metrics["loss"])

                        # Manually scale the gradients from unnormalized loss by total # of tokens
                        self._grad_scaler(
                            list(self._model.parameters()),
                            self.world_size / running_metrics["current_num_tokens"],
                            False if self.parallel_dims.tp_enabled else None,
                        )

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                            # If sharded, collect the DTensor here
                            if isinstance(grad_norm, DTensor):
                                grad_norm = grad_norm.full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)

                    # Update the number of steps when the weights are updated
                    self.global_step += 1

                    # Step the learning rate scheduler
                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # If float8 training is enabled, perform a single all-reduce to compute the
                    # scale for all float8 parameters efficiently instead of doing many small
                    # all-reduces for each parameter
                    if (
                        self._enable_fp8_training
                        and is_fp8_tensorwise_scaling(self._fp8_recipe_name)
                        and self.dp_degree > 1
                    ):
                        precompute_float8_dynamic_scale_for_fsdp(self._model)

                    total_metrics = self.calculate_total_metrics(running_metrics)
                    pbar.update(1)
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {total_metrics['loss']}"
                    )

                    # Log per-step metrics
                    if (
                        self.global_step % self._log_every_n_steps == 0
                        and self._is_rank_zero
                    ):
                        time_per_step = time.perf_counter() - t0
                        log_dict = {
                            "train/loss": total_metrics["loss"],
                            "train/mean_loss": total_metrics["mean"],
                            "train/sum_loss": total_metrics["sum"],
                            "train/distance_loss": total_metrics["distance"],
                            "train/var_loss": total_metrics["variance"],
                            "lr": get_lr(
                                (
                                    self._optimizer
                                    if not self._optimizer_in_bwd
                                    else self._optim_ckpt_wrapper
                                ),
                            ),
                            "tokens_per_second_per_gpu": (
                                total_metrics["current_num_tokens"]
                                / self.parallel_dims.non_data_parallel_size
                            )
                            / (time_per_step * self.world_size),
                        }
                        if self._log_peak_memory_stats:
                            log_dict.update(
                                training.get_memory_stats(device=self._device)
                            )
                        if self._clip_grad_norm is not None:
                            log_dict.update({"grad_norm": grad_norm})
                        self._metric_logger.log_dict(
                            log_dict,
                            step=self.global_step,
                        )

                    # Reset running stats for the next step
                    running_metrics = {}
                    t0 = time.perf_counter()

                    # Stop tracking CUDA memory now that active steps are complete
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx
                        == self.profiler_wait_steps
                        + self.profiler_warmup_steps
                        + self.profiler_active_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history(enabled=None)

                    # Step profiler
                    # Note that this is called within gradient accumulation block, hence
                    # will include multiple forward / backward passes if gradient accumulation > 1
                    self._profiler.step()

                    # Run validation after gradient update
                    if (
                        self._run_val_every_n_steps is not None
                        and self.global_step % self._run_val_every_n_steps == 0
                    ):
                        pbar.refresh()
                        self.save_checkpoint(curr_epoch, real_ckpt=False)
                        self.validate()

                if (
                    (idx + 1) // self._gradient_accumulation_steps
                ) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            self.save_checkpoint(curr_epoch)

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._logger.info("Training completed.")
            self._metric_logger.close()
        destroy_process_group()

    ### HELPER FUNCTIONS ###

    def save_checkpoint(self, curr_epoch: int, real_ckpt: bool = True) -> None:
        """
            Saves the current model and optimizer state to a checkpoint.
            Due to a hacky usage of the library we save the in between epoch checkpoints to 100 + max epochs. This makes saving ca. 10 seconds faster
        """
        epoch = curr_epoch if real_ckpt else self.total_epochs + 100
        self._checkpoint_client.save_checkpoint(
            model=self._model,
            optimizer=(
                self._optimizer
                if not self._optimizer_in_bwd
                else self._optim_ckpt_wrapper
            ),
            training_progress=TrainingProgress(
                seed=self.seed,
                epochs_run=self.epochs_run,
                total_epochs=self.total_epochs,
                max_steps_per_epoch=self.max_steps_per_epoch,
                dataloader_state_dict=self._dataloader.state_dict(),
            ),
            epoch=epoch,
        )
        # After saving the checkpoint, we can safely update the current checkpoint directory
        self.current_checkpoint_dir = self._output_dir + f"/epoch_{epoch}"

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
            "variance": sum(
                torch.var(losses, unbiased=False) for losses in per_group_loss
            )
            / len(per_group_loss),
            "sum": sum(losses.sum() for losses in per_group_loss) / len(per_group_loss),
            "mean": total_mean_loss,
        }

    def log_metrics_over_epoch(
        self,
        new_metrics: dict[str, float],
        running_metrics: dict[str, float],
    ) -> dict[str, float]:
        for key, value in new_metrics.items():
            if key not in running_metrics:
                running_metrics[key] = torch.tensor(0.0, device=self._device)
            if key == "current_num_tokens":
                running_metrics[key] += value
            else:
                running_metrics[key] += (
                    value.detach().float() * new_metrics["current_num_tokens"]
                )

        return running_metrics

    def calculate_total_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        total_metrics = {}
        for key, value in metrics.items():
            if key == "current_num_tokens":
                total_metrics[key] = value
                continue
            total_metrics[key] = (
                value / metrics["current_num_tokens"]
                if metrics["current_num_tokens"] > 0
                else float("inf")
            )
        return total_metrics

    def generate(self, batch, templates, batch_idx):
        """
        Generate while training to log EM and F1 scores.
        """
        prompts = batch["prompt"]
        labels = batch["generation_label"]
        attention_mask = prompts.ne(self._tokenizer.pad_id).long()

        predictions = self.generator.generate(
            prompts=prompts,
            attention_mask=attention_mask,
            max_len=labels.shape[1],
        )

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
        self.get_em_per_prompt(squad_pred, squad_label, templates)

        self.squad.update(preds=squad_pred, target=squad_label)
        self.bleu.update(preds=pred_text, target=label_text)
        self.rouge.update(preds=pred_text, target=label_text)

        if batch_idx == 0:
            self.populate_sample_table(
                prompts,
                pred_text,
                label_text,
                squad_pred,
                squad_label,
                templates,
            )

    def get_em_per_prompt(
        self, squad_pred: list[str], squad_label: list[str], templates: list[str]
    ):
        for i, prompt_id in enumerate(templates):
            self.individual_squad.reset()
            self.individual_squad.update(preds=[squad_pred[i]], target=[squad_label[i]])
            if prompt_id not in self.em_per_prompt:
                self.em_per_prompt[prompt_id] = []
            self.em_per_prompt[prompt_id].append(
                self.individual_squad.compute()["exact_match"].item()
            )

    def populate_sample_table(
        self,
        prompts: torch.Tensor,
        pred_text: list[str],
        label_text: list[str],
        squad_pred: list[str],
        squad_label: list[str],
        templates: list[str],
    ):
        prompts = [
            self._tokenizer.decode(prompt, truncate_at_eos=True)
            for prompt in prompts.tolist()
        ]
        self._logger.info(
            f"Generated example:\nPrompt:\n{prompts[0]}\nPred:{pred_text[0]}\nLabel:{label_text[0]}"
        )
        ems, f1s = [], []
        for i in range(len(prompts)):
            self.individual_squad.reset()
            self.individual_squad.update(preds=[squad_pred[i]], target=[squad_label[i]])
            individual_metrics = self.individual_squad.compute()
            ems.append(individual_metrics["exact_match"].item())
            f1s.append(individual_metrics["f1"].item())

        timestep = [self.global_step] * len(pred_text)
        sample_table = wandb.Table(
            columns=["timestep", "template", "prompt", "pred", "label", "em", "f1"],
            data=list(
                zip(
                    timestep,
                    templates,
                    prompts,
                    pred_text,
                    label_text,
                    ems,
                    f1s,
                )
            ),
        )
        if self._is_rank_zero:
            self._metric_logger.log_dict(
                    {"sample_table": sample_table},
                    step=self.global_step,
                )


def get_rank() -> int:
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        # LOCAL_RANK from pytorch-lightning
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)
    recipe = FullFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    current_process_rank = get_rank()
    # port = 56789
    # if current_process_rank == 0:
    #     debugpy.listen(("0.0.0.0", port))
    #     print(
    #         f"Waiting for client to attach on port {port}... NOTE: if using docker, you need to forward the port with -p {port}:{port}."
    #     )
    #     debugpy.wait_for_client()
    sys.exit(recipe_main())
