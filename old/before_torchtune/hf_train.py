from transformers import TrainingArguments, set_seed
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.configuration_auto import AutoConfig

import torch
import os

from simple_parsing import parse
from print_on_steroids import logger

from hf_test.data_loading import CustomDataLoader

# from custom_trainer import CustomTrainer
from transformers import Trainer
from args import TrainingArgs
from dlib import CUDAMetricsCallback, WandbCleanupDiskAndCloudSpaceCallback, get_rank, log_slurm_info, wait_for_debugger

WANDB_PROJECT = "mini"
WANDB_ENTITY = "transformersclub"


def main(args: TrainingArgs):

    if args.offline or args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"
    current_process_rank = get_rank()
    if current_process_rank == 0 and args.debug:
        wait_for_debugger()

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
    # ==== Set up ====
    if args.accelerator == "cuda":
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus > args.num_devices:
            logger.warning(
                f"Requested {args.num_devices} GPUs but {num_available_gpus} are available.",
                f"Using first {args.num_devices} GPUs. You should set CUDA_VISIBLE_DEVICES or the docker --gpus flag to the desired GPU ids.",
            )
        if not torch.cuda.is_available():
            logger.error("CUDA is not available, you should change the accelerator with --accelerator cpu|tpu|mps.")
            exit(1)

    set_seed(args.seed)

    # ==== Load dataset ====
    data_loader = CustomDataLoader(args)
    datasets = data_loader.load_datasets()
    collate = data_loader.get_collator()

    # ==== Training script ====
    config = AutoConfig.from_pretrained(args.hf_model_name, return_dict=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, config=config)
    training_args = TrainingArguments(
        max_steps=args.training_goal,
        eval_steps=args.eval_interval,
        save_steps=args.save_interval,
        warmup_steps=args.warmup_period,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=args.precision == "bf16-mixed",
        fp16=args.precision == "16-mixed",
        dataloader_num_workers=args.workers,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="wandb",
        run_name=args.run_name,
        remove_unused_columns=False,
        group_by_length=False,
        eval_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collate,
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main(parse(TrainingArgs, add_config_path_arg=True))
