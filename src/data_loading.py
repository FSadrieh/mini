import errno
import glob
import os
import shutil
import tempfile
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import itertools
from typing import Iterator, Tuple
from torch.utils import data
import torch

from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection

import datasets
import lightning as L
from print_on_steroids import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs

T0_HELDOUT_TASKS = ['copa', 'hellaswag']#, 'cb', 'rte', 'wsc', 'winogrande', 'wic']


class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.args = training_args

        self.tokenizer_path = self.args.tokenizer_path or self.args.hf_model_name
        self.local_rank = get_rank()

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.loader = PromptLoader()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage):
        train_loader = self.loader.iterate_prompts(split="train")
        val_loader = self.loader.iterate_prompts(split="validation")
        self.train_dataset = IterDataset(train_loader)
        self.val_dataset = IterDataset(val_loader)
        self.data_collator = self.collate

    def train_dataloader(self):
        common_args = dict(
            batch_size=self.args.micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.eval_micro_batch_size,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.val_dataset, collate_fn=self.data_collator, **common_args)

    def collate(self, examples):
        tokenized_inputs = []
        tokenized_labels = []
        for input_list, label_list in examples:
            tokenized_inputs.append(self.tokenizer(input_list, padding=False)["input_ids"])
            tokenized_labels.append(self.tokenizer(label_list, padding=False, add_special_tokens=False)["input_ids"])

        batch_size = len(tokenized_inputs)
        prompt_examples = len(tokenized_inputs[0])

        # Create a list of tensors the list has length prompt_examples and every tensor has length batch_size
        input_ids = []
        labels = []
        for j in range(prompt_examples):
            input_list = [torch.tensor(tokenized_inputs[i][j] + tokenized_labels[i][j] + [self.tokenizer.eos_token_id]) for i in range(batch_size)]
            input_len = torch.tensor([len(tokenized_inputs[i][j]) for i in range(batch_size)])
            max_len = max(len(t) for t in input_list)

            input_id = torch.stack([
                    torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=self.tokenizer.pad_token_id)
                    for t in input_list
                ])
            
            label = input_id.clone()
            mask = torch.arange(max_len).unsqueeze(0) >= input_len.unsqueeze(1)
            masked_label = -100 * ~mask + label * mask
            masked_label[masked_label == self.tokenizer.pad_token_id] = -100

            labels.append(masked_label)
            input_ids.append(input_id)

        attention_mask = [
            torch.where(
                input_ids[i] != self.tokenizer.pad_token_id,
                torch.tensor(1),
                torch.tensor(0),
            )
            for i in range(prompt_examples)
        ]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_masks": attention_mask,
        }


class PromptLoader:
    def __init__(self):
        self.collection = TemplateCollection()


    def iterate_prompts(self, split: str = "train") -> Iterator[Tuple[str, str]]:
        for dataset_ids, templates in self.collection.datasets_templates.items():
            dataset_id, subset = dataset_ids
            if split == "train" and dataset_id in T0_HELDOUT_TASKS:
                continue
            elif split != "train" and dataset_id not in T0_HELDOUT_TASKS:
                continue
            dataset = load_dataset(dataset_id, subset, split=split)
            for sample in dataset:
                inputs, labels = [], []
                for template_id, template in templates.templates.items():
                    input_prompt, label = template.apply(sample)
                    inputs.append(input_prompt)
                    labels.append(label)

                yield inputs, labels

class IterDataset(data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator