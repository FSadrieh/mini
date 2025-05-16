import os
from typing import TYPE_CHECKING
from tqdm import tqdm

import itertools
from typing import Iterator, Tuple
from torch.utils import data
import torch

from datasets import load_dataset, Dataset, load_from_disk
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection

import datasets
import lightning as L
from print_on_steroids import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast, AutoTokenizer

from dlib.frameworks.pytorch import get_rank
# from src.customize_dataset import _iter_pytorch
# # necessary since the original implementation of the dataloader iter function in combination with huggingface datasets does not retrieve contiguous data from memory
# setattr(datasets.IterableDataset, "_iter_pytorch", _iter_pytorch)

if TYPE_CHECKING:
    from train import TrainingArgs

T0_HELDOUT_TASKS = ['copa', 'hellaswag', 'cb', 'rte', 'wsc', 'winogrande', 'wic']

class CustomDataLoader():
    def __init__(
        self,
        args: "TrainingArgs",
        ):
        super().__init__()
        self.args = args

        self.tokenizer_path = self.args.tokenizer_path or self.args.hf_model_name
        self.local_rank = get_rank()

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.hf_model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenized_data_path = f"{self.args.data_location}/tokenized_{self.tokenizer_path.replace('/', '_')}_{self.args.firstn_datasets}"

        self.loader = PromptLoader(self.tokenizer, args.seed, self.tokenized_data_path, self.args.firstn_datasets)

    def load_datasets(self) -> None:
        splits = ["train", "validation"]
        datasets = {}
        for split in splits:
            try:
                datasets[split] = load_from_disk(self.processed_data_path(split))
                logger.info("Data already prepared, skipping preparation.")
            except Exception as e:
                datasets[split] = self.preprocess_data(split)
                logger.info("Data not prepared, preparing data.")

        return datasets

    def processed_data_path(self, split: str) -> str:
        base_path = f"{self.args.data_location}/processed_{self.tokenizer_path.replace('/', '_')}_{self.args.firstn_datasets}"
        batch_size = self.args.micro_batch_size if split == "train" else self.args.eval_micro_batch_size
        return os.path.join(f'{base_path}_{batch_size}', split)

    def preprocess_data(self, split: str) -> None:
        if not os.path.isdir(self.processed_data_path(split)):
            os.makedirs(self.processed_data_path(split))

        try:
            dataset = Dataset.load_from_disk(self.tokenized_data_path + f"/{split}/")
        except OSError as e:
            logger.info(f"Creating tokenized dataset for {split}...")
            dataset = self.loader.iterate_prompts(split=split)

        batch_size = self.args.micro_batch_size if split == "train" else self.args.eval_micro_batch_size

        dataset = dataset.map(
                make_preprocess_function(batch_size, self.tokenizer),
                batched=True,
                batch_size=16_000,
                remove_columns=dataset.column_names,
                num_proc=self.args.preprocessing_workers,
                desc="Preprocessing dataset",
            )

        dataset.save_to_disk(self.processed_data_path(split))

        logger.info(f"Saved {split} dataset to {self.processed_data_path(split)}")

        return dataset
    def get_collator(self):
        def collate(examples):
            input_ids = [torch.tensor(example["batches"]["input_ids"]) for example in examples]
            labels = [torch.tensor(example["batches"]["labels"]) for example in examples]
            attention_masks = [torch.tensor(example["batches"]["attention_masks"]) for example in examples]

            return {
                "input_ids": torch.tensor([example["batches"]["input_ids"] for example in examples]).squeeze(0),
                "labels": torch.tensor([example["batches"]["labels"] for example in examples]).squeeze(0),
                "attention_masks": torch.tensor([example["batches"]["attention_masks"] for example in examples]).squeeze(0),
                # "sample_count": examples["batches"]["sample_count"].tolist(),
            }
        return collate


def make_tokenize_function(tokenizer: PreTrainedTokenizerFast):
    def tokenize_function(examples):
        tokenized_inputs = []
        tokenized_labels = []
        for input_list in examples["inputs"]:
            tokenized_inputs.append(tokenizer(input_list, padding=False)["input_ids"])

        for label_list in examples["labels"]:
            tokenized_labels.append(tokenizer(label_list, padding=False, add_special_tokens=False)["input_ids"])

        return {
            "input_ids": tokenized_inputs,
            "labels": tokenized_labels,
        }
    return tokenize_function

def make_preprocess_function(batch_size:int, tokenizer: PreTrainedTokenizerFast):
    def preprocess_function(examples):
        batches = []
        batch_input_ids = []
        batch_lengths = []
        batch_counter = []

        input_ids = examples["input_ids"]
        for input_ids, labels in zip(input_ids, examples["labels"]):
            # If we have more examples for a give input than micro batch size we can not leave them in one batch
            if len(input_ids) > batch_size:
                logger.error(f"Examples per sample {len(input_ids)} is greater than micro batch size {batch_size}.")
            # If the next example would fill the batch up too much we close the batch
            if len(batch_input_ids) + len(input_ids) > batch_size:
                # We find the maximum length in the batch and pad the input
                max_len = max(len(t) for t in batch_input_ids)
                input_ids_tensor = torch.stack([
                    torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=tokenizer.pad_token_id)
                    for t in batch_input_ids
                ])

                # The labels are -100 for the input ids and the padding tokens only the labels are kept
                label = input_ids_tensor.clone()
                mask = torch.arange(max_len).unsqueeze(0) >= torch.tensor(batch_lengths).unsqueeze(1)
                masked_label = -100 * ~mask + label * mask
                masked_label[masked_label == tokenizer.pad_token_id] = -100
                #TODO: What happens to the eos token it will also be masked
                attention_mask = torch.where(
                    input_ids_tensor != tokenizer.pad_token_id,
                    torch.tensor(1),
                    torch.tensor(0),
                )
                batches.append(
                    {
                        "input_ids": input_ids_tensor,
                        "labels": masked_label,
                        "attention_masks": attention_mask,
                        "sample_count": batch_counter,
                    }
                )
                batch_input_ids = []
                batch_lengths = []
                batch_counter = []

            for example_input_ids, example_labels in zip(input_ids, labels):
                # We append each example to the batch and store from which sample it comes from
                batch_input_ids.append(torch.tensor(example_input_ids + example_labels + [tokenizer.eos_token_id]))
                batch_lengths.append(torch.tensor(len(example_input_ids)))

            batch_counter.append(len(input_ids))

        return {"batches": batches}

    return preprocess_function


class PromptLoader:
    def __init__(self, tokenizer, seed, tokenized_data_path, firstn_datasets: int,  # set for 0 when using all datasets
                 ):
        self.collection = TemplateCollection()
        self.firstn_datasets = firstn_datasets
        self.tokenizer = tokenizer
        self.seed = seed
        self.tokenized_data_path = tokenized_data_path

    def iterate_prompts(self, split: str = "train") -> Iterator[Tuple[str, str]]:
        datasets_iterator = self.collection.datasets_templates.items()
        if self.firstn_datasets:
            print("Limiting the iterated datasets to first %s ones" % self.firstn_datasets)
            datasets_iterator = itertools.islice(self.collection.datasets_templates.items(), self.firstn_datasets)

        sample_inputs = []
        sample_labels = []
        dataset_list = []
        for dataset_ids, templates in tqdm(datasets_iterator, desc="Iterating over datasets"):
            dataset_id, subset = dataset_ids
            dataset_path = os.path.join(self.tokenized_data_path, dataset_id, subset) if subset else os.path.join(self.tokenized_data_path, dataset_id)
            #TODO: DEBUG
            if dataset_id in ["tydiqa", "duorc", "multi_news"]:
                logger.info(f"Skipping {dataset_id} dataset, because Problems with it.")
                continue
            if split == "train" and dataset_id in T0_HELDOUT_TASKS:
                    continue
            elif split != "train" and dataset_id not in T0_HELDOUT_TASKS:
                continue
            try:
                logger.info(f"Processing {dataset_id}...")
                dataset = Dataset.load_from_disk(dataset_path)
                logger.info(f"Loaded {dataset_id} dataset from disk.")
                dataset_list.append(dataset)
            except OSError as e:
                try:
                    dataset = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
                except (ValueError, TypeError) as e:
                    dataset = load_dataset(dataset_id, subset, split=split)
                for sample in tqdm(dataset, desc=f"Processing {dataset_id}"):
                    inputs, labels = [], []
                    for template_id, template in templates.templates.items():
                        input_prompt, label = template.apply(sample)
                        inputs.append(input_prompt)
                        labels.append(label)

                    sample_inputs.append(self.tokenizer(inputs, padding=False)["input_ids"])
                    sample_labels.append(self.tokenizer(labels, padding=False, add_special_tokens=False)["input_ids"])
                dataset = Dataset.from_dict({"input_ids": sample_inputs, "labels": sample_labels})
                dataset.save_to_disk(dataset_path)
                dataset_list.append(dataset)

        full_dataset = datasets.concatenate_datasets(dataset_list)
        full_dataset = full_dataset.shuffle(seed=self.seed)
        full_dataset.save_to_disk(os.path.join(self.tokenized_data_path, split))
        return full_dataset