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
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from dlib.frameworks.pytorch import get_rank

# from src.customize_dataset import _iter_pytorch
# # necessary since the original implementation of the dataloader iter function in combination with huggingface datasets does not retrieve contiguous data from memory
# setattr(datasets.IterableDataset, "_iter_pytorch", _iter_pytorch)

if TYPE_CHECKING:
    from train import TrainingArgs

T0_HELDOUT_TASKS = ["copa", "hellaswag", "cb", "rte", "wsc", "winogrande", "wic"]

BAD_TASKS = [
    "tydiqa",
    "duorc",
    "multi_news",
    "amazon_us_reviews",
    "gigaword",
    "amazon_reviews_multi",
    "mdd",
    "story_cloze",
    "xquad_r",
    "emo",
    "fever",
    "paws",
    "paws",
    "paws",
    "quora",
    "acronym_identification",
    "paws-x",
    "anli",
    "jfleg",
    "circa",
    "xquad",
    "mc_taco",
    "wino_bias",
    "winograd_wsc",
    "openai_humaneval",
    "tab_fact",
    "climate_fever",
    "jigsaw_unintended_bias",
    "liar",
    "squadshifts",
    "glue",
    "conv_ai_3",
    "asnq",
    "wiki_qa",
    "great_code",
    "squad_adversarial",
    "docred",
    "crows_pairs",
    "wiki_hop",
    "super_glue",
    "conv_ai",
    "asset",
    "trec",
    "conv_ai_2",
    "craffel/openai_lambada",
    "discofuse",
    "turk",
    "e2e_nlg_cleaned",
]


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

        self.tokenized_data_path = (
            f"{self.args.data_location}/tokenized_{self.tokenizer_path.replace('/', '_')}_{self.args.firstn_datasets}"
        )

        self.loader = PromptLoader(tokenizer, training_args.seed, self.tokenized_data_path, self.args.firstn_datasets)

    def processed_data_path(self, split: str) -> str:
        base_path = f"{self.args.data_location}/processed_{self.tokenizer_path.replace('/', '_')}_{self.args.firstn_datasets}"
        batch_size = self.args.micro_batch_size if split == "train" else self.args.eval_micro_batch_size
        return os.path.join(f"{base_path}_{batch_size}", split)

    def prepare_data(self) -> None:
        splits = ["train", "validation"]
        if self.data_is_prepared(splits):
            logger.info("Data already prepared, skipping preparation.")
        else:
            logger.info("Data not prepared, preparing data.")
            for split in splits:
                self.preprocess_data(split)

    def data_is_prepared(self, splits: list[str]) -> bool:
        try:
            self.load_dataset(splits)
            return True
        except Exception as e:
            logger.info(f"Data not prepared: {e}")
            return False

    def load_dataset(self, splits) -> None:
        data_files_dict = {}
        for split in splits:
            logger.info("loading from datasets file", self.processed_data_path(split))
            # gather all data files in a dict if file ends in .arrow
            data_files_dict[split] = [
                os.path.join(self.processed_data_path(split), file)
                for file in os.listdir(self.processed_data_path(split))
                if file.endswith(".arrow")
            ]
        self.dataset_per_split = load_dataset("arrow", data_files=data_files_dict).with_format("torch")

        logger.info("Loaded dataset...")

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

    def setup(self, stage):
        splits = ["train", "validation"]
        self.dataset = self.load_dataset(splits)
        self.data_collator = self.collate

    def train_dataloader(self):
        common_args = dict(
            batch_size=1,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            shuffle=False,
        )
        return DataLoader(self.dataset_per_split["train"], collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=1,
            num_workers=self.args.workers,
            persistent_workers=(
                True if self.args.workers > 0 else False
            ),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
        )
        return DataLoader(self.dataset_per_split["validation"], collate_fn=self.data_collator, **common_args)

    def collate(self, examples):
        return {
            "input_ids": examples[0]["batches"]["input_ids"],
            "labels": examples[0]["batches"]["labels"],
            "attention_masks": examples[0]["batches"]["attention_masks"],
            "sample_count": examples[0]["batches"]["sample_count"].tolist(),
            "generation_input_ids": examples[0]["batches"]["generation_input_ids"],
            "generation_labels": examples[0]["batches"]["generation_labels"],
            "generation_attention_masks": examples[0]["batches"]["generation_attention_masks"],
        }


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


def make_preprocess_function(batch_size: int, tokenizer: PreTrainedTokenizerFast):
    """
    Return the preprocessing function for the dataset.
    As input we have a tokenized lists for each sample (a sample is one general question answer pair). The list contain varying amount of examples (a sample is a concrete prompt for a question answer pair).
    We want to batch the model input guaranteeing that all examples from a sample are in the same batch.
    Therefore, we greedily fill the batch until all examples from the next sample would not fit.
    For training we need the input_ids containing the prompt, the label, the eos token and maybe right padding; the labels (same size as the input_ids), but containing a -100 everywhere except for the label part;
    the attention mask 0 for padding; and the sample count (how many examples for each sample are in the batch).
    For generation we need left padded input_ids, the raw labels and the attention mask.
    """

    def padding_helper(to_pad: list[torch.tensor], pad_right=True) -> Tuple[torch.tensor, int]:
        max_len = max(len(t) for t in to_pad)
        if pad_right:
            padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=tokenizer.pad_token_id) for t in to_pad]
        else:
            padded = [torch.nn.functional.pad(t, (max_len - t.shape[0], 0), value=tokenizer.pad_token_id) for t in to_pad]
        return torch.stack(padded), max_len

    def preprocess_function(examples):
        batches = []
        batch_generation_input_ids = []
        batch_generation_labels = []
        batch_training_input_ids = []
        batch_lengths = []
        batch_counter = []

        # TODO: DEBUG renamed input_ids
        try:
            input_ids = examples["input_ids"]
        except KeyError:
            input_ids = examples["inputs"]

        for input_examples, label_examples in zip(input_ids, examples["labels"]):
            # If we have more examples for a give input than micro batch size we can not leave them in one batch
            if len(input_examples) > batch_size:
                raise ValueError(f"Examples per sample {len(input_examples)} is greater than micro batch size {batch_size}.")

            # If the examples of the next sample would not fit in the batch we create a new batch
            if len(batch_training_input_ids) + len(input_examples) > batch_size:
                # We find the maximum length in the batch and pad the input
                training_input_ids, training_max_len = padding_helper(batch_training_input_ids, pad_right=True)
                generation_input_ids, __ = padding_helper(batch_generation_input_ids, pad_right=False)
                generation_labels, __ = padding_helper(batch_generation_labels, pad_right=True)

                # The labels are -100 for the input ids and the padding tokens. Only the original labels are kept
                training_labels = training_input_ids.clone()
                mask = torch.arange(training_max_len).unsqueeze(0) >= torch.tensor(batch_lengths).unsqueeze(1)
                training_labels = torch.where(mask, training_labels, torch.full_like(training_labels, -100))
                training_labels[training_labels == tokenizer.pad_token_id] = -100
                # Replace the -1 with the eos_token_id
                training_input_ids[training_input_ids == -1] = tokenizer.eos_token_id
                training_labels[training_labels == -1] = tokenizer.eos_token_id
                batches.append(
                    {
                        "input_ids": training_input_ids.long(),
                        "labels": training_labels.long(),
                        "attention_masks": (training_input_ids != tokenizer.pad_token_id).long(),
                        "sample_count": batch_counter,
                        "generation_input_ids": generation_input_ids.long(),
                        "generation_labels": generation_labels.long(),
                        "generation_attention_masks": (generation_input_ids != tokenizer.pad_token_id).long(),
                    }
                )
                # Reset everything for the next batch
                batch_training_input_ids = []
                batch_generation_input_ids = []
                batch_generation_labels = []
                batch_lengths = []
                batch_counter = []

            for input_example, label_example in zip(input_examples, label_examples):
                # We append each example to the batch
                # The -1 is a placeholder for the first end of sentence token. It should receive attention
                batch_training_input_ids.append(torch.tensor(input_example + label_example + [-1]))
                batch_generation_input_ids.append(torch.tensor(input_example))
                batch_generation_labels.append(torch.tensor(label_example))
                batch_lengths.append(torch.tensor(len(input_example)))

            # For each samples we append how many examples it has
            batch_counter.append(len(input_examples))

        return {"batches": batches}

    return preprocess_function


class PromptLoader:
    def __init__(
        self,
        tokenizer,
        seed,
        tokenized_data_path,
        firstn_datasets: int,  # set for 0 when using all datasets
    ):
        self.collection = TemplateCollection()
        self.firstn_datasets = firstn_datasets
        self.tokenizer = tokenizer
        self.seed = seed
        self.tokenized_data_path = tokenized_data_path

        self.debug_tasks = [("winogrande", "winogrande_xs"), ("snips_built_in_intents", None), ("onestop_english", None)]

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
            if dataset_id in BAD_TASKS or (dataset_id, subset) not in self.debug_tasks:
                continue
            dataset_path = (
                os.path.join(self.tokenized_data_path, dataset_id, subset)
                if subset
                else os.path.join(self.tokenized_data_path, dataset_id)
            )
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
