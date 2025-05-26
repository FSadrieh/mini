import os
from tqdm import tqdm

import itertools
from typing import Iterator, Tuple
import torch

from datasets import load_dataset, Dataset, load_from_disk
from promptsource.templates import TemplateCollection

import datasets
from print_on_steroids import logger
from transformers import PreTrainedTokenizerFast

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


class CustomDataLoader():
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_name: str,
        batch_size: int,
        firstn_datasets: int,  # set for 0 when using all datasets
        seed: int,
        data_location: str,
        preprocessing_workers: int,
        val_batch_size: int = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_name
        self.tokenizer.pad_id = self.tokenizer.eos_id

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.firstn_datasets = firstn_datasets
        self.data_location = data_location
        self.preprocessing_workers = preprocessing_workers


        self.tokenized_data_path = (
            f"{data_location}/tokenized_{tokenizer_name}_{firstn_datasets}"
        )

        self.loader = PromptLoader(tokenizer, seed, self.tokenized_data_path, firstn_datasets)

    def load_dataset(self, split: str) -> None:
        try:
            dataset = load_from_disk(self.processed_data_path(split))
            logger.info("Data already prepared, skipping preparation.")
        except Exception:
            dataset = self.preprocess_data(split)
            logger.info("Data not prepared, preparing data.")

        return dataset

    def processed_data_path(self, split: str) -> str:
        base_path = f"{self.data_location}/processed_{self.tokenizer_path.replace('/', '_')}_{self.firstn_datasets}"
        batch_size = self.batch_size if split == "train" else self.val_batch_size
        return os.path.join(f"{base_path}_{batch_size}", split)

    def preprocess_data(self, split: str) -> Dataset:
        if not os.path.isdir(self.processed_data_path(split)):
            os.makedirs(self.processed_data_path(split), exist_ok=True)

        try:
            dataset = Dataset.load_from_disk(self.tokenized_data_path + f"/{split}/")
        except OSError:
            logger.info(f"Creating tokenized dataset for {split}...")
            dataset = self.loader.iterate_prompts(split=split)

        batch_size = self.batch_size if split == "train" else self.val_batch_size

        dataset = dataset.map(
            make_preprocess_function(batch_size, self.tokenizer),
            batched=True,
            batch_size=16_000,
            remove_columns=dataset.column_names,
            num_proc=self.preprocessing_workers,
            desc="Preprocessing dataset",
        )

        dataset.save_to_disk(self.processed_data_path(split))

        logger.info(f"Saved {split} dataset to {self.processed_data_path(split)}")
        return dataset

    def get_collator(self):
        def collate(examples):
            return {k: torch.tensor(v) for k, v in examples[0]["batches"].items()}

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
            padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=tokenizer.pad_id) for t in to_pad]
        else:
            padded = [torch.nn.functional.pad(t, (max_len - t.shape[0], 0), value=tokenizer.pad_id) for t in to_pad]
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
                training_labels[training_labels == tokenizer.pad_id] = -100
                attention_masks = (training_input_ids != tokenizer.pad_id).long()
                # Make the attention mask 3D
                causal_attention_mask = torch.tril(torch.ones((training_max_len, training_max_len), device=attention_masks.device)).unsqueeze(0)
                attention_mask = attention_masks.unsqueeze(1) * causal_attention_mask

                # Replace the -1 with the eos_id
                training_input_ids[training_input_ids == -1] = tokenizer.eos_id
                training_labels[training_labels == -1] = tokenizer.eos_id
                batches.append(
                    {
                        "input_ids": training_input_ids.long(),
                        "labels": training_labels.long(),
                        "attention_masks": attention_mask.long(),
                        "sample_count": batch_counter,
                        "generation_input_ids": generation_input_ids.long(),
                        "generation_labels": generation_labels.long(),
                        "generation_attention_masks": (generation_input_ids != tokenizer.pad_id).long(),
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
            # For debug purposes we only want to load a few datasets
            if dataset_id in BAD_TASKS or (dataset_id, subset) not in self.debug_tasks:
                continue
            if split == "train" and dataset_id in T0_HELDOUT_TASKS:
                continue
            elif split != "train" and dataset_id not in T0_HELDOUT_TASKS:
                continue
            dataset_path = (
                os.path.join(self.tokenized_data_path, dataset_id, subset)
                if subset
                else os.path.join(self.tokenized_data_path, dataset_id)
            )
            # We first try to load the dataset from disk if we have tokenized it before
            try:
                logger.info(f"Processing {dataset_id}...")
                dataset = Dataset.load_from_disk(dataset_path)
                logger.info(f"Loaded {dataset_id} dataset from disk.")
                dataset_list.append(dataset)
            except OSError:
                # If this fails we try to load the dataset from the hub
                try:
                    dataset = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
                except (ValueError, TypeError) as e:
                    dataset = load_dataset(dataset_id, subset, split=split)
                # For each sample in the dataset we apply the templates
                for sample in tqdm(dataset, desc=f"Processing {dataset_id}"):
                    inputs, labels = [], []
                    for template_id, template in templates.templates.items():
                        input_prompt, label = template.apply(sample)
                        inputs.append(self.tokenizer.encode(input_prompt, add_bos=True, add_eos=False))
                        labels.append(self.tokenizer.encode(label, add_bos=False, add_eos=False))

                    # We tokenize all examples of a sample in one list and add them to the sample list
                    # Note for the labels we do not want to add a BOS token
                    # TODO: Need to https://docs.pytorch.org/torchtune/0.2/generated/torchtune.models.llama3.Llama3Tokenizer.html different tokenizer
                    sample_inputs.append(inputs)
                    sample_labels.append(labels)
                # We save the individual dataset to disk and add it to the dataset list
                dataset = Dataset.from_dict({"input_ids": sample_inputs, "labels": sample_labels})
                dataset.save_to_disk(dataset_path)
                dataset_list.append(dataset)

        # We merge all datasets, shuffle them and save them to disk
        full_dataset = datasets.concatenate_datasets(dataset_list)
        full_dataset = full_dataset.shuffle(seed=self.seed)
        full_dataset.save_to_disk(os.path.join(self.tokenized_data_path, split))
        return full_dataset
