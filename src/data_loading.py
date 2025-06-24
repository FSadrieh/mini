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
from torchtune.data import Message, Role

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
    "cord19",
    "qed",
    "hate_speech18",
    "samsum",
    "sem_eval_2014_task_1",
    "squad_v2",
]


class CustomDataLoader:
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

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.firstn_datasets = firstn_datasets
        self.data_location = data_location
        self.preprocessing_workers = preprocessing_workers

        self.tokenized_data_path = (
            f"{data_location}/tokenized_{tokenizer_name}_{firstn_datasets}"
        )

        self.loader = PromptLoader(
            tokenizer, seed, self.tokenized_data_path, firstn_datasets
        )

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
            batch_size=1_000,
            remove_columns=dataset.column_names,
            num_proc=self.preprocessing_workers,
            desc="Preprocessing dataset",
        )

        dataset.save_to_disk(self.processed_data_path(split))

        logger.info(f"Saved {split} dataset to {self.processed_data_path(split)}")
        return dataset

    def get_collator(self):
        def collate(examples):
            if "templates" in examples[0]["batches"]:
                templates = examples[0]["batches"].pop("templates")
                batch = {k: torch.tensor(v) for k, v in examples[0]["batches"].items()}
                batch["templates"] = templates
                return batch
            return {k: torch.tensor(v) for k, v in examples[0]["batches"].items()}

        return collate


def make_tokenize_function(tokenizer: PreTrainedTokenizerFast):
    def tokenize_function(examples):
        tokenized_inputs = []
        tokenized_labels = []
        for input_list in examples["inputs"]:
            tokenized_inputs.append(tokenizer(input_list, padding=False)["input_ids"])

        for label_list in examples["labels"]:
            tokenized_labels.append(
                tokenizer(label_list, padding=False, add_special_tokens=False)[
                    "input_ids"
                ]
            )

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
    the attention mask is 1 everywhere for padding; and the sample count (how many examples for each sample are in the batch).
    For generation we need left padded input_ids, the raw labels and the attention mask.
    """

    def padding_helper(
        to_pad: list[torch.tensor], pad_right=True, pad_id: int = tokenizer.pad_id
    ) -> torch.tensor:
        max_len = max(len(t) for t in to_pad)
        if pad_right:
            padded = [
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id)
                for t in to_pad
            ]
        else:
            padded = [
                torch.nn.functional.pad(t, (max_len - t.shape[0], 0), value=pad_id)
                for t in to_pad
            ]
        return torch.stack(padded)

    def preprocess_function(examples):
        batches = []
        batch_tokens = []
        batch_generation_label = []
        batch_prompt = []
        batch_counter = []
        batch_templates = []

        include_templates = "templates" in examples

        def _flush_batch():
            # We find the maximum length in the batch and pad the input
            tokens = padding_helper(
                batch_tokens,
                pad_right=True,
                pad_id=tokenizer.special_tokens["<|finetune_right_pad_id|>"],
            )
            prompt = padding_helper(batch_prompt, pad_right=False)
            generation_label = padding_helper(batch_generation_label, pad_right=True)
            labels = torch.cat(
                [
                    tokens[:, 1:],
                    torch.full(
                        (tokens.size(0), 1),
                        fill_value=-100,
                        dtype=tokens.dtype,
                        device=tokens.device,
                    ),
                ],
                dim=-1,
            )

            labels[
                labels == tokenizer.special_tokens["<|finetune_right_pad_id|>"]
            ] = -100
            labels[labels == tokenizer.eos_id] = -100

            batch = {
                "tokens": tokens.long(),
                "labels": labels.long(),
                "sample_count": batch_counter,
                "prompt": prompt.long(),
                "generation_label": generation_label.long(),
            }
            if include_templates:
                batch["templates"] = batch_templates
            batches.append(batch)

        for sample_tokens, sample_prompt, sample_templates in zip(
            examples["tokens"],
            examples["prompt"],
            examples.get("templates", [])
            if include_templates
            else [None] * len(examples["tokens"]),
        ):
            # If we have more examples for a give input than micro batch size we can not leave them in one batch
            if len(sample_tokens) > batch_size:
                raise ValueError(
                    f"Examples per sample {len(sample_tokens)} is greater than micro batch size {batch_size}."
                )

            # If the examples of the next sample would not fit in the batch we create a new batch
            if len(batch_tokens) + len(sample_tokens) > batch_size:
                _flush_batch()
                # Reset everything for the next batch
                batch_tokens = []
                batch_prompt = []
                batch_generation_label = []
                batch_counter = []
                batch_templates = []

            for example_tokens, example_prompt in zip(sample_tokens, sample_prompt):
                # We append each example to the batch
                generation_label = example_tokens[
                    len(example_prompt) :
                ]  # The label for the generation is the original label

                # We do not want the model to generate end_of_text
                generation_label = generation_label[:-1]

                batch_tokens.append(torch.tensor(example_tokens))
                batch_prompt.append(torch.tensor(example_prompt))
                batch_generation_label.append(torch.tensor(generation_label))

            # For each samples we append how many examples it has
            batch_counter.append(len(sample_tokens))

            if include_templates:
                batch_templates.extend(sample_templates)

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

        self.debug_tasks = [
            ("winogrande", "winogrande_xs"),
            ("snips_built_in_intents", None),
            ("onestop_english", None),
        ]

        self.more_tasks = [
            "blbooksgenre",
            "hellaswag",
            "newspop",
            "winogrande",
            "wiqa",
        ]

    def iterate_prompts(self, split: str = "train") -> Iterator[Tuple[str, str]]:
        datasets_iterator = self.collection.datasets_templates.items()
        if self.firstn_datasets:
            print(
                "Limiting the iterated datasets to first %s ones" % self.firstn_datasets
            )
            datasets_iterator = itertools.islice(
                self.collection.datasets_templates.items(), self.firstn_datasets
            )

        sample_prompt = []
        sample_tokens = []
        template_list = []
        dataset_list = []
        for dataset_ids, templates in tqdm(
            datasets_iterator, desc="Iterating over datasets"
        ):
            try:
                dataset_id, subset = dataset_ids
                # For debug purposes we only want to load a few datasets
                if dataset_id in BAD_TASKS:
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
                    # logger.info(f"Processing {dataset_id}...")
                    dataset = Dataset.load_from_disk(dataset_path)
                    # logger.info(f"Loaded {dataset_id} dataset from disk.")
                    dataset_list.append(dataset)
                except OSError:
                    # If this fails we try to load the dataset from the hub
                    try:
                        dataset = load_dataset(
                            dataset_id, subset, split=split, trust_remote_code=True
                        )
                    except Exception:
                        dataset = load_dataset(dataset_id, subset, split=split)
                    # For each sample in the dataset we apply the templates

                    # Only take 10.000 random examples per dataset
                    if split == "train" and len(dataset) > 1_000:
                        dataset = dataset.shuffle(seed=self.seed).select(range(1_000))
                    for sample in tqdm(dataset, desc=f"Processing {dataset_id}"):
                        example_tokens, example_prompt, template_ids = [], [], []
                        for template_id, template in templates.templates.items():
                            tokens, label = template.apply(sample)
                            messages = [
                                Message(role="user", content=tokens, masked=True),
                                Message(role="assistant", content="", masked=True),
                            ]
                            example_prompt.append(
                                self.tokenizer({"messages": messages}, inference=True)[
                                    "tokens"
                                ]
                            )
                            messages[1] = Message(role="assistant", content=label)
                            tokens = self.tokenizer(
                                {"messages": messages}, inference=False
                            )["tokens"]
                            example_tokens.append(tokens)
                            template_ids.append(template_id)

                        sample_tokens.append(example_tokens)
                        sample_prompt.append(example_prompt)
                        template_list.append(
                            [
                                dataset_id + (f"_{subset}" if subset else "") + t
                                for t in template_ids
                            ]
                        )
                    # We save the individual dataset to disk and add it to the dataset list
                    dataset = Dataset.from_dict(
                        {
                            "tokens": sample_tokens,
                            "prompt": sample_prompt,
                        }
                        if split == "train"
                        else {
                            "tokens": sample_tokens,
                            "prompt": sample_prompt,
                            "templates": template_list,
                        }
                    )
                    dataset.save_to_disk(dataset_path)
                    dataset_list.append(dataset)
                    sample_tokens = []
                    sample_prompt = []
                    template_list = []
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {e}")
                sample_tokens = []
                sample_prompt = []
                template_list = []
                continue

        # We merge all datasets, shuffle them and save them to disk
        full_dataset = datasets.concatenate_datasets(dataset_list)
        del dataset_list
        full_dataset = full_dataset.shuffle(seed=self.seed)
        full_dataset.save_to_disk(os.path.join(self.tokenized_data_path, split))
        return full_dataset
