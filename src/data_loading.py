import os
from tqdm import tqdm

import itertools
from typing import Iterator, Tuple, Optional
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
        data_mode: str,
        batch_size: int,
        firstn_datasets: int,  # set for 0 when using all datasets
        seed: int,
        data_location: str,
        preprocessing_workers: int,
        val_batch_size: int = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name.replace("/", "_")
        self.data_mode = data_mode

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.firstn_datasets = firstn_datasets
        self.data_location = data_location
        self.preprocessing_workers = preprocessing_workers
        self.seed = seed

        self.tokenized_data_path = f"{data_location}/tokenized_{self.tokenizer_name}_{self.data_mode}_{firstn_datasets}"

        self.loader = PromptLoader(
            tokenizer, seed, self.tokenized_data_path, data_mode, firstn_datasets
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
        base_path = f"{self.data_location}/processed_{self.tokenizer_name}_{self.data_mode}_{self.firstn_datasets}"
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
            make_preprocess_function(batch_size, self.tokenizer, self.data_mode),
            batched=True,
            batch_size=1_000,
            remove_columns=dataset.column_names,
            num_proc=self.preprocessing_workers,
            desc="Preprocessing dataset",
        )
        if self.data_mode == "data_batches":
            dataset = dataset.shuffle(seed=self.seed)

        dataset.save_to_disk(self.processed_data_path(split))

        logger.info(f"Saved {split} dataset to {self.processed_data_path(split)}")
        return dataset

    def get_collator(self):

        def _collate(batch: dict[str, list[list[int|str]]]) -> dict[str, torch.Tensor]:
            if "templates" in batch:
                templates = batch.pop("templates")
                collated_batch = {k: torch.tensor(v) for k, v in batch.items()}
                collated_batch["templates"] = templates
                return collated_batch
            return {k: torch.tensor(v) for k, v in batch.items()}

        def collate_default(examples: tuple[dict[str, dict[str, list[list[int|str]]]]]) -> dict[str, torch.Tensor]:
            return _collate(examples[0]["batches"])

        def collate_with_data_batches(examples: tuple[dict[str, list[dict[str, list[list[int|str]]]]]]) -> list[dict[str, torch.Tensor]]:
            collated_batch = []
            for example in examples[0]['batches']:
                collated_batch.append(_collate(example))
            return collated_batch

        return collate_default if self.data_mode == "default" else collate_with_data_batches


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


def make_preprocess_function(
    batch_size: int, tokenizer: PreTrainedTokenizerFast, data_mode: str
):
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

    def make_generation_label(
        example_tokens: torch.tensor, example_prompt: torch.tensor
    ) -> torch.tensor:
        generation_label = example_tokens[
            len(example_prompt) :
        ]  # The label for the generation is the original label

        # We do not want the model to generate end_of_text
        return generation_label[:-1]

    def _flush_batch(
        batch_tokens: list[torch.tensor],
        batch_prompt: list[torch.tensor],
        batch_generation_label: list[torch.tensor],
        batch_templates: list[str],
        batch_counter: list[int] = None,
    ) -> dict:
        """
        Takes all lists and makes one batch out of them. Converts the lists to tensors and pads them.
        """
        # We find the maximum length in the batch and pad the input
        tokens = padding_helper(
            batch_tokens,
            pad_right=True,
            pad_id=tokenizer.special_tokens["<|finetune_right_pad_id|>"],
        )
        prompt = padding_helper(batch_prompt, pad_right=False)
        generation_label = padding_helper(batch_generation_label, pad_right=True)
        # Shift the tokens to the right by one position
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

        labels[labels == tokenizer.special_tokens["<|finetune_right_pad_id|>"]] = -100
        labels[labels == tokenizer.eos_id] = -100

        batch = {
            "tokens": tokens.long(),
            "labels": labels.long(),
            "prompt": prompt.long(),
            "generation_label": generation_label.long(),
            "templates": batch_templates,
        }
        if batch_counter:
            batch["sample_count"] = batch_counter
        return batch

    def default_preprocess_function(examples):
        """
        The input examples are lists of the tokens, prompts and templates for each sample (That means each sample gets one list with all examples for this sample).
        The goal for default preprocessing is to ensure that in each batch we have all examples for a given sample.
        We do this by greedily filling the batch until the next sample would not fit.
        Mixing datasets in a batch is ensured, by having the examples shuffled, prior to this function.
        """
        batches = []
        batch_tokens = []
        batch_generation_label = []
        batch_prompt = []
        batch_counter = []
        batch_templates = []

        for sample_tokens, sample_prompt, sample_templates in zip(
            examples["tokens"],
            examples["prompt"],
            examples["templates"],
        ):
            # If we have more examples for a give input than micro batch size we can not leave them in one batch
            if len(sample_tokens) > batch_size:
                raise ValueError(
                    f"Examples per sample {len(sample_tokens)} is greater than micro batch size {batch_size}."
                )

            # If the examples of the next sample would not fit in the batch we create a new batch
            if len(batch_tokens) + len(sample_tokens) > batch_size:
                batches.append(
                    _flush_batch(
                        batch_tokens,
                        batch_prompt,
                        batch_generation_label,
                        batch_counter,
                        batch_templates,
                    )
                )
                # Reset everything for the next batch
                batch_tokens = []
                batch_prompt = []
                batch_generation_label = []
                batch_counter = []
                batch_templates = []

            for example_tokens, example_prompt in zip(sample_tokens, sample_prompt):
                batch_tokens.append(torch.tensor(example_tokens))
                batch_prompt.append(torch.tensor(example_prompt))
                batch_generation_label.append(
                    torch.tensor(make_generation_label(example_tokens, example_prompt))
                )

            # For each samples we append how many examples it has
            batch_counter.append(len(sample_tokens))
            batch_templates.extend(sample_templates)

        return {"batches": batches}

    def data_batch_preprocess_function(examples):
        """
        The input examples are lists of the tokens, prompts and templates for each sample (That means each sample gets one list with all examples for this sample).
        The goal of the data batch approach is:
        1. Each template has its own batch
        2. All templates of a dataset are in one optimizer step
        3. Different datasets are in one optimizer step

        Therefore, we create data batches, which are lists of batches (of the specified batch size), where each batch contains the examples for one template.
        The data batch ensures 1 and 2. To ensure 3 we shuffle the data batches afterwards and need to set gradient accumulation steps to > 1.
        """
        data_batches = []
        # We want a batch per template. So data_batch needs to be the same length as the number of templates
        data_batch = [
            {"batch_tokens": [], "batch_prompt": [], "batch_generation_label": []}
            for _ in examples["templates"][0]
        ]
        start_of_batch_idx = 0
        for sample_idx, sample in enumerate(
            zip(
                examples["tokens"],
                examples["prompt"],
                examples["templates"],
            )
        ):
            sample_tokens, sample_prompt, sample_templates = sample

            # If the batch should be full we flush it
            # Note on dataset borders it does not need to be full
            if sample_idx % batch_size == 0 and sample_idx > 0:
                data_batches.append(
                    [
                        _flush_batch(
                            template_batch["batch_tokens"],
                            template_batch["batch_prompt"],
                            template_batch["batch_generation_label"],
                            [template_name] * len(template_batch["batch_tokens"]),
                        )
                        # We match each template_batch with its template_name
                        for template_batch, template_name in zip(
                            data_batch, examples["templates"][start_of_batch_idx]
                        )
                    ]
                )
                # Reset the data batch for the next sample
                start_of_batch_idx = sample_idx
                data_batch = [
                    {
                        "batch_tokens": [],
                        "batch_prompt": [],
                        "batch_generation_label": [],
                    }
                    for _ in examples["templates"][start_of_batch_idx]
                ]

            for example_idx, example in enumerate(
                zip(sample_tokens, sample_prompt, sample_templates)
            ):
                example_tokens, example_prompt, example_template = example
                # We make sure each batch has only one dataset. Thus we throw away the first examples of the next dataset and create a smaller batch
                if example_template not in examples["templates"][start_of_batch_idx]:
                    continue
                data_batch[example_idx]["batch_tokens"].append(
                    torch.tensor(example_tokens)
                )
                data_batch[example_idx]["batch_prompt"].append(
                    torch.tensor(example_prompt)
                )
                data_batch[example_idx]["batch_generation_label"].append(
                    torch.tensor(make_generation_label(example_tokens, example_prompt))
                )

        return {"batches": data_batches}

    if data_mode == "default":
        return default_preprocess_function
    if data_mode == "data_batches":
        return data_batch_preprocess_function
    raise ValueError(
        f"Unknown data mode {data_mode}. Please use 'default' or 'data_batches'."
    )


class PromptLoader:
    def __init__(
        self,
        tokenizer,
        seed,
        tokenized_data_path,
        data_mode: str,
        firstn_datasets: int,  # set for 0 when using all datasets
    ):
        self.collection = TemplateCollection()
        self.firstn_datasets = firstn_datasets
        self.tokenizer = tokenizer
        self.seed = seed
        self.tokenized_data_path = tokenized_data_path
        self.data_mode = data_mode

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

        dataset_list = []
        for dataset_ids, templates in tqdm(
            datasets_iterator, desc="Iterating over datasets"
        ):
            try:
                dataset_id, subset = dataset_ids
                dataset_path = (
                    os.path.join(self.tokenized_data_path, dataset_id, subset)
                    if subset
                    else os.path.join(self.tokenized_data_path, dataset_id)
                )
                load_was_successful, dataset = self.try_load_dataset(
                    dataset_id, subset, split, dataset_path
                )
                if not dataset:
                    continue
                if load_was_successful:
                    logger.info(f"Loaded {dataset_id} dataset from disk.")
                    dataset_list.append(dataset)
                else:
                    # Only take 1.000 random examples per dataset
                    # if split == "train" and len(dataset) > 1_000:
                    #     dataset = dataset.shuffle(seed=self.seed).select(range(1_000))
                    dataset = self.tokenize_dataset(
                        dataset, templates, dataset_id, subset
                    )
                    dataset.save_to_disk(dataset_path)
                    dataset_list.append(dataset)
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_id}: {e}")
                continue

        # We merge all datasets, shuffle them and save them to disk
        full_dataset = datasets.concatenate_datasets(dataset_list)
        del dataset_list
        if self.data_mode == "default":
            full_dataset = full_dataset.shuffle(seed=self.seed)
        full_dataset.save_to_disk(os.path.join(self.tokenized_data_path, split))
        return full_dataset

    def try_load_dataset(
        self, dataset_id: str, subset: str, split: str, dataset_path: str
    ) -> Tuple[bool, Optional[Dataset]]:
        """
        Try to load a dataset from disk or from the hub.
        """
        # Ignore bad tasks and the tasks from the other split
        if (
            dataset_id in BAD_TASKS
            or split == "train"
            and dataset_id in T0_HELDOUT_TASKS
            or split != "train"
            and dataset_id not in T0_HELDOUT_TASKS
            or (dataset_id, subset) not in self.debug_tasks  # TODO:
        ):
            return False, None
        # We first try to load the dataset from disk if we have tokenized it before
        try:
            # logger.info(f"Processing {dataset_id}...")
            dataset = Dataset.load_from_disk(dataset_path)
            # logger.info(f"Loaded {dataset_id} dataset from disk.")
            return True, dataset
        except OSError:
            # If this fails we try to load the dataset from the hub
            try:
                dataset = load_dataset(
                    dataset_id, subset, split=split, trust_remote_code=True
                )
            except Exception:
                dataset = load_dataset(dataset_id, subset, split=split)
            return False, dataset

    def tokenize_dataset(
        self,
        dataset: Dataset,
        templates: TemplateCollection,
        dataset_id: str,
        subset: str,
    ) -> Dataset:
        sample_prompt = []
        sample_tokens = []
        template_list = []
        for sample in tqdm(dataset, desc=f"Processing {dataset_id}"):
            example_tokens, example_prompt, template_ids = [], [], []
            for template_id, template in templates.templates.items():
                tokens, label = template.apply(sample)
                messages = [
                    Message(role="user", content=tokens, masked=True),
                    Message(role="assistant", content="", masked=True),
                ]
                example_prompt.append(
                    self.tokenizer({"messages": messages}, inference=True)["tokens"]
                )
                messages[1] = Message(role="assistant", content=label)
                tokens = self.tokenizer({"messages": messages}, inference=False)[
                    "tokens"
                ]
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
                "templates": template_list,
            }
        )

        return dataset
