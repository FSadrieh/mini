import itertools
from typing import Iterator, Tuple

from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection


T0_HELDOUT_TASKS = ['copa', 'hellaswag', 'story_cloze', 'anli', 'cb', 'rte', 'wsc', 'winogrande', 'wic']
# TODO: compared to T0 eval, our heldout tasks are missing bigbench coming from other source


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
            dataset = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
            for sample in dataset:
                inputs, labels = [], []
                for template_id, template in templates.templates.items():
                    input_prompt, label = template.apply(sample)
                    inputs.append(input_prompt)
                    labels.append(label)

                yield inputs, labels

loader = PromptLoader()
outputs = itertools.islice(loader.iterate_prompts("train"), 10)
print(list(outputs))
