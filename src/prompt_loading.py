import itertools
from typing import Iterator, Tuple

from datasets import load_dataset
from promptsource.templates import DatasetTemplates
from promptsource.templates import TemplateCollection


ALL_DATASET_IDS = ['ag_news']


class PromptLoader:

    def __init__(self):
        self.collection = TemplateCollection()


    def iterate_prompts(self, split: str = "train") -> Iterator[Tuple[str, str]]:
        for dataset_ids, templates in self.collection.datasets_templates.items():
            dataset_id, subset = dataset_ids
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
