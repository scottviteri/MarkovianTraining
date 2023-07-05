"""
Functions for getting formatted prompt datasets for SuperHF training.

Imports datasets from other RLHF experiments. Formats questions into lists:

[
    "\n\nHuman: What's the capital of France?\n\nAssistant:",
    "\n\nHuman: What's the best way to get to the airport?\n\nAssistant:",
    "\n\nHuman: How do I hotwire a car?\n\nAssistant:",
    ...
]

Which is the same format as from the Anthropic HH-RLHF dataset.
"""

from typing import Iterator, TypeVar

from datasets.load import load_dataset
from torch.utils.data import IterableDataset

from superhf.constants import SUPPORTED_DATASETS

T = TypeVar("T")

TEST_SET_SIZE_PER_DATASET = 2300


def get_superhf_prompts(
    dataset_name: str,
    split: str = "train",
    max_length_chars: int = -1,
    load_whole_completion: bool = False,
) -> list[str]:
    """
    Get a list of prompts from a dataset.

    Args:
        dataset_name: The name of the dataset to load. One of:
            - 'anthropic-red-team'
            - 'webgpt_comparisons'
            - 'anthropic-harmless-base'
            - 'anthropic-helpful-base'
            - 'mock'
        split: The split of the dataset to load.
        max_length_chars: Filter prompts longer than this many chars, or -1 for all prompts.
        load_whole_completion: Whether to load the whole completion or just the prompt.

    Returns:
        A list of prompts.
    """
    # pylint: disable=too-many-branches

    # Load the appropriate dataset then convert to a list of prompts
    output: list[str] = []
    if dataset_name == "anthropic-red-team":
        assert (
            not load_whole_completion
        ), f"{dataset_name} not supported for chosen completions."
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="red-team-attempts",
            split="train",
            keep_in_memory=False,
        )
        if split == "train":
            output = [
                dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ][:-TEST_SET_SIZE_PER_DATASET]
        elif split == "test":
            output = [
                dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ][-TEST_SET_SIZE_PER_DATASET:]
        else:
            raise ValueError(
                f"split must be one of 'test' or 'train'. Instead got {split}"
            )
    elif dataset_name == "openai/webgpt_comparisons":
        dataset = load_dataset(
            dataset_name,
            split="train",
            keep_in_memory=False,
        )
        if split == "train":
            assert (
                not load_whole_completion
            ), f"{dataset_name} not supported for chosen completions."
            output = [
                "\n\nHuman: " + row["question"]["full_text"] + "\n\nAssistant:"
                for row in dataset
            ][:-TEST_SET_SIZE_PER_DATASET]
        elif split == "test":
            output = [
                "\n\nHuman: " + row["question"]["full_text"] + "\n\nAssistant:"
                for row in dataset
            ][-TEST_SET_SIZE_PER_DATASET:]
        else:
            raise ValueError(
                f"split must be one of 'test' or 'train'. Instead got {split}"
            )
    elif dataset_name == "anthropic-harmless-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="harmless-base",
            split=split,
            keep_in_memory=False,
        )
        if load_whole_completion:
            output = [row["chosen"] for row in dataset]
        else:
            output = [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
    elif dataset_name == "anthropic-helpful-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split=split,
            keep_in_memory=False,
        )
        if load_whole_completion:
            output = [row["chosen"] for row in dataset]
        else:
            output = [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
    elif dataset_name == "self_instruct":
        assert (
            not load_whole_completion
        ), f"{dataset_name} not supported for chosen completions."
        dataset = load_dataset(
            "yizhongw/self_instruct",
            # data_dir="self_instruct",
            split="train",
            keep_in_memory=False,
        )
        output = [
            "\n\nHuman: " + row["prompt"].split("Output:")[0].strip() + "\n\nAssistant:"
            for row in dataset
        ]
        if split == "train":
            output = output[:-TEST_SET_SIZE_PER_DATASET]
        elif split == "test":
            output = output[-TEST_SET_SIZE_PER_DATASET:]
    elif dataset_name == "mock":
        output.extend(
            [
                f"{i}\n\nHuman: ...\n\nAssistant: Sphinx of black quartz, judge my vow."
                for i in range(50000)
            ]
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of the following "
            + f"supported datasets: {SUPPORTED_DATASETS}"
        )

    # Filter prompts longer than max_length_chars
    if max_length_chars > 0:
        old_length = len(output)
        output = [prompt for prompt in output if len(prompt) <= max_length_chars]
        print(
            f"Filtered {old_length - len(output)} prompts longer than"
            f" {max_length_chars} chars."
        )

    return output


def get_instruct_dataset(max_length_chars: int = -1) -> list[str]:
    """Get a dataset for supervised instruction tuning."""
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Format into text examples
    examples = [
        (
            (f'\n\nContext: {row["context"]}' if row["context"] else "")
            + f'\n\nHuman: {row["instruction"]}\n\nAssistant: {row["response"]}'
        )
        for row in dataset
    ]

    # Filter prompts longer than max_length_chars
    if max_length_chars > 0:
        old_length = len(examples)
        examples = [prompt for prompt in examples if len(prompt) <= max_length_chars]
        print(
            f"Filtered {old_length - len(examples)} prompts longer than"
            f" {max_length_chars} chars."
        )

    return examples


class ListDataset(IterableDataset[T]):
    """A Torch dataset that wraps a list of data."""

    def __init__(self, data: list[T]):
        self.data = data

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T:
        return self.data[index]
