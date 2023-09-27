"""
A file for testing the functions used in the mvp_loss_decrease.py file.
"""
import pytest
import tabulate
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np

import os

from collaborative_experiments.mvp_loss_decrease import (
    create_helpful_message_1,
    create_model_helpful_message,
    create_openai_helpful_message,
    train_step,
)
from collaborative_experiments.constants import (
    DEFAULT_MAX_CONTEXT_LENGTH,
    DEFAULT_MSG_CONTEXT_LENGTH,
)
from collaborative_experiments.utils import (
    get_device,
    load_and_format_dataset,
)

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from transformers import PreTrainedTokenizerFast
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from typing import Any


@pytest.fixture
def causal_lm_tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained("distilgpt2")


@pytest.fixture
def causal_lm():  # -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained("distilgpt2")


@pytest.fixture
def uncompressed_tokens() -> TensorType["batch":1, "seq_len":100]:
    return torch.tensor([list(range(100))])


def test_create_model_helpful_message(
    uncompressed_tokens: TensorType["batch", "seq_len"],
    causal_lm: AutoModelForCausalLM,
    causal_lm_tokenizer: PreTrainedTokenizerFast,
):
    helpful_message = create_model_helpful_message(
        uncompressed_tokens, causal_lm_tokenizer, causal_lm
    )
    assert helpful_message.shape[1] <= DEFAULT_MSG_CONTEXT_LENGTH


def test_train_step(causal_lm, causal_lm_tokenizer):
    # visualizes it
    sentence = "Hello, my name is john. I like apples. aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    tokens = causal_lm_tokenizer.encode(sentence, return_tensors="pt")
    device = get_device()
    tokens = tokens.to(device)  # (1, 24) == (1, seq_len)
    causal_lm = causal_lm.to(device)

    # batch, causal_lm, loss_fn, device, correct_probs_all
    loss_fn = torch.nn.CrossEntropyLoss()
    batch = {}
    batch["msg"] = tokens[:, :3]
    batch["content"] = tokens[:, 3:]
    loss, correct_probs, logits = train_step(
        batch, causal_lm, loss_fn, device, pytest=True
    )
    correct_probs = correct_probs.mean(dim=0)
    # logits have shape (batch_size, seq_len, vocab_size)
    # loss should have shape (batch_size, seq_len)
    # correct_probs should have shape (batch_size, seq_len - 1)

    decoded_sentence = []
    for token in batch["content"][0]:
        decoded_sentence.append(causal_lm_tokenizer.decode(token))
    predicted_tokens = ["N/A"]
    predicted_token_ids = [-1]
    for token in logits[0]:
        predicted_id = torch.argmax(token)
        predicted_token_ids.append(predicted_id)
        predicted_tokens.append(causal_lm_tokenizer.decode(predicted_id))
    correct_probs_list = ["N/A"]
    for i, prob in enumerate(correct_probs):
        correct_probs_list.append(prob.item())
        # correct_prob_idx = tokens[i+1]
        # correct_prob_manual = F.softmax(logits[0, i+1], dim=-1)[tokens[0, i+1]].item()
    # use tabulate to print a table of all this
    table = [
        ["Token", "Predicted Token", "Predicted Token ID", "Correct Probability"],
    ]
    if (
        len(decoded_sentence) != len(predicted_tokens)
        or len(decoded_sentence) != len(predicted_token_ids)
        or len(decoded_sentence) != len(correct_probs_list)
    ):
        raise ValueError(
            f"The lengths of the lists are not the same. decoded_sentence: {len(decoded_sentence)}, predicted_tokens: {len(predicted_tokens)}, predicted_token_ids: {len(predicted_token_ids)}, correct_probs: {len(correct_probs_list)}"
        )
    for i in range(len(decoded_sentence)):
        table.append(
            [
                decoded_sentence[i],
                predicted_tokens[i],
                predicted_token_ids[i],
                correct_probs_list[i],
            ]
        )
    print(tabulate.tabulate(table, headers="firstrow"))
    # save results to a file
    with open("tests/test_train_step.txt", "w") as f:
        f.write(tabulate.tabulate(table, headers="firstrow"))


def test_create_helpful_message_1(uncompressed_tokens: TensorType["batch", "seq_len"]):
    helpful_message = create_helpful_message_1(uncompressed_tokens)
    assert helpful_message.shape[1] <= DEFAULT_MSG_CONTEXT_LENGTH


def test_create_openai_helpful_msg(causal_lm_tokenizer):
    sentence = "Hi there, I am a textbook on working class americans. The world is full of people who work. And the color of the sky is blue, despite it being cloudy often. Don't let those clouds fool you. Often the clouds are really just a conspiracy from the illuminati. Listen here, you didn't hear this from me though."
    tokens = causal_lm_tokenizer.encode(sentence, return_tensors="pt")
    msg_context_length = 128
    helpful_message = create_openai_helpful_message(
        tokens,
        causal_lm_tokenizer=causal_lm_tokenizer,
        msg_context_length=msg_context_length,
    )
    assert helpful_message.shape == (1, msg_context_length)


def test_load_and_format_dataset(causal_lm, causal_lm_tokenizer):
    current_path = os.path.dirname(os.path.realpath(__file__))
    textbook_1_path = os.path.join(current_path, "../data/st_patrick_biography.txt")
    dataset, seq_len = load_and_format_dataset(
        textbook_1_path,
        causal_lm_tokenizer,
        train_context_length=causal_lm.config.n_positions,
        reduced_data=1,
    )
    assert len(dataset) == 1
    data_1 = []
    for datum in dataset:
        data_1.append(datum)
    assert seq_len == DEFAULT_MAX_CONTEXT_LENGTH


def test_ability_to_log():
    print("fhjaldksfkladsj;fkladsjfl;kadjs")


if __name__ == "__main__":
    tokens_1 = torch.tensor([list(range(1024))])
    test_create_helpful_message_1(tokens_1)
