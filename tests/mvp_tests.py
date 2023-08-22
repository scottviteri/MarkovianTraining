"""
A file for testing the functions used in the mvp_loss_decrease.py file.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from collaborative_experiments.mvp_loss_decrease import (
    get_device,
    load_and_format_dataset,
    create_helpful_message_1,
)
from collaborative_experiments.constants import MAX_CONTEXT_LENGTH, MSG_CONTEXT_LENGTH

@pytest.fixture
def causal_lm_tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")

@pytest.fixture
def causal_lm():
    return AutoModelForCausalLM.from_pretrained("gpt2")

@pytest.fixture
def tokens():
    return torch.tensor([list(range(1024))])

def test_create_helpful_message_1(tokens):
    """
    Get's a tokens vector of shape (1, 1024) from the fixture and then removes the last 
    MSG_CONTEXT_LENGTH tokens. We feed this into create_helpful_message_1 and check that
    the first MSG_CONTEXT_LENGTH tokens of the output are the same as the next MSG_CONTEXT_LENGTH
    tokens of the input.
    """
    tokens_input = tokens[0, :-MSG_CONTEXT_LENGTH]  # Removing the last MSG_CONTEXT_LENGTH tokens
    actual_output = create_helpful_message_1(tokens_input.unsqueeze(dim=0))

    # Check that the first MSG_CONTEXT_LENGTH tokens of the output are the same as expected
    assert torch.equal(actual_output[0, :MSG_CONTEXT_LENGTH], tokens_input[:MSG_CONTEXT_LENGTH]), "The tokens do not match as expected."
    assert actual_output.shape[1] == MAX_CONTEXT_LENGTH, "The output shape is not as expected."

if __name__ == "__main__":
    tokens_1 = torch.tensor([list(range(1024))])
    test_create_helpful_message_1(tokens_1)