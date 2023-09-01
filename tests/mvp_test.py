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

from collaborative_experiments.mvp_loss_decrease import (
    create_helpful_message_1,
    create_openai_helpful_message,
    train_step,
    combine_msg_and_data,
)
from collaborative_experiments.constants import DEFAULT_MAX_CONTEXT_LENGTH, DEFAULT_MSG_CONTEXT_LENGTH
from collaborative_experiments.utils import (
    get_device,
    load_and_format_dataset,
)

@pytest.fixture
def causal_lm_tokenizer():
    return AutoTokenizer.from_pretrained("distilgpt2")

@pytest.fixture
def causal_lm():
    return AutoModelForCausalLM.from_pretrained("distilgpt2")

@pytest.fixture
def tokens():
    return torch.tensor([list(range(1024))])

@pytest.mark.parametrize("msg_fn", [create_helpful_message_1])
def test_combine_msg_and_data_shape(tokens, msg_fn):
    # tokens have shape [1, 1024]
    og_tokens = tokens.clone()
    tokens = tokens[:, :-DEFAULT_MSG_CONTEXT_LENGTH]
    assert tokens.shape[1] == DEFAULT_MAX_CONTEXT_LENGTH - DEFAULT_MSG_CONTEXT_LENGTH
    helpful_msg = msg_fn(tokens, tokens_to_grab=DEFAULT_MSG_CONTEXT_LENGTH)
    combined = combine_msg_and_data(helpful_msg, tokens, msg_context_length=DEFAULT_MSG_CONTEXT_LENGTH)
    assert np.all(combined.shape == og_tokens.shape), "failed to reconstruct tokens"

def test_create_helpful_message_1(tokens):
    tokens_input = tokens[0, :-DEFAULT_MSG_CONTEXT_LENGTH]  # Removing the last MSG_CONTEXT_LENGTH tokens
    msg = create_helpful_message_1(tokens_input.unsqueeze(dim=0))
    actual_output = combine_msg_and_data(msg, tokens_input.unsqueeze(dim=0), msg_context_length=DEFAULT_MSG_CONTEXT_LENGTH)

    # Check that the first MSG_CONTEXT_LENGTH tokens of the output are the same as expected
    assert torch.equal(actual_output[0, :DEFAULT_MSG_CONTEXT_LENGTH], tokens_input[:DEFAULT_MSG_CONTEXT_LENGTH]), "The tokens do not match as expected."
    assert actual_output.shape[1] == DEFAULT_MAX_CONTEXT_LENGTH, "The output shape is not as expected."

def test_train_step(causal_lm, causal_lm_tokenizer):
    # visualizes it
    sentence = "Hello, my name is john. I like apples. aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    tokens = causal_lm_tokenizer.encode(sentence, return_tensors="pt")
    device = get_device()
    tokens = tokens.to(device) # (1, 24) == (1, seq_len)
    causal_lm = causal_lm.to(device)

    # batch, causal_lm, loss_fn, device, correct_probs_all
    loss_fn = torch.nn.CrossEntropyLoss()
    batch = {}
    batch["msg"] = tokens[:, :3]
    batch["content"] = tokens[:, 3:]
    loss, correct_probs, logits = train_step(batch, causal_lm, loss_fn, device, pytest=True)
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
    if len(decoded_sentence) != len(predicted_tokens) or len(decoded_sentence) != len(predicted_token_ids) or len(decoded_sentence) != len(correct_probs_list):
        raise ValueError(f"The lengths of the lists are not the same. decoded_sentence: {len(decoded_sentence)}, predicted_tokens: {len(predicted_tokens)}, predicted_token_ids: {len(predicted_token_ids)}, correct_probs: {len(correct_probs_list)}")
    for i in range(len(decoded_sentence)):
        table.append([decoded_sentence[i], predicted_tokens[i], predicted_token_ids[i], correct_probs_list[i]])
    print(tabulate.tabulate(table, headers="firstrow"))
    # save results to a file
    with open("tests/test_train_step.txt", "w") as f:
        f.write(tabulate.tabulate(table, headers="firstrow"))

@pytest.mark.parametrize("msg_fn", [create_helpful_message_1, create_openai_helpful_message])
def test_variable_context_lengths(causal_lm_tokenizer, msg_fn):
    """
    This test verifies that load_and_format_dataset handles different msg_context_lengths and total_context lengths properly.
    And then verifies that helpful_message_one does the proper thing with this. 
    """
    train_context_length = 256
    msg_context_length = 128
    dataset_path =  "data/st_patrick_biography.txt"
    dataset_tensor = load_and_format_dataset(dataset_path, causal_lm_tokenizer, reduced_data=2, train_context_length=train_context_length, msg_context_length=msg_context_length)
    # ^ hase shape [10, 32]
    print(dataset_tensor.shape)
    data_sample = dataset_tensor[-1, :]
    print(data_sample.shape) # [32,]
    data_sample_str = [causal_lm_tokenizer.decode(token) for token in data_sample]
    if msg_fn.__name__ == "create_openai_helpful_message":
        message = msg_fn(data_sample.unsqueeze(dim=0), causal_lm_tokenizer=causal_lm_tokenizer)
    else:
        message = msg_fn(data_sample.unsqueeze(dim=0), tokens_to_grab=msg_context_length)
    transformed_sample = combine_msg_and_data(message, data_sample.unsqueeze(dim=0), msg_context_length=msg_context_length)
    transformed_sample_str = [causal_lm_tokenizer.decode(token) for token in transformed_sample[0]]
    print(transformed_sample.shape) # [1, 64]
    # print(transformed_sample_str)

    logging_dict = defaultdict(list)
    # diff_in_size = len(transformed_sample_str) - len(data_sample_str)
    # if diff_in_size <= 0:
    #     logging_dict["data_sample_str"] = ["-"] * diff_in_size
    # else:
    #     logging_dict["transformed_sample_str"] = ["-"] * (-diff_in_size)
    logging_dict["data_sample_str"] = data_sample_str
    logging_dict["transformed_sample_str"] = transformed_sample_str
    logging_dict["message"] = [causal_lm_tokenizer.decode(token) for token in message[0]]

    def safe_get(list, i):
        if i >= len(list):
            return "-"
        return list[i]

    table = [list(logging_dict.keys())]
    for i in range(train_context_length + 1): # plus 1 to verify that it all ends in "-"
        table.append([safe_get(logging_dict[key], i) for key in logging_dict.keys()])
    print(tabulate.tabulate(table, headers="firstrow"))
    # save results to a file
    with open(f"tests/test_variable_context_lengths_{msg_fn.__name__}.txt", "w") as f:
        f.write(tabulate.tabulate(table, headers="firstrow"))

    assert transformed_sample.shape[0] == 1
    if msg_fn.__name__ == "create_openai_helpful_message":
        assert transformed_sample.shape[1] >= train_context_length - msg_context_length
        assert transformed_sample.shape[1] <= train_context_length + train_context_length//4
    else:
        assert transformed_sample.shape[1] == train_context_length

def test_create_openai_helpful_msg(causal_lm_tokenizer):
    sentence = "Hi there, I am a textbook on working class americans. The world is full of people who work. And the color of the sky is blue, despite it being cloudy often. Don't let those clouds fool you. Often the clouds are really just a conspiracy from the illuminati. Listen here, you didn't hear this from me though."
    tokens = causal_lm_tokenizer.encode(sentence, return_tensors="pt")
    helpful_message = create_openai_helpful_message(tokens, causal_lm_tokenizer=causal_lm_tokenizer)


def test_ability_to_log():
    print("fhjaldksfkladsj;fkladsjfl;kadjs")


if __name__ == "__main__":
    tokens_1 = torch.tensor([list(range(1024))])
    test_create_helpful_message_1(tokens_1)