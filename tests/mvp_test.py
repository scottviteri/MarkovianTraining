"""
A file for testing the functions used in the mvp_loss_decrease.py file.
"""
import pytest
import tabulate
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

from collaborative_experiments.mvp_loss_decrease import (
    get_device,
    load_and_format_dataset,
    create_helpful_message_1,
    create_helpful_message_2,
    create_openai_helpful_message,
    train_step,
)
from collaborative_experiments.constants import MAX_CONTEXT_LENGTH, MSG_CONTEXT_LENGTH

@pytest.fixture
def causal_lm_tokenizer():
    return AutoTokenizer.from_pretrained("distilgpt2")

@pytest.fixture
def causal_lm():
    return AutoModelForCausalLM.from_pretrained("distilgpt2")

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

def test_train_step(causal_lm, causal_lm_tokenizer):
    # visualizes it
    sentence = "Hello, my name is john. I like apples. aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    tokens = causal_lm_tokenizer.encode(sentence, return_tensors="pt")
    device = get_device()
    tokens = tokens.to(device) # (1, 24) == (1, seq_len)
    causal_lm = causal_lm.to(device)

    correct_probs_all = torch.zeros(tokens.shape[1] - 1).to(device) # (seq_len,)

    # batch, causal_lm, loss_fn, device, correct_probs_all
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, logits = train_step(tokens, causal_lm, loss_fn, device, correct_probs_all, pytest=True)
    # logits have shape (batch_size, seq_len, vocab_size)
    # loss should have shape (batch_size, seq_len)

    decoded_sentence = []
    for token in tokens[0]:
        decoded_sentence.append(causal_lm_tokenizer.decode(token))
    predicted_tokens = ["N/A"]
    predicted_token_ids = [-1]
    for token in logits[0]:
        predicted_id = torch.argmax(token)
        predicted_token_ids.append(predicted_id)
        predicted_tokens.append(causal_lm_tokenizer.decode(predicted_id))
    correct_probs = ["N/A"]
    for i, prob in enumerate(correct_probs_all):
        correct_probs.append(prob.item())
        # correct_prob_idx = tokens[i+1]
        # correct_prob_manual = F.softmax(logits[0, i+1], dim=-1)[tokens[0, i+1]].item()
    # use tabulate to print a table of all this
    table = [
        ["Token", "Predicted Token", "Predicted Token ID", "Correct Probability"],
    ]
    if len(decoded_sentence) != len(predicted_tokens) or len(decoded_sentence) != len(predicted_token_ids) or len(decoded_sentence) != len(correct_probs):
        raise ValueError("The lengths of the lists are not the same.")
    for i in range(len(decoded_sentence)):
        table.append([decoded_sentence[i], predicted_tokens[i], predicted_token_ids[i], correct_probs[i]])
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
    train_context_length = 64
    msg_context_length = 32
    dataset_path =  "data/st_patrick_biography.txt"
    dataset_tensor = load_and_format_dataset(dataset_path, causal_lm_tokenizer, reduced_data=2, train_context_length=train_context_length, msg_context_length=msg_context_length)
    # ^ hase shape [10, 32]
    print(dataset_tensor.shape)
    data_sample = dataset_tensor[-1, :]
    print(data_sample.shape) # [32,]
    data_sample_str = [causal_lm_tokenizer.decode(token) for token in data_sample]
    if msg_fn.__name__ == "create_openai_helpful_message":
        transformed_sample = msg_fn(data_sample.unsqueeze(dim=0), causal_lm_tokenizer=causal_lm_tokenizer)
    else:
        transformed_sample = msg_fn(data_sample.unsqueeze(dim=0), tokens_to_grab=msg_context_length)
    transformed_sample_str = [causal_lm_tokenizer.decode(token) for token in transformed_sample[0]]
    print(transformed_sample.shape) # [1, 64]
    # print(transformed_sample_str)

    logging_dict = defaultdict(list)
    diff_in_size = len(transformed_sample_str) - len(data_sample_str)
    if diff_in_size <= 0:
        logging_dict["data_sample_str"] = ["-"] * diff_in_size
    else:
        logging_dict["transformed_sample_str"] = ["-"] * (-diff_in_size)
    logging_dict["data_sample_str"] += data_sample_str
    logging_dict["transformed_sample_str"] += transformed_sample_str

    table = [
        ["Original", "Transformed Token"],
    ]
    for i in range(len(logging_dict["data_sample_str"])):
        table.append([logging_dict["data_sample_str"][i], logging_dict["transformed_sample_str"][i]])
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

def test_ability_to_log():
    print("fhjaldksfkladsj;fkladsjfl;kadjs")


if __name__ == "__main__":
    tokens_1 = torch.tensor([list(range(1024))])
    test_create_helpful_message_1(tokens_1)