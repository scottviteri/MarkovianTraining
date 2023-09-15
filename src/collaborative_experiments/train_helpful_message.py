"""
This is a file where we train a model to send more helpful messages to another model.

We will just simply use expert iteration to train the model to say more of the things that
created a more helpful
"""
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch
import fire
import wandb


from collaborative_experiments.mvp_loss_decrease import (
    create_model_helpful_message,
    msg_loss
)

from collaborative_experiments.utils import (
    get_device,
    load_and_format_dataset,
    tile_a_tensor,
)
from collaborative_experiments.constants import (
    DEFAULT_MSG_CONTEXT_LENGTH,
    DEFAULT_MAX_CONTEXT_LENGTH,
)
from collaborative_experiments.mocking import (
    mockCausalGPT2
)

def main(
        sample_size=2,
        model_name="distilgpt2",
        reduced_data=10,
        data_file_path="data/st_patrick_biography.txt",
        train_context_length=256,
        msg_context_length=64,
    ):

    # 0. get a data sample
    device = get_device(model_name)
    if model_name == "gpt-neo":
        causal_lm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    elif model_name == "mock":
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        causal_lm = mockCausalGPT2(causal_lm_tokenizer)
    else:
        causal_lm = AutoModelForCausalLM.from_pretrained("distilgpt2")
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    print("Loaded causal LM")
    print(causal_lm)
    causal_lm = causal_lm.to(device)
    # We are setting this token to be eos, so we must make sure to use attention masks
    # to not attend to these positions.
    causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
    print("Loaded causal LM to device")

    # load dataset
    # https://www.gutenberg.org/ebooks/71431
    current_path = os.path.dirname(os.path.realpath(__file__))
    textbook_1_path = os.path.join(current_path, "../../", data_file_path)
    data_loader, seq_len = load_and_format_dataset(
        textbook_1_path,
        causal_lm_tokenizer,
        # debug=debug,
        reduced_data=reduced_data,
        train_context_length=train_context_length,
    )
    data = []
    for datum in data_loader:
        data.append(datum)
        break
    
    # 1. Generate a bunch of messages given the same data
    data_msg_pairs = []
    for _ in range(sample_size):
        tokens = data[0]
        msg = create_model_helpful_message(
            tokens,
            causal_lm_tokenizer,
            causal_lm,
            msg_context_length=msg_context_length
        )
        print("Generated message")
        print(msg)
        example = {
            "content": tokens,
            "msg": msg,
        }
        data_msg_pairs.append(example)

    # 2. Get the loss for each of the messages
    loss_fn = torch.nn.CrossEntropyLoss()
    for example in data_msg_pairs:
        content = example["content"]
        msg = example["msg"]
        loss, logits_shifted, shifted_model_input, model_input = msg_loss(content, msg, causal_lm, loss_fn, device)
        print("Loss was", loss)
        example["loss"] = loss

    # 3. Rank the better ones, (also log all the losses in case we want to do RLHF eventually)
    data_msg_pairs.sort(key=lambda x: x["loss"])
    # fine tune on the best one
    best_example = data_msg_pairs[0]
    content = best_example["content"]
    msg = best_example["msg"]
    optimizer = torch.optim.Adam(causal_lm.parameters(), lr=5e-5)
    causal_lm.train()
    loss, logits_shifted, shifted_model_input, model_input = msg_loss(content, msg, causal_lm, loss_fn, device, requires_grad=True)
    
    print(f"Auto regressive loss on msg was {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # log the loss
    # wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    fire.Fire(main)