"""
This is a file where we train a model to send more helpful messages to another model.

We will just simply use expert iteration to train the model to say more of the things that
created a more helpful
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from dataclasses import dataclass
import fire
import wandb
import pandas as pd
from tqdm import tqdm

from collaborative_experiments.mvp_loss_decrease import (
    create_model_helpful_message,
    msg_loss,
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
from collaborative_experiments.mocking import mockCausalGPT2

LOG_COLUMNS = ["step", "loss", "msg", "content", "decoded_msg", "decoded_content"]


def generate_msg_data_pairs(
    data_loader,
    msg_context_length,
    causal_lm,
    causal_lm_tokenizer,
    sample_size=1,
    num_data_to_use=2,
):
    data_msg_pairs = []
    for i, datum in enumerate(data_loader):
        if i == num_data_to_use:
            break
        for _ in range(sample_size):
            with torch.no_grad():  # ensures no gradient info is saved
                msg = create_model_helpful_message(
                    datum,
                    causal_lm_tokenizer,
                    causal_lm,
                    max_helpful_message_length=msg_context_length,
                )
            print("Generated message")
            print(msg)
            example = {
                "content": datum,
                "msg": msg,
            }
            data_msg_pairs.append(example)
    return data_msg_pairs


def log_row_fn(example):
    new_example = []
    for key in LOG_COLUMNS:
        value = example[key]
        if key == "loss":
            value = str(value.item())
        elif isinstance(value, torch.Tensor):
            value = value.clone().detach().cpu().numpy()
        new_example.append(value)
    return new_example


def train_step(
    data_loader,
    msg_context_length,
    causal_lm,
    causal_lm_tokenizer,
    device,
    loss_fn,
    optimizer,
    sample_size,
    scheduler=None,
    num_data_to_use=1,
    log_table=None,
    step=-1,
):
    if num_data_to_use != 1:
        raise NotImplementedError(
            "num_data_to_use (a.k.a. super_batch_size) must be 1 for now"
        )
    data_msg_pairs = generate_msg_data_pairs(
        data_loader,
        msg_context_length,
        causal_lm,
        causal_lm_tokenizer,
        sample_size=sample_size,
        num_data_to_use=num_data_to_use,
    )
    for example in data_msg_pairs:
        content = example["content"]
        msg = example["msg"]
        loss, logits_shifted, shifted_model_input, model_input = msg_loss(
            content, msg, causal_lm, loss_fn, device
        )
        print("Loss was", loss)
        example["loss"] = loss
        if log_table is not None:
            example["decoded_msg"] = causal_lm_tokenizer.decode(msg[0])
            example["decoded_content"] = causal_lm_tokenizer.decode(content[0])
            example["step"] = step
            log_table.add_data(*log_row_fn(example))
    # 3. Rank the better ones, (also log all the losses in case we want to do RLHF eventually)
    data_msg_pairs.sort(key=lambda x: x["loss"])
    # fine tune on the best one
    best_example = data_msg_pairs[0]
    content = best_example["content"]
    msg = best_example["msg"]

    causal_lm.train()
    loss, logits_shifted, shifted_model_input, model_input = msg_loss(
        content, msg, causal_lm, loss_fn, device, requires_grad=True
    )
    log_dict = {}
    log_dict["loss"] = loss.item()
    log_dict["content"] = content
    log_dict["msg"] = msg
    log_dict["logits_shifted"] = logits_shifted
    log_dict["shifted_model_input"] = shifted_model_input
    log_dict["model_input"] = model_input
    print(f"Auto regressive loss on msg was {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if scheduler:
        scheduler.step()
    wandb.log(log_dict)


def train(cfg):
    if cfg.wandb:
        wandb.init(project="collaborative_training", config=cfg)

    device = get_device(cfg.device)
    if cfg.model_name == "gpt-neo":
        causal_lm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    if cfg.model_name == "gpt2":
        causal_lm = AutoModelForCausalLM.from_pretrained("gpt2")
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if cfg.model_name == "gpt2-medium":
        causal_lm = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        causal_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    elif cfg.model_name == "mock":
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
    textbook_1_path = os.path.join(current_path, "../../", cfg.data_file_path)
    data_loader, seq_len = load_and_format_dataset(
        textbook_1_path,
        causal_lm_tokenizer,
        # debug=debug,
        reduced_data=cfg.reduced_data,
        train_context_length=cfg.train_context_length,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(causal_lm.parameters(), lr=5e-5)

    log_table = wandb.Table(columns=LOG_COLUMNS)
    for step in tqdm(range(cfg.epochs), desc="Epochs"):
        train_step(
            data_loader,
            cfg.msg_context_length,
            causal_lm,
            causal_lm_tokenizer,
            device,
            loss_fn,
            optimizer,
            sample_size=cfg.sample_size,
            num_data_to_use=cfg.super_batch_size,
            log_table=log_table,
            step=step,
        )
    wandb.log({"log_table": log_table})


@dataclass
class TrainConfig:
    experiment_name: str = "train-helpful-message-via-expert-iteration"
    sample_size: int = 2
    super_batch_size: int = 1
    model_name: str = "distilgpt2"
    reduced_data: int = 10
    data_file_path: str = "data/st_patrick_biography.txt"
    train_context_length: int = 256
    msg_context_length: int = 64
    epochs: int = 1
    wandb: bool = True
    device: str = "cpu" # mps


def main(
    sample_size=4,
    super_batch_size=1,
    model_name= "distilgpt2", #"gpt2-medium", # "gpt-neo", # "distilgpt2",
    reduced_data=10,
    data_file_path="data/st_patrick_biography.txt",
    train_context_length=256,
    msg_context_length=64,
    epochs=2,
):
    """
    Args:
        sample size (int): the number of messages to generate per data point
        super_batch_size (int): the number of data points to use per batch
    """

    cfg = TrainConfig(
        sample_size=sample_size,
        super_batch_size=super_batch_size,
        model_name=model_name,
        reduced_data=reduced_data,
        data_file_path=data_file_path,
        train_context_length=train_context_length,
        msg_context_length=msg_context_length,
        epochs=epochs,
        # device="mps",  # mps
    )
    train(cfg)
    return True


if __name__ == "__main__":
    fire.Fire(main)
