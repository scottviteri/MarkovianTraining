import os

import torchtyping
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, Features, Value, Sequence
from peft import LoraConfig, get_peft_model

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from dataclasses import dataclass
import fire
import wandb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import List
from torchtyping import TensorType

DEVICE = "cuda"  # "mps"
MAX_SEQU = 50
IND_CUT = MAX_SEQU * 20
# distilgpt2  ;  EleutherAI/gpt-j-6b   ;
MODEL = "distilgpt2"

print("Loading Models")
causal_lm = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE)
causal_lm_tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")

causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id

print("Loading Data")

dataset = load_dataset("wikipedia", "20220301.frr")


def tokenization(example):
    return causal_lm_tokenizer(example["text"])


dataset = dataset.map(tokenization, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text"])


# Define the features of your dataset
features = Features(
    {
        "text": Value(dtype="string", id=None),
        "input_ids": Sequence(
            feature=Value(dtype="int32", id=None), length=-1, id=None
        ),
        "attention_mask": Sequence(
            feature=Value(dtype="int8", id=None), length=-1, id=None
        ),
    }
)

buffer = {
    "text": [],
    "input_ids": [],
    "attention_mask": [],
}
for smp in dataset["train"]:
    # Only accept long enough samples
    if smp["input_ids"].shape[-1] >= IND_CUT:
        for key in buffer:
            buffer[key].append(smp[key])

# Convert the list of dictionaries into a Dataset in one go
dataset_resampled = Dataset.from_dict(buffer, features=features)
dataset_resampled.set_format(
    type="torch", columns=["input_ids", "attention_mask", "text"]
)


print("Done Loading")
loss_fn = torch.nn.CrossEntropyLoss()


@dataclass
class MyRAO:
    r: torchtyping.TensorType
    a: torchtyping.TensorType
    o: torchtyping.TensorType


high_reward = causal_lm_tokenizer("0.0", return_tensors="pt").input_ids.to(DEVICE)
all_rao = []
for data in dataset_resampled:
    rao = torch.tensor([[]], device=DEVICE, dtype=torch.int32)
    rao_separated = []

    for smp in range(data["input_ids"].shape[-1] // MAX_SEQU):
        # Generate 100 tokens from causal lm and store it in "a"
        # Keep the logits around for later indexing

        curr_input_ids = data["input_ids"][smp * MAX_SEQU : (smp + 1) * MAX_SEQU]

        with torch.no_grad():
            incentive_rao = torch.cat((rao, high_reward), dim=-1)[
                :, -causal_lm.config.n_ctx // 2 :
            ]
            outputs = causal_lm.generate(
                inputs=incentive_rao,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=MAX_SEQU,
                pad_token_id=causal_lm_tokenizer.pad_token_id,
            )

        a: TensorType["batch", "seq_length"] = outputs.sequences
        o: TensorType["batch", "seq_length"] = curr_input_ids.view(1, -1).to(DEVICE)

        # Selecting the first element from the batch dimension
        logits: TensorType["seq_len", "vocab_size"] = torch.cat(outputs.scores)
        # Shape: ["seq_len", "vocab_size"]
        # Calculating the loss between the logits and the second part of the tensor
        # Using elements of o as indices into logits_first_element to calculate cross entropy loss
        scalar_reward: str = str(round(logits[range(o.shape[0]), o].mean().item(), 3))
        # Encode scalar reward as "r", a tensor of integers by tokenizing the scalar reward
        r: TensorType["batch", "seq_length"] = causal_lm_tokenizer(
            scalar_reward, return_tensors="pt"
        ).input_ids.to(DEVICE)

        rao = torch.concat((rao, r, a, o), dim=-1)
        curr_rao = MyRAO(r=r, a=a, o=o)
        rao_separated.append(curr_rao)

    all_rao.append(rao_separated)

    # if len(all_rao) >= 3:
    #    break


# Create a new data set for SFT from all_rao
features = Features(
    {
        "input_ids": Sequence(
            feature=Value(dtype="int32", id=None), length=-1, id=None
        ),
        "attention_mask": Sequence(
            feature=Value(dtype="int8", id=None), length=-1, id=None
        ),
    }
)

buffer = {
    "input_ids": [],
    "attention_mask": [],
}
for smp_rao in all_rao:
    sequ_ids = None
    print(len(smp_rao))
    for myrao in smp_rao:
        if sequ_ids is None:
            sequ_ids = torch.concat([myrao.r[0], myrao.a[0], myrao.o[0]])
        else:
            sequ_ids = torch.concat([sequ_ids, myrao.r[0], myrao.a[0], myrao.o[0]])

    buffer["input_ids"].append(sequ_ids)
    buffer["attention_mask"].append(torch.ones(sequ_ids.shape, dtype=torch.int8))

dataset_rao = Dataset.from_dict(buffer, features=features)
dataset_rao.set_format(type="torch", columns=["input_ids", "attention_mask"])
# dataset_rao.save_to_disk("training_rao_test")