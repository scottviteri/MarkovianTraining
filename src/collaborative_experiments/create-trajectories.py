# %%

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
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

# %%

causal_lm = AutoModelForCausalLM.from_pretrained("distilgpt2")
causal_lm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# https://www.gutenberg.org/ebooks/71431
current_path = os.path.dirname(os.path.realpath(__file__))
textbook_1_path = "/home/scottviteri/Projects/CollaborativeTraining/CollaborativeTraining/data/st_patrick_biography.txt"
data_loader, seq_len = load_and_format_dataset(
    textbook_1_path,
    causal_lm_tokenizer,
    debug_dataset_size=10,
    training_context_length=100
)

loss_fn = torch.nn.CrossEntropyLoss()

#%%

aor = torch.tensor([[causal_lm_tokenizer.bos_token_id]])
aor_separated = {"a":[], "o":[], "r":[]}
for data in data_loader:
    # Generate 100 tokens from causal lm and store it in "a"
    # Keep the logits around for later indexing
    with torch.no_grad():
        outputs = causal_lm.generate(aor[:,-causal_lm.config.n_ctx//2:], output_scores=True, return_dict_in_generate=True, max_length=100)
    # Let "a" be the argmax logit indices across outputs
    a: TensorType["batch", "seq_length"] = outputs.sequences
    o: TensorType["batch", "seq_length"] = data  # Second part of the tensor
    # Selecting the first element from the batch dimension
    logits_first_element: TensorType["seq_len", "vocab_size"] = torch.cat(outputs.scores) 
    # Shape: ["seq_len", "vocab_size"]
    # Calculating the loss between the logits and the second part of the tensor
    # Using elements of o as indices into logits_first_element to calculate cross entropy loss
    scalar_reward: str = str(logits_first_element[range(o.shape[0]), o].mean().item())
    # Encode scalar reward as "r", a tensor of integers by tokenizing the scalar reward
    r: TensorType["batch", "seq_length"] = causal_lm_tokenizer(scalar_reward, return_tensors="pt").input_ids
    aor = torch.concat((aor,a,o,r), dim=-1)
    aor_separated["a"].append(a)
    aor_separated["o"].append(o)
    aor_separated["r"].append(r)

# %%

r_list: List[TensorType] = aor_separated["r"]
a_list: List[TensorType] = aor_separated["a"]
o_list: List[TensorType] = aor_separated["o"]

#rao: TensorType = torch.tensor([]) 
#for r, a, o in zip(r_list, a_list, o_list):
#    print(r.shape, a.shape, o.shape)
#    #torch.Size([1, 9]) torch.Size([1, 100]) torch.Size([1, 100])
#    new_rao: TensorType = torch.cat((r, a, o), dim=-1)
#    rao = torch.concat((rao, new_rao))

# %%
