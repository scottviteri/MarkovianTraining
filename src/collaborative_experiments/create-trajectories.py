import os

import torchtyping
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset, Dataset, Features, Value, Sequence, Array2D
from peft import LoraConfig, get_peft_model

import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from einops import rearrange

from typing import List
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
TOKENS_PER_ACTION = 10
TOKENS_PER_OBSERVATION = 50
# make each have 20 substrings 
OBSERVATIONS_PER_DOCUMENT = 2 
TOKENS_PER_DOCUMENT = TOKENS_PER_OBSERVATION * OBSERVATIONS_PER_DOCUMENT 
MODEL = "distilgpt2" #"gpt2-xl" # distilgpt2  ;  EleutherAI/gpt-j-6b   ;
BATCH_SIZE = 2
NUM_BATCHES=10

"""
We will pull in passages from wikipedia articles. 
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""

print("Loading Models")
if MODEL == "distilgpt2":
    causal_lm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "distilgpt2", padding_side="left"
    )
elif MODEL == "gptj":
    causal_lm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6b", padding_side="left"
    )
elif MODEL == "gpt2-large":
    causal_lm = AutoModelForCausalLM.from_pretrained("gpt2-large").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-large", padding_side="left"
    )
elif MODEL == "gpt2-xl":
    causal_lm = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", padding_side="left")

causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id

print("Loading Data")

dataset = load_dataset("wikipedia", "20220301.frr")
dataset = dataset.map(lambda example: causal_lm_tokenizer(example["text"]), batched=True) 
# removing attention mask from features because the segments are all valid tokens
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text"])
dataset.set_format(type="torch", columns=["input_ids", "text"])

# Define your features
truncated_document_features = Features({
    'text': Value(dtype='string', id=None),
    'input_ids': Array2D(shape=(TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT), dtype='int32')
})

truncated_documents = {
    "text": [],
    "input_ids": []
}

for document in dataset["train"]:
    # Only accept long enough samples
    if document["input_ids"].shape[-1] >= TOKENS_PER_DOCUMENT:
        # truncate documents
        truncated_tokens = document["input_ids"][:TOKENS_PER_DOCUMENT]
        # view() places elements in reverse dimension order, aka into the TOKENS_PER_OBSERVATION dim
        truncated_documents["input_ids"].append(truncated_tokens.view((TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT)))
        # leave the text alone (not truncated)
        truncated_documents["text"].append(document["text"])

# Convert the list of dictionaries into a Dataset in one go
truncated_dataset = Dataset.from_dict(truncated_documents, features=truncated_document_features)
truncated_dataset.set_format(
    type="torch", columns=["input_ids", "text"] 
)

loss_fn = torch.nn.CrossEntropyLoss()

def save_traj_to_drive(rao_list, bdebug: bool = False):
    # Create a new data set for SFT from all_rao
    features = Features(
        {
            "input_ids": Sequence(
                feature=Value(dtype="int32", id=None), length=-1, id=None
            ),
        }
    )

    buffer = {
        "input_ids": [],
    }
    for smp_rao in rao_sequences:
        sequ_ids = None
        print(len(smp_rao))
        for myrao in smp_rao:
            if sequ_ids is None:
                sequ_ids = torch.concat([myrao.r, myrao.a, myrao.o])
            else:
                sequ_ids = torch.concat([sequ_ids, myrao.r, myrao.a, myrao.o])

        buffer["input_ids"].append(sequ_ids)

    dataset_rao = Dataset.from_dict(buffer, features=features)
    dataset_rao.set_format(type="torch", columns=["input_ids"])
    dataset_rao.save_to_disk("training_rao_test")

@dataclass
class MyRAO:
    r: torchtyping.TensorType
    a: torchtyping.TensorType
    o: torchtyping.TensorType

dataloader = DataLoader(truncated_dataset, batch_size=BATCH_SIZE)
high_reward = causal_lm_tokenizer(["0.1 " for _ in range(BATCH_SIZE)], return_tensors="pt").input_ids.to(DEVICE)
rao_sequences = []
i = 0
optimizer = torch.optim.Adam(causal_lm.parameters())
for data in dataloader:
    if i > NUM_BATCHES: break
    i += 1
    rao_tensor = torch.tensor([[] for _ in range(BATCH_SIZE)], device=DEVICE, dtype=torch.int32)
    rao_sequence = []
    aggregate_reward = 0
    for observation_index in range(OBSERVATIONS_PER_DOCUMENT): 
        optimizer.zero_grad()
        incentive_rao = torch.cat((rao_tensor, high_reward), dim=-1)
        full_action = causal_lm.generate(
            inputs=incentive_rao,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=TOKENS_PER_ACTION,
            pad_token_id=causal_lm_tokenizer.pad_token_id,
            eos_token_id=None # disable the EOS token
        ) 
        action : TensorType["batch", "seq_length"] = full_action.sequences[
            :, -TOKENS_PER_ACTION:
        ]
        true_obs: TensorType["batch", "seq_length"] = data["input_ids"][:,:,observation_index] 
        true_obs = true_obs.to(DEVICE)
        prediction = causal_lm(true_obs, labels=true_obs)
        predicted_logits = prediction.logits
        predicted_obs : TensorType["batch", "seq_length"] = predicted_logits.argmax(dim=-1)
        selected_logits = predicted_logits.gather(dim=2, index=true_obs.unsqueeze(-1)).squeeze(-1)
        batch_loss = selected_logits.mean(dim=-1)
        aggregate_loss = batch_loss.mean()
        print("aggregate loss: ", aggregate_loss)
        print("action: ", causal_lm_tokenizer.batch_decode(action))
        print("predicted obs: ", causal_lm_tokenizer.batch_decode(predicted_obs))
        print("true obs:", causal_lm_tokenizer.batch_decode(true_obs))
        aggregate_loss.backward()
        optimizer.step()
        string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
        losses : TensorType["batch", "seq_length"] = causal_lm_tokenizer(
            string_losses, return_tensors="pt", padding=True
        ).input_ids.to(DEVICE)
        rao_tensor = torch.cat((rao_tensor, losses, action, true_obs), dim=-1)[:, -causal_lm.config.n_ctx//3:]
        rao_sequence.append([MyRAO(r=losses[i], a=action[i], o=true_obs[i]) for i in range(BATCH_SIZE)])

    for b in range(BATCH_SIZE):
        rao_sequences.append([rao_batch[b] for rao_batch in rao_sequence])
        
    if len(rao_sequences) % 5 == 0 and len(rao_sequences) > 2:
        save_traj_to_drive(rao_sequences, bdebug=False)

    if len(rao_sequences) % 10 == 0 and len(rao_sequences) > 2:
        save_traj_to_drive(rao_sequences, bdebug=True)

