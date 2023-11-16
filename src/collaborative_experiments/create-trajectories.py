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
import numpy as np
import math

from transformers import GPTJPreTrainedModel
from transformers import GPT2LMHeadModel
from typing import List
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
TOKENS_PER_ACTION = 10
TOKENS_PER_OBSERVATION = 20
# make each have 20 substrings
OBSERVATIONS_PER_DOCUMENT = 10
TOKENS_PER_DOCUMENT = TOKENS_PER_OBSERVATION * OBSERVATIONS_PER_DOCUMENT
MODEL = "gptj" #"gpt2-xl" #"distilgpt2" #gpt2-large" # distilgpt2  ;  EleutherAI/gpt-j-6b   
BATCH_SIZE = 4 
NUM_BATCHES = 1000
NUM_DATAPOINTS = BATCH_SIZE * NUM_BATCHES
ENTROPY_PENALTY = False

"""
We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""

print("Loading Models")

peft_config = LoraConfig(
        #base_model_name_or_path=MODEL,
        r = 64,
        lora_alpha=32,
        lora_dropout=0.1
        #target_modules=["query","values"]
    )


if MODEL == "distilgpt2":
    causal_lm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "distilgpt2", padding_side="left"
    )
    CTXT_WINDOW_SIZE = causal_lm.config.n_ctx

elif MODEL == "gptj":
    causal_lm = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-j-6b", padding_side="left"
    )
    CTXT_WINDOW_SIZE = causal_lm.config.n_positions

elif MODEL == "gpt2-large":
    causal_lm = AutoModelForCausalLM.from_pretrained("gpt2-large").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-large", padding_side="left"
    )
    CTXT_WINDOW_SIZE = causal_lm.config.n_ctx

elif MODEL == "gpt2-xl":
    causal_lm = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", padding_side="left")
    CTXT_WINDOW_SIZE = causal_lm.config.n_ctx

elif MODEL == "gpt2":
    causal_lm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    CTXT_WINDOW_SIZE = causal_lm.config.n_ctx


causal_lm = get_peft_model(causal_lm, peft_config)
causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id

POINTS_FROM_DATASET = NUM_DATAPOINTS
while 1:
    #dataset = load_dataset("bookcorpus", split=f"train[:{POINTS_FROM_DATASET}]")
    dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{POINTS_FROM_DATASET}]")
    dataset = dataset.map(lambda example: causal_lm_tokenizer(example["text"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "text"])

    # Define your features
    truncated_document_features = Features({
        'text': Value(dtype='string', id=None),
        #'input_ids': Array2D(shape=(TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT), dtype='int32')
        'input_ids': Array2D(shape=(OBSERVATIONS_PER_DOCUMENT, TOKENS_PER_OBSERVATION), dtype='int32')
    })

    truncated_documents = {
        "text": [],
        "input_ids": []
    }

    i = 0
    for document in dataset:
        if i == NUM_DATAPOINTS: break
        # Only accept long enough samples
        if document["input_ids"].shape[-1] >= TOKENS_PER_DOCUMENT:
            i += 1
            # truncate documents
            truncated_tokens = document["input_ids"][:TOKENS_PER_DOCUMENT]
            # view() places elements in reverse dimension order, aka into the TOKENS_PER_OBSERVATION dim
            #truncated_documents["input_ids"].append(truncated_tokens.view((TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT)))
            truncated_documents["input_ids"].append(truncated_tokens.view((OBSERVATIONS_PER_DOCUMENT, TOKENS_PER_OBSERVATION)))
            # leave the text alone (not truncated)
            truncated_documents["text"].append(document["text"])

    if i == NUM_DATAPOINTS: break
    POINTS_FROM_DATASET *= 2


# Convert the list of dictionaries into a Dataset in one go
truncated_dataset = Dataset.from_dict(truncated_documents, features=truncated_document_features)
truncated_dataset.set_format(
    type="torch", columns=["input_ids", "text"]
)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

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
        #print(len(smp_rao))
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

dataloader = DataLoader(truncated_dataset, batch_size=BATCH_SIZE, drop_last=True)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=total_steps, pct_start=warmup_steps/total_steps)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
#                     base_lr = 1e-4, # Initial learning rate which is the lower boundary in the cycle for each parameter group
#                     max_lr = 1e-3, # Upper learning rate boundaries in the cycle for each parameter group
#                     step_size_up = 4, # Number of training iterations in the increasing half of a cycle
#                     cycle_momentum = False,
#                     mode = "triangular")
#scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, )
rao_sequences = []
i = 0
aggregate_losses = []
optimizer = torch.optim.Adam(causal_lm.parameters(), lr=1e-4)
total_steps = NUM_BATCHES * OBSERVATIONS_PER_DOCUMENT
warmup_steps = int(0.1 * total_steps)

for data in tqdm(dataloader, total=NUM_BATCHES):
    if i > NUM_BATCHES: break
    i += 1
    rao_tensor = torch.tensor([[] for _ in range(BATCH_SIZE)], device=DEVICE, dtype=torch.int32)
    rao_sequence = []
    for observation_index in range(OBSERVATIONS_PER_DOCUMENT):
        #input("hi")
        optimizer.zero_grad()
        high_reward_value = round(np.mean(aggregate_losses) - np.std(aggregate_losses),3) if aggregate_losses else 6.0
        #start_reward = causal_lm_tokenizer([" Reward: " for _ in range(BATCH_SIZE)], return_tensors="pt").input_ids.to(DEVICE)
        high_reward = causal_lm_tokenizer([str(high_reward_value) for _ in range(BATCH_SIZE)], return_tensors="pt").input_ids.to(DEVICE)
        incentive_rao = torch.cat((rao_tensor, high_reward), dim=-1)
        full_action = causal_lm.generate(
            inputs=incentive_rao,
            output_scores=True,
            do_sample=True,
            return_dict_in_generate=True,
            max_new_tokens=TOKENS_PER_ACTION,
            pad_token_id=causal_lm_tokenizer.pad_token_id,
            eos_token_id=None # disable the EOS token
        )
        action : TensorType["batch", "seq_length"] = full_action.sequences[
            :, -TOKENS_PER_ACTION:
        ]
        true_obs: TensorType["batch", "seq_length"] = data["input_ids"][:,observation_index, :]
        true_obs = true_obs.to(DEVICE)
        prediction = causal_lm(torch.cat((incentive_rao, action, true_obs), dim=-1)[:, -CTXT_WINDOW_SIZE//3:])
        predicted_logits = prediction.logits[:,-TOKENS_PER_OBSERVATION-1:-1,:]
        # out shape = (batch seq_length)
        out = loss_fn(
            input = rearrange(predicted_logits, 'batch seq_length vocab_size -> batch vocab_size seq_length'),
            target = true_obs
        )
        batch_loss = out.mean(dim=-1)
        if ENTROPY_PENALTY:
            scores=torch.stack(full_action.scores,dim=1)
            scores.requires_grad = True
            action_probs = torch.softmax(scores[:,-TOKENS_PER_ACTION:], dim=-1) + 1e-10
            batch_entropy = ENTROPY_PENALTY * torch.e ** ( torch.log2(action_probs) * action_probs).sum(dim=-1).mean(dim=-1)
            aggregate_loss = batch_loss.mean() + batch_entropy.mean()
        else:
            aggregate_loss = batch_loss.mean()
        aggregate_losses.append(aggregate_loss.item())
        predicted_obs : TensorType["batch", "seq_length"] = predicted_logits.argmax(dim=-1)
        if observation_index == OBSERVATIONS_PER_DOCUMENT - 1 and i%20==0:
        #if i%(math.ceil(NUM_BATCHES/20.0))==0:
        #if aggregate_loss.item() < (np.mean(aggregate_losses) - np.std(aggregate_losses)):
        #if True:
            print()
            print()
            print("loss: ", batch_loss[0])
            if ENTROPY_PENALTY: print("e ^ negentropy:", batch_entropy[0])
            print("average loss: ", np.mean(aggregate_losses))
            print("action: ", repr(causal_lm_tokenizer.batch_decode(action)[0]))
            print("predicted obs: ", repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]))
            print("true obs:", repr(causal_lm_tokenizer.batch_decode(true_obs)[0]))
        aggregate_loss.backward()
        optimizer.step()
        #scheduler.step()
        string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
        losses : TensorType["batch", "seq_length"] = causal_lm_tokenizer(
            string_losses, return_tensors="pt", padding=True
        ).input_ids.to(DEVICE)
        rao_tensor = torch.cat((rao_tensor, losses, action, true_obs), dim=-1)[:, -CTXT_WINDOW_SIZE//3:]
        rao_sequence.append([MyRAO(r=losses[i], a=action[i], o=true_obs[i]) for i in range(BATCH_SIZE)])

    for b in range(BATCH_SIZE):
        rao_sequences.append([rao_batch[b] for rao_batch in rao_sequence])

    #if len(rao_sequences) % 5 == 0 and len(rao_sequences) > 2:
    #    save_traj_to_drive(rao_sequences, bdebug=False)

    #if len(rao_sequences) % 10 == 0 and len(rao_sequences) > 2:
    #    save_traj_to_drive(rao_sequences, bdebug=True)
