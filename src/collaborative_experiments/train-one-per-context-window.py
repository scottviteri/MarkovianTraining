#mamba activate menv
#pip install transformers datasets==2.14.6 torchtyping==0.1.4  
#pip install peft einops apache_beam==2.51.0 matplotlib wandb
#pip install -U flash-attn --no-build-isolation
import os

import torchtyping
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from datasets import load_dataset, Dataset, Features, Value, Sequence, Array2D
from peft import LoraConfig, get_peft_model

import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from einops import rearrange, repeat
import numpy as np
import math
import wandb

from typing import List
from torch.nn.utils.rnn import pad_sequence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
LOAD_MODEL = False
MODEL = "gpt2" # "gpt2" #"gpt2-xl" #"distilgpt2" #gpt2-large" # distilgpt2  ;  EleutherAI/gpt-j-6b   
WANDB = False 

if WANDB:
    run = wandb.init(project="collaborative-training-many-per-context-window", entity="scottviteri")
    wandb_table = wandb.Table(data = [], columns=["Previous Observation", "Action", "Predicted Observation", "Actual Observation"])
else:
    wandb_table = None

"""
We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""

if  LOAD_MODEL:
    causal_lm_tokenizer = AutoTokenizer.from_pretrained(f"/content/drive/MyDrive/CollaborativeTrainingModelWeights/tokenizer_{MODEL}", padding_size="left")
    causal_lm = AutoModelForCausalLM.from_pretrained(f"/content/drive/MyDrive/CollaborativeTrainingModelWeights/trained_{MODEL}")
    causal_lm.to(DEVICE)
    if MODEL == "gptj": CTXT_WINDOW_SIZE = causal_lm.config.n_positions 
    elif MODEL == "gptj": CTXT_WINDOW_SIZE = causal_lm.config.sliding_window
    else: CTXT_WINDOW_SIZE = causal_lm.config.n_ctx

elif MODEL == "mistral":
    causal_lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, use_flash_attention_2=True).to(DEVICE)
    causal_lm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
    CTXT_WINDOW_SIZE = causal_lm.config.sliding_window

elif MODEL == "distilgpt2":
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

def get_linear_layers(model):
    return set(map(lambda x:x[0].split('.')[-1], 
                   filter(lambda x:isinstance(x[1],torch.nn.Linear), causal_lm.named_modules())))

peft_config = LoraConfig(
        #base_model_name_or_path=MODEL,
        r = 64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=get_linear_layers(causal_lm)
    )

print("Loading Models")

causal_lm = get_peft_model(causal_lm, peft_config)
causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id


TOKENS_PER_REWARD = 10 
TOKENS_PER_ACTION = 100
TOKENS_PER_OBSERVATION = CTXT_WINDOW_SIZE - TOKENS_PER_ACTION - TOKENS_PER_REWARD
OBSERVATIONS_PER_DOCUMENT =  5
TOKENS_PER_RAO = TOKENS_PER_REWARD + TOKENS_PER_ACTION + TOKENS_PER_OBSERVATION
TOKENS_PER_DOCUMENT = TOKENS_PER_OBSERVATION * OBSERVATIONS_PER_DOCUMENT
BATCH_SIZE = 2
NUM_BATCHES = 4 # None
NUM_DATAPOINTS = BATCH_SIZE * NUM_BATCHES if NUM_BATCHES else None
SAVE_WEIGHTS_INTERVAL = 30 
PRINT_INTERVAL = 5 if MODEL == "gptj" or MODEL == "mistral" else 10
SAVE_DIRECTORY = "/home/scottviteri/Projects/CollaborativeTraining/CollaborativeTraining/saved_weights_and_losses"

POINTS_FROM_DATASET = NUM_DATAPOINTS
truncated_documents = {
    "text": [],
    "input_ids": []
}

while len(truncated_documents["input_ids"]) < NUM_DATAPOINTS:
    # This while loop is used to load the dataset. It will keep running until a valid dataset is loaded.
    # The termination condition is when a valid dataset is loaded without any exceptions.
    # not creating enough datapoints
    dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{POINTS_FROM_DATASET}]" if POINTS_FROM_DATASET else "train")
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

    observation_prefix = "Observation: "
    observation_prefix_tokens = causal_lm_tokenizer.encode(observation_prefix, add_special_tokens=False)
    tokens_per_pure_observation = TOKENS_PER_OBSERVATION - len(observation_prefix_tokens)

    # currently need to use tensors in input_ids as per the truncated_document_features
    for document in dataset:
        if NUM_DATAPOINTS and len(truncated_documents["input_ids"]) == NUM_DATAPOINTS: break
        # Only accept long enough samples
        if document["input_ids"].shape[-1] <= TOKENS_PER_DOCUMENT: continue 
        # truncate documents
        tokens = document["input_ids"].tolist()
        # Inject "Observation: " prefix every TOKENS_PER_OBSERVATION - observation_prefix_tokens
        new_tokens = []
        for j in range(0, len(tokens), tokens_per_pure_observation):
            new_tokens.extend(observation_prefix_tokens)
            new_tokens.extend(tokens[j:j+tokens_per_pure_observation])
            #if len(new_tokens) >= TOKENS_PER_DOCUMENT: break
        new_document = torch.tensor(new_tokens[:TOKENS_PER_DOCUMENT])
        reshaped_document = new_document.reshape((OBSERVATIONS_PER_DOCUMENT, TOKENS_PER_OBSERVATION))
        truncated_documents["input_ids"].append(reshaped_document)
        assert len(new_tokens[:TOKENS_PER_DOCUMENT]) == TOKENS_PER_DOCUMENT
        # leave the text alone (not truncated)
        truncated_documents["text"].append(document["text"])
    if not NUM_DATAPOINTS: 
        NUM_DATAPOINTS = len(truncated_documents["text"])
        NUM_BATCHES = NUM_DATAPOINTS // BATCH_SIZE
        break
    POINTS_FROM_DATASET *= 2

assert len(truncated_documents["input_ids"]) == NUM_DATAPOINTS

# Convert the list of dictionaries into a Dataset in one go
truncated_dataset = Dataset.from_dict(truncated_documents, features=truncated_document_features)
truncated_dataset.set_format(
    type="torch", columns=["input_ids", "text"]
)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

@dataclass
class MyRAO:
    r: torchtyping.TensorType
    a: torchtyping.TensorType
    o: torchtyping.TensorType

def log_and_print_info(batch_index, observation_index, batch_loss, aggregate_losses, prev_obs, action, predicted_obs, true_obs, optimizer, wandb_table, causal_lm_tokenizer, MODEL):
    if batch_index % PRINT_INTERVAL == 0 and observation_index % PRINT_INTERVAL == 0:
        print(f"\nBatch number {batch_index}")
        print("batch loss: ", batch_loss[0])
        if aggregate_losses: print("aggregate loss: ", aggregate_losses[-1])
        print("previous obs:", repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]))
        print("action: ", repr(causal_lm_tokenizer.batch_decode(action)[0]))
        print("predicted obs: ", repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]))
        print("true obs:", repr(causal_lm_tokenizer.batch_decode(true_obs)[0]))
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group["lr"])
    with open(f'{SAVE_DIRECTORY}/{MODEL}_training_info.txt', 'a') as f:
        print(f"\nBatch number {batch_index}", file=f)
        print("batch loss: ", batch_loss[0], file=f)
        if aggregate_losses: print("aggregate loss: ", aggregate_losses[-1], file=f)
        print("previous obs:", repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]), file=f)
        print("action: ", repr(causal_lm_tokenizer.batch_decode(action)[0]), file=f)
        print("predicted obs: ", repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]), file=f)
        print("true obs:", repr(causal_lm_tokenizer.batch_decode(true_obs)[0]), file=f)
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group["lr"], file=f)
    if WANDB:
        wandb.log({
            "Batch number": batch_index,
            "Batch Loss": batch_loss[0].item(),
            #"Aggregate loss": aggregate_losses[-1] if aggregate_losses else -1,
            "Current learning rate": [g["lr"] for g in optimizer.param_groups if "lr" in g][0]
        })
        wandb_table.add_data(
            repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]), 
            repr(causal_lm_tokenizer.batch_decode(action)[0]), 
            repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]),
            repr(causal_lm_tokenizer.batch_decode(true_obs)[0]))

dataloader = DataLoader(truncated_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
i = 0
aggregate_losses = []
optimizer = torch.optim.Adam(causal_lm.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_BATCHES)
with open(f'{SAVE_DIRECTORY}/{MODEL}_training_info.txt', 'w') as f: pass

action_prefix = "Action: "
action_prefix_tokens = causal_lm_tokenizer.encode(action_prefix, add_special_tokens=False)
action_prefix_tensor = repeat(torch.tensor(action_prefix_tokens), 'tokens -> batch tokens', batch=BATCH_SIZE).to(DEVICE)
tokens_per_pure_action = TOKENS_PER_ACTION - len(action_prefix_tokens)

for data in tqdm(dataloader, total=NUM_BATCHES) if NUM_BATCHES else tqdm(dataloader):
    if NUM_BATCHES and i > NUM_BATCHES: break
    i += 1
    if i > 1 and i%SAVE_WEIGHTS_INTERVAL == 0: 
        print(f"Saving trained_{MODEL}")
        causal_lm_tokenizer.save_pretrained(f"./saved_weights_and_losses/tokenizer_{MODEL}")
        causal_lm.save_pretrained(f"./saved_weights_and_losses/trained_{MODEL}")
    rao_sequence = []
    for observation_index in range(OBSERVATIONS_PER_DOCUMENT):
        optimizer.zero_grad()
        high_reward_value = round(np.mean(aggregate_losses) - np.std(aggregate_losses),3) if aggregate_losses else 6.0
        high_reward = causal_lm_tokenizer(["Reward: " + str(high_reward_value) for _ in range(BATCH_SIZE)], 
                                          return_tensors="pt", padding="max_length", max_length=TOKENS_PER_REWARD).input_ids
        high_reward = high_reward.to(DEVICE)
        incentive_rao = torch.cat((high_reward, action_prefix_tensor), dim=-1)
        full_action = causal_lm.generate(
            inputs=incentive_rao,
            output_scores=True,
            do_sample=True,
            return_dict_in_generate=True,
            max_new_tokens=tokens_per_pure_action,
            pad_token_id=causal_lm_tokenizer.pad_token_id,
            eos_token_id=None
        )
        action : TensorType["batch", "seq_length"] = full_action.sequences[:, -TOKENS_PER_ACTION:]
        if observation_index > 1:
            prev_obs: TensorType["batch", "seq_length"] = data["input_ids"][:,observation_index-1, :]
        else:
            prev_obs: TensorType["batch", "seq_length"] = torch.full_like(data["input_ids"][:,0, :], causal_lm_tokenizer.pad_token_id)
        true_obs: TensorType["batch", "seq_length"] = data["input_ids"][:,observation_index, :]
        true_obs = true_obs.to(DEVICE)
        with torch.no_grad():
            prediction = causal_lm(torch.cat((high_reward, action, true_obs), dim=-1))
            predicted_logits = prediction.logits[:,-TOKENS_PER_OBSERVATION-1:-1,:]
            predicted_obs = predicted_logits.argmax(dim=-1)
            out = loss_fn(
                input = rearrange(predicted_logits, 'batch seq_length vocab_size -> batch vocab_size seq_length'),
                target = true_obs
            )
            batch_loss = out.mean(dim=-1)
        string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
        losses : TensorType["batch", "seq_length"] = causal_lm_tokenizer(
            string_losses, return_tensors="pt", padding=True
        ).input_ids.to(DEVICE)
        rao_sequence.append([MyRAO(r=losses[i], a=action[i], o=true_obs[i]) for i in range(BATCH_SIZE)])
        log_and_print_info(i, observation_index, batch_loss, aggregate_losses, prev_obs, action, predicted_obs, true_obs, optimizer, wandb_table, causal_lm_tokenizer, MODEL)
    
        # Compute the loss on the whole rao_tensor sequence and perform backpropagation
        rao_tensor = torch.cat((losses, action, true_obs), dim=-1)
        rao_tensor_logits = causal_lm(rao_tensor).logits
        rao_tensor_loss = loss_fn(
            input = rearrange(rao_tensor_logits, 'batch seq_length vocab_size -> batch vocab_size seq_length'),
            target = rao_tensor
        )
        aggregate_loss = rao_tensor_loss.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()
        print("Aggregate loss: ", aggregate_loss)
        if WANDB: wandb.log({"Aggregate loss": aggregate_loss})
        optimizer.step()
    scheduler.step()

if WANDB:
    run.log({"Prediction Accuracy Table": wandb_table})
    wandb.finish()
