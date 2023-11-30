# mamba activate menv
# pip install transformers datasets==2.14.6 torchtyping==0.1.4
# pip install peft einops apache_beam==2.51.0 matplotlib wandb
# pip install -U flash-attn --no-build-isolation

import torchtyping
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, Features, Value, Array2D
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType
from dataclasses import dataclass
from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
import wandb
from collaborative_experiments.rao_tools import MyRAO, RaoConfig

cfg = RaoConfig(
    load_model=False,
    wandb=False,  # True,
    model_name="distilgpt2",  # "mistral",  # "gpt2" #"gpt2-xl" #"distilgpt2" #gpt2-large" # distilgpt2  ;  EleutherAI/gpt-j-6b
    save_dir=".",  # "/home/scottviteri/Projects/CollaborativeTraining/CollaborativeTraining/saved_weights_and_losses"
    tok_p_reward=10,
    tok_p_action=100,
    tok_p_obs=None,  # CTXT_WINDOW_SIZE - TOKENS_PER_ACTION - TOKENS_PER_REWARD
    obs_p_doc=5,  # 20,
    batch_size=2,
    num_batches=4,
    interval_save_weights=30,
)

if cfg.wandb:
    run = wandb.init(
        project="collaborative-training-many-per-context-window", entity="scottviteri"
    )
    wandb_table = wandb.Table(
        data=[],
        columns=[
            "Previous Observation",
            "Action",
            "Predicted Observation",
            "Actual Observation",
        ],
    )
else:
    wandb_table = None

"""
We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""


NUM_DATAPOINTS = cfg.batch_size * cfg.num_batches if cfg.num_batches else None
causal_lm = cfg.model
causal_lm_tokenizer = cfg.tokenizer
POINTS_FROM_DATASET = NUM_DATAPOINTS
truncated_documents = {"text": [], "input_ids": []}

while len(truncated_documents["input_ids"]) < NUM_DATAPOINTS:
    # This while loop is used to load the dataset. It will keep running until a valid dataset is loaded.
    # The termination condition is when a valid dataset is loaded without any exceptions.
    # not creating enough datapoints
    dataset = load_dataset(
        "wikipedia",
        "20220301.simple",
        split=f"train[:{POINTS_FROM_DATASET}]" if POINTS_FROM_DATASET else "train",
    )
    dataset = dataset.map(
        lambda example: causal_lm_tokenizer(example["text"]), batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "text"])

    # Define your features
    truncated_document_features = Features(
        {
            "text": Value(dtype="string", id=None),
            #'input_ids': Array2D(shape=(TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT), dtype='int32')
            "input_ids": Array2D(shape=(cfg.obs_p_doc, cfg.tok_p_obs), dtype="int32"),
        }
    )

    truncated_documents = {"text": [], "input_ids": []}

    observation_prefix = "Observation: "
    observation_prefix_tokens = causal_lm_tokenizer.encode(
        observation_prefix, add_special_tokens=False
    )
    tokens_per_pure_observation = cfg.tok_p_obs - len(observation_prefix_tokens)

    # currently need to use tensors in input_ids as per the truncated_document_features
    for document in dataset:
        if NUM_DATAPOINTS and len(truncated_documents["input_ids"]) == NUM_DATAPOINTS:
            break
        # Only accept long enough samples
        if document["input_ids"].shape[-1] <= cfg.tok_p_doc:
            continue
        # truncate documents
        tokens = document["input_ids"].tolist()
        # Inject "Observation: " prefix every TOKENS_PER_OBSERVATION - observation_prefix_tokens
        new_tokens = []
        for j in range(0, len(tokens), tokens_per_pure_observation):
            new_tokens.extend(observation_prefix_tokens)
            new_tokens.extend(tokens[j : j + tokens_per_pure_observation])
            # if len(new_tokens) >= TOKENS_PER_DOCUMENT: break
        new_document = torch.tensor(new_tokens[: cfg.tok_p_doc])
        reshaped_document = new_document.reshape((cfg.obs_p_doc, cfg.tok_p_obs))
        truncated_documents["input_ids"].append(reshaped_document)
        assert len(new_tokens[: cfg.tok_p_doc]) == cfg.tok_p_doc
        # leave the text alone (not truncated)
        truncated_documents["text"].append(document["text"])
    if not NUM_DATAPOINTS:
        NUM_DATAPOINTS = len(truncated_documents["text"])
        NUM_BATCHES = NUM_DATAPOINTS // cfg.batch_size
        break
    POINTS_FROM_DATASET *= 2

assert len(truncated_documents["input_ids"]) == NUM_DATAPOINTS

# Convert the list of dictionaries into a Dataset in one go
truncated_dataset = Dataset.from_dict(
    truncated_documents, features=truncated_document_features
)
truncated_dataset.set_format(type="torch", columns=["input_ids", "text"])

loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


def log_and_print_info(
    batch_index,
    observation_index,
    batch_loss,
    aggregate_losses,
    prev_obs,
    action,
    predicted_obs,
    true_obs,
    optimizer,
    wandb_table,
    causal_lm_tokenizer,
):
    if (
        batch_index % cfg.interval_print == 0
        and observation_index % cfg.interval_print == 0
    ):
        print(f"\nBatch number {batch_index}")
        print("batch loss: ", batch_loss[0])
        if aggregate_losses:
            print("aggregate loss: ", aggregate_losses[-1])
        print("previous obs:", repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]))
        print("action: ", repr(causal_lm_tokenizer.batch_decode(action)[0]))
        print(
            "predicted obs: ", repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0])
        )
        print("true obs:", repr(causal_lm_tokenizer.batch_decode(true_obs)[0]))
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group["lr"])
    with open(f"{cfg.save_dir}/{cfg.model_name}_training_info.txt", "a") as f:
        print(f"\nBatch number {batch_index}", file=f)
        print("batch loss: ", batch_loss[0], file=f)
        if aggregate_losses:
            print("aggregate loss: ", aggregate_losses[-1], file=f)
        print(
            "previous obs:", repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]), file=f
        )
        print("action: ", repr(causal_lm_tokenizer.batch_decode(action)[0]), file=f)
        print(
            "predicted obs: ",
            repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]),
            file=f,
        )
        print("true obs:", repr(causal_lm_tokenizer.batch_decode(true_obs)[0]), file=f)
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group["lr"], file=f)
    if cfg.wandb:
        wandb.log(
            {
                "Batch number": batch_index,
                "Batch Loss": batch_loss[0].item(),
                # "Aggregate loss": aggregate_losses[-1] if aggregate_losses else -1,
                "Current learning rate": [
                    g["lr"] for g in optimizer.param_groups if "lr" in g
                ][0],
            }
        )
        wandb_table.add_data(
            repr(causal_lm_tokenizer.batch_decode(prev_obs)[0]),
            repr(causal_lm_tokenizer.batch_decode(action)[0]),
            repr(causal_lm_tokenizer.batch_decode(predicted_obs)[0]),
            repr(causal_lm_tokenizer.batch_decode(true_obs)[0]),
        )


dataloader = DataLoader(
    truncated_dataset, batch_size=cfg.batch_size, drop_last=True, shuffle=True
)
i = 0
aggregate_losses = []
optimizer = torch.optim.Adam(causal_lm.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.num_batches
)
with open(f"{cfg.save_dir}/{cfg.model_name}_training_info.txt", "w") as f:
    pass

action_prefix = "Action: "
action_prefix_tokens = causal_lm_tokenizer.encode(
    action_prefix, add_special_tokens=False
)
action_prefix_tensor = repeat(
    torch.tensor(action_prefix_tokens), "tokens -> batch tokens", batch=cfg.batch_size
).to(cfg.device)
tokens_per_pure_action = cfg.tok_p_action - len(action_prefix_tokens)

for data in (
    tqdm(dataloader, total=cfg.num_batches) if cfg.num_batches else tqdm(dataloader)
):
    if cfg.num_batches and i > cfg.num_batches:
        break
    i += 1
    if i > 1 and i % cfg.interval_save_weights == 0:
        print(f"Saving trained_{cfg.model_name}")
        causal_lm_tokenizer.save_pretrained(
            f"./saved_weights_and_losses/tokenizer_{cfg.model_name}"
        )
        causal_lm.save_pretrained(
            f"./saved_weights_and_losses/trained_{cfg.model_name}"
        )
    rao_sequence = []
    for observation_index in range(cfg.obs_p_doc):
        optimizer.zero_grad()
        high_reward_value = (
            round(np.mean(aggregate_losses) - np.std(aggregate_losses), 3)
            if aggregate_losses
            else 6.0
        )
        high_reward = causal_lm_tokenizer(
            ["Reward: " + str(high_reward_value) for _ in range(cfg.batch_size)],
            return_tensors="pt",
            padding="max_length",
            max_length=cfg.tok_p_reward,
        ).input_ids
        high_reward = high_reward.to(cfg.device)
        incentive_rao = torch.cat((high_reward, action_prefix_tensor), dim=-1)
        full_action = causal_lm.generate(
            inputs=incentive_rao,
            output_scores=True,
            do_sample=True,
            return_dict_in_generate=True,
            max_new_tokens=tokens_per_pure_action,
            pad_token_id=causal_lm_tokenizer.pad_token_id,
            eos_token_id=None,
        )
        action: TensorType["batch", "seq_length"] = full_action.sequences[
            :, -cfg.tok_p_action :
        ]
        if observation_index > 1:
            prev_obs: TensorType["batch", "seq_length"] = data["input_ids"][
                :, observation_index - 1, :
            ]
        else:
            prev_obs: TensorType["batch", "seq_length"] = torch.full_like(
                data["input_ids"][:, 0, :], causal_lm_tokenizer.pad_token_id
            )
        true_obs: TensorType["batch", "seq_length"] = data["input_ids"][
            :, observation_index, :
        ]
        true_obs = true_obs.to(cfg.device)
        with torch.no_grad():
            prediction = causal_lm(torch.cat((high_reward, action, true_obs), dim=-1))
            predicted_logits = prediction.logits[:, -cfg.tok_p_obs - 1 : -1, :]
            predicted_obs = predicted_logits.argmax(dim=-1)
            out = loss_fn(
                input=rearrange(
                    predicted_logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=true_obs,
            )
            batch_loss = out.mean(dim=-1)
        string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
        losses: TensorType["batch", "seq_length"] = causal_lm_tokenizer(
            string_losses, return_tensors="pt", padding=True
        ).input_ids.to(cfg.device)
        rao_sequence.append(
            [
                MyRAO(r=losses[i], a=action[i], o=true_obs[i])
                for i in range(cfg.batch_size)
            ]
        )
        log_and_print_info(
            i,
            observation_index,
            batch_loss,
            aggregate_losses,
            prev_obs,
            action,
            predicted_obs,
            true_obs,
            optimizer,
            wandb_table,
            causal_lm_tokenizer,
        )

        # Compute the loss on the whole rao_tensor sequence and perform backpropagation
        rao_tensor = torch.cat((losses, action, true_obs), dim=-1)
        rao_tensor_logits = causal_lm(rao_tensor).logits
        rao_tensor_loss = loss_fn(
            input=rearrange(
                rao_tensor_logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=rao_tensor,
        )
        aggregate_loss = rao_tensor_loss.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()
        print("Aggregate loss: ", aggregate_loss)
        if cfg.wandb:
            wandb.log({"Aggregate loss": aggregate_loss})
        optimizer.step()
    scheduler.step()

if cfg.wandb:
    run.log({"Prediction Accuracy Table": wandb_table})
    wandb.finish()
