# mamba activate menv
# pip install transformers datasets==2.14.6 torchtyping==0.1.4
# pip install peft einops apache_beam==2.51.0 matplotlib wandb
# pip install -U flash-attn --no-build-isolation

from datasets import load_dataset, Dataset, Features, Value, Array2D
import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
import wandb
from collaborative_experiments.rao_tools import MyRAO, RaoConfig, log_and_print_info
from collaborative_experiments.rao_generator import RaoGenerator

cfg = RaoConfig(
    load_model=False,
    wandb=False,  # True,
    model_name="distilgpt2",  # "mistral",  # "gpt2" #"gpt2-xl" #"distilgpt2" #gpt2-large" # distilgpt2  ;  EleutherAI/gpt-j-6b
    save_dir=".",  # "/home/scottviteri/Projects/CollaborativeTraining/CollaborativeTraining/saved_weights_and_losses"
    tok_p_reward=10,
    tok_p_action=100,
    # if None, calculates: CTXT_WINDOW_SIZE - TOKENS_PER_ACTION - TOKENS_PER_REWARD
    tok_p_obs=None,
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


NUM_DATAPOINTS = cfg.batch_size * cfg.num_batches if cfg.num_batches else None
causal_lm = cfg.model
causal_lm_tokenizer = cfg.tokenizer

raogen = RaoGenerator(
    cfg=cfg,
    points_from_data=NUM_DATAPOINTS,
    num_data_points=NUM_DATAPOINTS,
)
dataloader = raogen.dataloader
tokens_per_pure_reward = raogen.tokens_per_pure_reward
reward_prefix_tensor = raogen.reward_prefix_tensor
action_prefix_tensor = raogen.action_prefix_tensor
tokens_per_pure_action = raogen.tokens_per_pure_action


i = 0
aggregate_losses = []
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(causal_lm.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.num_batches
)

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
            cfg,
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
