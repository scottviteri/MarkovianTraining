import torch
from tqdm import tqdm
import wandb
import einops
from typing import List
from datasets import load_dataset
from prepare_dataset import group_pairs, take

from src.training_types import *


def train_autoregressive(cfg):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    losses = []
    data = take(cfg.num_batches-1, group_pairs(cfg.dataset.dataloader))
    for batch_index, pair in tqdm(enumerate(data), total=cfg.num_batches):
        d1, d2 = pair
        o1, o2 = d1["Observation"], d2["Observation"]
        obs = torch.cat([o1, o2], dim=1)
        optimizer.zero_grad()
        logits = cfg.causal_lm(obs).logits[:, -cfg.tok_p_obs-1:-1, :]
        loss = loss_fn(
            input=einops.rearrange(
                logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=obs[:, -cfg.tok_p_obs:]
        )
        loss.backward()
        if not isinstance(cfg.debug, NoWeightUpdates):
            optimizer.step()
        losses.append(loss.item())
        if cfg.wandb:
            wandb.log({
                "Batch Index": batch_index,
                "Observation Loss": loss.item()
                })
        if batch_index % cfg.interval_print == 0:
            print(f"Batch {batch_index}, Loss: {loss.item()}")
    return losses

