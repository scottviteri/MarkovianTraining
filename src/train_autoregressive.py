import torch
from tqdm import tqdm
import wandb
import einops
from typing import List
from datasets import load_dataset

from src.training_types import *


def train_autoregressive(cfg):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    losses = []
    for batch_index, datapt in tqdm(enumerate(cfg.dataloader), total=cfg.num_batches):
        obs = datapt["Observation"]
        optimizer.zero_grad()
        logits = cfg.causal_lm(obs).logits[:, :-1, :]
        loss = loss_fn(
            input=einops.rearrange(
                logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=obs[:, 1:]
        )
        loss.backward()
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

