import torch
from tqdm import tqdm
import wandb
import einops
from typing import List
from datasets import load_dataset

from src.types_and_utilities import InitialConfig, InitTrainingType, Config
from src.types_and_utilities import AR, GptEval, AO, AOA, RAOInit
from src.types_and_utilities import log_and_print_info



def train_autoregressive(cfg):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    losses = []
    for batch_index, obs in tqdm(enumerate(cfg.dataloader), total=cfg.num_batches):
        optimizer.zero_grad()
        logits = cfg.causal_lm(obs).logits[:, :-1, :]
        loss = loss_fn(
            input=einops.rearrange(
                logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=obs[:, 1:],
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if cfg.wandb:
            wandb.log({"Observation Loss": loss.item()})
        if batch_index % cfg.interval_print == 0:
            print(f"Batch {batch_index}, Loss: {loss.item()}")
    return losses

