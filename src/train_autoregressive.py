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
    causal_lm = cfg.causal_lm
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(causal_lm.parameters(), lr=cfg.lr)
    #itr_ds = load_dataset(
    #    cfg.dataset_name, cfg.task_name, split="train", streaming=True
    #)
    #ds_tokenized = map(
    #    lambda x: causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(
    #        cfg.device
    #    ),
    #    itr_ds,
    #)
    #pure_obs = get_pure_obs(cfg.batch_size, cfg.tok_p_obs, cfg.device, ds_tokenized)
    #obs_ds = take(cfg.num_batches, pure_obs)
    obs_ds = cfg.dataloader
    # Initialize the list to store the losses
    losses = []
    for batch_index, obs in tqdm(enumerate(obs_ds), total=cfg.num_batches):
        optimizer.zero_grad()
        logits = causal_lm(obs).logits[:, :-1, :]
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

