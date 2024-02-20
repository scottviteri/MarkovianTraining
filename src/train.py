# pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import einops
import wandb
import json
import random
import os
from datasets import load_dataset
from openai import OpenAI
from matplotlib import pyplot as plt
import functools

import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel

from src.training_types import *
from src.utilities import extend_initial_config, log_and_print_info
from src.utilities import create_run_name, multi_print
from src.config_examples import configs 


from src.evaluate_via_gpt import evaluate_via_gpt
import src.config_examples
import torch.distributed as dist

def save_weights(cfg, batch_index):
    if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
        print(f"Saving trained_{cfg.model_name} \n\n")
        cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
        cfg.causal_lm.save_pretrained(cfg.path_2_model)

def default_action(cfg):
    initial_helpful_msg = cfg.causal_lm_tokenizer("Use StepByStep spaces to help predict your next observation.",
                                          return_tensors="pt")["input_ids"].repeat(cfg.batch_size, 1).to(cfg.device)
    assert initial_helpful_msg.shape[-1] < cfg.tok_p_pure_obs
    prev_action = torch.cat(
        (
            cfg.action_prefix_tensor,
            initial_helpful_msg, 
            torch.full(
                (cfg.batch_size, cfg.tok_p_pure_action - initial_helpful_msg.shape[-1]),
                fill_value=cfg.causal_lm_tokenizer.pad_token_id,
                dtype=torch.int64,
                device=cfg.device,
            ),
        ),
        dim=1,
    )
    return prev_action

def log_wandb(cfg, batch_index, aggregate_loss, losses):
    prev_action_loss, prev_observation_loss, action_loss, observation_loss = losses
    if cfg.wandb and dist.get_rank() == 0:
        wandb.log(
            {
                "Batch Index": batch_index,
                "Aggregate Loss": aggregate_loss,
                "Previous Action Loss": prev_action_loss,
                "Previous Observation Loss": prev_observation_loss,
                "Action Loss": action_loss,
                "Observation Loss": observation_loss
            }
        )

def log_print_losses(cfg, batch_index, aggregate_loss, losses):
    prev_action_loss, prev_observation_loss, action_loss, obs_loss = losses
    if batch_index % cfg.interval_print == 0 and dist.get_rank() == 0:
        with open(cfg.path_2_log, "a") as f:
            multi_print(f"Aggregate loss: {aggregate_loss}", f)
            multi_print(
                f"PrevAction/PrevObservation/Action/Obs loss: {prev_action_loss}/{prev_observation_loss}/{action_loss}/{obs_loss}",
                f,
            )
            multi_print("______________________________________________________", f)

def log_print_oa(cfg, batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first):
    if batch_index % cfg.interval_print == 0 and dist.get_rank() == 0:
        with open(cfg.path_2_log, "a") as f:
            multi_print(f"Batch Index: {batch_index}", f)
            multi_print(f"Is First: {is_first}", f)
            multi_print(
                f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0]))}", f
            )
            multi_print(
                f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                f,
            )
            if not is_first:
                if is_guidance_action:
                    multi_print(
                        f"Guidance Action: {repr(cfg.causal_lm_tokenizer.decode(action[0]))}",
                        f,
                    )
                else:
                    multi_print(
                        f"Action: {repr(cfg.causal_lm_tokenizer.decode(action[0]))}",
                        f,
                    )
            multi_print(
                f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f
            )

def sample(cfg, prev_action, prev_obs, observation):
    sampling_cfg = cfg.sampling_cfg
    # currently not using filter_best_action parameters
    cfg.causal_lm.eval()
    with torch.no_grad():
        input_sequence = torch.cat([prev_action, prev_obs, cfg.action_prefix_tensor], dim=1)
        attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
        with FullyShardedDataParallel.summon_full_params(cfg.causal_lm, writeback=False, recurse=False):
            action_candidates = cfg.causal_lm.generate(
                    inputs=input_sequence,
                    attention_mask=attention_mask,
                    num_beams=cfg.num_beams,
                    bad_words_ids=[[cfg.causal_lm_tokenizer.pad_token_id]],
                    output_scores=True,
                    do_sample=True,
                    temperature=1.0,
                    min_new_tokens=cfg.tok_p_pure_action,
                    max_new_tokens=cfg.tok_p_pure_action,
                    pad_token_id=cfg.causal_lm_tokenizer.pad_token_id,
                )[:, -cfg.tok_p_action :]
        return action_candidates

def update_weights(cfg, optimizer, prev_action, prev_obs, action, obs):
    training_cfg = cfg.training_cfg
    optimizer.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    cfg.causal_lm.train()
    with autocast(cache_enabled=False, dtype=torch.bfloat16 if cfg.model_name in ["llama", "mistral"] else torch.float16):
        if training_cfg.train_O_given_prev_O:
            input_sequence = torch.cat([prev_obs, obs], dim=1)
            attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
            logits = cfg.causal_lm(input_sequence, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
            loss_tensor = loss_fn(
                input=einops.rearrange(
                    logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=input_sequence[:, 1:]
            )
            aggregate_loss = loss_tensor[:,-cfg.tok_p_pure_obs:].mean()
        else:
            input_sequence = torch.cat([prev_action, prev_obs, action], dim=1)
            attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
            logits = cfg.causal_lm(input_sequence, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
            loss_tensor = loss_fn(
                input=einops.rearrange(
                    logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=input_sequence[:, 1:]
            )

            prev_action_tensor = loss_tensor[:, : cfg.tok_p_action]
            prev_observation_tensor = loss_tensor[:, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs]
            action_tensor = loss_tensor[:, -cfg.tok_p_pure_action :]
            prev_action_loss = prev_action_tensor.mean()
            prev_observation_loss = prev_observation_tensor.mean()
            action_loss = action_tensor.mean()

            #with torch.no_grad():
            mkv_input_sequence = torch.cat([action, obs], dim=1)
            mkv_attention_mask = (mkv_input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
            mkv_logits = cfg.causal_lm(mkv_input_sequence, attention_mask=mkv_attention_mask, use_cache=False).logits[:, :-1, :]
            mkv_loss_tensor = loss_fn(
                input=einops.rearrange(
                    mkv_logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=mkv_input_sequence[:, 1:]
            )
            obs_tensor = (mkv_loss_tensor * mkv_attention_mask[:, 1:])[:, -cfg.tok_p_pure_obs:]
            obs_loss = obs_tensor.sum() / mkv_attention_mask[:, 1:][:,-cfg.tok_p_pure_obs:].sum()

            aggregate_loss = obs_loss
            #aggregate_loss =  action_loss*obs_loss
            #aggregate_loss = sum(map(lambda x: x[1] if x[0] else 0.0, 
            #                         zip([training_cfg.train_A_given_AO, training_cfg.train_O_given_A],
            #                                [action_loss, obs_loss])))

    if not isinstance(cfg.debug, NoWeightUpdates):
        aggregate_loss.backward()
        optimizer.step()
    
    if training_cfg.train_O_given_prev_O: return aggregate_loss, None, None
    loss_tensors = prev_action_tensor, prev_observation_tensor, action_tensor, obs_tensor
    losses = prev_action_loss, prev_observation_loss, action_loss, obs_loss
    return aggregate_loss, loss_tensors, losses

def log_and_save(cfg, batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first, aggregate_loss, losses):
    save_weights(cfg, batch_index)
    log_print_oa(cfg, batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first)
    if cfg.training_cfg.train_O_given_prev_O: 
        if cfg.wandb and dist.get_rank() == 0: wandb.log({"Batch Index": batch_index, "Observation Loss": aggregate_loss})
    else:
        log_wandb(cfg, batch_index, aggregate_loss, losses)
        log_print_losses(cfg, batch_index, aggregate_loss, losses)

def trainer(cfg, optimizer):
    state = [default_action(cfg), 0, None]

    def update(datapt_pair):
        nonlocal state
        prev_datapt, datapt = datapt_pair
        is_first = "First" in datapt and datapt["First"]
        prev_action, batch_index, _ = state
        prev_obs, obs = prev_datapt["Observation"], datapt["Observation"]
        if is_first: 
            prev_action = default_action(cfg)
            log_print_oa(cfg, batch_index, prev_action, prev_obs, None, obs, "Action" in datapt, is_first)
            state = [prev_action, batch_index + 1, None]
            return
        # now can assume that prev_datapt contains the question and datapt contains the Answer
        if "Action" in datapt: 
            action = datapt["Action"]
        elif cfg.training_cfg.train_O_given_prev_O: 
            action = prev_action
        else:
            action = sample(cfg, prev_action, prev_obs, obs) 
        aggregate_loss, loss_tensors, losses = update_weights(cfg, optimizer, prev_action, prev_obs, action, obs)
        log_and_save(cfg, batch_index, prev_action, prev_obs, action, obs, "Action" in datapt, is_first, aggregate_loss, losses)
        state = [action, batch_index + 1, aggregate_loss]
        return

    def pi(): 
        nonlocal state
        return state[-1]
    
    return update, pi

def train_via_update(cfg, optimizer):
    aggregate_losses = []
    update, pi = trainer(cfg, optimizer)
    for datapt_pair in tqdm(cfg.dataset.dataloader, total=cfg.num_batches):
        aggregate_loss = pi()
        if aggregate_loss is not None: aggregate_losses.append(aggregate_loss)
        update(datapt_pair)
    return aggregate_losses

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        print("Dev", self.device)
        self.cfg = cfg
        self.cfg.device = self.device
        self.cfg.action_prefix_tensor = self.cfg.action_prefix_tensor.to(self.device)
        self.cfg.obs_prefix_tensor = self.cfg.obs_prefix_tensor.to(self.device)
        self.model = self.cfg.causal_lm
        self.state = [default_action(self.cfg), 0, None]

    def forward(self, prev_action, prev_obs, obs):
        return sample(self.cfg, prev_action, prev_obs, obs)

    def training_step(self, datapt_pair, batch_idx):
        #if batch_idx == 0:
        #    print(self)
        self.cfg.device = self.device
        self.cfg.action_prefix_tensor = self.cfg.action_prefix_tensor.to(self.device)
        self.cfg.obs_prefix_tensor = self.cfg.obs_prefix_tensor.to(self.device)
        prev_datapt, datapt = datapt_pair
        is_first = "First" in datapt and datapt["First"]
        prev_action, batch_index, _ = self.state
        prev_action = prev_action.to(self.device)
        prev_obs, obs = prev_datapt["Observation"], datapt["Observation"]
        if is_first: 
            prev_action = default_action(self.cfg)
            log_print_oa(self.cfg, batch_index, prev_action, prev_obs, None, obs, "Action" in datapt, is_first)
            self.state = [prev_action, batch_index + 1, None]
            return torch.tensor(0.0, device=self.device, requires_grad=True) #self.cfg.causal_lm(torch.zeros((1,1), requires_grad=True)).sum() * 0 #None #torch.tensor(5.0)
        # now can assume that prev_datapt contains the question and datapt contains the Answer
        if "Action" in datapt: 
            action = datapt["Action"]
        elif self.cfg.training_cfg.train_O_given_prev_O: 
            action = prev_action
        else:
            action = self.forward(prev_action, prev_obs, obs) 
        aggregate_loss, loss_tensors, losses = update_weights(self.cfg, prev_action, prev_obs, action, obs)
        log_and_save(self.cfg, batch_index, prev_action, prev_obs, action, obs, "Action" in datapt, is_first, aggregate_loss, losses)
        self.state = [action, batch_index + 1, aggregate_loss]
        return aggregate_loss 

    def configure_optimizers(self):
        if self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.cfg.causal_lm.parameters(), lr=self.cfg.lr)#, momentum=0.01)
        elif self.cfg.optimizer == "adam":
            optimizer = torch.optim.AdamW(self.cfg.causal_lm.parameters(), lr=self.cfg.lr)
        elif self.cfg.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(self.cfg.causal_lm.parameters(), lr=self.cfg.lr)
        return optimizer

def custom_auto_wrap_policy(
    module: torch.nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
) -> bool:
    return nonwrapped_numel >= min_num_params
# Configure a custom `min_num_params`
my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e1))

def pl_train_model(init_cfg):
    cfg = extend_initial_config(init_cfg)
    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")
    with open(cfg.path_2_log, "a") as f:
        f.write("")
    if cfg.wandb and dist.get_rank() == 0: 
        wandb.init(
            project="collaborative-training-many-per-context-window", 
            name=create_run_name(cfg))
    model = LitModel(cfg)
    trainer = pl.Trainer(num_nodes=1,
        max_epochs=1, limit_train_batches=cfg.num_batches, accelerator="cuda", 
        devices=2, 
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            #auto_wrap_policy=my_auto_wrap_policy, 
            auto_wrap_policy = size_based_auto_wrap_policy, 
            mixed_precision = MixedPrecision(param_dtype=torch.bfloat16)
            ))
    trainer.fit(model, cfg.dataset.dataloader)
    if cfg.wandb and dist.get_rank() == 0: wandb.finish()

def train_model(init_cfg):
    cfg = extend_initial_config(init_cfg)
    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")
    with open(cfg.path_2_log, "a") as f:
        f.write("")
    if cfg.wandb and dist.get_rank() == 0: 
        wandb.init(
            project="collaborative-training-many-per-context-window", 
            name=create_run_name(cfg))
    #cfg.causal_lm = FullyShardedDataParallel(
    #    cfg.causal_lm, 
    #    auto_wrap_policy=size_based_auto_wrap_policy,
    ##    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16)
    #)
    print(cfg.causal_lm)
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr, momentum=0.01)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(cfg.causal_lm.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(cfg.causal_lm.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}. Please choose either 'sgd' or 'adam'.")
    train_via_update(cfg, optimizer)
    if cfg.wandb and dist.get_rank() == 0: wandb.finish()

def setup_distributed(backend='nccl', port='12355'):
    # Set the environment variable for master address and port
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = port
    ## Determine the rank of the process and total number of processes
    #rank = int(os.environ.get("RANK", "0"))
    #world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Initialize the process group
    #dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.init_process_group(backend="nccl")

if __name__ == "__main__":
    # Setup distributed environment
    setup_distributed()
    torch.cuda.set_device(dist.get_rank())
    print("rank", dist.get_rank())
    for init_cfg in configs:
        train_model(init_cfg)
