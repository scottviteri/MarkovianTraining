import torch
import einops
from datasets import load_dataset

from src.training_types import InitialConfig, InitTrainingType, Config
from src.training_types import AR, GptEval, AO, AOA, RAOInit


def prepare_dataset(init_cfg, task_name, causal_lm_tokenizer, device, tok_p_pure_obs, obs_prefix):
    itr_ds = load_dataset(init_cfg.dataset_name, task_name, split="train", streaming=True)
    ds_tokenized = map(
        lambda x: causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(device), 
        itr_ds)
    pure_obs = get_pure_obs(init_cfg.batch_size, tok_p_pure_obs, device, ds_tokenized)
    obs_ds = prepend_obs_tensor(obs_prefix, pure_obs)
    # Note: is it true that RAO uses the same num_batches to count the number of weight updates?
    #  this would make them not comparable to AO, unless we divide by obs per weight update
    if isinstance(init_cfg.training_type, RAOInit):
        obs_ds = group_obs_tensors(init_cfg.training_type.obs_between_weight_updates, obs_ds)
    if init_cfg.repeat_first_datapoint:
        obs_ds = repeat(obs_ds)
    return take(init_cfg.num_batches, obs_ds)

def get_pure_obs(batch_size, tok_per_pure_obs, device, itr_ds):
    batches = [torch.empty((1,0), dtype=torch.int64, device=device) 
        for _ in range(batch_size)]
    while 1:
        for i in range(len(batches)):
            while batches[i].shape[1] < tok_per_pure_obs:
                batches[i] = torch.cat((batches[i], next(itr_ds)),dim=1)
        for batch in batches: assert  batch.shape[-1] >= tok_per_pure_obs
        out_tensor = torch.cat([batch[:,:tok_per_pure_obs] for batch in batches], dim=0)
        for i in range(len(batches)):
            batches[i] = batches[i][:, tok_per_pure_obs:]
        yield out_tensor
    return batches 

def prepend_obs_tensor(obs_prefix_tensor, itr_ds):
    return map(lambda x: torch.cat((obs_prefix_tensor, x), dim=1), itr_ds)

def group_obs_tensors(obs_per_weight_update, itr_ds):
    while 1:
        grouped = torch.stack([next(itr_ds) for _ in range(obs_per_weight_update)])
        yield einops.rearrange(grouped, 'buffer batch tokens -> batch buffer tokens')

def take(num_batches, itr_ds): 
    for _ in range(num_batches): yield next(itr_ds)

def repeat(itr):
    first_val = next(itr)
    while 1: yield first_val

