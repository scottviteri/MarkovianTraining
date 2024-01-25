import torch
import einops
from datasets import load_dataset
from itertools import islice, tee

from src.training_types import *

def prepare_dataset(init_cfg, task_name, causal_lm_tokenizer, device, tok_p_pure_obs, obs_prefix):
    itr_ds = iter(load_dataset(init_cfg.dataset_name, task_name, split="train", streaming=True))
    if init_cfg.dataset_name == "wikipedia":
        ds_tokenized = map(
            lambda x: causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(device), 
            itr_ds)
        pure_obs = get_pure_obs(init_cfg.batch_size, tok_p_pure_obs, device, ds_tokenized)
    elif init_cfg.dataset_name == "bigbench":
        # could have also batched first, then impose pairs ordering
        itr = batch(init_cfg.batch_size, itr_ds)
        itr = group_pairs(itr)
        itr = map(lambda x: map(mergeQA, x[0], x[1]), itr)
        itr = map(
            lambda batch: list(map(lambda i: fill_to_size(causal_lm_tokenizer, i[0], i[1], tok_p_pure_obs), batch)), 
            itr)
        itr = map(lambda batch: torch.stack(batch, dim=0).to(device), itr)
        pure_obs = concat_batches_to_len(tok_p_pure_obs, itr)
    else:
        assert False, "Unknown dataset"
    if isinstance(init_cfg.debug, ReplaceWithRandomTokens):
        pure_obs = replace_with_random_tokens(causal_lm_tokenizer.vocab_size, pure_obs)
    obs_ds = prepend_obs_tensor(obs_prefix, pure_obs)
    # Note: is it true that RAO uses the same num_batches to count the number of weight updates?
    #  this would make them not comparable to AO, unless we divide by obs per weight update
    if isinstance(init_cfg.training_type, RAOInit):
        obs_ds = group_obs_tensors(init_cfg.debug.num_points, obs_ds)
    if isinstance(init_cfg.debug, RepeatNPoints):
        obs_ds = repeat_every_n_points(init_cfg.debug.num_points, obs_ds)
    elif isinstance(init_cfg.debug, RepeatPointNTimes): #1 time is identity
        obs_ds = repeat_point_n_times(init_cfg.debug.num_times, obs_ds)
    return take(init_cfg.num_batches, obs_ds)

def concat_batches_to_len(length, itr):
    batch = next(itr)
    while 1:
        while batch.shape[1] < length:
            batch = torch.cat((batch, next(itr)), dim=1)
        yield batch[:,:length]
        batch = batch[:,length:]

def batch(batch_size, itr):
    while 1:
        yield [next(itr) for _ in range(batch_size)]

def flatten(itrs):
    while 1:
        for itr in itrs:
            yield next(itr)

def replace_with_random_tokens(vocab_size, pure_obs):
   while True:
       batch = next(pure_obs)
       random_tokens = torch.randint(0, vocab_size, batch.shape, device=batch.device, dtype=batch.dtype)
       yield random_tokens

def split3(itr):
    def i1():
        while 1: 
            yield next(itr)
            i2()
    def i2():
        while 1: 
            yield next(itr)
        i3()
    def i3():
        while 1: yield next(itr)
        i3()
    return i1(itr), i2(itr), i3(itr)

#def nth_iterators(n, itr):
#    return (islice(itr, i, None, n) for i in range(n))
#

def nth_iterators(n, itr):
    itrs = tee(itr, n)
    return (islice(itr, i, None, n) for i, itr in enumerate(itrs))

def group_pairs(itr):
    first = next(itr)
    while 1:
        second = next(itr)
        yield (first, second)
        first = second

def fill_to_size(tokenizer, begin, end, size):
    begin_tok = tokenizer(begin, return_tensors="pt")["input_ids"][0]
    end_tok = tokenizer(end, return_tensors="pt")["input_ids"][0]
    middle_tok = torch.full((size - len(begin_tok) - len(end_tok),), tokenizer.pad_token_id, dtype=torch.int64)
    return torch.cat((begin_tok, middle_tok, end_tok))

def prep_bb(tokenizer, device, tok_per_pure_obs, itr):
    itr = group_pairs(itr)
    itr = map(lambda x:("A: "+x[0]["targets"][0] + "\n", x[1]["inputs"].split("?")[0] + "?\n"), itr)
    itr = map(lambda x: fill_to_size(tokenizer, x[0], x[1], tok_per_pure_obs), itr)
    return map(lambda x:x.to(device), itr)

def mergeQA(d1, d2):
    return ("A: "+d1["targets"][0] + "\n", d2["inputs"].split("?")[0] + "?\n")
    
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

def repeat_every_n_points(n, itr):
    first_n_vals = [next(itr) for _ in range(n)]
    i = 0
    while 1: 
        yield first_n_vals[i]
        i = (i+1) % n

def repeat_point_n_times(n, itr):
    while 1:
        next_val = next(itr)
        for _ in range(n): 
            yield next_val


