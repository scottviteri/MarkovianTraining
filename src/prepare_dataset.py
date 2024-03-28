import torch
import einops
from datasets import load_dataset
from itertools import islice, tee
from typing import Iterator, Dict

import json
import operator
import numpy as np

from prepare_dataset import *
from training_types import *

"""
This file controlles dataset loading. 
Think of the input data as consisting of trajectories of observations.
Each observation is a fixed number of tokens.
The high level logic is in init_arithmetic_dataset and prepare_dataset.
arithmetic_generator creates an iter of {"Action":_, "Observation": _} dicts.
Actions (if provided) override the transformer generated action. 
to_qa_traj_itr creates an iter of iters, and each inner iter is a trajectory.
The question-answer format consists of trajectories of length two.
traj_itr_to_batch_lst creates a |batch|-len list.
    Each element pulls from one trajs at a time, never splitting a traj.
tokenize_batches tokenizes action and observations.
apply_debug_transformations: optionally repeat obs or replace w/ random toks.
finalize_dataset contains peek_every_n, stack_batch, group_pairs, and take.
peek_every_n filters all but every nth Action suggestion.
stack_batch turns the list of iters into a single iter, by batching the tensors.
group_pairs turns (a,b,c,d,...) into ((a,b),(b,c),(c,d),...).
    So the main loops sees the observation and previous obs at once.
take restricts the dataset to the cfg-specified number of datapoints. 
place debug(itr) at the point where you want to inspect the dataflow
    It helps to set a breakpoint inside of the debug function
"""


def prepare_dataset(
    init_cfg,
    tokenizer,
    device,
    pure_ctxt_sizes,
    prefix_tensors,
):
    task = init_cfg.dataset.task
    dict_ds = initialize_dataset(
        task,
        prefix_tensors,
        pure_ctxt_sizes,
        init_cfg,
        tokenizer,
        device,
    )
    dict_ds = apply_debug_transformations(dict_ds, init_cfg, tokenizer)
    return finalize_dataset(dict_ds, init_cfg)


def initialize_dataset(
    task,
    prefix_tensors,
    pure_ctxt_sizes,
    init_cfg,
    tokenizer,
    device,
):

    if isinstance(task, ArithmeticTask):
        return init_arithmetic_dataset(
            task,
            prefix_tensors,
            pure_ctxt_sizes,
            init_cfg,
            tokenizer,
            device,
        )
    elif isinstance(task, WikipediaTask):
        return init_wikipedia_dataset()
    else:
        raise ValueError("Unknown dataset")


def init_arithmetic_dataset(
    task,
    prefix_tensors,
    pure_ctxt_sizes,
    init_cfg,
    tokenizer,
    device,
):

    def to_qa_traj_itr(itr):
        while 1:
            qa = next(itr)
            yield iter(
                [
                    Datapt(
                        action="Work through the following question step by step, concisely decomposing problems into subproblems.",
                        obs=qa.question,
                        is_first=True,
                    ),
                    Datapt(action=qa.explanation, obs=qa.answer, is_first=False),
                ]
            )

    def tokenize_and_pad(
        device,
        tokenizer,
        prefix_tensors,
        pure_ctxt_sizes,
        datapt,
    ):
        obs_tok = tokenizer(datapt.obs, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].to(device)
        tok_per_pure_action = (
            pure_ctxt_sizes.first_action_size
            if datapt.is_first
            else pure_ctxt_sizes.action_size
        )
        tok_per_pure_obs = (
            pure_ctxt_sizes.first_obs_size
            if datapt.is_first
            else pure_ctxt_sizes.obs_size
        )
        assert len(obs_tok) < tok_per_pure_obs
        obs_pad_tok = torch.full(
            (tok_per_pure_obs - len(obs_tok),),
            tokenizer.pad_token_id,
            dtype=torch.int64,
            device=device,
        )
        action_prefix_tensor = (
            prefix_tensors.first_action_prefix_tensor
            if datapt.is_first
            else prefix_tensors.action_prefix_tensor
        )
        obs_prefix_tensor = (
            prefix_tensors.first_obs_prefix_tensor
            if datapt.is_first
            else prefix_tensors.obs_prefix_tensor
        )
        if datapt.action:
            action_tok = tokenizer(
                datapt.action, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].to(device)
            assert len(action_tok) < tok_per_pure_action
            action_pad_tok = torch.full(
                (tok_per_pure_action - len(action_tok),),
                tokenizer.pad_token_id,
                dtype=torch.int64,
                device=device,
            )
            return Datapt(
                obs=torch.cat([obs_prefix_tensor[0], obs_tok, obs_pad_tok]),
                action=torch.cat([action_prefix_tensor[0], action_tok, action_pad_tok]),
                is_first=datapt.is_first,
            )
        else:
            return Datapt(
                obs=torch.cat([obs_prefix_tensor[0], obs_tok, obs_pad_tok]),
                action=None,
                is_first=datapt.is_first,
            )

    def traj_itr_to_batch_lst(batch_size, traj_itr):
        batch_itrs = [next(traj_itr) for _ in range(batch_size)]
        while 1:
            out_lst = []
            for i in range(batch_size):
                try:
                    next_item = next(batch_itrs[i])
                except StopIteration:
                    batch_itrs[i] = next(traj_itr)
                    next_item = next(batch_itrs[i])
                out_lst.append(next_item)
            yield out_lst

    def tokenize_batches(qa_batch_lst_itr):
        qa_tokenized_itr_ds = map(
            lambda batch_lst: [
                tokenize_and_pad(
                    device,
                    tokenizer,
                    prefix_tensors,
                    pure_ctxt_sizes,
                    b,
                )
                for b in batch_lst
            ],
            qa_batch_lst_itr,
        )
        return qa_tokenized_itr_ds

    itr_ds = arithmetic_generator(
        task.num_terms,
        task.num_digits,
        task.operations,
        task.probs,
    )
    # itr_ds = debug(itr_ds)
    itr_ds = to_qa_traj_itr(itr_ds)
    # itr_ds = debug(itr_ds)
    itr_ds = traj_itr_to_batch_lst(init_cfg.batch_size, itr_ds)
    # itr_ds = debug(itr_ds)
    itr_ds = tokenize_batches(itr_ds)
    # itr_ds = debug(itr_ds)
    return itr_ds


def init_wikipedia_dataset(init_cfg, tokenizer, device):
    # I don't know if this works anymore
    # only need is_first at the beginning
    def gen_wiki_datapts(batch_size, tok_per_pure_obs, itr_ds):
        # batch in a way that keeps state so that you only add to a batch index
        #  with fewer than tok_per_pure_obs tokens
        batches = [
            torch.empty((1, 0), dtype=torch.int64, device=device)
            for _ in range(batch_size)
        ]
        first = True
        while 1:
            for i in range(len(batches)):
                while batches[i].shape[1] < tok_per_pure_obs:
                    batches[i] = torch.cat((batches[i], next(itr_ds)), dim=1)
            for batch in batches:
                assert batch.shape[-1] >= tok_per_pure_obs
            out_tensor = torch.cat(
                [batch[:, :tok_per_pure_obs] for batch in batches], dim=0
            )
            for i in range(len(batches)):
                batches[i] = batches[i][:, tok_per_pure_obs:]
            yield Datapt(action=None, obs=out_tensor, is_first=first)
            first = False
        assert False, "Ran out of data"

    itr_ds = iter(
        load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    )
    ds_tokenized = map(
        lambda x: tokenizer(x["text"], return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device),
        itr_ds,
    )
    # use an underestimate of the total number of allowed tokens
    return gen_wiki_datapts(
        init_cfg.batch_size, init_cfg.ctxt_sizes // 2, device, ds_tokenized
    )


def apply_debug_transformations(dict_ds, init_cfg, tokenizer):
    def replace_with_random_tokens(dict_ds):
        dict_ds = map(
            lambda d: {
                **d,
                "Observation": torch.randint(
                    0,
                    tokenizer.vocab_size,
                    d["Observation"].shape,
                    device=d["Observation"].device,
                    dtype=d["Observation"].dtype,
                ),
            },
            dict_ds,
        )
        return dict_ds

    def repeat_every_n_points(n, dict_ds):
        first_n_vals = [next(dict_ds) for _ in range(n)]
        i = 0
        while 1:
            yield first_n_vals[i]
            i = (i + 1) % n

    def repeat_point_n_times(n, itr):
        while 1:
            next_val = next(itr)
            for _ in range(n):
                yield next_val

    if isinstance(init_cfg.debug, ReplaceWithRandomTokens):
        return replace_with_random_tokens(dict_ds)
    elif isinstance(init_cfg.debug, RepeatNPoints):
        return repeat_every_n_points(init_cfg.debug.num_points, dict_ds)
    elif isinstance(init_cfg.debug, RepeatPointNTimes):
        return repeat_point_n_times(init_cfg.debug.num_times, dict_ds)
    return dict_ds


def finalize_dataset(dict_ds, init_cfg):
    def peek_every_n(n, dict_itr):
        """
        If batch_idx % n != 0, filter out the Action.
        We will use an action iff it remains unfiltered through prepare_dataset.
        """
        i = 0
        for datapt in dict_itr:
            if i % n != 0:
                yield Datapt(action=None, obs=datapt.obs, is_first=datapt.is_first)
            else:
                yield datapt
            i += 1

    def stack_batch(batch):
        # do I want each batch to separately be able to add actions? Seems unnecessary for now
        grouped_obs = torch.stack([d.obs for d in batch])
        return Datapt(
            action=(
                torch.stack([d.action for d in batch])
                if batch[0].action is not None
                else None
            ),
            obs=grouped_obs,
            is_first=batch[0].is_first,
        )

    def group_pairs(itr):
        first = next(itr)
        while 1:
            second = next(itr)
            yield (first, second)
            first = second

    def take(num_batches, itr_ds):
        for _ in range(num_batches):
            yield next(itr_ds)

    if init_cfg.dataset.peek_every is not None:
        dict_ds = peek_every_n(init_cfg.dataset.peek_every, dict_ds)
        # dict_ds = debug(dict_ds)
    dict_ds = map(stack_batch, dict_ds)
    # dict_ds = debug(dict_ds)
    # dict_ds = group_pairs(dict_ds)
    # dict_ds = debug(dict_ds)
    return take(init_cfg.num_batches, dict_ds)


def arithmetic_generator(num_terms, num_digits, operations, probs):
    # If not specified, use simple addition
    if operations is None:
        operations = ["+"]
    # Use uniform distribution for operators if not set
    if probs is None:
        probs = [(1.0 / len(operations)) for _ in range(len(operations))]
    # Check for valid operations
    valid_ops = {"+": operator.add, "-": operator.sub, "*": operator.mul}
    for op in operations:
        assert (
            op in valid_ops.keys()
        ), f"Invalid operation {op} not in {valid_ops.keys()}"
    assert len(probs) == len(operations), "len(Operations) != len(probs)"

    while 1:
        question = ""
        total = 0.0
        nums = torch.randint(0, 10**num_digits - 1, (num_terms,))
        ops_rand = np.random.choice(operations, num_terms - 1, p=probs)

        for i in range(num_terms):
            num = nums[i]
            if i == 0:
                total = nums[0].item()
                question += f"{num} "
            else:
                op_rand = ops_rand[i - 1]
                total = valid_ops[op_rand](total, num)
                if op_rand == "-":
                    question += f"+ (-{num}) "
                else:
                    question += f"{op_rand} {num} "
        question = question[:-1] + "."

        answer = f"{total}"
        yield QADatapt(question=question, explanation=None, answer=answer)


def debug(itr):
    while 1:
        next_val = next(itr)
        print(next_val)
        yield next_val


def unused_functions():
    def prepend_prefix_tensors(prefix_tensors, itr_ds):
        out_d = {}
        for d in itr_ds:
            if "Observation" in d:
                out_d["Observation"] = torch.cat(
                    (prefix_tensors.obs_prefix_tensor, d["Observation"]), dim=1
                )
            if "Action" in d:
                out_d["Action"] = torch.cat(
                    (prefix_tensors.action_prefix_tensor, d["Action"]), dim=1
                )
        return out_d

    def jsonl_to_dict_iterator(filename: str) -> Iterator[Dict]:
        with open(filename, "r") as infile:
            for line in infile:
                yield json.loads(line)

    def concat_batches_to_len(length, itr):
        batch = next(itr)
        while 1:
            while batch.shape[1] < length:
                batch = torch.cat((batch, next(itr)), dim=1)
            yield batch[:, :length]
            batch = batch[:, length:]

    def batch(batch_size, itr):
        while 1:
            new_list = [next(itr) for _ in range(batch_size)]
            yield new_list

    def flatten(itrs):
        while 1:
            for itr in itrs:
                yield next(itr)

    def nth_iterators(n, itr):
        itrs = tee(itr, n)
        return (islice(itr, i, None, n) for i, itr in enumerate(itrs))

    def stack_buffer(batch):
        # do I want each batch to separately be able to add actions? Seems unnecessary for now
        grouped_obs = torch.stack([d["Observation"] for d in batch])
        if "Action" in batch[0]:
            grouped_actions = torch.stack([d["Action"] for d in batch])
            return {
                "Observation": einops.rearrange(
                    grouped_obs, "buffer batch tokens -> batch buffer tokens"
                ),
                "Action": einops.rearrange(
                    grouped_actions, "buffer batch tokens -> batch buffer tokens"
                ),
            }
        return {
            "Observation": einops.rearrange(
                grouped_obs, "buffer batch tokens -> batch buffer tokens"
            )
        }

    def debug_shape(itr):
        next_val = next(itr)
        print(next_val.shape)
        yield next_val
