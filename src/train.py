#, pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
import torch.nn as nn
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
import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel, BackwardPrefetch, FullStateDictConfig, StateDictType

from src.training_types import *
from src.utilities import extend_initial_config, log_and_print_info
from src.utilities import create_run_name, multi_print
from src.config_examples import configs 

from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer 
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


from src.evaluate_via_gpt import evaluate_via_gpt
import src.config_examples
import torch.distributed as dist

def save_weights(cfg, batch_index):
    if batch_index > 0 and batch_index % cfg.interval_save_weights == 0 and dist.get_rank()==0:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
                    cfg.inference_lm, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    model_state = cfg.inference_lm.state_dict()
                    os.makedirs(cfg.path_2_model, exist_ok=True)
                    save_name = cfg.path_2_model + "/pytorch_model.bin"
                    torch.save(model_state, save_name)
                    #config_save_path = os.path.join(cfg.path_2_model, "config.json")
                    cfg.inference_lm.config.save_pretrained(cfg.path_2_model)
    #    with FullyShardedDataParallel.summon_full_params(cfg.predictor_lm, writeback=True, recurse=True):
    #        print(f"Saving trained_{cfg.model_name} \n\n")
    #        cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
    #        cfg.predictor_lm.save_pretrained(cfg.path_2_model)

def default_action(cfg):
    initial_helpful_msg = (
        cfg.causal_lm_tokenizer(
            "Use StepByStep spaces to help predict your next observation.",
            return_tensors="pt",
        )["input_ids"]
        .repeat(cfg.batch_size, 1)
        .to(cfg.device)
    )
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
    (
        prev_action_loss,
        prev_observation_loss,
        action_loss,
        observation_loss,
        perturbed_loss,
    ) = losses
    if cfg.wandb and dist.get_rank() == 0:
        wandb.log(
            {
                "Batch Index": batch_index,
                "Aggregate Loss": aggregate_loss,
                "Perturbed Loss": perturbed_loss,
                "Previous Action Loss": prev_action_loss,
                "Previous Observation Loss": prev_observation_loss,
                "Action Loss": action_loss,
                "Observation Loss": observation_loss,
            },
            step = batch_index
        )


def log_print_losses(cfg, batch_index, aggregate_loss, losses):
    (
        prev_action_loss,
        prev_observation_loss,
        action_loss,
        observation_loss,
        perturbed_loss,
    ) = losses
    if batch_index % cfg.interval_print == 0 and dist.get_rank() == 0:
        with open(cfg.path_2_log, "a") as f:
            multi_print(f"Aggregate loss: {aggregate_loss}", f)
            multi_print(
                f"PrevAction/PrevObservation/Action/Obs/Pert loss: {prev_action_loss:0.4f}/{prev_observation_loss:0.4f}/{action_loss:0.4f}/{observation_loss:0.4f}/{perturbed_loss}",
                f,
            )
            multi_print("______________________________________________________", f)


def log_print_oa(
    cfg, batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first
):
    if batch_index % cfg.interval_print == 0 and dist.get_rank() == 0:
        with open(cfg.path_2_log, "a") as f:
            multi_print(f"Batch Index: {batch_index}", f)
            multi_print(f"Is First: {is_first}", f)
            multi_print(
                f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0]))}",
                f,
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
    inference_cfg = cfg.inference_cfg
    cfg.inference_lm.eval()
    with torch.inference_mode():
        #with autocast(cache_enabled=True, dtype=torch.bfloat16 if cfg.model_name in ["llama", "mistral"] else torch.float16):
        with FullyShardedDataParallel.summon_full_params(cfg.inference_lm, recurse=False):
            input_sequence = torch.cat([prev_action, prev_obs, cfg.action_prefix_tensor], dim=1)
            attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
            action_candidates = cfg.inference_lm.generate(
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
                    num_return_sequences=cfg.inference_cfg.num_return_sequences
                )[:, -cfg.tok_p_action :]
        return action_candidates

def compute_loss_tensor(cfg, input_sequence):
    """
    Computes the loss tensor for a given input sequence.
    
    Args:
        cfg: Configuration object containing model and tokenizer information.
        input_sequence: The input sequence tensor for which the loss is to be computed.
    
    Returns:
        The computed loss tensor.
    """
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    logits = cfg.inference_lm(input_sequence, attention_mask=attention_mask, use_cache=False).logits[:, :-1, :]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_tensor = loss_fn(
        input=einops.rearrange(
            logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:]
    )
    return loss_tensor  


def update_weights(
    cfg, batch_index, prev_action, prev_obs, action, obs, do_weight_update=True
):
    prediction_cfg = cfg.prediction_cfg

    prev_action = prev_action.repeat_interleave(cfg.inference_cfg.num_return_sequences, dim=0)
    prev_obs = prev_obs.repeat_interleave(cfg.inference_cfg.num_return_sequences, dim=0)
    obs = obs.repeat_interleave(cfg.inference_cfg.num_return_sequences, dim=0)

    # this can be pulled out into a synchronize_weights function before sample
    update_every, fraction_to_update = (
        cfg.inference_cfg.update_every,
        cfg.inference_cfg.fraction_to_update,
    )
    if update_every is not None and batch_index % update_every == 0:
        with torch.no_grad():
            if dist.get_rank() == 0: print("Updating Inference Model\n")
            with FullyShardedDataParallel.summon_full_params(cfg.inference_lm, writeback=True, recurse=True):
                for param_inference, param_prediction in zip(cfg.inference_lm.parameters(), cfg.predictor_lm.parameters()):
                    param_inference.data.mul_(1 - fraction_to_update).add_(param_prediction.data.reshape(param_inference.data.shape) * fraction_to_update)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    cfg.predictor_lm.eval()
    cfg.inference_lm.train()
    with autocast(cache_enabled=False, dtype=torch.bfloat16 if cfg.model_name in ["llama", "mistral"] else torch.float16):
        if prediction_cfg.train_O_given_prev_O:
            loss_tensor = compute_loss_tensor(cfg, torch.cat([prev_obs, obs], dim=1))
            aggregate_loss = loss_tensor[:,-cfg.tok_p_pure_obs:].mean()
        else:
            loss_tensor = compute_loss_tensor(cfg, torch.cat([prev_action, prev_obs, action], dim=1))

        mkv_input_sequence = torch.cat([action, obs], dim=1)
        mkv_attention_mask = (mkv_input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
        mkv_logits = cfg.predictor_lm(mkv_input_sequence, attention_mask=mkv_attention_mask, use_cache=False).logits[:, :-1, :]
        mkv_loss_tensor = loss_fn(
            input=einops.rearrange(
                mkv_logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=mkv_input_sequence[:, 1:]
        )
        obs_tensor = (mkv_loss_tensor * mkv_attention_mask[:, 1:])[:, -cfg.tok_p_pure_obs:]
        obs_losses = obs_tensor.sum(dim=-1) / mkv_attention_mask[:, 1:][:,-cfg.tok_p_pure_obs:].sum(dim=-1)

        if cfg.prediction_cfg.filter_best_actions:
            best_loss_indices = torch.topk(obs_losses, k=cfg.prediction_cfg.filter_best_actions, largest=False).indices
        else:
            best_loss_indices = torch.arange(obs_losses.size(0))

        prev_action_tensor = loss_tensor[best_loss_indices, : cfg.tok_p_action]
        prev_observation_tensor = loss_tensor[best_loss_indices, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs]
        action_tensor = loss_tensor[best_loss_indices, -cfg.tok_p_pure_action :]
        prev_action_loss = prev_action_tensor.mean()
        prev_observation_loss = prev_observation_tensor.mean()
        action_loss = action_tensor.mean()
        obs_loss = obs_losses[best_loss_indices].mean()
        #aggregate_loss = obs_loss
        aggregate_loss =  action_loss + obs_loss#.detach()
        #aggregate_loss = sum(map(lambda x: x[1] if x[0] else 0.0, 
        #                         zip([prediction_cfg.train_A_given_AO, prediction_cfg.train_O_given_A],
        #                                [action_loss, obs_loss])))

    if do_weight_update:
        cfg.optimizer.zero_grad()
        aggregate_loss.backward()
        cfg.optimizer.step()
        cfg.optimizer.zero_grad()

    if prediction_cfg.train_O_given_prev_O:
        return aggregate_loss, None, [] 
    loss_tensors = (
        prev_action_tensor,
        prev_observation_tensor,
        action_tensor,
        obs_tensor,
    )
    losses = [prev_action_loss, prev_observation_loss, action_loss, obs_loss]
    return aggregate_loss, loss_tensors, losses


def log_and_save(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action,
    obs,
    is_guidance_action,
    is_first,
    aggregate_loss,
    losses,
):
    save_weights(cfg, batch_index)
    log_print_oa(
        cfg,
        batch_index,
        prev_action,
        prev_obs,
        action,
        obs,
        is_guidance_action,
        is_first,
    )
    if cfg.prediction_cfg.train_O_given_prev_O:
        if cfg.wandb and dist.get_rank() == 0:
            wandb.log({"Batch Index": batch_index, "Observation Loss": aggregate_loss}, step=batch_index)
    else:
        log_wandb(cfg, batch_index, aggregate_loss, losses)
        log_print_losses(cfg, batch_index, aggregate_loss, losses)


def perturb_action(action, cfg):
    offset = cfg.action_prefix_tensor.shape[-1]
    # PERTURBATION 1
    # Given n <= cfg.tok_p_pure_action, change token through randomization
    frac_randomize = cfg.perturbation_cfg.frac_of_tokens_to_randomize
    assert 1.0 >= frac_randomize >= 0.0, f"frac_randomize is {frac_randomize}"
    perturb_target_inds = torch.randint(
        low=offset,
        high=action.shape[-1],
        size=[int(frac_randomize * (action.shape[-1] - offset))],
    )
    action[:, perturb_target_inds] = torch.randint(
        low=0,
        high=cfg.causal_lm_tokenizer.vocab_size,
        size=[int(frac_randomize * (action.shape[-1] - offset))],
    )

    # PERTURBATION 2
    # Given a fraction of cfg.tok_p_pure_action, replace with spaces/padding
    frac_spaces = cfg.perturbation_cfg.frac_of_tokens_to_pad
    assert 1.0 >= frac_spaces >= 0.0, f"frac_randomize is {frac_spaces}"
    token_id_space = cfg.causal_lm_tokenizer.encode(" ")[-1]
    action[
        :, offset + int((1.0 - frac_spaces) * (action.shape[-1] - offset)) :
    ] = token_id_space

    return action


def trainer(cfg):
    state = [default_action(cfg), 0, None]

    def update(datapt_pair):
        nonlocal state
        prev_datapt, datapt = datapt_pair
        is_first = "First" in datapt and datapt["First"]
        prev_action, batch_index, _ = state
        prev_obs, obs = prev_datapt["Observation"], datapt["Observation"]
        if is_first:
            prev_action = default_action(cfg)
            log_print_oa(
                cfg,
                batch_index,
                prev_action,
                prev_obs,
                None,
                obs,
                "Action" in datapt,
                is_first,
            )
            state = [prev_action, batch_index + 1, None]
            return
        # now can assume that prev_datapt contains the question and datapt contains the Answer
        if "Action" in datapt:
            action = datapt["Action"]
        elif cfg.prediction_cfg.train_O_given_prev_O:
            action = prev_action
        else:
            action = sample(cfg, prev_action, prev_obs, obs)

        aggregate_loss, loss_tensors, losses = update_weights(
            cfg,
            batch_index,
            prev_action,
            prev_obs,
            action,
            obs,
            do_weight_update=not isinstance(cfg.debug, NoWeightUpdates),
        )

        if (
            cfg.perturbation_cfg is not None
            and batch_index % cfg.perturbation_cfg.eval_every == 0
        ):
            perturbed_action = perturb_action(action, cfg)
            aggregate_loss0, loss_tensors0, losses0 = update_weights(
                cfg,
                batch_index,
                prev_action,
                prev_obs,
                perturbed_action,
                obs,
                do_weight_update=False,
            )
            perturbed_loss = losses0[-1]
        else:
            perturbed_loss = None
        losses.append(perturbed_loss)

        log_and_save(
            cfg,
            batch_index,
            prev_action,
            prev_obs,
            action,
            obs,
            "Action" in datapt,
            is_first,
            aggregate_loss,
            losses,
        )
        state = [action, batch_index + 1, aggregate_loss]
        return

    def pi():
        nonlocal state
        return state[-1]

    return update, pi


def train_via_update(cfg):
    aggregate_losses = []
    update, pi = trainer(cfg)
    for datapt_pair in tqdm(cfg.dataset.dataloader, total=cfg.num_batches):
        aggregate_loss = pi()
        if aggregate_loss is not None:
            aggregate_losses.append(aggregate_loss)
        update(datapt_pair)
    return aggregate_losses


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
            name=create_run_name(cfg),
        )
    if cfg.model_name == "gptj":
        block_name = GPTJBlock
    elif cfg.model_name == "mistral":
        block_name = MistralDecoderLayer
    elif cfg.model_name == "llama":
        block_name = LlamaDecoderLayer
    elif "gpt2" in cfg.model_name:
        block_name = GPT2Block
    else:
        assert "Unsupported model name for fdsp wrap policy"
    transformer_auto_wrapper_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            block_name,
        },
    )
    cfg.inference_lm = FullyShardedDataParallel(
        cfg.inference_lm, 
        auto_wrap_policy=transformer_auto_wrapper_policy,
        sharding_strategy=fsdp.ShardingStrategy.FULL_SHARD,
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        device_id=torch.cuda.current_device()
    )

    print("init", cfg.inference_lm._fsdp_wrapped_module.transformer.h[0]._is_root)
    print("Inference: ", cfg.inference_lm)
    print("Predictor: ", cfg.predictor_lm)
    if cfg.optimizer == "sgd":
        cfg.optimizer = torch.optim.SGD(
            cfg.predictor_lm.parameters(), lr=cfg.lr
        )  # , momentum=0.01)
    elif cfg.optimizer == "adam":
        cfg.optimizer = torch.optim.Adam(cfg.predictor_lm.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "rmsprop":
        cfg.optimizer = torch.optim.RMSprop(cfg.predictor_lm.parameters(), lr=cfg.lr)
    else:
        raise ValueError(
            f"Unsupported optimizer: {cfg.optimizer}. Please choose 'sgd', 'adam', or 'rmsprop'."
        )
    train_via_update(cfg)
    if cfg.wandb and dist.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    print("rank", dist.get_rank())
    for init_cfg in configs:
        train_model(init_cfg)
