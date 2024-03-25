# , pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
import einops
import wandb
import json
import random
import os
from datasets import load_dataset
from datetime import datetime, timezone

from openai import OpenAI
from matplotlib import pyplot as plt
import functools
from contextlib import nullcontext
import bitsandbytes
import transformers
from transformers.generation import beam_search
from collections import UserDict

from training_types import *
from utilities import (
    extend_initial_config,
    log_and_print_info,
    predict_action,
    predict_observation,
    get_neg_log_probs,
    get_masked_mean,
    create_run_name,
    multi_print,
    wrap_input_tokens,
)
from config_examples import configs
from beam import BeamSearchScorer

from transformers.models.gptj.modeling_gptj import GPTJBlock
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from evaluate_via_gpt import evaluate_via_gpt
import config_examples


def save_weights(cfg, batch_index):
    if (
        batch_index > 0
        and batch_index % cfg.interval_save_weights == 0
        and (cfg.use_mac or dist.get_rank() == 0)
    ):
        print(f"Saving trained_{cfg.model_name} \n\n")
        cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
        cfg.causal_lm.save_pretrained(cfg.path_2_model)


def log_wandb(cfg, batch_index, aggregate_loss, losses):
    if cfg.wandb and (cfg.use_mac or dist.get_rank() == 0):
        if cfg.prediction_cfg.train_O_given_prev_O:
            observation_loss = losses[0]
            wandb.log(
                {"Batch Index": batch_index, "Observation Loss": aggregate_loss},
                step=batch_index,
            )
        else:
            (action_loss, observation_loss, value_loss, negentropy, perturbed_loss) = (
                losses
            )
            if value_loss:
                wandb.log(
                    {
                        "Batch Index": batch_index,
                        "Aggregate Loss": aggregate_loss,
                        "Perturbed Loss": perturbed_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss.mean(),
                        "Value Loss": value_loss,
                        "Negentropy": negentropy,
                    },
                    step=batch_index,
                )
            else:
                wandb.log(
                    {
                        "Batch Index": batch_index,
                        "Aggregate Loss": aggregate_loss,
                        "Perturbed Loss": perturbed_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss.mean(),
                        "Negentropy": negentropy,
                    },
                    step=batch_index,
                )


def log_print_losses(cfg, batch_index, aggregate_loss, losses):

    if batch_index % cfg.interval_print == 0 and (cfg.use_mac or dist.get_rank() == 0):
        if cfg.prediction_cfg.train_O_given_prev_O:
            observation_loss = losses[0]
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Obs loss: {observation_loss:0.4f}", f)
        else:
            (
                action_loss,
                observation_loss,
                value_loss,
                negentropy,
                perturbed_loss,
            ) = losses
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                if value_loss:
                    multi_print(
                        f"Action/Obs/Q/NegEnt/Pert loss: {action_loss:0.4f}/{observation_loss.mean():0.4f}/{value_loss:0.4f}/{negentropy:0.4f}/{perturbed_loss}",
                        f,
                    )
                else:
                    multi_print(
                        f"Action/Obs/Pert loss: {action_loss:0.4f}/{observation_loss.mean():0.4f}/{perturbed_loss}",
                        f,
                    )


def log_print_oa(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action,
    default_action,
    obs,
    is_guidance_action,
    is_first,
    aggregate_loss,
):
    if batch_index % cfg.interval_print == 0 and (cfg.use_mac or dist.get_rank() == 0):
        with open(cfg.path_2_log, "a", encoding="utf-8") as f:
            multi_print(f"Batch Index: {batch_index}", f)
            if aggregate_loss:
                multi_print(f"Aggregate Loss: {aggregate_loss}", f)
            multi_print(f"Training predictor mode: {cfg.training_predictor_mode}", f)
            multi_print(f"Is First: {is_first}", f)
            multi_print(
                f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0])) if prev_action is not None else None}",
                f,
            )
            if not is_first:
                multi_print(
                    f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                    f,
                )
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
                    f"Default Action: {repr(cfg.causal_lm_tokenizer.decode(default_action[0]))}",
                    f,
                )
            multi_print(
                f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f
            )


def init_traj_storage(traj_path, initial_config):
    """Initialize the JSON file with general config values if it doesn't exist."""
    if not os.path.exists(traj_path):
        with open(traj_path, "w") as file:
            json.dump(initial_config, file, indent=4)


def append_traj_to_storage(traj_path, traj_data):
    """Append data for a training step to the JSON file."""
    # Load existing data
    with open(traj_path, "r") as file:
        data = json.load(file)

    # Append new trajectory
    if "trajectory" not in data:
        data["trajectory"] = []
    data["trajectory"].append(traj_data)

    # Write updated data back to the file
    with open(traj_path, "w") as file:
        json.dump(data, file, indent=4)


def save_trajectory(cfg, batch_index, prev_action, prev_obs, action, obs, losses):
    if cfg.use_mac or dist.get_rank() == 0:
        if batch_index == 0:
            # Adding a UID to the file name to avoid overwriting
            cfg.traj_path += f"_{datetime.now(timezone.utc).timestamp():0.0f}.json"

        # Does nothing if file already exists
        init_traj_storage(
            cfg.traj_path,
            {
                "model": cfg.model_name,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_batches": cfg.num_batches,
                "num_beams": cfg.num_beams,
                "optimizer": repr(cfg.optimizer),
                "dataset": repr(cfg.dataset),
                "perturbation": repr(cfg.perturbation_cfg),
                "trainer": repr(cfg.trainer_cfg),
                "inference": repr(cfg.inference_cfg),
                "prediction": repr(cfg.prediction_cfg),
                "wandb_run_id": wandb.run.id if cfg.wandb else False,
                "wandb_run_name": wandb.run.name if cfg.wandb else False,
            },
        )

        (
            action_loss,
            observation_loss,
            value_loss,
            negentropy,
            perturbed_loss,
        ) = losses

        traj_data = {
            "batch_index": batch_index,
            # "prev_action": repr(cfg.causal_lm_tokenizer.decode(prev_action[0])),
            "prev_obs": repr(cfg.causal_lm_tokenizer.decode(prev_obs[0])),
            "action": repr(cfg.causal_lm_tokenizer.decode(action[0])),
            "obs": repr(cfg.causal_lm_tokenizer.decode(obs[0])),
            # "action_loss": action_loss.item(),
            "observation_loss": observation_loss.mean().item(),
            "value_loss": value_loss.item() if value_loss is not None else 0.0,
            "negentropy": negentropy.item() if negentropy is not None else 0.0,
            "perturbed_loss": (
                perturbed_loss.item() if perturbed_loss is not None else 0.0
            ),
        }
        append_traj_to_storage(cfg.traj_path, traj_data)


def sample(cfg, prev_action, prev_obs, observation, add_q_head=True):
    inference_cfg = cfg.inference_cfg
    # cfg.causal_lm.eval()
    with torch.inference_mode():
        with (
            nullcontext
            if cfg.use_mac
            else autocast(
                # cache_enabled=False,
                dtype=(
                    torch.bfloat16
                    if cfg.model_name in ["llama", "mistral"]
                    else torch.float16
                ),
            )
        ):
            # input_ids = torch.cat(
            #    [prev_action, prev_obs, cfg.prefix_tensors.action_prefix_tensor], dim=1
            # )
            input_ids = wrap_input_tokens(
                cfg,
                [prev_action, prev_obs],
                [cfg.prefix_tensors.action_prefix_tensor],
                use_start_token=True,
                use_instruct_tokens=True,
                is_prediction=False,
            )
            attention_mask = (input_ids != cfg.causal_lm_tokenizer.pad_token_id).long()

            bad_words_ids = [
                [
                    cfg.causal_lm_tokenizer.bos_token_id,
                    cfg.causal_lm_tokenizer.eos_token_id,
                ]
            ]

            logits_warper = transformers.generation.LogitsProcessorList(
                [
                    transformers.generation.NoBadWordsLogitsProcessor(
                        bad_words_ids,
                        eos_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                    ),
                    # transformers.generation.MinNewTokensLengthLogitsProcessor(
                    #    prompt_length_to_skip=input_ids.shape[-1],
                    #    min_new_tokens=cfg.pure_ctxt_sizes.action_size,
                    #    eos_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                    # ),
                    transformers.generation.TemperatureLogitsWarper(0.6),
                    transformers.generation.InfNanRemoveLogitsProcessor(),
                    transformers.LogitNormalization(),
                ]
            )
            # stopping_criteria = transformers.generation.StoppingCriteriaList(
            #    [
            #        transformers.generation.MaxLengthCriteria(
            #            max_length=input_ids.shape[-1] + cfg.pure_ctxt_sizes.action_size
            #        )
            #    ]
            # )

            generation_config = transformers.GenerationConfig(
                # max_new_tokens=cfg.pure_ctxt_sizes.action_size,
                # min_new_tokens=cfg.pure_ctxt_sizes.action_size,
                min_length=2 * cfg.ctxt_sizes.action_size + cfg.ctxt_sizes.obs_size,
                max_length=2 * cfg.ctxt_sizes.action_size + cfg.ctxt_sizes.obs_size,
                do_sample=True,
                logits_warper=logits_warper,
                # num_beams=cfg.num_beams,
                # renormalize_logits=True,
                # remove_invalid_values=True,
                num_return_sequences=cfg.inference_cfg.num_return_sequences,
                output_scores=True,
                pad_token_id=cfg.causal_lm_tokenizer.pad_token_id,
                eos_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                return_dict_in_generate=False,
                # length_penalty=1.0,
                # early_stopping=False,
            )
            cfg.causal_lm.generation_config = generation_config

            beam_input_ids = input_ids.repeat_interleave(cfg.num_beams, dim=0)
            beam_observations = observation.repeat_interleave(cfg.num_beams, dim=0)
            action_candidates = cfg.causal_lm.generate(
                beam_input_ids,
                add_q_head=add_q_head,
                get_v_head=False,
                attention_mask=attention_mask.repeat_interleave(cfg.num_beams, dim=0),
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                # synced_gpus=False,
            )[:, -cfg.ctxt_sizes.action_size :]
            return action_candidates


def get_neg_log_probs(cfg, input_sequence):
    """
    Computes the loss tensor for a given input sequence.

    Args:
        cfg: Configuration object containing model and tokenizer information.
        input_sequence: The input sequence tensor for which the loss is to be computed.

    Returns:
        The computed loss tensor.
    """
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    logits = cfg.causal_lm(input_sequence, attention_mask=attention_mask).logits[
        :, :-1, :
    ]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_tensor = loss_fn(
        input=einops.rearrange(
            logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )
    return loss_tensor


def get_masked_mean(arr, mask):
    return (arr * mask).sum() / mask.sum()


def log_manual_observations(cfg, obs):
    repeat = "210 210 210 210 210 210 210 210 210 210 210 210 210 210 210"
    in_order = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 37. Then, add 37 and 81 to get 118. Finally, add 118 and 92 to arrive at the final result 210"
    in_pieces = "Let's break down the expression 23 + 14 + 81 + 92 by evaluating the tens place first: 20 + 10 + 80 + 90 = 200. Now, let's add the ones place: 3 + 4 + 1 + 2 = 10. Combining the results from the tens and ones places gives us the final answer 210"
    in_order_corrupted = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 27. Then, add 27 and 81 to get 108. Finally, add 108 and 92 to arrive at the final result 210"
    direct_question = "The solution to 23 + 14 + 81 + 92 is 210"
    random_test = "I am a flying banana 210"
    input_strings = [
        repeat,
        in_order,
        in_pieces,
        in_order_corrupted,
        direct_question,
        random_test,
    ]
    obs_losses = []
    for action in input_strings:
        obs_loss = predict_observation(
            cfg, action, obs, add_q_head=False, per_batch=True
        )
        obs_losses.append(obs_loss)
    print(obs_losses)


def update_weights(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action,
    default_action,
    obs,
    do_weight_update=True,
):
    prediction_cfg = cfg.prediction_cfg

    assert (
        cfg.causal_lm.qhead.base_model.model.model.layers[
            -3
        ].mlp.up_proj.base_layer.weight
        - cfg.causal_lm.transformer.model.layers[-3].mlp.up_proj.weight
    ).abs().sum().item() == 0, "Frozen weight copies should be equal"

    # assert (cfg.causal_lm.qhead.base_model.model.model.layers[-3].mlp.up_proj.base_layer.weight == cfg.causal_lm.transformer.model.layers[-3].mlp.up_proj.weight).all(), "These weights should be frozen and equal."

    with (
        nullcontext
        if cfg.use_mac
        else autocast(
            dtype=(
                torch.bfloat16
                if cfg.model_name in ["llama", "mistral"]
                else torch.float16
            ),
        )
    ):
        if prediction_cfg.train_O_given_prev_O:
            assert not prediction_cfg.train_O_given_A
            loss_tensor = get_neg_log_probs(cfg, torch.cat([prev_obs, obs], dim=1))
            aggregate_loss = loss_tensor[:, -cfg.pure_ctxt_sizes.obs_size :].mean()
            return aggregate_loss, None, []

        prev_action = prev_action.repeat_interleave(
            cfg.inference_cfg.num_return_sequences, dim=0
        )
        prev_obs = prev_obs.repeat_interleave(
            cfg.inference_cfg.num_return_sequences, dim=0
        )
        obs = obs.repeat_interleave(cfg.inference_cfg.num_return_sequences, dim=0)
        action_loss, values, negentropy = predict_action(
            cfg, prev_action, prev_obs, action, add_q_head=True
        )
        with torch.no_grad():  # just to be sure
            old_critic_action_loss, _, old_critic_negentropy = predict_action(
                cfg, prev_action, prev_obs, action, add_q_head=False
            )
            obs_loss = predict_observation(
                cfg,
                action,
                obs,
                add_q_head=False,
                per_batch=True,
                is_default_action=False,
            )
            default_obs_loss = predict_observation(
                cfg,
                default_action,
                obs,
                add_q_head=False,
                per_batch=True,
                is_default_action=True,
            )
        normalized_negentropy = negentropy - old_critic_negentropy
        normalized_obs_loss = obs_loss - default_obs_loss
        repeated_obs_losses = normalized_obs_loss.unsqueeze(1).repeat(
            1, values.shape[1]
        )
        value_loss = torch.mean(torch.abs(values - repeated_obs_losses))
        action_log_prob = -action_loss
        obs_log_prob = -obs_loss
        old_critic_action_log_prob = -old_critic_action_loss
        action_prob_ratio = torch.exp(action_log_prob - old_critic_action_log_prob)
        clipped_ratio = torch.clamp(action_prob_ratio, 0.7, 1.3)
        value_loss = torch.abs(values - repeated_obs_losses).mean()
        # negentropy +
        neg_advantage = (repeated_obs_losses - values.detach()).mean()
        # neg_advantage = obs_loss.mean()
        # aggregate_loss = action_loss * (normalized_obs_loss.mean() + negentropy * .1)
        # aggregate_loss = -torch.min(
        #    action_prob_ratio * obs_log_prob.mean(),
        #    clipped_ratio * obs_log_prob.mean(),
        # )
        # (negentropy - old_critic_negentropy) * 0.1 +
        aggregate_loss = (
            torch.max(action_prob_ratio * neg_advantage, clipped_ratio * neg_advantage)
            + value_loss
        )
        if cfg.wandb:
            wandb.log(
                {
                    "Values": values.mean(),
                    "Normalized Obs Loss": normalized_obs_loss.mean(),
                    "Value Loss": value_loss,
                    "Old Critic Action Loss": old_critic_action_loss,
                    "Action Prob Ratio": action_prob_ratio,
                },
                step=batch_index,
            )

        if do_weight_update:
            cfg.optimizer.zero_grad()
            aggregate_loss.backward()
            cfg.optimizer.step()
            cfg.optimizer.zero_grad()
        losses = [action_loss, obs_loss, value_loss, negentropy]
        return aggregate_loss, losses


def log_and_save(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action,
    default_action,
    obs,
    is_guidance_action,
    is_first,
    aggregate_loss,
    losses,
):

    # if cfg.training_predictor_mode:
    #    # Save trajectories for post-training eval to .json
    #    save_trajectory(cfg, batch_index, prev_action, prev_obs, action, obs, losses)
    #    save_weights(cfg, batch_index)
    #    log_wandb(cfg, batch_index, aggregate_loss, losses)
    #    log_print_losses(cfg, batch_index, aggregate_loss, losses)
    # else:
    save_trajectory(cfg, batch_index, prev_action, prev_obs, action, obs, losses)
    save_weights(cfg, batch_index)
    log_wandb(cfg, batch_index, aggregate_loss, losses)
    log_print_losses(cfg, batch_index, aggregate_loss, losses)

    log_print_oa(
        cfg=cfg,
        batch_index=batch_index,
        prev_action=prev_action,
        prev_obs=prev_obs,
        action=action,
        default_action=default_action,
        obs=obs,
        is_guidance_action=is_guidance_action,
        is_first=is_first,
        aggregate_loss=aggregate_loss,  # if not cfg.training_predictor_mode else None,
    )

    with open(cfg.path_2_log, "a") as f:
        multi_print("______________________________________________________", f)


def perturb_action(action, cfg):
    offset = cfg.prefix_tensors.action_prefix_tensor.shape[-1]
    # PERTURBATION 1
    # Given n <= cfg.pure_ctxt_sizes.action_size, change token through randomization
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
    # Given a fraction of cfg.pure_ctxt_sizes.action_size, replace with spaces/padding
    frac_spaces = cfg.perturbation_cfg.frac_of_tokens_to_pad
    assert 1.0 >= frac_spaces >= 0.0, f"frac_randomize is {frac_spaces}"
    token_id_space = cfg.causal_lm_tokenizer.encode(" ")[-1]
    action[:, offset + int((1.0 - frac_spaces) * (action.shape[-1] - offset)) :] = (
        token_id_space
    )

    return action


def trainer(cfg):
    # state = [get_default_action(cfg), 0, None]
    state = TrainerState(action=None, obs=None, batch_index=0, aggregate_loss=None)

    def update(datapt):
        nonlocal state
        # prev_datapt, datapt = datapt_pair
        # prev_obs, obs = prev_datapt.obs, datapt.obs
        if state.batch_index > 0:
            pred_len, inf_len = (
                cfg.trainer_cfg.prediction_training_length,
                cfg.trainer_cfg.inference_training_length,
            )
            mode_index = state.batch_index % (pred_len + inf_len)
            # if mode_index == 0:  # switch from inf mode to pred mode
            #    assert not cfg.training_predictor_mode
            #    # for name, param in cfg.causal_lm.named_parameters():
            #    #    param.requires_grad = "q_head" not in name
            #    cfg.training_predictor_mode = True
            # elif mode_index == pred_len:
            #    assert cfg.training_predictor_mode
            #    # cfg.causal_lm.load_state_dict(cfg.causal_lm.state_dict())
            #    # for name, param in cfg.causal_lm.named_parameters():
            #    #    param.requires_grad = "q_head" in name
            #    cfg.training_predictor_mode = False

        if datapt.is_first:
            # get_default_action(cfg)
            log_print_oa(
                cfg=cfg,
                batch_index=state.batch_index,
                prev_action=None,
                prev_obs=state.obs,
                action=datapt.action,
                default_action=datapt.action,
                obs=datapt.obs,
                is_guidance_action=datapt.action is not None,
                is_first=datapt.is_first,
                aggregate_loss=state.aggregate_loss,
            )
            state = TrainerState(
                action=datapt.action,
                obs=datapt.obs,
                batch_index=state.batch_index,
                aggregate_loss=None,
            )

            return
        else:
            # now can assume that prev_datapt contains the question and datapt contains the Answer
            if datapt.action:
                action = datapt.action
            elif cfg.prediction_cfg.train_O_given_prev_O:
                action = state.action  # why?
            else:
                action = sample(
                    cfg, state.action, state.obs, datapt.obs, add_q_head=True
                )
                default_action = sample(
                    cfg, state.action, state.obs, datapt.obs, add_q_head=False
                )
            # action : [batch * beam, cfg.ctxt_sizes.action_size]

            aggregate_loss, losses = update_weights(
                cfg,
                state.batch_index,
                state.action,
                state.obs,
                action,
                default_action,
                datapt.obs,
                do_weight_update=not isinstance(cfg.debug, NoWeightUpdates),
            )

            if (
                cfg.perturbation_cfg is not None
                and state.batch_index % cfg.perturbation_cfg.eval_every == 0
            ):
                perturbed_action = perturb_action(action, cfg)
                aggregate_loss0, losses0 = update_weights(
                    cfg,
                    state.batch_index,
                    state.action,
                    state.obs,
                    perturbed_action,
                    datapt.obs,
                    do_weight_update=False,
                )
                perturbed_loss = losses0[-1]
            else:
                perturbed_loss = None
            losses.append(perturbed_loss)

            log_and_save(
                cfg,
                state.batch_index,
                state.action,
                state.obs,
                action,
                default_action,
                datapt.obs,
                datapt.action is not None,
                datapt.is_first,
                aggregate_loss,
                losses,
            )
            state = TrainerState(
                action=action,
                obs=datapt.obs,
                batch_index=state.batch_index + 1,
                aggregate_loss=aggregate_loss,
            )
            return

    def pi():
        nonlocal state
        return state.aggregate_loss

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
    if not cfg.use_mac:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        print("rank", dist.get_rank())
    # cfg.causal_lm = DDP(
    #    cfg.causal_lm, device_ids=[dist.get_rank()]  # , find_unused_parameters=True
    # )
    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")
    with open(cfg.path_2_log, "a") as f:
        f.write("")
    if cfg.wandb and (cfg.use_mac or dist.get_rank() == 0):
        wandb.init(
            project="collaborative-training-many-per-context-window",
            name=create_run_name(cfg),
        )
    print("Causal LM: ", cfg.causal_lm)
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD if cfg.use_mac else bitsandbytes.optim.SGD8bit
        cfg.optimizer = optimizer(
            cfg.causal_lm.parameters(), lr=cfg.lr
        )  # , momentum=0.01)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.AdamW if cfg.use_mac else bitsandbytes.optim.AdamW8bit
        cfg.optimizer = optimizer(  # torch.optim.Adam(  #
            list(cfg.causal_lm.qhead.parameters())
            + list(cfg.causal_lm.v_head_group.parameters()),
            lr=cfg.lr,
        )

    elif cfg.optimizer == "rmsprop":
        cfg.optimizer = torch.optim.RMSprop(cfg.causal_lm.parameters(), lr=cfg.lr)
    else:
        raise ValueError(
            f"Unsupported optimizer: {cfg.optimizer}. Please choose 'sgd', 'adam', or 'rmsprop'."
        )
    train_via_update(cfg)
    if cfg.wandb and (cfg.use_mac or dist.get_rank() == 0):
        wandb.finish()


if __name__ == "__main__":
    for init_cfg in configs:
        train_model(init_cfg)
