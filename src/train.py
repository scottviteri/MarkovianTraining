# , pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed as dist

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
from contextlib import nullcontext
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
        and (cfg.use_mac or cfg.rank == 0)
    ):
        print(f"Saving trained_{cfg.model_name} \n\n")
        cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
        torch.save(cfg.causal_lm, cfg.path_2_model + ".pth")
        # cfg.causal_lm.save_pretrained(cfg.path_2_model)


def log_wandb(cfg, batch_index, aggregate_loss, losses):
    if cfg.wandb and (cfg.use_mac or cfg.rank == 0):
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


def log_print_losses(cfg, batch_index, action_is_generated, aggregate_loss, losses):

    if batch_index % cfg.interval_print == 0 and (cfg.use_mac or cfg.rank == 0):
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
                if not action_is_generated:
                    multi_print(f"Pre-Generated Action", f)
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
    is_first,
    aggregate_loss,
):
    if batch_index % cfg.interval_print == 0 and (cfg.use_mac or cfg.rank == 0):
        if not is_first:
            with open(cfg.path_2_log, "a", encoding="utf-8") as f:
                multi_print(f"Batch Index: {batch_index}", f)
                if aggregate_loss:
                    multi_print(f"Aggregate Loss: {aggregate_loss}", f)
                multi_print(
                    f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0])) if prev_action is not None else None}",
                    f,
                )
                if not is_first:
                    multi_print(
                        f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                        f,
                    )
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


def save_trajectory(
    cfg, batch_index, prev_action, prev_obs, action, obs, losses, aggregate_loss
):
    if cfg.use_mac or cfg.rank == 0:

        (
            action_loss,
            observation_loss,
            value_loss,
            negentropy,
            perturbed_loss,
        ) = losses

        traj_data = {
            "batch_index": batch_index,
            "prev_action": cfg.causal_lm_tokenizer.decode(prev_action[0]),
            "prev_obs": cfg.causal_lm_tokenizer.decode(prev_obs[0]),
            "action": cfg.causal_lm_tokenizer.decode(action[0]),
            "obs": cfg.causal_lm_tokenizer.decode(obs[0]),
            "aggregate_loss": aggregate_loss.item(),
            "action_loss": action_loss.item(),
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
                    transformers.generation.TemperatureLogitsWarper(1.0),
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

            beam_input_ids = input_ids.repeat_interleave(cfg.num_beams, dim=0)
            beam_observations = observation.repeat_interleave(cfg.num_beams, dim=0)
            action_candidates = cfg.causal_lm.module.generate(
                beam_input_ids,
                add_q_head=add_q_head,
                get_v_head=False,
                attention_mask=attention_mask.repeat_interleave(cfg.num_beams, dim=0),
                pad_token_id=cfg.causal_lm.module.generation_config.pad_token_id,
                eos_token_id=cfg.causal_lm.module.generation_config.eos_token_id,
                output_scores=cfg.causal_lm.module.generation_config.output_scores,
                return_dict_in_generate=cfg.causal_lm.module.generation_config.return_dict_in_generate,
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
    action_is_generated,
    action,
    default_action,
    obs,
    do_weight_update=True,
):
    prediction_cfg = cfg.prediction_cfg

    if "llama" in cfg.model_name or "mistral" in cfg.model_name:
        assert (
            cfg.causal_lm.module.qhead.base_model.model.model.layers[
                -3
            ].mlp.up_proj.base_layer.weight
            - cfg.causal_lm.module.transformer.model.layers[-3].mlp.up_proj.weight
        ).abs().sum().item() == 0, "Frozen weight copies should be equal"

    # assert (cfg.causal_lm.module.qhead.base_model.model.model.layers[-3].mlp.up_proj.base_layer.weight == cfg.causal_lm.module.transformer.model.layers[-3].mlp.up_proj.weight).all(), "These weights should be frozen and equal."

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
        # aggregate_loss = action_prob_ratio * neg_advantage + value_loss
        aggregate_loss = (
            torch.max(action_prob_ratio * neg_advantage, clipped_ratio * neg_advantage)
            + value_loss
        )
        if cfg.wandb and cfg.rank == 0 and action_is_generated:
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

        if "llama" in cfg.model_name or "mistral" in cfg.model_name:
            weights_before = cfg.causal_lm.module.transformer.model.layers[
                -3
            ].mlp.up_proj.weight
            non_qhead_weights_before = (
                cfg.causal_lm.module.qhead.base_model.model.model.layers[
                    -3
                ].mlp.up_proj.weight
            )

        if do_weight_update:
            cfg.optimizer.zero_grad()
            aggregate_loss.backward()
            cfg.optimizer.step()
            cfg.optimizer.zero_grad()

        if "llama" in cfg.model_name or "mistral" in cfg.model_name:
            weights_after = cfg.causal_lm.module.transformer.model.layers[
                -3
            ].mlp.up_proj.weight
            non_qhead_weights_after = (
                cfg.causal_lm.module.qhead.base_model.model.model.layers[
                    -3
                ].mlp.up_proj.weight
            )
            assert (weights_before == weights_after).all(), "Should be frozen"
            assert (
                non_qhead_weights_before == non_qhead_weights_after
            ).all(), "Should be frozen"

        losses = [action_loss, obs_loss, value_loss, negentropy]
        return aggregate_loss, losses


def log_and_save(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action_is_generated,
    action,
    default_action,
    obs,
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

    save_weights(cfg, batch_index)
    # if action_is_generated:
    save_trajectory(
        cfg, batch_index, prev_action, prev_obs, action, obs, losses, aggregate_loss
    )
    log_wandb(cfg, batch_index, aggregate_loss, losses)
    log_print_losses(cfg, batch_index, action_is_generated, aggregate_loss, losses)

    log_print_oa(
        cfg=cfg,
        batch_index=batch_index,
        prev_action=prev_action,
        prev_obs=prev_obs,
        action=action,
        default_action=default_action,
        obs=obs,
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
    state = TrainerState(
        action=None, obs=None, batch_index=0, aggregate_loss=None, replay_buffer=[]
    )
    # replay buffer as trajectory and a score

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
                is_first=datapt.is_first,
                aggregate_loss=state.aggregate_loss,
            )
            state = TrainerState(
                action=datapt.action,
                obs=datapt.obs,
                batch_index=state.batch_index,
                aggregate_loss=None,
                replay_buffer=state.replay_buffer,
            )

            return
        else:
            # now can assume that prev_datapt contains the question and datapt contains the Answer
            action_is_generated = False
            if datapt.action is not None:
                action = datapt.action
                default_action = sample(
                    cfg, state.action, state.obs, datapt.obs, add_q_head=False
                )
            elif cfg.prediction_cfg.train_O_given_prev_O:
                action = state.action  # why?
            else:
                action_is_generated = True
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
                action_is_generated,
                action,
                default_action,
                datapt.obs,
                do_weight_update=not isinstance(cfg.debug, NoWeightUpdates),
            )

            # not currently using this functionality, assume None
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
                    action_is_generated,  # we could pass True if needed
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
                action_is_generated,
                action,
                default_action,
                datapt.obs,
                datapt.is_first,
                aggregate_loss,
                losses,
            )
            new_buffer_element = ScoredTrajectory(
                prev_datapt=Datapt(action=state.action, obs=state.obs, is_first=True),
                datapt=Datapt(action=action, obs=datapt.obs, is_first=datapt.is_first),
                loss=aggregate_loss,
            )
            if action_is_generated and cfg.replay_buffer_size is not None:
                new_replay_buffer = (
                    state.replay_buffer[:-1]
                    if len(state.replay_buffer) == cfg.replay_buffer_size
                    else state.replay_buffer
                ) + [new_buffer_element]
            else:
                new_replay_buffer = state.replay_buffer

            state = TrainerState(
                action=action,
                obs=datapt.obs,
                batch_index=state.batch_index + (1 if action_is_generated else 0),
                aggregate_loss=aggregate_loss,
                replay_buffer=new_replay_buffer,
            )
            return

    def pi():
        nonlocal state
        return state

    return update, pi

    # return aggregate_losses


def process_replay_buffer(cfg, state, update):
    if (
        cfg.replay_buffer_size is not None
        and state.batch_index % cfg.replay_buffer_size == 0
        and len(state.replay_buffer) >= cfg.replay_buffer_size
    ):
        top_trajectories = sorted(state.replay_buffer, key=lambda x: x.loss)[
            : cfg.replay_buffer_size // 10
        ]
        for trajectory in top_trajectories:
            update(trajectory.prev_datapt)
            update(trajectory.datapt)


def train_model(init_cfg):
    cfg = extend_initial_config(init_cfg)
    update, pi = trainer(cfg)
    for datapt in tqdm(cfg.dataset.dataloader, total=cfg.num_batches):
        state = pi()
        if datapt.is_first:
            process_replay_buffer(cfg, state, update)
        update(datapt)

    if cfg.wandb and (cfg.use_mac or cfg.rank == 0):
        wandb.finish()


()

if __name__ == "__main__":
    for init_cfg in configs:
        train_model(init_cfg)
