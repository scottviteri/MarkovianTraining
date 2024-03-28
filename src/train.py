# , pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
from torch.cuda.amp import autocast

from tqdm import tqdm
import wandb
import json
from contextlib import nullcontext

from training_types import *
from utilities import (
    extend_initial_config,
    log_print_oa,
    predict_action,
    predict_observation,
    get_neg_log_probs,
    multi_print,
    wrap_input_tokens,
)
from config_examples import configs
from evaluate_actions import perturb_action


def save_weights(cfg, batch_index):
    if (
        batch_index > 0
        and batch_index % cfg.interval_save_weights == 0
        and (cfg.use_mac or cfg.rank == 0)
    ):
        print(f"Saving trained_{cfg.model_name} \n\n")
        cfg.tokenizer.save_pretrained(cfg.path_2_tokenizer)
        torch.save(cfg.causal_lm, cfg.path_2_model + ".pth")


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
            "prev_action": cfg.tokenizer.decode(prev_action[0]),
            "prev_obs": cfg.tokenizer.decode(prev_obs[0]),
            "action": cfg.tokenizer.decode(action[0]),
            "obs": cfg.tokenizer.decode(obs[0]),
            "aggregate_loss": aggregate_loss.item(),
            "action_loss": action_loss.item(),
            "observation_loss": observation_loss.mean().item(),
            "value_loss": value_loss.item() if value_loss is not None else 0.0,
            "negentropy": negentropy.item() if negentropy is not None else 0.0,
            "perturbed_loss": (
                perturbed_loss.item() if perturbed_loss is not None else 0.0
            ),
        }
        with open(cfg.traj_path, "r") as file:
            data = json.load(file)

        # Append new trajectory
        if "trajectory" not in data:
            data["trajectory"] = []
        data["trajectory"].append(traj_data)

        # Write updated data back to the file
        with open(cfg.traj_path, "w") as file:
            json.dump(data, file, indent=4)


def sample(cfg, prev_action, prev_obs, add_q_head=True):
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
            input_ids = wrap_input_tokens(
                cfg,
                [prev_action, prev_obs],
                [cfg.prefix_tensors.action_prefix_tensor],
                use_start_token=True,
                use_instruct_tokens=True,
                is_prediction=False,
            )
            attention_mask = (input_ids != cfg.tokenizer.pad_token_id).long()
            beam_input_ids = input_ids.repeat_interleave(cfg.num_beams, dim=0)
            action_candidates = cfg.causal_lm.module.generate(
                beam_input_ids,
                add_q_head=add_q_head,
                get_v_head=False,
                attention_mask=attention_mask.repeat_interleave(cfg.num_beams, dim=0),
                pad_token_id=cfg.causal_lm.module.generation_config.pad_token_id,
                eos_token_id=cfg.causal_lm.module.generation_config.eos_token_id,
                output_scores=cfg.causal_lm.module.generation_config.output_scores,
                return_dict_in_generate=cfg.causal_lm.module.generation_config.return_dict_in_generate,
            )[:, -cfg.ctxt_sizes.action_size :]
            return action_candidates


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


def handle_perturbation(cfg, state, action_is_generated, action, datapt):
    if (
        cfg.perturbation_cfg is not None
        and state.batch_index % cfg.perturbation_cfg.eval_every == 0
    ):
        perturbed_action = perturb_action(action, cfg)
        _, losses0 = update_weights(
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
    return perturbed_loss


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


def trainer(cfg):
    state = TrainerState(
        action=None, obs=None, batch_index=0, aggregate_loss=None, replay_buffer=[]
    )

    def update(datapt):
        nonlocal state
        if datapt.is_first:
            state = TrainerState(
                action=datapt.action,
                obs=datapt.obs,
                batch_index=state.batch_index,
                aggregate_loss=None,
                replay_buffer=state.replay_buffer,
            )

            return
        # now can assume that prev_datapt contains the question and datapt contains the Answer
        action_is_generated = False
        if datapt.action is not None:
            action = datapt.action
            default_action = sample(cfg, state.action, state.obs, add_q_head=False)
        elif cfg.prediction_cfg.train_O_given_prev_O:
            action = state.action  # why?
        else:
            action_is_generated = True
            action = sample(cfg, state.action, state.obs, add_q_head=True)
            default_action = sample(cfg, state.action, state.obs, add_q_head=False)

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

        perturbed_loss = handle_perturbation(
            cfg, state, action_is_generated, action, datapt
        )
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


if __name__ == "__main__":
    for init_cfg in configs:
        train_model(init_cfg)
