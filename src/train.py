import torch
from torch.cuda.amp import autocast

from tqdm import tqdm
import wandb
import json
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta

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
        current_time = datetime.now(timezone(timedelta(hours=-7)))
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        torch.save(cfg.causal_lm, f"{cfg.path_2_model}_qhead_{timestamp}.pth")
        torch.save(cfg.v_head, f"{cfg.path_2_model}_vhead_{timestamp}.pth")


def log_wandb(cfg, batch_index, aggregate_loss, losses):
    if cfg.wandb and (cfg.use_mac or cfg.rank == 0):
        if cfg.prediction_cfg.train_O_given_prev_O:
            observation_loss = losses[0]
            wandb.log(
                {"Batch Index": batch_index, "Observation Loss": aggregate_loss},
                step=batch_index,
            )
        else:
            (
                action_losses,
                observation_losses,
                value_losses,
                negentropies,
                perturbed_losses,
            ) = losses
            wandb.log(
                {
                    "Batch Index": batch_index,
                    "Aggregate Loss": aggregate_loss.item(),
                    "Action Loss": action_losses.mean(),
                    "Observation Loss": observation_losses.mean(),
                    "Value Loss": value_losses.mean(),
                    "Negentropy": negentropies.mean(),
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
                action_losses,
                observation_losses,
                value_losses,
                negentropies,
                perturbed_losses,
            ) = losses
            with open(cfg.path_2_log, "a") as f:
                if not action_is_generated:
                    multi_print(f"Pre-Generated Action", f)
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                multi_print(
                    f"Action/Obs/Value/NegEnt loss: {action_losses.mean():0.4f}/{observation_losses.mean():0.4f}/{value_losses.mean():0.4f}/{negentropies.mean():0.4f}",
                    f,
                )


def save_trajectory(
    cfg, batch_index, prev_action, prev_obs, action, obs, losses, aggregate_loss
):
    if cfg.use_mac or cfg.rank == 0:

        (
            action_losses,
            observation_losses,
            value_losses,
            negentropies,
            perturbed_losses,
        ) = losses

        traj_data = {
            "batch_index": batch_index,
            "prev_action": cfg.tokenizer.batch_decode(prev_action),
            "prev_obs": cfg.tokenizer.batch_decode(prev_obs),
            "action": cfg.tokenizer.batch_decode(action),
            "obs": cfg.tokenizer.batch_decode(obs),
            "aggregate_loss": aggregate_loss.item(),
            "action_loss": action_losses.tolist(),
            "observation_losses": observation_losses.tolist(),
            "value_loss": value_losses.tolist() if value_losses is not None else 0.0,
            "negentropy": negentropies.tolist() if negentropies is not None else 0.0,
            "perturbed_loss": (
                perturbed_losses.tolist() if perturbed_losses is not None else 0.0
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
            model = (
                cfg.causal_lm.module.qhead
                if add_q_head
                else cfg.causal_lm.module.transformer
            )
            action_candidates = model.generate(
                beam_input_ids,
                min_new_tokens=cfg.pure_ctxt_sizes.action_size,
                max_new_tokens=cfg.pure_ctxt_sizes.action_size,
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
    trained_sender_receiver_obs_losses= []
    for action in input_strings:
        obs_loss_batch = predict_observation(cfg, action, obs, add_q_head=False)
        obs_losses.append(obs_loss_batch)
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
        action_losses, values, negentropies = predict_action(
            cfg, prev_action, prev_obs, action, add_q_head=True, add_v_head=True
        )
        with torch.no_grad():  # just to be sure
            old_critic_action_losses, _ = predict_action(
                cfg, prev_action, prev_obs, action, add_q_head=False, add_v_head=False
            )
            trained_sender_obs_losses = predict_observation(
                cfg,
                action,
                obs,
                add_q_head=False,
                is_default_action=False,
            )
            obs_losses = predict_observation(
                cfg,
                default_action,
                obs,
                add_q_head=False,
                is_default_action=True,
            )
            trained_receiver_obs_losses = predict_observation(
                cfg,
                default_action,
                obs,
                add_q_head=True,
                is_default_action=False,
            )
        trained_sender_receiver_obs_losses= predict_observation(
            cfg,
            action,
            obs,
            add_q_head=True,
            is_default_action=False,
        )
        normalized_obs_losses = trained_sender_receiver_obs_losses- trained_receiver_obs_losses
        repeated_obs_losses = normalized_obs_losses.unsqueeze(1).repeat(
            1, values.shape[1]
        )
        action_log_probs = -action_losses
        old_critic_action_log_probs = -old_critic_action_losses
        action_prob_ratios = torch.exp(action_log_probs - old_critic_action_log_probs)
        clipped_ratios = torch.clamp(action_prob_ratios, 0.9, 1.1)
        value_losses = torch.abs(values - repeated_obs_losses).mean(dim=1)
        neg_advantages = (repeated_obs_losses - values.detach()).mean(dim=1)
        unclipped = action_prob_ratios * neg_advantages
        clipped = clipped_ratios * neg_advantages
        max_branch = torch.max(unclipped, clipped)
        aggregate_losses = max_branch + value_losses
        aggregate_loss = (trained_sender_receiver_obs_losses.detach() * action_log_probs + trained_sender_receiver_obs_losses).mean()
        #aggregate_losses = trained_sender_receiver_obs_losses* action_log_probs
        if cfg.wandb and cfg.rank == 0 and action_is_generated:
            wandb.log(
                {
                    "Values": values.mean(),
                    "Normalized Obs Loss": normalized_obs_losses.mean(),
                    "Normalized Sender Score": trained_sender_obs_losses.mean()-obs_losses.mean(),
                    "Normalized Receiver Score": trained_receiver_obs_losses.mean()-obs_losses.mean(), 
                    "Value Loss": value_losses.mean(),
                    "Old Critic Action Loss": old_critic_action_losses.mean(),
                    "Action Prob Ratio": action_prob_ratios.mean(),
                    "Unclipped": unclipped.mean(),
                    "Clipped": clipped.mean(),
                    "Max Branch": max_branch.mean(),
                    "ClipFrac": (unclipped != max_branch).float().mean(),
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

        aggregate_loss.backward()
        accumulate_steps = 10
        if do_weight_update and batch_index > 0 and batch_index % accumulate_steps == 0:
            for param in cfg.causal_lm.module.parameters():
                if param.grad is not None:
                    param.grad /= float(accumulate_steps)
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

        losses = [action_losses, obs_losses, value_losses, negentropies]
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
