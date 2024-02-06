import torch
from tqdm import tqdm
import wandb
import einops
from datasets import load_dataset

from src.training_types import *
from src.utilities import log_and_print_info, multi_print


def train_ei(cfg: Config):

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    
    aggregate_losses = []
    prev_obs, prev_action = reset_obs_action(cfg)
    with open(cfg.path_2_log, "a") as f:
        f.write("")

    for batch_index, datapt in tqdm(enumerate(cfg.dataset.dataloader), total=cfg.num_batches):
        obs = datapt["Observation"]
        is_first = "First" in datapt and datapt["First"]
        if is_first:
            prev_obs, prev_action = reset_obs_action(cfg)
            if "Action" in datapt:
                action = datapt["Action"]
            else:
                action = cfg.causal_lm.generate(
                            inputs=torch.cat([obs, cfg.action_prefix_tensor], dim=1),
                            output_scores=True,
                            do_sample=True,
                            temperature=1.0,
                            min_new_tokens=cfg.tok_p_pure_action,
                            max_new_tokens=cfg.tok_p_pure_action,
                            pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                        )[:, -cfg.tok_p_action :]

        else:
            if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
                print(f"Saving trained_{cfg.model_name} \n\n")
                cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
                cfg.causal_lm.save_pretrained(cfg.path_2_model)

            if "Action" in datapt:
                action = datapt["Action"]
            else:
                with torch.no_grad():
                    action_candidates = [
                        cfg.causal_lm.generate(
                            inputs=torch.cat([prev_action, prev_obs, cfg.action_prefix_tensor], dim=1),
                            output_scores=True,
                            do_sample=True,
                            temperature=1.0,
                            min_new_tokens=cfg.tok_p_pure_action,
                            max_new_tokens=cfg.tok_p_pure_action,
                            pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                        )[:, -cfg.tok_p_action :]
                        for _ in range(cfg.training_type.num_samples)
                    ]
                    losses = []
                    for action_candidate in action_candidates:
                        input_sequence = torch.cat([action_candidate, obs], dim=1)
                        logits = cfg.causal_lm(input_sequence).logits[:, :-1, :]
                        loss_tensor = loss_fn(
                            input=einops.rearrange(
                                logits,
                                "batch seq_length vocab_size -> batch vocab_size seq_length",
                            ),
                            target=input_sequence[:, 1:]
                        )
                        loss = loss_tensor[:,-cfg.tok_p_obs:].mean().item()
                        losses.append(loss)
                    min_loss_index = losses.index(min(losses))
                    action = action_candidates[min_loss_index]

            input_sequence = torch.cat([prev_action, prev_obs, action], dim=1)
            logits = cfg.causal_lm(input_sequence).logits[:, :-1, :]
            loss_tensor = loss_fn(
                input=einops.rearrange(
                    logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=input_sequence[:, 1:]
            )

            prev_action_tensor = loss_tensor[:, : cfg.tok_p_action]
            prev_observation_tensor = loss_tensor[:, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs]
            action_tensor = loss_tensor[:, cfg.tok_p_action + cfg.tok_p_obs :]
            prev_action_loss = prev_action_tensor.mean()
            prev_observation_loss = prev_observation_tensor.mean()
            action_loss = action_tensor.mean()

            aggregate_loss = sum(map(lambda x: x[1] if x[0] else 0.0, zip(
                [cfg.training_type.prev_action, cfg.training_type.prev_observation, cfg.training_type.action], 
                [prev_action_loss, prev_observation_loss, action_loss])))

            aggregate_losses.append(aggregate_loss.item())

            if not isinstance(cfg.debug, NoWeightUpdates):
                aggregate_loss.backward()
                optimizer.step()

            if cfg.wandb:
                wandb.log(
                    {
                        "Batch Index": batch_index,
                        "Aggregate Loss": aggregate_loss,
                        "Previous Action Loss": prev_action_loss,
                        "Previous Observation Loss": prev_observation_loss,
                        "Action Loss": action_loss
                    }
                )

            # printing
            if batch_index % cfg.interval_print == 0:
                with open(cfg.path_2_log, "a") as f:
                    multi_print(f"Batch {batch_index}", f)
                    multi_print(f"Aggregate loss: {aggregate_loss}", f)
                    multi_print(
                        f"PrevAction/PrevObservation/Action loss: {prev_action_loss}/{prev_observation_loss}/{action_loss}",
                        f,
                    )
                    if prev_obs is not None:
                        multi_print(
                            f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                            f,
                        )
                    multi_print(
                        f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0]))}", f
                    )
                    multi_print(
                        f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f
                    )
                    multi_print(
                        f"Action: {repr(cfg.causal_lm_tokenizer.decode(action[0]))}",
                        f,
                    )
                    multi_print("______________________________________________________", f)
        prev_action = action
        prev_obs = obs

    return aggregate_losses

def reset_obs_action(cfg):
    prev_obs = torch.cat(
        (
            cfg.action_prefix_tensor,
            torch.full(
                (cfg.batch_size, cfg.tok_p_pure_obs),
                fill_value=cfg.causal_lm_tokenizer.pad_token_id,
                dtype=torch.int64,
                device=cfg.device,
            ),
        ),
        dim=1,
    )
    prev_action = torch.cat(
        (
            cfg.action_prefix_tensor,
            torch.full(
                (cfg.batch_size, cfg.tok_p_pure_action),
                fill_value=cfg.causal_lm_tokenizer.pad_token_id,
                dtype=torch.int64,
                device=cfg.device,
            ),
        ),
        dim=1,
    )
    return prev_obs, prev_action

