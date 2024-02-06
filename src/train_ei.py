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
    prev_obs = None
    with open(cfg.path_2_log, "a") as f:
        f.write("")

    for batch_index, datapt in tqdm(enumerate(cfg.dataset.dataloader), total=cfg.num_batches):
        obs = datapt["Observation"]
        if prev_obs is None:
            next_action = torch.cat(
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
        else:
            if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
                print(f"Saving trained_{cfg.model_name} \n\n")
                cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
                cfg.causal_lm.save_pretrained(cfg.path_2_model)

            if "Action" in datapt:
                next_action = datapt["Action"]
            else:
                with torch.no_grad():
                    next_action_candidates = [
                        cfg.causal_lm.generate(
                            inputs=torch.cat([action, prev_obs, cfg.action_prefix_tensor], dim=1),
                            output_scores=True,
                            do_sample=True,
                            min_new_tokens=cfg.tok_p_pure_action,
                            max_new_tokens=cfg.tok_p_pure_action,
                            pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                        )[:, -cfg.tok_p_action :]
                        for _ in range(cfg.training_type.num_samples)
                    ]
                    losses = []
                    for next_action in next_action_candidates:
                        input_sequence = torch.cat([next_action, obs], dim=1)
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
                    next_action = next_action_candidates[min_loss_index]

            input_sequence = [action, prev_obs, next_action]
            logits = cfg.causal_lm(torch.cat(input_sequence, dim=1)).logits[
                :, :-1, :
            ]
            loss_tensor = loss_fn(
                input=einops.rearrange(
                    logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=torch.cat(input_sequence, dim=1)[:, 1:]
            )

            action_tensor = loss_tensor[:, : cfg.tok_p_action]
            observation_tensor = loss_tensor[:, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs]
            next_action_tensor = loss_tensor[:, cfg.tok_p_action + cfg.tok_p_obs :]
            action_loss = action_tensor.mean()
            observation_loss = observation_tensor.mean()
            next_action_loss = next_action_tensor.mean()

            aggregate_loss = torch.tensor(0, device=cfg.device, dtype=torch.float32)
            if cfg.training_type.action:
                aggregate_loss += action_loss
            if cfg.training_type.observation:
                aggregate_loss += observation_loss
            if cfg.training_type.next_action: 
                aggregate_loss += next_action_loss

            is_first = "First" in datapt and datapt["First"]
            if not is_first: aggregate_losses.append(aggregate_loss.item())

            if not isinstance(cfg.debug, NoWeightUpdates) and not is_first:
                aggregate_loss.backward()
                optimizer.step()

            if cfg.wandb and not is_first:
                wandb.log(
                    {
                        "Batch Index": batch_index,
                        "Aggregate Loss": aggregate_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss,
                        "Next Action Loss": next_action_loss
                    }
                )

            # printing
            if batch_index % cfg.interval_print == 0 and not is_first:
                with open(cfg.path_2_log, "a") as f:
                    multi_print(f"Batch {batch_index}", f)
                    multi_print(f"Aggregate loss: {aggregate_loss}", f)
                    multi_print(
                        f"Action/Observation/NextAction loss: {action_loss}/{observation_loss}/{next_action_loss}",
                        f,
                    )
                    if prev_obs is not None:
                        multi_print(
                            f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                            f,
                        )
                    multi_print(
                        f"Action: {repr(cfg.causal_lm_tokenizer.decode(action[0]))}", f
                    )
                    multi_print(
                        f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f
                    )
                    multi_print(
                        f"Next Action: {repr(cfg.causal_lm_tokenizer.decode(next_action[0]))}",
                        f,
                    )
                    multi_print("______________________________________________________", f)
        action = next_action
        prev_obs = obs

    return aggregate_losses
