import torch
from tqdm import tqdm
import wandb
import einops
from datasets import load_dataset

from src.training_types import *
from src.utilities import log_and_print_info, multi_print


def train_aoa(cfg: Config):
    if cfg.training_type.use_gumbel:
        return train_gumbel(cfg)
    else:
        return train_without_gumbel(cfg)

def train_without_gumbel(cfg):

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    action = torch.cat(
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
    aggregate_losses = []
    prev_obs = None
    with open(cfg.path_2_log, "a") as f:
        f.write("")

    for batch_index, datapt in tqdm(enumerate(cfg.dataset.dataloader), total=cfg.num_batches):
        obs = datapt["Observation"]
        next_action = datapt["Action"] if "Action" in datapt else None

        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            cfg.causal_lm.save_pretrained(cfg.path_2_model)

        if next_action is None:
            with torch.no_grad():
                next_action = cfg.causal_lm.generate(
                inputs=torch.cat([action, obs, cfg.action_prefix_tensor], dim=1),
                output_scores=True,
                do_sample=True,
                min_new_tokens=cfg.tok_p_pure_action,
                max_new_tokens=cfg.tok_p_pure_action,
                pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
            )[:, -cfg.tok_p_action :]

        optimizer.zero_grad()
        input_sequence = (
            [action, obs]
            if cfg.training_type.ignore_second_action
            else [action, obs, next_action]
        )
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
        if not cfg.training_type.ignore_second_action:
            next_action_tensor = loss_tensor[:, cfg.tok_p_action + cfg.tok_p_obs :]

        if cfg.training_type.ignore_first_action and cfg.training_type.ignore_second_action:
            aggregate_loss = observation_tensor.mean()
        elif cfg.training_type.ignore_first_action and not cfg.training_type.ignore_second_action:
            aggregate_loss = torch.cat([observation_tensor, next_action_tensor], dim=1).mean()
        elif not cfg.training_type.ignore_first_action and cfg.training_type.ignore_second_action:
            aggregate_loss = torch.cat([action_tensor, observation_tensor], dim=1).mean()
        else:
            aggregate_loss = loss_tensor.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()
        if not isinstance(cfg.debug, NoWeightUpdates):
            optimizer.step()

        with torch.no_grad():
            action_loss = action_tensor.mean()
            observation_loss = observation_tensor.mean()
            if not cfg.training_type.ignore_second_action:
                next_action_loss = next_action_tensor.mean()

            if cfg.wandb:
                if not cfg.training_type.ignore_second_action:
                    wandb.log({"Next Action Loss": next_action_loss})
                wandb.log(
                    {
                        "Batch Index": batch_index,
                        "Aggregate Loss": aggregate_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss,
                    }
                )

        # printing
        if batch_index % cfg.interval_print == 0:
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Batch {batch_index}", f)
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                if not cfg.training_type.ignore_second_action:
                    multi_print(
                        f"Action/Observation/NextAction loss: {action_loss}/{observation_loss}/{next_action_loss}",
                        f,
                    )
                else:
                    multi_print(
                        f"Action/Observation loss: {action_loss}/{observation_loss}", f
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
                    f"Next action: {repr(cfg.causal_lm_tokenizer.decode(next_action[0]))}",
                    f,
                )
                multi_print("______________________________________________________", f)
        action = next_action
        prev_obs = obs
    return aggregate_losses

def train_gumbel(cfg: Config):
    causal_lm = cfg.causal_lm

    if not cfg.load_model:
       with open(cfg.path_2_log, "w") as f:
           print("")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    with torch.no_grad():
        action = torch.cat(
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
        embed_weight = causal_lm.get_input_embeddings().weight
        one_hot_action_prefix = torch.nn.functional.one_hot(
            cfg.action_prefix_tensor, num_classes=cfg.causal_lm_tokenizer.vocab_size
        )
        embedded_action_prefix = one_hot_action_prefix.float() @ embed_weight
        one_hot_action = torch.nn.functional.one_hot(action, num_classes=cfg.causal_lm_tokenizer.vocab_size).float()
        embedded_action = one_hot_action @ embed_weight
        aggregate_losses = []
        prev_obs = None
        with open(cfg.path_2_log, "a") as f:
            f.write("")

    #with torch.autograd.detect_anomaly():
    for batch_index, obs in tqdm(enumerate(cfg.dataloader), total=cfg.num_batches):

        optimizer.zero_grad()

        one_hot_obs = torch.nn.functional.one_hot(obs, num_classes=cfg.causal_lm_tokenizer.vocab_size).float() 
        embedded_obs = one_hot_obs @ embed_weight
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            causal_lm.save_pretrained(cfg.path_2_model)

        embedded_next_action, one_hot_next_action = gumbel_generate(cfg, causal_lm, embedded_action, embedded_obs, embedded_action_prefix, embed_weight)

        input_sequence = (
            [embedded_action, embedded_obs, embedded_next_action]
            if isinstance(cfg.training_type, AOA)
            else [embedded_action, embedded_obs]
        )
        next_action = torch.argmax(one_hot_next_action, dim=-1)
        ids_sequence = (
            [action, obs, next_action]
            if isinstance(cfg.training_type, AOA)
            else [one_hot_action, one_hot_obs]
        )

        ao_logits = causal_lm(torch.cat([action, obs], dim=1)).logits
        loss_tensor = loss_fn(
            input=einops.rearrange(
                torch.cat([ao_logits, one_hot_next_action], dim=1),
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            )[:, :, :-1],
            target=torch.cat(ids_sequence, dim=1)[:, 1:],
        )

        aggregate_loss = loss_tensor.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()

        optimizer.step()
        with torch.no_grad():
            action_loss = loss_tensor[:, : cfg.tok_p_action].mean()
            observation_loss = loss_tensor[
                :, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs
            ].mean()
            if isinstance(cfg.training_type, AOA):
                next_action_loss = loss_tensor[:, cfg.tok_p_obs :].mean()

            if cfg.wandb:
                if isinstance(cfg.training_type, AOA):
                    wandb.log({"Next Action Loss": next_action_loss})
                wandb.log(
                    {
                        "Aggregate Loss": aggregate_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss,
                    }
                )

        # printing
        if batch_index % cfg.interval_print == 0:
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Batch {batch_index}", f)
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                if isinstance(cfg.training_type, AOA):
                    multi_print(
                        f"Action/Observation/NextAction loss: {action_loss}/{observation_loss}/{next_action_loss}",
                        f,
                    )
                else:
                    multi_print(
                        f"Action/Observation loss: {action_loss}/{observation_loss}", f
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
                    f"Next action: {repr(cfg.causal_lm_tokenizer.decode(next_action[0]))}",
                    f,
                )
                multi_print("______________________________________________________", f)
        action = next_action.clone()
        prev_obs = obs.clone()

    return aggregate_losses

def gumbel_generate(cfg: Config, causal_lm, embedded_action, embedded_obs, embedded_action_prefix, embed_weight):
    embedded_inputs = torch.cat([embedded_action, embedded_obs, embedded_action_prefix], dim=1)
    one_hot_next_action = torch.zeros((cfg.batch_size, cfg.tok_p_pure_action, cfg.causal_lm_tokenizer.vocab_size), device=cfg.device)
    for i in range(cfg.tok_p_pure_action):
        logits = causal_lm(inputs_embeds=embedded_inputs).logits[:, -1, :] 
        next_token = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True, dim=-1)
        one_hot_next_action[:, i, :] = next_token 
        embedded_inputs = torch.cat([embedded_inputs, (next_token @ embed_weight).unsqueeze(1)], dim=1)
    embedded_next_action = embedded_inputs[:, -cfg.tok_p_action :]
    return embedded_next_action,  one_hot_next_action # action  embedded, and action logits

