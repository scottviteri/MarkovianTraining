import torch
from tqdm import tqdm
import wandb
import einops
from datasets import load_dataset

from src.types_and_utilities import InitialConfig, InitTrainingType, Config
from src.types_and_utilities import AR, GptEval, AO, AOA, RAOInit
from src.types_and_utilities import log_and_print_info, multi_print
from src.types_and_utilities import get_pure_obs, prepend_obs_tensor, take


def train_ao_or_aoa(cfg: Config):

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    training_ctxt_size = (
        cfg.training_ctxt_size if cfg.training_ctxt_size else cfg.ctxt_size
    )

    tok_p_action = int(training_ctxt_size / (2 + cfg.obs_to_action_ratio))
    tok_p_obs = int(cfg.obs_to_action_ratio * tok_p_action)
    tok_p_pure_action = tok_p_action - cfg.action_prefix_tensor.shape[1]
    tok_p_pure_obs = tok_p_obs - cfg.obs_prefix_tensor.shape[1]
    assert tok_p_pure_action > 0 and tok_p_pure_obs > 0

    itr_ds = load_dataset(
        cfg.dataset_name, cfg.task_name, split="train", streaming=True
    )
    ds_tokenized = map(
        lambda x: cfg.causal_lm_tokenizer(x["text"], return_tensors="pt")[
            "input_ids"
        ].to(cfg.device),
        itr_ds,
    )
    pure_obs = get_pure_obs(cfg.batch_size, tok_p_pure_obs, cfg.device, ds_tokenized)
    obs_ds = take(cfg.num_batches, prepend_obs_tensor(cfg.obs_prefix_tensor, pure_obs))
    action = torch.cat(
        (
            cfg.action_prefix_tensor,
            torch.full(
                (cfg.batch_size, tok_p_pure_action),
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

    for batch_index, obs in tqdm(enumerate(obs_ds), total=cfg.num_batches):
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            cfg.causal_lm.save_pretrained(cfg.path_2_model)

        with torch.no_grad():
            next_action = cfg.causal_lm.generate(
                inputs=torch.cat([action, obs, cfg.action_prefix_tensor], dim=1),
                output_scores=True,
                do_sample=True,
                min_new_tokens=tok_p_pure_action,
                max_new_tokens=tok_p_pure_action,
                pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
            )[:, -cfg.tok_p_action :]

        optimizer.zero_grad()
        input_sequence = (
            [action, obs, next_action]
            if isinstance(cfg.training_type, AOA)
            else [action, obs]
        )
        rao_tensor_logits = cfg.causal_lm(torch.cat(input_sequence, dim=1)).logits[
            :, :-1, :
        ]
        rao_tensor_loss = loss_fn(
            input=einops.rearrange(
                rao_tensor_logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=torch.cat(input_sequence, dim=1)[:, 1:],
        )

        aggregate_loss = rao_tensor_loss.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()
        optimizer.step()

        with torch.no_grad():
            action_loss = rao_tensor_loss[:, : cfg.tok_p_action].mean()
            observation_loss = rao_tensor_loss[
                :, cfg.tok_p_action : cfg.tok_p_action + cfg.tok_p_obs
            ].mean()
            if isinstance(cfg.training_type, AOA):
                next_action_loss = rao_tensor_loss[:, cfg.tok_p_obs :].mean()

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
        action = next_action
        prev_obs = obs
    return aggregate_losses
