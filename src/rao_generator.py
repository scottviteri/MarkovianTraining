import einops
import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value, Array3D
from einops import repeat, rearrange
from itertools import islice
from functools import reduce
import numpy as np
from src.types_and_utilities import (
    Config,
    log_and_print_info,
    condense_triples,
    create_loss_tokens_tensor,
)

def inject_noise(loss: torch.Tensor) -> torch.Tensor:
    return loss * (1.0 + torch.randn(1) * 0.05)

def gen_rao_tensor(
    cfg:Config, input_ids, loss_fn, aggregate_losses, optimistic_loss: float, prev_obs, batch_index
):
    rao_tensor_triples = []
    default_tensor = torch.tensor(
        [[] for _ in range(cfg.batch_size)],
        dtype=torch.int64,
        device=cfg.device,
    )
    losses = torch.tensor(
        [[] for _ in range(cfg.batch_size)],
        dtype=torch.float32,
        device=cfg.device,
    )
    optimistic_loss_tokens_tensor = create_loss_tokens_tensor(
        torch.tensor(
            [[optimistic_loss] for _ in range(cfg.batch_size)],
            dtype=torch.float32,
            device=cfg.device
        ),
        cfg.causal_lm_tokenizer,
        cfg.device,
        cfg.training_type.tok_p_pure_loss,
    )

    for observation_index in range(cfg.training_type.obs_between_weight_updates):
        incentive_rao = torch.cat(
            (
                condense_triples(
                    rao_tensor_triples[-cfg.training_type.num_rao :], default_tensor
                ),
                optimistic_loss_tokens_tensor,
                cfg.action_prefix_tensor,
            ),
            dim=-1,
        )

        # RAOR_A
        # argmax in Action (P_theta_t (helpful_msg_{t+1} | lhes ++ optimistic_loss_{t+1})))
        if cfg.training_type.use_multirao_for_action_gen:
            action_input = incentive_rao[
                :,
                -(cfg.ctxt_size - cfg.tok_p_pure_action) :,
            ]
        else:
            action_input = incentive_rao[
                :,
                -(
                    cfg.training_type.tok_p_rao * (cfg.training_type.num_rao + 1)
                    + cfg.training_type.tok_p_loss
                    + cfg.action_prefix_tensor.shape[-1]
                ) :,
            ]
        with torch.no_grad():
            full_action = cfg.causal_lm.generate(
                inputs=action_input,
                output_scores=True,
                do_sample=True,
                min_new_tokens=cfg.tok_p_pure_action,
                max_new_tokens=cfg.tok_p_pure_action,
                pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
            )
        action: TensorType["batch", "seq_length"] = full_action[
            :, -cfg.tok_p_action :
        ]

        true_obs: TensorType["batch", "seq_length"] = input_ids[
            :, observation_index, :
        ].to(cfg.device)

        # Calculate loss for the actual observation, using only the loss and action as context
        # actual_loss_t = log P_theta (external_text_t | lhes ++ optimistic_loss_t ++ helpful_msg_t)
        # target num_rao = 0
        context = torch.cat(
            (
                condense_triples(
                    rao_tensor_triples[-cfg.training_type.num_rao:], default_tensor
                )
                if cfg.training_type.num_rao > 0
                else torch.tensor(
                    [[] for _ in range(cfg.batch_size)],
                    dtype=torch.int32,
                    device=cfg.device,
                ),
                optimistic_loss_tokens_tensor,
                action,
                true_obs,
            ),
            dim=-1,
        )
        prediction = cfg.causal_lm(context)
        predicted_logits = prediction.logits[:, -cfg.tok_p_obs - 1 : -1, :]
        predicted_obs = predicted_logits.argmax(dim=-1)
        out = loss_fn(
            input=rearrange(
                predicted_logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=true_obs,
        )
        batch_loss = out.mean(dim=-1, keepdim=True)

        # Calculate loss for the filler action
        if cfg.training_type.use_loss_difference:
            with torch.no_grad():
                prediction = cfg.causal_lm(true_obs)
                predicted_logits = prediction.logits[:, :-1, :]
                predicted_obs = predicted_logits.argmax(dim=-1)
                out = loss_fn(
                    input=rearrange(
                        predicted_logits,
                        "batch seq_length vocab_size -> batch vocab_size seq_length",
                    ),
                    target=true_obs[:, 1:],
                )
                batch_loss = batch_loss - out.mean(dim=-1)

        optimistic_loss_tokens_tensor = cfg.causal_lm_tokenizer.batch_encode_plus(
            [
                str(round(inject_noise(optimistic_loss).item(), 3))
                for _ in range(cfg.batch_size)
            ],
            return_tensors="pt",
            truncation="longest_first",
            padding="max_length",
            max_length=cfg.training_type.tok_p_pure_loss,
        ).input_ids.to(cfg.device)
        assert (
            optimistic_loss_tokens_tensor.shape[-1] == cfg.training_type.tok_p_pure_loss
        )
        optimistic_loss_tokens_tensor = torch.cat(
            (cfg.training_type.loss_prefix_tensor, optimistic_loss_tokens_tensor), dim=-1
        )

        losses = torch.cat((losses, batch_loss), dim=-1)
        rao_tensor_triples.append((optimistic_loss_tokens_tensor, action, true_obs))

        log_and_print_info(
            cfg,
            batch_index,
            observation_index,
            batch_loss,
            aggregate_losses,
            prev_obs,
            action,
            predicted_obs,
            true_obs,
        )
        prev_obs = true_obs

    return rao_tensor_triples, losses, true_obs 

