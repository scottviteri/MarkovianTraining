import torch
from tqdm import tqdm
import wandb
import einops
from typing import List

from src.training_types import *
from src.utilities import log_and_print_info

def train_rao(cfg : Config):

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    average_losses = []
    aggregate_losses = []
    optimistic_loss = 0.5
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    prev_obs = torch.cat(
        (
            cfg.obs_prefix_tensor,
            torch.full(
                (cfg.batch_size, cfg.tok_p_pure_obs),
                fill_value=cfg.causal_lm_tokenizer.pad_token_id,
                dtype=torch.int64,
                device=cfg.device,
            ),
        ),
        dim=1,
    )

    for batch_index, input_ids in (
        tqdm(enumerate(cfg.dataloader), total=cfg.num_batches)
        if cfg.num_batches
        else tqdm(cfg.dataloader)
    ):
        if cfg.num_batches and batch_index > cfg.num_batches:
            break
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            cfg.causal_lm.save_pretrained(cfg.path_2_model)

        with torch.no_grad():
            rao_tensor_optimistic_triples, new_losses, prev_obs = gen_rao_tensor(
                cfg=cfg,
                input_ids=input_ids,
                loss_fn=loss_fn,
                aggregate_losses=aggregate_losses,
                optimistic_loss=optimistic_loss,
                prev_obs=prev_obs,
                batch_index=batch_index,
            )

        if cfg.training_type.use_rewards_to_go:
            new_losses = compute_cumulative_averages(new_losses)  # (batch, num_rao)

        rao_tensor_triples = []
        for i, (_, action, observation) in enumerate(rao_tensor_optimistic_triples):
            loss_tokens_tensor = create_loss_tokens_tensor(
                new_losses[:, i],
                cfg.causal_lm_tokenizer,
                cfg.device,
                cfg.training_type.tok_p_pure_loss,
            )
            loss_tokens_tensor = torch.cat(
                (cfg.training_type.loss_prefix_tensor, loss_tokens_tensor), dim=-1
            )
            rao_tensor_triples.append((loss_tokens_tensor, action, observation))

        optimizer.zero_grad()
        # if num_rao = 0, I want to be slicing by 1
        default_tensor = torch.tensor(
            [[] for _ in range(cfg.batch_size)], dtype=torch.int64, device=cfg.device
        )
        for i in range(0, len(rao_tensor_triples), cfg.training_type.num_rao + 1):
            rao_tensor_slice = condense_triples(
                rao_tensor_triples[i : i + cfg.training_type.num_rao + 1], default_tensor
            )
            rao_tensor_logits = cfg.causal_lm(rao_tensor_slice).logits[:, :-1, :]
            rao_tensor_loss = loss_fn(
                input=einops.rearrange(
                    rao_tensor_logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=rao_tensor_slice[:, 1:],
            )
            if cfg.training_type.use_loss_difference:
                obs_slice = torch.cat(
                    [obs for _, _, obs in rao_tensor_triples[i : i + cfg.training_type.num_rao + 1]],
                    dim=1,
                )

            # Calculate the relative weights for each loss component
            with torch.no_grad():
                sections = rao_tensor_loss.split(cfg.training_type.tok_p_rao, dim=-1)
                loss_triples = [
                    (
                        section[:, : cfg.training_type.tok_p_loss],
                        section[:, cfg.training_type.tok_p_loss : cfg.training_type.tok_p_loss + cfg.tok_p_action],
                        section[:, cfg.training_type.tok_p_loss + cfg.tok_p_action :],
                    )
                    for section in sections
                ]
                loss_loss, action_loss, observation_loss = zip(*loss_triples)
                loss_loss = torch.cat(loss_loss, dim=-1).mean()
                action_loss = torch.cat(action_loss, dim=-1).mean()
                observation_loss = torch.cat(observation_loss, dim=-1).mean()
            # Compute the mean of rao_tensor_loss and backward pass as usual
            aggregate_loss = rao_tensor_loss.mean()
            aggregate_losses.append(aggregate_loss.item())
            optimistic_loss = torch.tensor(aggregate_losses).mean().item() - \
                torch.tensor(aggregate_losses).std().item()
            aggregate_loss.backward()
            # print("Aggregate loss: ", aggregate_loss)
            optimizer.step()

            if cfg.wandb:
                wandb.log(
                    {
                        "Aggregate Loss": aggregate_loss,
                        "Loss Loss": loss_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss,
                    }
                )
    return aggregate_losses


def gen_rao_tensor(
    cfg : Config, input_ids : torch.Tensor, loss_fn, aggregate_losses : List[float], optimistic_loss: float, prev_obs, batch_index
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
            input=einops.rearrange(
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
                    input=einops.rearrange(
                        predicted_logits,
                        "batch seq_length vocab_size -> batch vocab_size seq_length",
                    ),
                    target=true_obs[:, 1:],
                )
                batch_loss = batch_loss - out.mean(dim=-1)

        optimistic_loss_tokens_tensor = cfg.causal_lm_tokenizer.batch_encode_plus(
            [
                str(round((optimistic_loss * (1.0 + torch.randn(1) * 0.05)).item(), 3))
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

def create_loss_tokens_tensor(
    batch_loss: torch.Tensor, tokenizer, device: torch.device, tokens_per_pure_reward
):
    string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
    loss_tokens_tensor: TensorType["batch", "seq_length"] = tokenizer.batch_encode_plus(
        string_losses,
        return_tensors="pt",
        padding="max_length",
        truncation="longest_first",
        max_length=tokens_per_pure_reward,
    ).input_ids.to(device)
    assert tokenizer.eos_token_id not in loss_tokens_tensor
    assert loss_tokens_tensor.shape[-1] == tokens_per_pure_reward
    return loss_tokens_tensor

def condense_triples(rao_tensor_triples, default_tensor):
    if not rao_tensor_triples:
        return default_tensor
    return torch.cat(
        [torch.cat(triple, dim=-1) for triple in rao_tensor_triples], dim=-1
    )

def compute_cumulative_averages(losses: torch.Tensor) -> torch.Tensor:
    # Flip the tensor to compute right-sided cumulative averages
    losses = torch.flip(losses, dims=[1])
    cumulative_averages = torch.cumsum(losses, dim=1) / torch.arange(
        1, losses.shape[1] + 1, device=losses.device
    )
    # Flip the tensor back
    cumulative_averages = torch.flip(cumulative_averages, dims=[1])
    return cumulative_averages
