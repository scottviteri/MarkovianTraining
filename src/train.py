# pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
import torch
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import einops
import wandb
import json
import random
from datasets import load_dataset
from openai import OpenAI
from matplotlib import pyplot as plt

from src.training_types import *
from src.utilities import extend_initial_config, log_and_print_info
from src.utilities import create_run_name, multi_print

from src.evaluate_via_gpt import evaluate_via_gpt
import src.config_examples


def run_training(cfg: Config):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = cfg.optimizer(cfg.causal_lm.parameters(), lr=cfg.lr)

    def save_weights(batch_index):
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            cfg.causal_lm.save_pretrained(cfg.path_2_model)

    def default_action():
        initial_helpful_msg = cfg.causal_lm_tokenizer("Use StepByStep spaces to help predict your next observation.",
                                              return_tensors="pt")["input_ids"].repeat(cfg.batch_size, 1).to(cfg.device)
        assert initial_helpful_msg.shape[-1] < cfg.tok_p_pure_obs
        prev_action = torch.cat(
            (
                cfg.action_prefix_tensor,
                initial_helpful_msg, 
                torch.full(
                    (cfg.batch_size, cfg.tok_p_pure_action - initial_helpful_msg.shape[-1]),
                    fill_value=cfg.causal_lm_tokenizer.pad_token_id,
                    dtype=torch.int64,
                    device=cfg.device,
                ),
            ),
            dim=1,
        )
        return prev_action

    def log_wandb(batch_index, aggregate_loss, losses):
        prev_action_loss, prev_observation_loss, action_loss, observation_loss = losses
        if cfg.wandb:
            wandb.log(
                {
                    "Batch Index": batch_index,
                    "Aggregate Loss": aggregate_loss,
                    "Previous Action Loss": prev_action_loss,
                    "Previous Observation Loss": prev_observation_loss,
                    "Action Loss": action_loss,
                    "Observation Loss": observation_loss
                }
            )

    def log_print_losses(batch_index, aggregate_loss, losses):
        prev_action_loss, prev_observation_loss, action_loss, obs_loss = losses
        if batch_index % cfg.interval_print == 0:
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                multi_print(
                    f"PrevAction/PrevObservation/Action/Obs loss: {prev_action_loss}/{prev_observation_loss}/{action_loss}/{obs_loss}",
                    f,
                )
 
    def log_print_oa(batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first):
        if batch_index % cfg.interval_print == 0:
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Batch Index: {batch_index}", f)
                multi_print(f"Is First: {is_first}", f)
                multi_print(
                    f"Prev Action: {repr(cfg.causal_lm_tokenizer.decode(prev_action[0]))}", f
                )
                multi_print(
                    f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}",
                    f,
                )
                if not is_first:
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
                    f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f
                )
                multi_print("______________________________________________________", f)

    def sample(sampling_cfg, prev_action, prev_obs, observation):
        # currently not using filter_best_action parameters
        cfg.causal_lm.eval()
        with torch.no_grad():
            action_candidates = cfg.causal_lm.generate(
                    inputs=torch.cat([prev_action, prev_obs, cfg.action_prefix_tensor], dim=1),
                    output_scores=True,
                    do_sample=True,
                    temperature=1.0,
                    min_new_tokens=cfg.tok_p_pure_action,
                    max_new_tokens=cfg.tok_p_pure_action,
                    pad_token_id=cfg.causal_lm_tokenizer.eos_token_id,
                )[:, -cfg.tok_p_action :]
            return action_candidates
    
    def update_weights(training_cfg, prev_action, prev_obs, action, obs):
        cfg.causal_lm.train()
        with autocast():
            if training_cfg.train_O_given_prev_O:
                input_sequence = torch.cat([prev_obs, obs], dim=1)
                logits = cfg.causal_lm(input_sequence).logits[:, :-1, :]
                loss_tensor = loss_fn(
                    input=einops.rearrange(
                        logits,
                        "batch seq_length vocab_size -> batch vocab_size seq_length",
                    ),
                    target=input_sequence[:, 1:]
                )
                aggregate_loss = loss_tensor[:,-cfg.tok_p_obs:].mean()
            else:
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

                mkv_input_sequence = torch.cat([action, obs], dim=1)
                mkv_logits = cfg.causal_lm(mkv_input_sequence).logits[:, :-1, :]
                mkv_loss_tensor = loss_fn(
                    input=einops.rearrange(
                        mkv_logits,
                        "batch seq_length vocab_size -> batch vocab_size seq_length",
                    ),
                    target=mkv_input_sequence[:, 1:]
                )
                obs_tensor = mkv_loss_tensor[:,-cfg.tok_p_obs:]
                obs_loss = obs_tensor.mean()

                aggregate_loss = sum(map(lambda x: x[1] if x[0] else 0.0, 
                                         zip([training_cfg.train_A_given_AO, training_cfg.train_O_given_A],
                                                [action_loss, obs_loss])))

            if not isinstance(cfg.debug, NoWeightUpdates):
                aggregate_loss.backward()
                optimizer.step()
        
            if training_cfg.train_O_given_prev_O: return aggregate_loss.item(), None, None
            loss_tensors = prev_action_tensor, prev_observation_tensor, action_tensor, obs_tensor
            losses = prev_action_loss, prev_observation_loss, action_loss, obs_loss
            return aggregate_loss.item(), loss_tensors, losses

    def log_and_save(batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first, aggregate_loss, losses):
        save_weights(batch_index)
        log_print_oa(batch_index, prev_action, prev_obs, action, obs, is_guidance_action, is_first)
        if cfg.training_cfg.train_O_given_prev_O: 
            if cfg.wandb: wandb.log({"Batch Index": batch_index, "Previous Observation Loss": aggregate_loss})
        else:
            log_wandb(batch_index, aggregate_loss, losses)
            log_print_losses(batch_index, aggregate_loss, losses)

    def trainer():
        state = [default_action(), 0, None]

        def update(datapt_pair):
            nonlocal state
            prev_datapt, datapt = datapt_pair
            is_first = "First" in datapt and datapt["First"]
            prev_action, batch_index, _ = state
            prev_obs, obs = prev_datapt["Observation"], datapt["Observation"]
            if is_first: 
                prev_action = default_action()
                log_print_oa(batch_index, prev_action, prev_obs, None, obs, "Action" in datapt, is_first)
                state = [prev_action, batch_index + 1, None]
                return
            # now can assume that prev_datapt contains the question and datapt contains the Answer
            if "Action" in datapt: 
                action = datapt["Action"]
            elif cfg.training_cfg.train_O_given_prev_O: 
                action = prev_action
            else:
                action = sample(cfg.sampling_cfg, prev_action, prev_obs, obs) 
            aggregate_loss, loss_tensors, losses = update_weights(cfg.training_cfg, prev_action, prev_obs, action, obs)
            log_and_save(batch_index, prev_action, prev_obs, action, obs, "Action" in datapt, is_first, aggregate_loss, losses)
            state = [action, batch_index + 1, aggregate_loss]
            return

        def pi(): 
            nonlocal state
            return state[-1]
        
        return update, pi

    def train_via_update():
        aggregate_losses = []
        update, pi = trainer()
        for datapt_pair in tqdm(cfg.dataset.dataloader, total=cfg.num_batches):
            aggregate_loss = pi()
            if aggregate_loss is not None: aggregate_losses.append(aggregate_loss)
            update(datapt_pair)
        return aggregate_losses

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    with open(cfg.path_2_log, "a") as f:
        f.write("")

    aggregate_losses = train_via_update()
    return aggregate_losses
    

if __name__ == "__main__":
    for initial_cfg in src.config_examples.example_configs:
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cfg = extend_initial_config(initial_cfg)
        if cfg.wandb:
            #sweep_id = wandb.sweep(
            #    sweep_config, project="collaborative-training-many-per-context-window"
            #)
            run = wandb.init(project="collaborative-training-many-per-context-window")
            run.name = create_run_name(cfg)
            #wandb.agent(sweep_id, function=train_specific_type)
            aggregate_losses = run_training(cfg)
            run.finish()
        else:
            aggregate_losses = run_training(cfg)
            plt.figure()
            plt.plot(aggregate_losses)
            plt.show()

