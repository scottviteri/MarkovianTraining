# pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login
"""  
Main Definitions:

1. sweep_config class instance
2. train function

High level structure:

1. Import necessary libraries and modules:  This includes libraries like torch, tqdm, einops, wandb, and your custom modules RaoConfig and RaoGenerator.

2. Define sweep configuration: This is a dictionary that contains the parameters for your training process, such as learning rate, batch size, number of batches, and model parameters. If you're using Weights & Biases (wandb), this configuration is used to set up a hyperparameter sweep.

3. Initialize wandb sweep: If you're using wandb, this line sets up a hyperparameter sweep with the given sweep configuration and project name.

4. Define the train function: This is the main function where the training process happens. It includes the following steps:

- Initialize wandb run: If you're using wandb, this step initializes a new run and sets up the configuration parameters for the run.

- Create RaoConfig instance: This step creates an instance of the RaoConfig class with the parameters from the sweep configuration.

- Create RaoGenerator instance: This step creates an instance of the RaoGenerator class, which is responsible for generating the reward-action-observation (RAO) triples.

- Initialize dataloader, loss function, and optimizer: This step initializes the dataloader for your dataset, the loss function for your model, and the optimizer for your training process.

- Training loop: This is the main loop where the training happens. For each batch in the dataloader, it generates a RAO tensor, calculates the loss, and updates the model parameters. If you're using wandb, it also logs the loss values to wandb.

5. Finish wandb run: If you're using wandb, this step finishes the current run after the training process is complete.

6. Start training process: This is the entry point of your program. If you're using wandb, it starts a wandb agent to run the train function. Otherwise, it directly calls the train function.
"""

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import wandb
from src.rao_tools import RaoConfig, condense_triples, multi_print
from src.rao_tools import compute_cumulative_averages, create_loss_tokens_tensor
from src.rao_generator import RaoGenerator, get_pure_obs, take, prepend_obs_tensor
import json
from datasets import load_dataset

with open("sweep_config.json") as f:
    sweep_config = json.load(f)

sweep_id = wandb.sweep(
    sweep_config, project="collaborative-training-many-per-context-window"
)

def train_alternate(raogen, cfg):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.tokenizer
    optimizer = torch.optim.SGD(causal_lm.parameters(), lr=cfg.lr)

    training_ctxt_size = cfg.training_ctxt_size if cfg.training_ctxt_size else cfg.ctxt_size
    tok_p_action = int(training_ctxt_size / (2 + cfg.obs_to_action_ratio))
    tok_p_obs = int(cfg.obs_to_action_ratio * tok_p_action)
    tok_p_pure_action = tok_p_action - raogen._action_prefix_tensor.shape[1]
    tok_p_pure_obs = tok_p_obs - raogen._observation_prefix_tensor.shape[1]
    assert tok_p_pure_action > 0 and tok_p_pure_obs > 0

    itr_ds = load_dataset(cfg.dataset_name, cfg.task_name, split="train", streaming=True)
    ds_tokenized = map(
            lambda x: cfg.tokenizer(x["text"], return_tensors="pt")["input_ids"].to(cfg.device), 
            itr_ds)
    pure_obs = get_pure_obs(cfg.batch_size, tok_p_pure_obs, cfg.device, ds_tokenized)
    obs_ds = take(cfg.num_batches, prepend_obs_tensor(raogen._observation_prefix_tensor, pure_obs))
    action = torch.cat((raogen._action_prefix_tensor, 
        torch.full((cfg.batch_size, tok_p_pure_action), fill_value=cfg.tokenizer.pad_token_id, 
            dtype=torch.int64, device=cfg.device)), dim=1)
    aggregate_losses = []
    prev_obs = None
    with open(f"saved_weights_and_losses/{cfg.model_name}_log.txt", "a") as f:
        f.write("")

    for batch_index, obs in (
        tqdm(enumerate(obs_ds), total=cfg.num_batches)
    ):
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            causal_lm.save_pretrained(cfg.path_2_model)

        with torch.no_grad():
            next_action = causal_lm.generate(
                inputs=torch.cat([action, obs, raogen._action_prefix_tensor], dim=1),
                output_scores=True,
                do_sample=True,
                num_beams=cfg.num_beams,
                min_new_tokens=tok_p_pure_action,
                max_new_tokens=tok_p_pure_action,
                pad_token_id=causal_lm_tokenizer.eos_token_id,
            )[:, -cfg.tok_p_action:]

        optimizer.zero_grad()
        input_sequence = [action, obs, next_action] if cfg.alternate_training == 1 else [action, obs]
        rao_tensor_logits = causal_lm(torch.cat(input_sequence, dim=1)).logits[:,:-1,:]
        rao_tensor_loss = loss_fn(
            input=rearrange(
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
            action_loss = rao_tensor_loss[:, :cfg.tok_p_action].mean()
            observation_loss = rao_tensor_loss[:, cfg.tok_p_action:cfg.tok_p_action+cfg.tok_p_obs].mean()
            if cfg.alternate_training == 1:
                next_action_loss = rao_tensor_loss[:, cfg.tok_p_obs:].mean()
 
            if cfg.wandb:
               wandb.log(
                    {
                        "Aggregate Loss": aggregate_loss,
                        "Action Loss": action_loss,
                        "Observation loss": observation_loss / observation_weight,
                    }
                )
                if cfg.alternate_training == 1: 
                    wandb.log({"Next Action Loss": next_action_loss})

        #printing
        if batch_index % cfg.interval_print == 0:
            with open(f"saved_weights_and_losses/{cfg.model_name}_log.txt", "a") as f:
                multi_print(f"Batch {batch_index}", f)
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                if cfg.alternate_training == 1:
                    multi_print(f"Action/Observation/NextAction loss: {action_loss}/{observation_loss}/{next_action_loss}", f)
                else:
                    multi_print(f"Action/Observation loss: {action_loss}/{observation_loss}", f)
                if prev_obs is not None: 
                    multi_print(f"Prev Observation: {repr(causal_lm_tokenizer.decode(prev_obs[0]))}", f)
                multi_print(f"Action: {repr(causal_lm_tokenizer.decode(action[0]))}", f)
                multi_print(f"Observation: {repr(causal_lm_tokenizer.decode(obs[0]))}", f)
                multi_print(f"Next action: {repr(causal_lm_tokenizer.decode(next_action[0]))}", f)
                multi_print("______________________________________________________", f)

        action = next_action
        prev_obs = obs
    
    return 1

def train_regular(raogen, cfg):
    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.tokenizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(causal_lm.parameters(), lr=cfg.lr)
    itr_ds = load_dataset(cfg.dataset_name, cfg.task_name, split="train", streaming=True)
    ds_tokenized = map(
            lambda x: causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(cfg.device), 
            itr_ds)
    pure_obs = get_pure_obs(cfg.batch_size, 
        raogen._tokens_per_pure_observation, cfg.device, ds_tokenized)
    obs_ds = take(cfg.num_batches, pure_obs)
    # Initialize the list to store the losses
    losses = []
    for batch_index, obs in (tqdm(enumerate(obs_ds), total=cfg.num_batches)):
        optimizer.zero_grad()
        logits = causal_lm(obs).logits[:,:-1,:]
        loss = loss_fn(
            input=rearrange(
                logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=obs[:, 1:],
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if cfg.wandb: wandb.log({"Observation Loss": loss.item()})
        if batch_index % cfg.interval_print == 0:
            print(f"Batch {batch_index}, Loss: {loss.item()}")
    return losses

def train():
    run = None
    if sweep_config["parameters"]["wandb"]["values"][0]:
        run = wandb.init(resume=sweep_config["parameters"]["load_model"]["values"][0])
        wb_cfg = run.config
        config_params = {
            param: getattr(wb_cfg, param) for param in sweep_config["parameters"]
        }
    else:
        wb_cfg = None
        config_params = {
            param: sweep_config["parameters"][param]["values"][0]
            for param in sweep_config["parameters"]
        }

    cfg = RaoConfig(
        model_name=config_params.get("model_name"),
        lr=config_params.get("lr"),
        num_rao=config_params.get("num_rao"),
        batch_size=config_params.get("batch_size"),
        num_batches=config_params.get("num_batches"),
        obs_between_weight_updates=config_params.get("obs_between_weight_updates"),
        obs_to_action_ratio=config_params.get("obs_to_action_ratio"),
        interval_save_weights=config_params.get("interval_save_weights"),
        interval_print=config_params.get("interval_print"),
        wandb=config_params.get("wandb"),
        load_model=config_params.get("load_model"),
        do_lora=config_params.get("do_lora"),
        use_loss_difference=config_params.get("use_loss_difference"),
        training_ctxt_size=config_params.get("training_ctxt_size"),
        use_multirao_for_action_gen=config_params.get("use_multirao_for_action_gen"),
        use_rewards_to_go=config_params.get("use_rewards_to_go"),
        dataset_name=config_params.get("dataset_name"),
        alternate_training = config_params.get("alternate_training"),
        regular_training = config_params.get("regular_training")
    )
    if run is not None:
        run_name = ""
        if cfg.alternate_training: run_name += f"ALT{cfg.alternate_training}_"
        run_name += f"{cfg.model_name[:4]}_"
        if cfg.lr != 1e-4: run_name += f"lr{cfg.lr}_"
        if cfg.num_rao != 0:
            run_name += f"nr{cfg.num_rao}_"
        if cfg.batch_size != 1:
            run_name += f"bs{cfg.batch_size}_"
        run_name += f"nb{cfg.num_batches}_"
        if cfg.obs_to_action_ratio != 1:
            run_name += f"o:a={cfg.obs_to_action_ratio}:1_"
        if cfg.load_model: run_name += f"lm_"
        if cfg.do_lora: run_name += "lora_"
        run_name += f"rao{cfg.tok_p_loss}/{cfg.tok_p_action}/{cfg.tok_p_obs}_"
        if cfg.training_ctxt_size:
            run_name += f"ics{cfg.training_ctxt_size}_"
        if not cfg.alternate_training:
            run_name += f"obwu{cfg.obs_between_weight_updates}_"
            if cfg.use_loss_difference: run_name += "ld_"
            if cfg.use_multirao_for_action_gen:
                run_name += f"mr{cfg.use_multirao_for_action_gen}_"
            if cfg.use_rewards_to_go: run_name += "rtg_"
        if cfg.regular_training:
            run.name = f"{cfg.model_name[:4]}_regular_{cfg.training_ctxt_size}"
        else:
            run.name = run_name

    if not cfg.load_model:
        with open(f"saved_weights_and_losses/{cfg.model_name}", "w") as f:
            print("")

    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.tokenizer
    print(cfg.model_name)

    raogen = RaoGenerator(cfg=cfg)
    dataloader = raogen.dataset

    if cfg.regular_training:
        return train_regular(raogen, cfg)
    if cfg.alternate_training:
        return train_alternate(raogen, cfg)
        
    average_losses = []
    aggregate_losses = []
    optimistic_loss = 0.5
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(causal_lm.parameters(), lr=cfg.lr)
    prev_obs = torch.cat((raogen._observation_prefix_tensor, 
        torch.full((cfg.batch_size, raogen._tokens_per_pure_observation), fill_value=cfg.tokenizer.pad_token_id, 
            dtype=torch.int64, device=cfg.device)), dim=1)

    for batch_index, input_ids in (
        tqdm(enumerate(dataloader), total=cfg.num_batches)
        if cfg.num_batches
        else tqdm(dataloader)
    ):
        if cfg.num_batches and batch_index > cfg.num_batches:
            break
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name} \n\n")
            causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            causal_lm.save_pretrained(cfg.path_2_model)

        with torch.no_grad():
            rao_tensor_optimistic_triples, new_losses, prev_obs = raogen.gen_rao_tensor(
                input_ids=input_ids,
                loss_fn=loss_fn,
                aggregate_losses=aggregate_losses,
                optimistic_loss=optimistic_loss,
                prev_obs=prev_obs,
                batch_index=batch_index,
            )

        if cfg.use_rewards_to_go:
            new_losses = compute_cumulative_averages(new_losses)  # (batch, num_rao)

        rao_tensor_triples = []
        for i, (_, action, observation) in enumerate(rao_tensor_optimistic_triples):
            loss_tokens_tensor = create_loss_tokens_tensor(
                new_losses[:, i],
                causal_lm_tokenizer,
                cfg.device,
                raogen.tokens_per_pure_reward,
            )
            loss_tokens_tensor = torch.cat(
                (raogen._reward_prefix_tensor, loss_tokens_tensor), dim=-1
            )
            rao_tensor_triples.append((loss_tokens_tensor, action, observation))

        optimizer.zero_grad()
        # if num_rao = 0, I want to be slicing by 1
        default_tensor = torch.tensor(
            [[] for _ in range(cfg.batch_size)], dtype=torch.int64, device=cfg.device
        )
        for i in range(0, len(rao_tensor_triples), cfg.num_rao + 1):
            rao_tensor_slice = condense_triples(
                rao_tensor_triples[i : i + cfg.num_rao + 1], default_tensor
            )
            rao_tensor_logits = causal_lm(rao_tensor_slice).logits[:, :-1, :]
            rao_tensor_loss = loss_fn(
                input=rearrange(
                    rao_tensor_logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=rao_tensor_slice[:, 1:],
            )
            if cfg.use_loss_difference:
                obs_slice = torch.cat([obs for _, _, obs in rao_tensor_triples[i : i+ cfg.num_rao + 1]], dim=1)

            # Calculate the relative weights for each loss component
            with torch.no_grad():
                sections = rao_tensor_loss.split(cfg.tok_p_rao, dim=-1)
                loss_triples = [
                    (
                        section[:, : cfg.tok_p_loss],
                        section[:, cfg.tok_p_loss : cfg.tok_p_loss + cfg.tok_p_action],
                        section[:, cfg.tok_p_loss + cfg.tok_p_action :],
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
            optimistic_loss = np.mean(aggregate_losses) - np.std(aggregate_losses)
            aggregate_loss.backward()
            #print("Aggregate loss: ", aggregate_loss)
            optimizer.step()

            if wb_cfg:
                wandb.log(
                    {
                        "Aggregate Loss": aggregate_loss,
                        "Loss Loss": loss_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss,
                    }
                )

    if wb_cfg:
        run.finish()


if sweep_config["parameters"]["wandb"]["values"][0]:
    wandb.agent(sweep_id, function=train)
else:
    train()

