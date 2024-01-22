# pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import wandb
import json
from datasets import load_dataset
from openai import OpenAI
from matplotlib import pyplot as plt

from src.rao_tools import AR, GptEval, AO, AOA, RAOInit, RAO, InitTrainingType, TrainingType
from src.rao_tools import InitialConfig, Config, condense_triples, multi_print
from src.rao_tools import compute_cumulative_averages, create_loss_tokens_tensor
from src.rao_tools import extend_initial_config
from src.rao_tools import get_pure_obs, take, prepend_obs_tensor
from src.rao_generator import gen_rao_tensor


def train():
    with open("../sweep_config.json") as f:
        sweep_config = json.load(f)
    cfg_dict = {k:v["values"][0] for k,v in sweep_config["parameters"].items()}
    cfg_dict["training_type"] = eval(cfg_dict["training_type"])
    initial_config = InitialConfig(**cfg_dict)
    cfg = extend_initial_config(initial_config)
    train_specific_type = None
    if isinstance(cfg.training_type, AR):
        train_specific_type = lambda: train_autoregressive(cfg)
    elif isinstance(cfg.training_type, GptEval):
        train_specific_type =  lambda: train_gpt_eval(cfg)
    elif isinstance(cfg.training_type, AO) or isinstance(cfg.training_type, AOA):
        train_specific_type = lambda: train_ao_or_aoa(cfg)
    elif isinstance(cfg.training_type, RAO):
        train_specific_type = lambda: train_rao(cfg)
    else:
        assert "Invalid training type"
    if cfg.wandb:
        sweep_id = wandb.sweep(
            sweep_config, project="collaborative-training-many-per-context-window"
        )
        run = wandb.init()
        wandb.agent(sweep_id, function=train_specific_type)
        run.finish()
    else:
        aggregate_losses = train_specific_type()
        plt.figure()
        plt.plot(aggregate_losses)

def train_rao(cfg):

    if not cfg.load_model:
        with open(f"saved_weights_and_losses/{cfg.model_name}", "w") as f:
            print("")

    average_losses = []
    aggregate_losses = []
    optimistic_loss = 0.5
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    prev_obs = torch.cat((cfg.obs_prefix_tensor, 
        torch.full((cfg.batch_size, cfg.tokens_per_pure_observation), fill_value=cfg.causal_lm_tokenizer.pad_token_id, 
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
            cfg.causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            cfg.causal_lm.save_pretrained(cfg.path_2_model)

        with torch.no_grad():
            rao_tensor_optimistic_triples, new_losses, prev_obs = gen_rao_tensor(
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
                cfg.training_type.tokens_per_pure_loss,
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
        for i in range(0, len(rao_tensor_triples), cfg.num_rao + 1):
            rao_tensor_slice = condense_triples(
                rao_tensor_triples[i : i + cfg.num_rao + 1], default_tensor
            )
            rao_tensor_logits = cfg.causal_lm(rao_tensor_slice).logits[:, :-1, :]
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
    return aggregate_losses


def train_ao_or_aoa(cfg : Config):

    if not cfg.load_model:
        with open(cfg.path_2_log, "w") as f:
            print("")

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(cfg.causal_lm.parameters(), lr=cfg.lr)
    training_ctxt_size = cfg.training_ctxt_size if cfg.training_ctxt_size else cfg.ctxt_size

    tok_p_action = int(training_ctxt_size / (2 + cfg.obs_to_action_ratio))
    tok_p_obs = int(cfg.obs_to_action_ratio * tok_p_action)
    tok_p_pure_action = tok_p_action - cfg.action_prefix_tensor.shape[1]
    tok_p_pure_obs = tok_p_obs - cfg.obs_prefix_tensor.shape[1]
    assert tok_p_pure_action > 0 and tok_p_pure_obs > 0

    itr_ds = load_dataset(cfg.dataset_name, cfg.task_name, split="train", streaming=True)
    ds_tokenized = map(
            lambda x: cfg.causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(cfg.device), 
            itr_ds)
    pure_obs = get_pure_obs(cfg.batch_size, tok_p_pure_obs, cfg.device, ds_tokenized)
    obs_ds = take(cfg.num_batches, prepend_obs_tensor(cfg.obs_prefix_tensor, pure_obs))
    action = torch.cat((cfg.action_prefix_tensor, 
        torch.full((cfg.batch_size, tok_p_pure_action), fill_value=cfg.causal_lm_tokenizer.pad_token_id, 
            dtype=torch.int64, device=cfg.device)), dim=1)
    aggregate_losses = []
    prev_obs = None
    with open(cfg.path_2_log, "a") as f:
        f.write("")

    for batch_index, obs in (
        tqdm(enumerate(obs_ds), total=cfg.num_batches)
    ):
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
            )[:, -cfg.tok_p_action:]

        optimizer.zero_grad()
        input_sequence = [action, obs, next_action] if isinstance(cfg.training_type, AOA) else [action, obs]
        rao_tensor_logits = cfg.causal_lm(torch.cat(input_sequence, dim=1)).logits[:,:-1,:]
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
            if isinstance(cfg.training_type, AOA):
                next_action_loss = rao_tensor_loss[:, cfg.tok_p_obs:].mean()
 
            if cfg.wandb:
                if isinstance(cfg.training_type, AOA):
                    wandb.log({"Next Action Loss": next_action_loss})
                wandb.log({
                        "Aggregate Loss": aggregate_loss,
                        "Action Loss": action_loss,
                        "Observation Loss": observation_loss})

        #printing
        if batch_index % cfg.interval_print == 0:
            with open(cfg.path_2_log, "a") as f:
                multi_print(f"Batch {batch_index}", f)
                multi_print(f"Aggregate loss: {aggregate_loss}", f)
                if isinstance(cfg.training_type, AOA):
                    multi_print(f"Action/Observation/NextAction loss: {action_loss}/{observation_loss}/{next_action_loss}", f)
                else:
                    multi_print(f"Action/Observation loss: {action_loss}/{observation_loss}", f)
                if prev_obs is not None: 
                    multi_print(f"Prev Observation: {repr(cfg.causal_lm_tokenizer.decode(prev_obs[0]))}", f)
                multi_print(f"Action: {repr(cfg.causal_lm_tokenizer.decode(action[0]))}", f)
                multi_print(f"Observation: {repr(cfg.causal_lm_tokenizer.decode(obs[0]))}", f)
                multi_print(f"Next action: {repr(cfg.causal_lm_tokenizer.decode(next_action[0]))}", f)
                multi_print("______________________________________________________", f)
        action = next_action
        prev_obs = obs
    return aggregate_losses

def train_autoregressive(cfg):
    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.causal_lm_tokenizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(causal_lm.parameters(), lr=cfg.lr)
    itr_ds = load_dataset(cfg.dataset_name, cfg.task_name, split="train", streaming=True)
    ds_tokenized = map(
            lambda x: causal_lm_tokenizer(x["text"], return_tensors="pt")["input_ids"].to(cfg.device), 
            itr_ds)
    pure_obs = get_pure_obs(cfg.batch_size, cfg.tok_p_obs, cfg.device, ds_tokenized)
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

def evaluate_via_gpt(model_name, num_batches, use_wandb, gpt_eval):
    def log_filter(line):
        if line == "\n": return False
        if line.startswith("Batch"): return True
        if line.startswith("Action:"): return True
        if line.startswith("Observation:"): return True
        if line.startswith("True Obs:"): return True
        return False

    def reformat(line):
        if line.startswith("True Obs:"):
            return "Observation:"+line[9:]
        elif line.startswith("Batch"):
            batch_number = line.split(" ")[-1]
            return f"Batch: {batch_number}"
        return line

    def remove_duplicates(itr):
        order, i = ["Batch", "Action", "Observation"], 0
        while 1:
            next_line = next(itr)
            if next_line.startswith(order[i]):
                yield next_line
                i = (i + 1) % len(order)

    def collect_dictionaries(itr):
        dict_keys = ["Batch: ", "Action: ", "Observation: "]
        while 1:
            d = {}
            for k in dict_keys:
                next_line = next(itr)
                d[k[:-2]] = next_line[len(k):-1] # remove the newline character
            yield d

    def throttle(itr):
        thresholds = torch.linspace(0, num_batches, gpt_eval).tolist()
        current_index = -1
        for t in thresholds:
            #print("threshold: ",t)
            while current_index < t:
                d = next(itr)
                current_index = int(d["Batch"])
            yield d

    def openai_rating(d):
        act, obs = d["Action"], d["Observation"]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Output a single number with no other text."},
                {"role": "user", "content": 
            f"""Look at the following pair of strings, and give a numerical response from 1 to 10 saying how much the first string would help you predict the second string. 
            String 1: {act} 
            String 2: {obs}
            """}]
        )
        return (int(d["Batch"]), float(response.choices[0].message.content))

    def wandb_log(line):
        if use_wandb:
            wandb.log({"Batch": line[0], "Rating": line[1]})
        return line

    def take(n, itr):
        for _ in range(n):
            yield next(itr)
    
    def print_all(itr):
        for i in itr:
            print(i)

    def log_result(log_itr):
        a = log_itr
        a = filter(log_filter, a)
        a = map(reformat, a)
        a = remove_duplicates(a)
        a = collect_dictionaries(a)
        a = throttle(a)
        a = map(openai_rating, a)
        a = map(wandb_log, a)
        print_all(enumerate(a))

    client = OpenAI()
    with open(cfg.path_2_log, "r") as file:
        log_itr = iter(file.readlines())
    log_result(log_itr)

if __name__ == "__main__":
    train()