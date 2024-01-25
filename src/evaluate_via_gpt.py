import torch
from tqdm import tqdm
import wandb
import einops
from typing import List
from openai import OpenAI
import time

from src.training_types import *
from src.utilities import log_and_print_info

def evaluate_via_gpt(cfg : Config):
    def log_filter(line):
        if line == "\n":
            return False
        if line.startswith("Batch"):
            return True
        if line.startswith("Action:"):
            return True
        if line.startswith("Observation:"):
            return True
        if line.startswith("True Obs:"):
            return True
        return False

    def reformat(line):
        if line.startswith("True Obs:"):
            return "Observation:" + line[9:]
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
                d[k[:-2]] = next_line[len(k) : -1]  # remove the newline character
            yield d

    def throttle(num_batches, itr):
        thresholds = torch.linspace(0, num_batches, cfg.training_type.num_evals).tolist()
        current_index = -1
        for t in thresholds:
            # print("threshold: ",t)
            while current_index < t:
                d = next(itr)
                current_index = int(d["Batch"])
            yield d

    def openai_rating(d):
        act, obs = d["Action"], d["Observation"]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": 
"""
Look at helpful_msg and observation, and give a numerical response from 0 to 1 saying how much the helpful_msg helps you predict the observation. 
0 denotes that the helpful_msg that you would do just as well or better at predicting the observation without the helpful_msg.
1 denotes that you can now completely predict the observation, and that you would otherwise have no idea what the observation would be.
Output a single floating point number from 0 to 1 with no other text.
""",
                },
                {
                    "role": "user",
                    "content": 
f"""
    <helpful_msg> {act} </helpful_msg>
    <observation> {obs} </observation>
""",
                },
            ],
        )
        return (int(d["Batch"]), float(response.choices[0].message.content))

    def gptj_rating(tokenizer, d):
        act, obs = d["Action"], d["Observation"]
        input_ids = tokenizer.encode(act, return_tensors='pt')
        output_ids = tokenizer.encode(obs, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        logits = outputs.logits
        mean_logit = torch.mean(logits[:, -len(output_ids):])
        return (int(d["Batch"]), float(mean_logit.item()))

    def wandb_log(line):
        if cfg.wandb:
            wandb.log({"Batch": line[0], "Rating": line[1]})
        return line

    def take(n, itr):
        for _ in range(n):
            yield next(itr)

    def collect_and_print(itr):
        out_lst = []
        for i in itr:
            print(i)
            time.sleep(0.1)
            out_lst.append(i[1][1])
        return out_lst


    def log_result(tokenizer, num_batches, log_itr):
        a = log_itr
        a = filter(log_filter, a)
        a = map(reformat, a)
        a = remove_duplicates(a)
        a = collect_dictionaries(a)
        a = throttle(num_batches, a)
        if cfg.training_type.use_gptj:
            a = map(lambda x: gptj_rating(tokenizer, x), a)
        else:
            a = map(openai_rating, a)
        a = map(wandb_log, a)
        return collect_and_print(enumerate(a))

    def get_num_batches():
        with open(cfg.path_2_log, "r") as file:
            lines = file.readlines()
        batch_lines = filter(lambda x: x.startswith("Batch"), lines)
        batch_numbers = map(lambda x: int(x.split(" ")[-1]), batch_lines)
        return max(batch_numbers)

    client = OpenAI()
    num_batches = get_num_batches()
    with open(cfg.path_2_log, "r") as file:
        log_itr = iter(file.readlines())
    return log_result(cfg.causal_lm_tokenizer, num_batches, log_itr)

