import torch
from tqdm import tqdm
import wandb
import einops
from typing import List

from src.types_and_utilities import InitialConfig, InitTrainingType, Config
from src.types_and_utilities import AR, GptEval, AO, AOA, RAOInit
from src.types_and_utilities import log_and_print_info

def evaluate_via_gpt(model_name, num_batches, use_wandb, gpt_eval):
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

    def throttle(itr):
        thresholds = torch.linspace(0, num_batches, gpt_eval).tolist()
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
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Output a single number with no other text.",
                },
                {
                    "role": "user",
                    "content": f"""Look at the following pair of strings, and give a numerical response from 1 to 10 saying how much the first string would help you predict the second string. 
            String 1: {act} 
            String 2: {obs}
            """,
                },
            ],
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

