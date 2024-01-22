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

from src.types_and_utilities import *
from src.train_rao import train_rao
from src.train_ao_or_aoa import train_ao_or_aoa
from src.evaluate_via_gpt import evaluate_via_gpt
from src.config_examples import gpt2_AOA, gpt2_RAO, example_configs


def train(initial_cfg : InitialConfig):
    cfg = extend_initial_config(initial_cfg)
    train_specific_type = None
    if isinstance(cfg.training_type, AR):
        train_specific_type = train_autoregressive
    elif isinstance(cfg.training_type, GptEval):
        train_specific_type = evaluate_via_gpt
    elif isinstance(cfg.training_type, AO) or isinstance(cfg.training_type, AOA):
        train_specific_type = train_ao_or_aoa
    elif isinstance(cfg.training_type, RAO):
        train_specific_type = train_rao
    else:
        assert "Invalid training type"
    if cfg.wandb:
        #sweep_id = wandb.sweep(
        #    sweep_config, project="collaborative-training-many-per-context-window"
        #)
        run = wandb.init(project="collaborative-training-many-per-context-window")
        #wandb.agent(sweep_id, function=train_specific_type)
        aggregate_losses = train_specific_type(cfg)
        run.finish()
    else:
        aggregate_losses = train_specific_type(cfg)
        plt.figure()
        plt.plot(aggregate_losses)



def test():
    for config in example_configs:
        train(config)

if __name__ == "__main__":
   #train(gpt2_RAO) 
   test()