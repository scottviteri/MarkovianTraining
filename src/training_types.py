from dataclasses import dataclass
from typing import Optional, Union, NamedTuple, Iterable
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import torch
from enum import Enum

AR = NamedTuple("AR", [("observation_size", int)])
GptEval = NamedTuple("GptEval", [("num_evals", int)])
AOA = NamedTuple("AOA", 
[("use_gumbel", bool), ("ignore_first_action", bool), ("ignore_second_action", bool)])
RAOInit = NamedTuple("RAO",
    [("num_rao", int),  ("obs_between_weight_updates", int), 
    ("use_loss_difference", bool), ("use_multirao_for_action_gen", bool), 
    ("use_rewards_to_go", bool)])
RAO = NamedTuple("RAO",
    [("num_rao", int),  ("obs_between_weight_updates", int), 
    ("use_loss_difference", bool), ("use_multirao_for_action_gen", bool), ("use_rewards_to_go", bool),
    ("tok_p_loss", int), ("tok_p_pure_loss", int), ("loss_prefix_tensor", torch.Tensor), ("tok_p_doc", int), ("tok_p_rao", int)])

InitTrainingType = Union[AR, GptEval, RAOInit, AOA]
TrainingType = Union[AR, GptEval, RAO, AOA]

RepeatNPoints = NamedTuple("RepeatNPoints", [("num_points", int)])
RepeatPointNTimes = NamedTuple("RepeatPointNTimes", [("num_times", int)])
ReplaceWithRandomTokens = NamedTuple("ReplaceWithRandomTokens", [])
NoWeightUpdates = NamedTuple("NoWeightUpdates", [])

DebugType = Union[RepeatNPoints, RepeatPointNTimes, ReplaceWithRandomTokens, NoWeightUpdates]

@dataclass
class InitialConfig:
    model_name: str
    lr: float
    batch_size: int
    num_batches: int
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    wandb: bool
    load_model: bool
    do_lora : bool
    training_ctxt_size: Optional[int]
    dataset_name: str
    task_name: Optional[str]
    training_type : InitTrainingType
    debug : Optional[DebugType]

@dataclass
class Config:
    model_name: str
    causal_lm: Optional[PreTrainedModel]
    causal_lm_tokenizer: Optional[PreTrainedTokenizer]
    lr: float
    batch_size: int
    num_batches: int
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    wandb: bool
    load_model: bool
    do_lora: bool
    training_ctxt_size: int
    device: str
    dataset_name: str
    task_name: str
    path_2_log: str
    path_2_model: str
    path_2_tokenizer: str
    tok_p_action: Optional[int]
    tok_p_obs: Optional[int]
    tok_p_pure_action : Optional[int]
    tok_p_pure_obs : Optional[int]
    action_prefix_tensor: Optional[torch.Tensor]
    obs_prefix_tensor: Optional[torch.Tensor]
    ctxt_size: Optional[int]
    dataloader : Iterable[torch.Tensor]
    training_type: TrainingType
    debug : Optional[DebugType]
