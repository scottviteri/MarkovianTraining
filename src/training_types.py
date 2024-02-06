from dataclasses import dataclass
from typing import Optional, Union, NamedTuple, Iterable, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import torch
from enum import Enum

AR = NamedTuple("AR", [])
GptEval = NamedTuple("GptEval", [("num_evals", int), ("use_gptj", bool)])
AOA = NamedTuple("AOA", 
[("use_gumbel", bool), ("ignore_first_action", bool), ("ignore_second_action", bool)])
EI = NamedTuple("EI", 
  [("ignore_first_action", bool), ("ignore_second_action", bool), 
   ("ignore_observation", bool), ("num_samples", int)
   ])
RAOInit = NamedTuple("RAO",
    [("num_rao", int),  ("obs_between_weight_updates", int), 
    ("use_loss_difference", bool), ("use_multirao_for_action_gen", bool), 
    ("use_rewards_to_go", bool)])
RAO = NamedTuple("RAO",
    [("num_rao", int),  ("obs_between_weight_updates", int), 
    ("use_loss_difference", bool), ("use_multirao_for_action_gen", bool), ("use_rewards_to_go", bool),
    ("tok_p_loss", int), ("tok_p_pure_loss", int), ("loss_prefix_tensor", torch.Tensor), ("tok_p_doc", int), ("tok_p_rao", int)])

InitTrainingType = Union[AR, GptEval, RAOInit, AOA, EI]
TrainingType = Union[AR, GptEval, RAO, AOA, EI]

RepeatNPoints = NamedTuple("RepeatNPoints", [("num_points", int)])
RepeatPointNTimes = NamedTuple("RepeatPointNTimes", [("num_times", int)])
ReplaceWithRandomTokens = NamedTuple("ReplaceWithRandomTokens", [])
NoWeightUpdates = NamedTuple("NoWeightUpdates", [])
InitDatasetType = NamedTuple("InitDatasetType", 
  [("name", str), ("task", Optional[str]), ("peek_every", Optional[int])])

DatasetType = NamedTuple("DatasetType", 
  [("name", str), ("task", str), ("peek_every", Optional[int]), ("dataloader", Iterable[Dict[str, torch.Tensor]])])

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
    dataset: InitDatasetType
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
    dataset : DatasetType
    training_type: TrainingType
    debug : Optional[DebugType]
