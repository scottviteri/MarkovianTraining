from dataclasses import dataclass
from typing import Optional, Union, NamedTuple, Iterable, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch
from enum import Enum

GptEval = NamedTuple("GptEval", [("num_evals", int), ("use_gptj", bool)])
PredictionConfig = NamedTuple(
    "PredictionConfig",
    [
        ("filter_best_actions", Optional[int]),
        ("train_A_given_AO", bool),
        ("train_O_given_A", bool),
        ("train_O_given_prev_O", bool),
    ],
)
InferenceConfig = NamedTuple(
    "InferenceConfig",
    [
        ("num_return_sequences", int),
        # ("update_every", Optional[int]),
        # ("fraction_to_update", Optional[float]),
    ],
)
TrainerConfig = NamedTuple(
    "TrainerConfig",
    [
        ("prediction_training_length", Optional[int]),
        ("inference_training_length", Optional[int]),
    ],
)
PerturbationConfig = NamedTuple(
    "PerturbationConfig",
    [
        ("eval_every", int),
        ("frac_of_tokens_to_pad", float),
        ("frac_of_tokens_to_randomize", float),
    ],
)

RepeatNPoints = NamedTuple("RepeatNPoints", [("num_points", int)])
RepeatPointNTimes = NamedTuple("RepeatPointNTimes", [("num_times", int)])
ReplaceWithRandomTokens = NamedTuple("ReplaceWithRandomTokens", [])
NoWeightUpdates = NamedTuple("NoWeightUpdates", [])

ArithmeticTask = NamedTuple(
    "ArithmeticTask",
    [
        ("num_digits", int),
        ("num_terms", int),
        ("cumulative", bool),
        ("operations", Optional[list]),
        ("probs", Optional[list]),
    ],
)
WikipediaTask = NamedTuple("WikipediaTask", [])
TaskType = Union[ArithmeticTask, WikipediaTask]
InitDatasetType = NamedTuple(
    "InitDatasetType", [("task", TaskType), ("peek_every", Optional[int])]
)

DatasetType = NamedTuple(
    "DatasetType",
    [
        ("task", TaskType),
        ("peek_every", Optional[int]),
        ("dataloader", Iterable[Dict[str, torch.Tensor]]),
    ],
)

DebugType = Union[
    RepeatNPoints, RepeatPointNTimes, ReplaceWithRandomTokens, NoWeightUpdates
]


@dataclass
class InitialConfig:
    model_name: str
    lr: float
    optimizer: str
    batch_size: int
    num_batches: int
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    wandb: bool
    load_model: bool
    do_lora: bool
    num_beams: int
    inference_cfg: InferenceConfig
    prediction_cfg: PredictionConfig
    trainer_cfg: TrainerConfig
    training_ctxt_size: Optional[int]
    dataset: InitDatasetType
    perturbation_cfg: Optional[PerturbationConfig]
    debug: Optional[DebugType]


@dataclass
class Config:
    model_name: str
    causal_lm: PreTrainedModel
    causal_lm_tokenizer: Optional[PreTrainedTokenizer]
    lr: float
    optimizer: torch.optim.Optimizer
    qhead_optimizer: torch.optim.Optimizer
    batch_size: int
    num_batches: int
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    wandb: bool
    load_model: bool
    do_lora: bool
    num_beams: int
    training_ctxt_size: int
    device: str
    path_2_log: str
    traj_path: str
    path_2_model: str
    path_2_tokenizer: str
    tok_p_action: Optional[int]
    tok_p_obs: Optional[int]
    tok_p_pure_action: Optional[int]
    tok_p_pure_obs: Optional[int]
    action_prefix_tensor: Optional[torch.Tensor]
    obs_prefix_tensor: Optional[torch.Tensor]
    ctxt_size: Optional[int]
    dataset: DatasetType
    inference_cfg: InferenceConfig
    prediction_cfg: PredictionConfig
    trainer_cfg: TrainerConfig
    training_predictor_mode: bool
    perturbation_cfg: Optional[PerturbationConfig]
    debug: Optional[DebugType]
