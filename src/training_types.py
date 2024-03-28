from dataclasses import dataclass
from typing import Optional, Union, NamedTuple, Iterable, Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch
from enum import Enum


@dataclass
class QADatapt:
    question: str
    explanation: Optional[str]
    answer: str


@dataclass
class Datapt:
    action: Optional[torch.tensor]
    obs: torch.tensor
    is_first: bool


@dataclass
class ContextSizes:
    first_action_size: int
    first_obs_size: int
    action_size: int
    obs_size: int


@dataclass
class ScoredTrajectory:
    prev_datapt: Datapt
    datapt: Datapt
    loss: float


@dataclass
class TrainerState:
    action: Optional[torch.tensor]
    obs: Optional[torch.tensor]
    batch_index: int
    aggregate_loss: Optional[float]
    replay_buffer: List[ScoredTrajectory]


GptEval = NamedTuple("GptEval", [("num_evals", int), ("use_gptj", bool)])
PredictionConfig = NamedTuple(
    "PredictionConfig",
    [
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
    replay_buffer_size: Optional[int]
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    use_mac: bool
    wandb: bool
    load_model: bool
    do_lora: bool
    num_beams: int
    inference_cfg: InferenceConfig
    prediction_cfg: PredictionConfig
    trainer_cfg: TrainerConfig
    ctxt_sizes: ContextSizes
    dataset: InitDatasetType
    perturbation_cfg: Optional[PerturbationConfig]
    debug: Optional[DebugType]


@dataclass
class PrefixTensors:
    first_action_prefix_tensor: torch.Tensor
    first_obs_prefix_tensor: torch.Tensor
    action_prefix_tensor: torch.Tensor
    obs_prefix_tensor: torch.Tensor


@dataclass(frozen=True)
class Config:
    model_name: str
    causal_lm: PreTrainedModel
    causal_lm_tokenizer: Optional[PreTrainedTokenizer]
    rank: int
    lr: float
    optimizer: torch.optim.Optimizer
    batch_size: int
    num_batches: int
    replay_buffer_size: Optional[int]
    obs_to_action_ratio: float
    interval_save_weights: int
    interval_print: int
    use_mac: bool
    wandb: bool
    load_model: bool
    do_lora: bool
    num_beams: int
    device: str
    path_2_log: str
    traj_path: str
    path_2_model: str
    path_2_tokenizer: str
    prefix_tensors: PrefixTensors
    pure_ctxt_sizes: ContextSizes
    ctxt_sizes: ContextSizes
    dataset: DatasetType
    inference_cfg: InferenceConfig
    prediction_cfg: PredictionConfig
    trainer_cfg: TrainerConfig
    perturbation_cfg: Optional[PerturbationConfig]
    debug: Optional[DebugType]
