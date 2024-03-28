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


@dataclass(frozen=True)
class QADatapt:
    question: str
    explanation: Optional[str]
    answer: str


@dataclass(frozen=True)
class Datapt:
    action: Optional[torch.tensor]
    obs: torch.tensor
    is_first: bool


@dataclass(frozen=True)
class ContextSizes:
    first_action_size: int
    first_obs_size: int
    action_size: int
    obs_size: int


@dataclass(frozen=True)
class ScoredTrajectory:
    prev_datapt: Datapt
    datapt: Datapt
    loss: float


@dataclass(frozen=True)
class TrainerState:
    action: Optional[torch.tensor]
    obs: Optional[torch.tensor]
    batch_index: int
    aggregate_loss: Optional[float]
    replay_buffer: List[ScoredTrajectory]


@dataclass(frozen=True)
class GptEval:
    num_evals: int
    use_gptj: bool


@dataclass(frozen=True)
class PredictionConfig:
    train_O_given_A: bool
    train_O_given_prev_O: bool


@dataclass(frozen=True)
class InferenceConfig:
    num_return_sequences: int
    # update_every: Optional[int]
    # fraction_to_update: Optional[float]


@dataclass(frozen=True)
class TrainerConfig:
    prediction_training_length: Optional[int]
    inference_training_length: Optional[int]


@dataclass(frozen=True)
class PerturbationConfig:
    eval_every: int
    frac_of_tokens_to_pad: float
    frac_of_tokens_to_randomize: float


@dataclass(frozen=True)
class RepeatNPoints:
    num_points: int


@dataclass(frozen=True)
class RepeatPointNTimes:
    num_times: int


@dataclass(frozen=True)
class ReplaceWithRandomTokens:
    pass


@dataclass(frozen=True)
class NoWeightUpdates:
    pass


@dataclass(frozen=True)
class ArithmeticTask:
    num_digits: int
    num_terms: int
    operations: Optional[list]
    probs: Optional[list]


@dataclass(frozen=True)
class WikipediaTask:
    pass


TaskType = Union[ArithmeticTask, WikipediaTask]


@dataclass(frozen=True)
class InitDatasetType:
    task: TaskType
    peek_every: Optional[int]


@dataclass(frozen=True)
class DatasetType:
    task: TaskType
    peek_every: Optional[int]
    dataloader: Iterable[Dict[str, torch.Tensor]]


DebugType = Union[
    RepeatNPoints, RepeatPointNTimes, ReplaceWithRandomTokens, NoWeightUpdates
]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class PrefixTensors:
    first_action_prefix_tensor: torch.Tensor
    first_obs_prefix_tensor: torch.Tensor
    action_prefix_tensor: torch.Tensor
    obs_prefix_tensor: torch.Tensor


@dataclass(frozen=True)
class Config:
    model_name: str
    causal_lm: PreTrainedModel
    tokenizer: Optional[PreTrainedTokenizer]
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
