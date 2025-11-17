import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

# Ensure src/ is on sys.path so tests can import project modules without env vars
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reduce noisy HF warnings during tests
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from utils import generate_question_answer_batches, load_model  # noqa: E402


@pytest.fixture(scope="session")
def gpt2_components() -> Dict[str, object]:
    """Load GPT-2 actor/critic/tokenizer once per session for integration tests."""
    # Use tiny LoRA ranks to keep memory requirements small for CI
    hyperparameters = {"lora_rank": 2, "lora_alpha": 4}
    actor_model, critic_model, tokenizer, device = load_model("gpt2", hyperparameters)
    yield {
        "actor": actor_model,
        "critic": critic_model,
        "tokenizer": tokenizer,
        "device": device,
    }


@pytest.fixture(scope="session")
def tiny_training_hparams() -> Dict[str, object]:
    """Hyperparameters tuned for very small integration tests."""
    return {
        "task_type": "arithmetic",
        "model_type": "gpt2",
        "batch_size": 2,
        "cot_length": 48,
        "question_length": 32,
        "target_length": 32,
        "temperature": 0.8,
        "r": 0.9,
        "markovian": True,
        "actor_reward_weight": 0.0,
        "kl_penalty": 0.0,
        "gradient_accumulation_steps": 1,
        "normalize_loss": False,
        "use_ei": None,
        "use_pg": True,
        "use_ppo": False,
        "parallel": False,
        "num_batches": 1,
        "checkpoint_frequency": 1,
        "eval_frequency": 1,
        "enable_weight_verification": False,
    }


@pytest.fixture
def arithmetic_batch() -> Tuple[str, str]:
    """Single arithmetic QA batch used by multiple tests."""
    batch = next(
        generate_question_answer_batches(
            num_batches=1,
            batch_size=2,
            task_type="arithmetic",
            tokenizer=None,
            hyperparameters={"parallel": False},
        )
    )
    return batch

