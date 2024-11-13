import pytest
import torch
import numpy as np
from src.policy_gradient_normalized import *


# Fixtures
@pytest.fixture
def model_setup():
    """Setup model, tokenizer and device for testing"""
    model, frozen_model, tokenizer, device = load_model("mistral")
    return model, frozen_model, tokenizer, device


@pytest.fixture
def sample_qa_batch():
    """Generate a sample QA batch for testing"""
    return generate_question_answer_batch(batch_size=2)


# Model Loading Tests
def test_load_model_mistral():
    """Test loading Mistral model"""
    model, frozen_model, tokenizer, device = load_model("mistral")
    assert model is not None
    assert frozen_model is not None
    assert tokenizer is not None
    assert device is not None
    assert next(model.parameters()).requires_grad
    assert not next(frozen_model.parameters()).requires_grad


def test_load_model_llama():
    """Test loading Llama model"""
    model, frozen_model, tokenizer, device = load_model("llama")
    assert model is not None
    assert frozen_model is not None
    assert tokenizer is not None
    assert device is not None


def test_load_model_invalid():
    """Test loading invalid model type"""
    with pytest.raises(ValueError):
        load_model("invalid_model")


# Data Generation Tests
def test_generate_question_answer_batch():
    """Test arithmetic question generation"""
    batch = generate_question_answer_batch(batch_size=3)
    assert len(batch) == 3
    for question, answer in batch:
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert "+" in question
        assert answer.isdigit()


def test_load_gsm8k_dataset():
    """Test GSM8K dataset loading"""
    data = load_gsm8k_dataset()
    assert len(data) > 0
    assert isinstance(data[0], tuple)
    assert len(data[0]) == 2


# Answer Extraction Tests
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The answer is 42", 42),
        ("= 123", 123),
        ("Final answer: -15", -15),
        ("Invalid", "[invalid]"),
        ("42,000", 42000),
    ],
)
def test_extract_answer(answer, expected):
    """Test answer extraction from various formats"""
    assert extract_answer(answer) == expected


# Advantage Calculation Tests
def test_calculate_threshold():
    """Test advantage threshold calculation"""
    advantages = [1.0, 2.0, 3.0, 4.0, 5.0]
    threshold = calculate_threshold(advantages)
    assert isinstance(threshold, float)
    assert threshold > np.mean(advantages)


def test_exponential_weighted_average():
    """Test exponential weighted average calculation"""
    values = [1.0, 2.0, 3.0]
    r = 0.9
    result = exponential_weighted_average(values, r)
    assert isinstance(result, float)
    assert result > min(values) and result < max(values)


# Loss Calculation Tests
def test_calculate_losses_ppo():
    """Test PPO loss calculation"""
    hyperparameters = {
        "use_ei": False,
        "use_pg": False,
        "use_ppo": True,
        "ppo_epsilon": 0.2,
    }
    unfrozen_probs = torch.tensor([0.1, 0.2])
    frozen_probs = torch.tensor([0.1, 0.2])
    advantage = torch.tensor([1.0, 1.0])

    total_loss, policy_loss, ppo_ratio, clipped_ratio = calculate_losses(
        unfrozen_probs, frozen_probs, advantage, hyperparameters
    )
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(policy_loss, torch.Tensor)
    assert ppo_ratio is not None
    assert clipped_ratio is not None


# Integration Tests
def test_training_step(model_setup):
    """Test a single training step"""
    model, frozen_model, tokenizer, device = model_setup
    batch = generate_question_answer_batch(batch_size=2)
    questions, answers = zip(*batch)

    prompts = [f"{q} \nReasoning:" for q in questions]
    tokenized_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=400,
            min_new_tokens=400,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    assert outputs.shape[0] == 2
    assert outputs.shape[1] > tokenized_inputs.input_ids.shape[1]


# Utility Function Tests
def test_get_grad_norm():
    """Test gradient norm calculation"""
    model = torch.nn.Linear(10, 1)
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    loss = torch.nn.MSELoss()(model(x), y)
    loss.backward()

    grad_norm = get_grad_norm(model.parameters())
    assert isinstance(grad_norm, float)
    assert grad_norm >= 0


def test_tensor_to_python():
    """Test tensor conversion to Python types"""
    tensor_cases = [
        (torch.tensor(1.0), 1.0),
        (torch.tensor([1.0, 2.0]), [1.0, 2.0]),
        (np.array(1.0), 1.0),
        (np.array([1.0, 2.0]), [1.0, 2.0]),
    ]

    for input_tensor, expected in tensor_cases:
        result = tensor_to_python(input_tensor)
        assert result == expected
