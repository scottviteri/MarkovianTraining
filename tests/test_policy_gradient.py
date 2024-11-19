import pytest
import torch
import numpy as np
from src.train import *


# Fixtures
@pytest.fixture
def model_setup():
    """Setup model, tokenizer and device for testing"""
    model, frozen_model, tokenizer, device = load_model("mistral")
    return model, frozen_model, tokenizer, device


@pytest.fixture
def sample_qa_batch():
    """Generate a sample QA batch for testing"""
    return next(
        generate_question_answer_batches(
            num_batches=1,
            batch_size=2,
            task_type="arithmetic",
            tokenizer=None,
            hyperparameters=None,
        )
    )


@pytest.fixture
def sample_hyperparameters():
    """Setup default hyperparameters for testing"""
    return {
        # Model configuration
        "model_type": "mistral",
        "model_learning_rate": 0.0001,
        
        # Training configuration
        "batch_size": 6,
        "gradient_accumulation_steps": 8,
        "num_batches": 10000,
        "normalize_loss": True,
        
        # Task configuration
        "task_type": "arithmetic",
        "cot_length": 150,
        "question_length": 500,
        "target_length": 500,
        
        # Method configuration
        "use_ppo": False,
        "use_ei": False,
        "use_pg": True,
        "ppo_epsilon": 0.2,
        
        # Generation parameters
        "temperature": 0.7,
        "r": 0.9,
        "shrink_cot": False,
        "ei_threshold": None,
    }


# Add new fixtures for our dataclasses
@pytest.fixture
def training_state(model_setup, sample_hyperparameters):
    """Setup TrainingState for testing"""
    model, frozen_model, tokenizer, device = model_setup
    return TrainingState(
        batch_index=0,
        previous_normalized_rewards=[],
        previous_advantages=[],
        grad_accum_count=0,
        actor_model=model,
        critic_model=frozen_model,
        actor_optimizer=bitsandbytes.optim.AdamW8bit(
            model.parameters(), 
            lr=sample_hyperparameters["model_learning_rate"]
        ),
        tokenizer=tokenizer,
        device=device,
        model_save_path="test_model_path",
        log_file="test_log.jsonl",
        hyperparameters=sample_hyperparameters
    )

@pytest.fixture
def batch_data(training_state, sample_qa_batch):
    """Generate sample BatchData for testing"""
    questions, answers = zip(*sample_qa_batch)
    reasoning_output = generate_reasoning_and_kl(training_state, questions)
    advantage_output = calculate_advantages(
        training_state,
        questions,
        answers,
        reasoning_output,
    )
    losses, training_mask, metrics = calculate_losses(
        reasoning_output.kl,
        reasoning_output.R_mean_actor_logprobs,
        reasoning_output.R_mean_critic_logprobs,
        advantage_output.advantages,
        training_state.previous_advantages,
        training_state.hyperparameters,
    )
    
    return BatchData(
        questions=questions,
        answers=answers,
        actor_reasoning=reasoning_output.actor_reasoning,
        critic_reasoning=reasoning_output.critic_reasoning,
        R_mean_actor_logprobs=reasoning_output.R_mean_actor_logprobs,
        R_mean_critic_logprobs=reasoning_output.R_mean_critic_logprobs,
        kl=reasoning_output.kl,
        advantages=advantage_output.advantages,
        normalized_rewards=advantage_output.normalized_rewards,
        losses=losses,
        training_mask=training_mask,
        metrics=metrics
    )


## Model Loading Tests
# def test_load_model_mistral():
#    """Test loading Mistral model"""
#    model, frozen_model, tokenizer, device = load_model("mistral")
#    assert model is not None
#    assert frozen_model is not None
#    assert tokenizer is not None
#    assert device is not None
#    assert next(model.parameters()).requires_grad
#    assert not next(frozen_model.parameters()).requires_grad


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


def test_generate_question_answer_batch():
    """Test arithmetic question generation"""
    # Use generate_question_answer_batches with num_batches=1
    batch = next(
        generate_question_answer_batches(
            num_batches=1,
            batch_size=3,
            task_type="arithmetic",
            tokenizer=None,  # Tokenizer is optional for this test
            hyperparameters={},  # Empty hyperparameters
        )
    )

    assert len(batch) == 3
    for question, answer in batch:
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert any(op in question for op in ["+", "-"])  # Include negative numbers
        assert str(int(float(answer))) == answer  # Check if answer is a valid number


def test_generate_negative_arithmetic_batch():
    """Test negative arithmetic question generation"""
    # Use generate_question_answer_batches with num_batches=1
    batch = next(
        generate_question_answer_batches(
            num_batches=1,
            batch_size=3,
            task_type="arithmetic-negative",
            tokenizer=None,  # Tokenizer is optional for this test
            hyperparameters={},  # Empty hyperparameters
        )
    )

    assert len(batch) == 3
    for question, answer in batch:
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert "-" in question  # Should include negative numbers
        assert str(int(float(answer))) == answer


def test_generate_wiki_batch(model_setup, sample_hyperparameters):
    """Test Wikipedia task generation"""
    model, frozen_model, tokenizer, device = model_setup

    for task_type in ["wiki_compression", "wiki_continuation"]:
        qa_batch = next(
            generate_question_answer_batches(
                num_batches=1,
                batch_size=2,
                task_type=task_type,
                tokenizer=tokenizer,
                hyperparameters=sample_hyperparameters,
            )
        )
        assert len(qa_batch) > 0

        # Check token lengths
        for q, a in qa_batch:
            q_tokens = len(tokenizer(q, return_tensors="pt").input_ids[0])
            a_tokens = len(tokenizer(a, return_tensors="pt").input_ids[0])

            if task_type == "wiki_compression":
                assert q_tokens == sample_hyperparameters["target_length"]
            else:  # wiki_continuation
                assert q_tokens == sample_hyperparameters["question_length"]
                assert a_tokens == sample_hyperparameters["target_length"]


def test_load_gsm8k_dataset():
    """Test GSM8K dataset loading"""
    data_iterator = load_gsm8k_dataset()

    # Try to get the first item from the iterator
    first_item = next(data_iterator)

    # Check the first item
    assert isinstance(first_item, tuple)
    assert len(first_item) == 2

    # Optionally, you can check the types of the tuple elements
    question, answer = first_item
    assert isinstance(question, str)
    assert isinstance(answer, str)


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
def test_calculate_losses_ppo(training_state):
    """Test PPO loss calculation"""
    training_state.hyperparameters.update({
        "use_ei": False,
        "use_pg": False,
        "use_ppo": True,
        "ppo_epsilon": 0.2,
    })
    
    kl = torch.tensor([0.1, 0.1])
    actor_logprobs = torch.tensor([0.1, 0.2])
    critic_logprobs = torch.tensor([0.1, 0.2])
    advantages = torch.tensor([1.0, 1.0])
    
    losses, training_mask, metrics = calculate_losses(
        kl,
        actor_logprobs,
        critic_logprobs,
        advantages,
        training_state.previous_advantages,
        training_state.hyperparameters
    )
    
    assert isinstance(losses, torch.Tensor)
    assert isinstance(metrics['pg_losses'], torch.Tensor)
    assert 'prob_ratios' in metrics
    assert 'clipped_ratios' in metrics


# Integration Tests
def test_training_step(training_state, sample_qa_batch):
    """Test a single training step"""
    # Ensure hyperparameters are complete
    assert "model_type" in training_state.hyperparameters
    assert "task_type" in training_state.hyperparameters
    
    batch_data = process_batch(training_state, sample_qa_batch)
    grad_norm = update_model(training_state, batch_data)
    
    assert isinstance(grad_norm, float)
    assert grad_norm >= 0
    assert batch_data.actor_reasoning[0] != batch_data.critic_reasoning[0]
    assert len(batch_data.questions) == len(batch_data.answers)


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


def test_colored_print(capsys):
    """Test colored print functionality"""
    test_label = "Test"
    test_text = "Hello World"
    colored_print(test_label, test_text, Colors.BLUE)
    captured = capsys.readouterr()
    assert Colors.BLUE in captured.out
    assert Colors.END in captured.out
    assert repr(test_text) in captured.out


def test_get_default_hyperparameters():
    """Test default hyperparameter generation"""
    default_params = {
        "task_type": "arithmetic",
        "model_type": "llama",
        "training_methods": {"use_ppo": False, "use_ei": False, "use_pg": True},
        "cot_length": 150,
        "r": 0.9,
        "temperature": 0.7,
        "question_length": 500,
        "target_length": 500,
        "shrink_cot": False,
        "ei_threshold": None,
        "gradient_accumulation_steps": 8
    }
    
    params = get_default_hyperparameters(**default_params)
    
    # Verify required parameters are present
    assert params["model_type"] == "llama"
    assert params["task_type"] == "arithmetic"
    assert params["cot_length"] == 150
    assert params["normalize_loss"] is True


def test_get_text_with_token_length(model_setup):
    """Test binary search for text with desired token length"""
    _, _, tokenizer, _ = model_setup

    # Test case 1: Text long enough for desired tokens
    long_text = "This is a very long text " * 100
    desired_tokens = 50
    chunk, actual_tokens = get_text_with_token_length(
        long_text, desired_tokens, tokenizer
    )

    assert chunk is not None
    assert isinstance(chunk, str)
    # Allow small deviation due to binary search approximation
    assert abs(actual_tokens - desired_tokens) <= 5

    # Test case 2: Text too short for desired tokens
    short_text = "This is too short"
    desired_tokens = 100
    chunk, actual_tokens = get_text_with_token_length(
        short_text, desired_tokens, tokenizer
    )

    assert chunk is None
    assert actual_tokens == 0

    # Test case 3: Binary search convergence
    medium_text = "This is a test sentence. " * 20
    desired_tokens = 20
    chunk, actual_tokens = get_text_with_token_length(
        medium_text, desired_tokens, tokenizer
    )

    assert chunk is not None
    tokens = tokenizer(chunk, return_tensors="pt").input_ids[0]
    # Verify binary search found a close approximation
    assert len(tokens) == desired_tokens


def test_arithmetic_lazy_loading(model_setup, sample_hyperparameters):
    """Test lazy loading behavior of arithmetic dataset"""
    _, _, tokenizer, _ = model_setup

    gen = generate_question_answer_batches(
        num_batches=10,
        batch_size=4,
        task_type="arithmetic",
        tokenizer=tokenizer,
        hyperparameters=sample_hyperparameters,
    )

    # Get first chunk
    first_chunk = []
    for _ in range(25):  # 100/4 = 25 batches in first chunk
        batch = next(gen)
        assert len(batch) == 4
        first_chunk.extend(batch)

    # Verify uniqueness within chunk
    questions = [q for q, _ in first_chunk]
    assert len(set(questions)) == len(
        questions
    ), "Questions should be unique within chunk"

    # Get start of next chunk to verify different questions
    next_batch = next(gen)
    assert len(next_batch) == 4
    assert all(
        q not in questions for q, _ in next_batch
    ), "Next chunk should have different questions"


def test_gsm8k_lazy_loading(model_setup, sample_hyperparameters):
    """Test lazy loading behavior of GSM8K dataset"""
    _, _, tokenizer, _ = model_setup

    gen = generate_question_answer_batches(
        num_batches=10,
        batch_size=4,
        task_type="gsm8k",
        tokenizer=tokenizer,
        hyperparameters=sample_hyperparameters,
    )

    # Get first chunk
    first_chunk = []
    chunk_batches = 0
    for batch in gen:
        assert len(batch) == 4
        first_chunk.extend(batch)
        chunk_batches += 1
        if chunk_batches >= 25:  # 100/4 = 25 batches
            break

    assert len(first_chunk) == 100, "Should get 100 examples in first chunk"


def test_wiki_compression_lazy_loading(model_setup, sample_hyperparameters):
    """Test lazy loading behavior of Wikipedia compression dataset"""
    _, _, tokenizer, _ = model_setup

    gen = generate_question_answer_batches(
        num_batches=10,
        batch_size=4,
        task_type="wiki_compression",
        tokenizer=tokenizer,
        hyperparameters=sample_hyperparameters,
    )

    # Get first chunk
    first_chunk = []
    chunk_batches = 0
    for batch in gen:
        assert len(batch) == 4
        first_chunk.extend(batch)
        chunk_batches += 1
        if chunk_batches >= 25:  # 100/4 = 25 batches
            break

    # Verify token lengths
    for q, a in first_chunk[:5]:  # Only check first 5 to speed up test
        q_tokens = len(tokenizer(q, return_tensors="pt").input_ids[0])
        assert abs(q_tokens - sample_hyperparameters["target_length"]) <= 5


def test_wiki_continuation_lazy_loading(model_setup, sample_hyperparameters):
    """Test lazy loading behavior of Wikipedia continuation dataset"""
    _, _, tokenizer, _ = model_setup

    gen = generate_question_answer_batches(
        num_batches=10,
        batch_size=4,
        task_type="wiki_continuation",
        tokenizer=tokenizer,
        hyperparameters=sample_hyperparameters,
    )

    # Get first chunk
    first_chunk = []
    chunk_batches = 0
    for batch in gen:
        assert len(batch) == 4
        first_chunk.extend(batch)
        chunk_batches += 1
        if chunk_batches >= 25:  # 100/4 = 25 batches
            break

    # Verify token lengths for first 5 examples to speed up test
    for q, a in first_chunk[:5]:
        q_tokens = len(tokenizer(q, return_tensors="pt").input_ids[0])
        a_tokens = len(tokenizer(a, return_tensors="pt").input_ids[0])

        assert abs(q_tokens - sample_hyperparameters["question_length"]) <= 5
        assert abs(a_tokens - sample_hyperparameters["target_length"]) <= 5


def test_construct_prompts(model_setup, sample_hyperparameters):
    """Test prompt construction for different tasks and models"""
    _, _, tokenizer, _ = model_setup

    test_cases = [
        {
            "question": "15 + 23 + 45",
            "hyperparameters": {**sample_hyperparameters, "task_type": "arithmetic"},
            "reasoning": "Let me solve this step by step:\n15 + 23 = 38\n38 + 45 = 83",
        },
        {
            "question": "The quick brown fox jumps over the lazy dog.",
            "hyperparameters": {**sample_hyperparameters, "task_type": "wiki_compression"},
            "reasoning": "This sentence contains all letters of the alphabet.",
        },
    ]

    for case in test_cases:
        # Test without reasoning
        prompt = construct_prompts(
            question=case["question"],
            hyperparameters=case["hyperparameters"]
        )
        assert case["question"] in prompt
        
        # Test with reasoning
        full_prompt = construct_prompts(
            question=case["question"],
            hyperparameters=case["hyperparameters"],
            reasoning=case["reasoning"]
        )
        assert case["reasoning"] in full_prompt

def test_prompt_edge_cases(model_setup, sample_hyperparameters):
    """Test edge cases in prompt construction"""
    _, _, tokenizer, _ = model_setup

    # Test empty question
    prompt = construct_prompts(
        question="",
        hyperparameters=sample_hyperparameters
    )
    assert "Question: " in prompt

    # Test very long question
    long_question = "x + " * 1000
    prompt = construct_prompts(
        question=long_question,
        hyperparameters=sample_hyperparameters
    )
    assert len(prompt) > 0

def test_log_metrics(training_state, batch_data):
    """Test creation and usage of LogMetrics"""
    metrics = LogMetrics.from_batch(
        batch_data=batch_data,
        grad_norm=1.0,
        grad_accum_count=4,
        previous_advantages=training_state.previous_advantages,
        batch_size=training_state.hyperparameters["batch_size"]
    )
    
    # Verify metric types and ranges
    assert isinstance(metrics.loss, float)
    assert isinstance(metrics.pg_loss, float)
    assert isinstance(metrics.advantage, float)
    assert isinstance(metrics.fraction_active, float)
    assert 0 <= metrics.fraction_active <= 1
    
    # Verify optional metrics
    if training_state.hyperparameters["use_ppo"]:
        assert metrics.ppo_ratio is not None
        assert metrics.ppo_clipped_ratio is not None
    else:
        assert metrics.ppo_ratio is None
        assert metrics.ppo_clipped_ratio is None
