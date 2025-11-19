"""
Integration tests for evaluation pipeline.

Tests the full evaluation flow end-to-end with mocked models:
- MCQ evaluation (MMLU-style)
- Numeric evaluation (GSM8K-style)
- Haiku integration (mocked)
- Deterministic sampling
- Result structure and backward compatibility
"""

import pytest
import torch
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evaluation import (
    evaluate_model_generic,
    evaluate_model_on_mcq,
    evaluate_model_on_numeric,
    evaluate_model_on_mmlu,
    evaluate_model_on_gsm8k,
)


class MockBatchEncoding:
    """Minimal BatchEncoding stand-in with .to() support."""
    
    def __init__(self, batch_size: int, seq_len: int = 10):
        self.input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        self.attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class MockModel:
    """Mock model that returns predetermined outputs."""
    
    def __init__(self, outputs: List[str]):
        self.outputs = outputs
        self.call_count = 0
        
    def generate(self, input_ids, attention_mask, **kwargs):
        """Return predetermined output as token IDs."""
        # Simple mock: return input_ids + new tokens
        output = input_ids.clone()
        self.call_count += 1
        return output
    
    def eval(self):
        """Mock eval mode."""
        return self


class MockTokenizer:
    """Mock tokenizer that encodes/decodes predetermined strings."""
    
    def __init__(self, cot_outputs: List[str], answer_outputs: List[str]):
        self.cot_outputs = cot_outputs
        self.answer_outputs = answer_outputs
        self.cot_call_count = 0
        self.answer_call_count = 0
        self.pad_token_id = 0
        
    def __call__(self, texts, **kwargs):
        """Mock tokenization that returns a batch encoding with .to()."""
        batch_size = len(texts) if isinstance(texts, list) else 1
        return MockBatchEncoding(batch_size=batch_size)
    
    def batch_decode(self, token_ids, **kwargs):
        """Mock decoding - return predetermined outputs."""
        batch_size = token_ids.shape[0]
        
        # Determine if this is CoT or answer generation based on call sequence
        # First decode calls are CoT, later ones are answers
        if self.cot_call_count < len(self.cot_outputs):
            outputs = self.cot_outputs[self.cot_call_count:self.cot_call_count + batch_size]
            self.cot_call_count += batch_size
        else:
            outputs = self.answer_outputs[self.answer_call_count:self.answer_call_count + batch_size]
            self.answer_call_count += batch_size
        
        return outputs


@pytest.fixture
def mock_device():
    """Mock device."""
    return torch.device("cpu")


@pytest.fixture
def basic_hyperparameters():
    """Basic hyperparameters for testing."""
    return {
        "task_type": "gsm8k",
        "cot_length": 50,
        "question_length": 256,
        "temperature": 1.0,
        "batch_size": 2,
        "markovian": True,
        "actor_reward_weight": 0.0,
        "model_type": "llama",
    }


class TestMCQEvaluation:
    """Test MCQ evaluation pipeline."""
    
    def test_mcq_basic_evaluation(self, mock_device, basic_hyperparameters):
        """Test basic MCQ evaluation with correct answers."""
        basic_hyperparameters["task_type"] = "aqua"
        # Prepare test data
        test_data = [
            ("What is 2+2?", "A"),
            ("What is 3+3?", "B"),
        ]
        
        # Mock tokenizer with predetermined outputs
        cot_outputs = [
            "Let me calculate: 2+2=4",
            "Let me calculate: 3+3=6",
        ]
        answer_outputs = [
            "Answer: A",  # Correct
            "Answer: B",  # Correct
        ]
        
        tokenizer = MockTokenizer(cot_outputs, answer_outputs)
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        # Run evaluation
        accuracy, results, haiku_metrics = evaluate_model_on_mcq(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=2,
            num_samples=None,
            num_choices=4,
            enable_haiku_metric=False,
        )
        
        # Check accuracy (word boundary extraction should work)
        assert accuracy == 1.0  # Both correct
        
        # Check results structure
        assert len(results) == 2
        assert results[0]["question"] == "What is 2+2?"
        assert results[0]["predicted"] == "A"
        assert results[0]["answer"] == "A"
        assert results[0]["correct"] is True
        
        # Check backward compatibility fields
        assert "gold" in results[0]
        assert "is_correct" in results[0]
    
    def test_mcq_word_boundary_vs_legacy(self, mock_device, basic_hyperparameters):
        """Test that word boundary extraction ignores letters inside words."""
        basic_hyperparameters["task_type"] = "aqua"
        test_data = [
            ("Question 1", "B"),
        ]
        
        # Answer that triggers legacy false positive
        cot_outputs = ["Reasoning text"]
        answer_outputs = ["The answer is B"]  # 'E' in "The" triggers legacy false positive
        
        tokenizer = MockTokenizer(cot_outputs, answer_outputs)
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        accuracy, results, _ = evaluate_model_on_mcq(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=1,
            enable_haiku_metric=False,
        )
        
        # Word boundary extraction should still capture the isolated 'B'
        assert accuracy == 1.0
        assert results[0]["predicted"] == "B"


class TestNumericEvaluation:
    """Test numeric evaluation pipeline."""
    
    def test_numeric_basic_evaluation(self, mock_device, basic_hyperparameters):
        """Test basic numeric evaluation with correct answers."""
        test_data = [
            ("What is 5+5?", "10"),
            ("What is 3*4?", "12"),
        ]
        
        cot_outputs = [
            "Let me add: 5+5",
            "Let me multiply: 3*4",
        ]
        answer_outputs = [
            "Answer: 10",  # Correct
            "Answer: 12",  # Correct
        ]
        
        tokenizer = MockTokenizer(cot_outputs, answer_outputs)
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        accuracy, results, _ = evaluate_model_on_numeric(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=2,
            answer_extraction_method="anchor",
            enable_haiku_metric=False,
        )
        
        assert accuracy == 1.0
        assert len(results) == 2
        assert results[0]["predicted"] == 10
        assert results[1]["predicted"] == 12
    
    def test_numeric_anchor_priority(self, mock_device, basic_hyperparameters):
        """Test that anchor method prioritizes 'Answer:' label."""
        test_data = [
            ("Question", "30"),
        ]
        
        cot_outputs = ["Reasoning"]
        # Multiple numbers, but Answer label should win
        answer_outputs = ["First 10, then 20, Answer: 30"]
        
        tokenizer = MockTokenizer(cot_outputs, answer_outputs)
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        accuracy, results, _ = evaluate_model_on_numeric(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=1,
            answer_extraction_method="anchor",
            enable_haiku_metric=False,
        )
        
        # Should extract 30 (from Answer label), not 10 or 20
        assert results[0]["predicted"] == 30
        assert accuracy == 1.0


class TestDeterministicSampling:
    """Test that evaluation sampling is deterministic."""
    
    def test_deterministic_num_samples(self, mock_device, basic_hyperparameters):
        """Test that num_samples uses deterministic slicing, not random sampling."""
        test_data = [
            ("Q1", "A"),
            ("Q2", "B"),
            ("Q3", "C"),
            ("Q4", "D"),
        ]
        
        cot_outputs = ["R1", "R2", "R3", "R4"]
        answer_outputs = ["A", "B", "C", "D"]
        
        tokenizer = MockTokenizer(cot_outputs, answer_outputs)
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        # Evaluate with num_samples=2 multiple times
        basic_hyperparameters["task_type"] = "aqua"
        results_list = []
        for _ in range(3):
            tokenizer.cot_call_count = 0
            tokenizer.answer_call_count = 0
            
            _, results, _ = evaluate_model_on_mcq(
                actor_model=actor_model,
                critic_model=critic_model,
                tokenizer=tokenizer,
                device=mock_device,
                test_data=test_data,
                hyperparameters=basic_hyperparameters,
                batch_size=2,
                num_samples=2,  # Limit to first 2
                enable_haiku_metric=False,
            )
            results_list.append([r["question"] for r in results])
        
        # All runs should evaluate same first 2 questions (deterministic)
        assert results_list[0] == ["Q1", "Q2"]
        assert results_list[1] == ["Q1", "Q2"]
        assert results_list[2] == ["Q1", "Q2"]


class TestActorVsCriticSelection:
    """Test model selection based on actor_reward_weight."""
    
    def test_critic_used_when_actor_reward_weight_zero(self, mock_device):
        """Critic should generate answers when actor_reward_weight=0.0."""
        hyperparams = {
            "task_type": "test",
            "actor_reward_weight": 0.0,  # Critic baseline
            "markovian": True,
            "cot_length": 50,
            "question_length": 256,
            "temperature": 1.0,
        }
        
        test_data = [("Q", "A")]
        tokenizer = MockTokenizer(["CoT"], ["A"])
        
        actor_model = Mock()
        critic_model = Mock()
        
        # Set up models to track which one generated answers
        actor_model.eval.return_value = actor_model
        critic_model.eval.return_value = critic_model
        
        actor_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        critic_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        
        # Mock construct_prompts to avoid import issues
        with patch('evaluation.construct_prompts', return_value="prompt"):
            _, _, _ = evaluate_model_generic(
                actor_model=actor_model,
                critic_model=critic_model,
                tokenizer=tokenizer,
                device=mock_device,
                test_data=test_data,
                hyperparameters=hyperparams,
                answer_extractor_fn=lambda x: "A",
                batch_size=1,
                enable_haiku_metric=False,
            )
        
        # Critic should be called for answer generation (second generate call)
        assert critic_model.generate.called
    
    def test_actor_used_when_actor_reward_weight_positive(self, mock_device):
        """Actor should generate answers when actor_reward_weight>0."""
        hyperparams = {
            "task_type": "test",
            "actor_reward_weight": 0.5,  # Actor-only mode
            "markovian": True,
            "cot_length": 50,
            "question_length": 256,
            "temperature": 1.0,
        }
        
        test_data = [("Q", "A")]
        tokenizer = MockTokenizer(["CoT"], ["A"])
        
        actor_model = Mock()
        critic_model = Mock()
        
        actor_model.eval.return_value = actor_model
        critic_model.eval.return_value = critic_model
        
        actor_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)
        
        with patch('evaluation.construct_prompts', return_value="prompt"):
            _, _, _ = evaluate_model_generic(
                actor_model=actor_model,
                critic_model=critic_model,
                tokenizer=tokenizer,
                device=mock_device,
                test_data=test_data,
                hyperparameters=hyperparams,
                answer_extractor_fn=lambda x: "A",
                batch_size=1,
                enable_haiku_metric=False,
            )
        
        # Actor should be called twice (CoT + answer)
        assert actor_model.generate.call_count == 2
        # Critic should not be called
        assert not critic_model.generate.called


class TestHaikuIntegration:
    """Test Haiku metric integration (with mocking)."""
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @patch('evaluation.extract_answer')
    def test_haiku_metric_enabled(self, mock_extract, mock_device, basic_hyperparameters):
        """Test that Haiku metric is computed when enabled and API key is set."""
        test_data = [("Q", "10")]
        
        tokenizer = MockTokenizer(["CoT"], ["Answer: 10"])
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        # Mock extract_answer to avoid actual API call
        def extract_side_effect(text, method="simple", answer_format="numeric"):
            if method == "llm":
                return 10  # Mock Haiku extraction
            elif method == "anchor":
                return 10  # Mock anchor extraction
            return 10
        
        mock_extract.side_effect = extract_side_effect
        
        _, _, haiku_metrics = evaluate_model_on_numeric(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=1,
            enable_haiku_metric=True,
        )
        
        # Haiku metrics should be present
        assert haiku_metrics is not None
        assert "accuracy" in haiku_metrics
        assert "cost_usd" in haiku_metrics
        assert "num_calls" in haiku_metrics
        
        # Should have called extract_answer with method="llm"
        llm_calls = [call for call in mock_extract.call_args_list 
                     if call[1].get("method") == "llm"]
        assert len(llm_calls) > 0
    
    @patch.dict(os.environ, {}, clear=True)  # No API key
    def test_haiku_metric_disabled_no_api_key(self, mock_device, basic_hyperparameters):
        """Test that Haiku metric is skipped when API key is not set."""
        test_data = [("Q", "10")]
        
        tokenizer = MockTokenizer(["CoT"], ["Answer: 10"])
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        _, _, haiku_metrics = evaluate_model_on_numeric(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=1,
            enable_haiku_metric=True,  # Enabled but no API key
        )
        
        # Haiku metrics should be None (no API key)
        assert haiku_metrics is None


class TestBackwardCompatibility:
    """Test backward compatibility of result structure."""
    
    def test_result_structure_has_all_fields(self, mock_device, basic_hyperparameters):
        """Test that results contain all expected fields for backward compatibility."""
        test_data = [("Question", "A")]
        
        tokenizer = MockTokenizer(["CoT"], ["Answer: A"])
        actor_model = MockModel([])
        critic_model = MockModel([])
        
        basic_hyperparameters["task_type"] = "aqua"
        _, results, _ = evaluate_model_on_mcq(
            actor_model=actor_model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            device=mock_device,
            test_data=test_data,
            hyperparameters=basic_hyperparameters,
            batch_size=1,
            enable_haiku_metric=False,
        )
        
        result = results[0]
        
        # Check all expected fields
        assert "question" in result
        assert "reasoning" in result
        assert "generated_answer" in result
        assert "predicted" in result
        assert "answer" in result
        assert "gold" in result  # Alias for answer
        assert "correct" in result
        assert "is_correct" in result  # Alias for correct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

