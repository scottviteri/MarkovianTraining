"""
Unified evaluation module for all tasks.

This module consolidates evaluation logic from train.py and evaluate_gsm8k.py
to provide a single source of truth for model evaluation across all datasets.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional, Callable
import re
import json
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from utils import construct_prompts, colored_print, Colors


# ============================================================================
# Helper Functions
# ============================================================================

def get_default_eval_batch_size(train_batch_size: int) -> int:
    """Default evaluation batch size: floor(1.5x train batch size)."""
    return max(1, int(train_batch_size * 1.5))


def extract_answer(answer):
    """Extract numerical answer from various text formats."""
    if "=" in answer:
        answer = answer.split("=")[-1].strip()
    answer = answer.replace(",", "")
    try:
        matches = re.findall(r"-?\d+", answer.strip())
        if matches:
            answer = int(matches[0])
        else:
            answer = "[invalid]"
    except:
        answer = "[invalid]"
    return answer


def extract_letter(text: str) -> str:
    """Extract first letter A-E from text. Returns 'X' if none found.
    
    NOTE: This is the buggy version without word boundaries.
    Kept for backward compatibility with existing evaluation results.
    Use extract_letter_word_boundary for correct extraction.
    """
    matches = re.findall(r"[A-E]", text.upper())
    return matches[0] if matches else "X"


def extract_letter_word_boundary(text: str) -> str:
    """Extract first letter A-E with word boundaries. Returns 'X' if none found.
    
    This correctly extracts the intended choice letter, avoiding false matches
    in common words like 'The' (contains E) or 'Select' (contains E).
    """
    # Try uppercase A-E with word boundary
    match = re.search(r"\b([A-E])\b", text.upper())
    if match:
        return match.group(1)
    # Try lowercase a-e with word boundary
    match = re.search(r"\b([a-e])\b", text)
    if match:
        return match.group(1).upper()
    return "X"


# ============================================================================
# Core Generic Evaluation Function
# ============================================================================

def evaluate_model_generic(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    answer_extractor_fn: Callable[[str], Any],
    answer_comparator_fn: Optional[Callable[[Any, str], bool]] = None,
    answer_extractor_fn_wb: Optional[Callable[[str], Any]] = None,
    batch_size: int = 16,
    num_samples: Optional[int] = None,
    task_name: str = "Task",
    max_answer_tokens: int = 10
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Generic evaluation function for all task types using critic model for answer generation.
    
    This follows the Markovian framework:
    1. Actor generates CoT at training temperature
    2. Critic generates answer deterministically (temp 0) after "Answer:"
    3. Extract and compare answers using task-specific functions
    
    Args:
        actor_model: The model to evaluate
        critic_model: The critic model (frozen)
        tokenizer: Tokenizer for the models
        device: Device to run on
        test_data: List of (question: str, gold_answer: str) tuples
        hyperparameters: Training hyperparameters
        answer_extractor_fn: Function to extract predicted answer from generated text
                             Signature: (generated_text: str) -> extracted_answer: Any
        answer_comparator_fn: Optional function to compare predicted vs gold answers
                             Signature: (extracted_pred: Any, gold_answer: str) -> bool
                             If None, uses simple equality (predicted == gold)
        answer_extractor_fn_wb: Optional word-boundary extractor for dual metrics (MCQ only)
        batch_size: Evaluation batch size
        num_samples: Optional limit on number of samples to evaluate
        task_name: Name for progress bar
        max_answer_tokens: Maximum tokens to generate for answer
        
    Returns:
        tuple: (accuracy: float, detailed_results: List[Dict], accuracy_wb: Optional[float])
    """
    # Limit number of samples if specified
    if num_samples and num_samples < len(test_data):
        test_data = random.sample(test_data, num_samples)
    
    # Default comparator is simple equality
    if answer_comparator_fn is None:
        answer_comparator_fn = lambda pred, gold: pred == gold
    
    actor_model.eval()
    
    correct = 0
    correct_wb = 0  # Track word boundary accuracy
    results = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Evaluating {task_name}"):
        batch = test_data[i:i+batch_size]
        questions, answers = zip(*batch)
        
        # Stage 1: Generate CoT with actor model
        reasoning_prompts = [
            construct_prompts(question=q, hyperparameters=hyperparameters)
            for q in questions
        ]
        
        inputs = tokenizer(
            reasoning_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=hyperparameters.get("question_length", 512)
        ).to(device)
        
        with torch.no_grad():
            reasoning_outputs = actor_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=hyperparameters.get("cot_length", 100),
                min_new_tokens=hyperparameters.get("cot_length", 100),
                do_sample=True,
                temperature=hyperparameters.get("temperature", 1.0),
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        cot_texts = tokenizer.batch_decode(
            reasoning_outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Stage 2: Generate answer with appropriate model (deterministic)
        # Use actor if actor_reward_weight > 0 (actor was trained to generate answers)
        # Use critic otherwise (standard Markovian baseline)
        actor_reward_weight = hyperparameters.get("actor_reward_weight", 0.0)
        answer_model = actor_model if actor_reward_weight > 0 else critic_model
        
        include_question_in_eval = not hyperparameters.get("markovian", True)
        answer_prompts = [
            construct_prompts(
                question=q,
                hyperparameters=hyperparameters,
                reasoning=cot,
                include_question=include_question_in_eval
            )
            for q, cot in zip(questions, cot_texts)
        ]
        
        answer_inputs = tokenizer(
            answer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=hyperparameters.get("question_length", 512) + hyperparameters.get("cot_length", 100)
        ).to(device)
        
        with torch.no_grad():
            answer_outputs = answer_model.generate(
                input_ids=answer_inputs.input_ids,
                attention_mask=answer_inputs.attention_mask,
                max_new_tokens=max_answer_tokens,
                do_sample=False,  # Deterministic
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract predicted answers using task-specific function
        # Type: generated_answers is List[str] (raw text from model)
        # Type: predicted_answers is List[Any] (extracted answers, e.g., int for SVAMP, str for MMLU)
        predicted_answers = [answer_extractor_fn(ans) for ans in generated_answers]
        
        # Word boundary extraction if provided
        if answer_extractor_fn_wb is not None:
            predicted_answers_wb = [answer_extractor_fn_wb(ans) for ans in generated_answers]
        else:
            predicted_answers_wb = [None] * len(generated_answers)
        
        # Calculate accuracy using task-specific comparator
        for q, cot, gen_ans, pred, pred_wb, gold in zip(questions, cot_texts, generated_answers, predicted_answers, predicted_answers_wb, answers):
            is_correct = answer_comparator_fn(pred, gold)
            if is_correct:
                correct += 1
            
            # Word boundary correctness (if applicable)
            if pred_wb is not None:
                is_correct_wb = answer_comparator_fn(pred_wb, gold)  # Use same comparator
                if is_correct_wb:
                    correct_wb += 1
            else:
                is_correct_wb = None
                
            # Use both "is_correct" and "correct" for backwards compatibility
            results.append({
                "question": q,
                "reasoning": cot,
                "generated_answer": gen_ans,
                "predicted": pred,
                "predicted_wb": pred_wb,  # Word boundary prediction
                "answer": gold,
                "gold": gold,  # Alias for backwards compatibility
                "correct": is_correct,
                "correct_wb": is_correct_wb,  # Word boundary correctness
                "is_correct": is_correct,  # Alias for backwards compatibility
            })
    
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0
    accuracy_wb = correct_wb / len(test_data) if answer_extractor_fn_wb and len(test_data) > 0 else None
    return accuracy, results, accuracy_wb


# ============================================================================
# Task-Specific Evaluation Functions
# ============================================================================

def evaluate_model_on_mcq(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None,
    num_choices: int = 4,
    task_name: str = "MCQ"
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Generic MCQ evaluation for any number of choices.
    
    Args:
        num_choices: 4 for A-D (MMLU, ARC), 5 for A-E (AQuA, MathQA)
        task_name: Name for progress bar
    
    Returns both original and word-boundary-based accuracy for comparison.
    """
    choice_letter = chr(64 + num_choices)  # D for 4, E for 5
    
    def extract_letter(text: str) -> str:
        """Extract first letter A-{choice_letter} from text. Returns 'X' if none found."""
        pattern = f"[A-{choice_letter}]"
        matches = re.findall(pattern, text.upper())
        return matches[0] if matches else "X"
    
    def extract_letter_wb(text: str) -> str:
        """Extract first letter with word boundaries."""
        # Try uppercase A-{choice_letter} with word boundary
        pattern = f"\\b([A-{choice_letter}])\\b"
        match = re.search(pattern, text.upper())
        if match:
            return match.group(1)
        # Try lowercase a-{choice_letter} with word boundary
        pattern_lower = f"\\b([a-{choice_letter.lower()}])\\b"
        match = re.search(pattern_lower, text)
        if match:
            return match.group(1).upper()
        return "X"
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, extract_letter,
        answer_extractor_fn_wb=extract_letter_wb,
        batch_size=batch_size, num_samples=num_samples,
        task_name=task_name, max_answer_tokens=10
    )


def evaluate_model_on_mmlu(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: int = 500
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate MMLU - 4-choice MCQ (A-D).
    
    Returns both original and word-boundary-based accuracy for comparison.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=4, task_name="MMLU"
    )


def evaluate_model_on_arc(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate ARC - 4-choice MCQ (A-D).
    
    Returns both original and word-boundary-based accuracy for comparison.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=4, task_name="ARC"
    )


def evaluate_model_on_aqua(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate AQuA - 5-choice MCQ (A-E).
    
    Returns both original and word-boundary-based accuracy for comparison.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=5, task_name="AQuA"
    )


def evaluate_model_on_mathqa(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate MathQA - 5-choice MCQ (A-E).
    
    Returns both original and word-boundary-based accuracy for comparison.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=5, task_name="MathQA"
    )


def evaluate_model_on_numeric(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate numeric QA tasks (GSM8K, SVAMP, MATH).
    
    Pipeline:
    1. extract_answer: str -> Union[int, str]  (extracts first number or "[invalid]")
    2. compare_normalized: Union[int, str], str -> bool  (normalizes and compares)
    """
    def normalize_numeric(text: str) -> str:
        """Normalize numeric text by removing LaTeX, whitespace, etc."""
        s = text.strip()
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        s = s.replace("$", "").replace("\\", "")
        s = re.sub(r"\s+", "", s)
        return s
    
    def compare_normalized(pred: Any, gold: str) -> bool:
        """Compare extracted prediction with gold answer after normalization.
        
        Args:
            pred: Extracted answer from extract_answer (int or "[invalid]")
            gold: Gold answer string from dataset
        """
        return normalize_numeric(str(pred)) == normalize_numeric(str(gold))
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters,
        answer_extractor_fn=extract_answer,  # str -> Union[int, str]
        answer_comparator_fn=compare_normalized,  # Union[int, str], str -> bool
        batch_size=batch_size,
        task_name="Numeric", max_answer_tokens=16
    )


def evaluate_model_on_gsm8k(
    actor_model,  
    critic_model, 
    tokenizer,
    device,
    test_data,
    hyperparameters,
    num_samples=None,
    batch_size=None,
    baseline_mode=False,
    baseline_thinking_tokens=None,
    baseline_temperature=None,
):
    """Evaluate model on GSM8K test set.
    
    Args:
        actor_model: Model for generating reasoning (with temperature) or baseline thinking
        critic_model: Frozen model for generating answers (deterministic)
        tokenizer: Tokenizer for both models
        device: torch device
        test_data: List of (question, answer) tuples
        hyperparameters: Configuration dictionary
        num_samples: Optional limit on number of samples to evaluate
        batch_size: Batch size for evaluation
        baseline_mode: If True, use standard baseline prompting
        baseline_thinking_tokens: Max thinking tokens for baseline (caps new tokens for stage 1)
        baseline_temperature: Temperature for baseline thinking generation
    """
    from utils import construct_baseline_prompts
    
    # Determine default eval batch size: floor(1.5x) of training batch size (defaults to 8 if absent)
    if batch_size is None:
        try:
            batch_size = max(1, int(hyperparameters.get("batch_size", 8) * 1.5))
        except Exception:
            batch_size = 12
    if num_samples:
        test_data = test_data[:num_samples]
    
    all_results = []
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        questions, answers = zip(*batch)
        
        # Create prompts for stage 1 (reasoning/thinking)
        if baseline_mode:
            prompts = [
                construct_baseline_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                )
                for q in questions
            ]
        else:
            prompts = [
                construct_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                )
                for q in questions
            ]
        
        # Tokenize
        tokenized_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        # 1. Generate CoT/Thinking using actor model (with temperature)
        with torch.no_grad():
            cot_outputs = actor_model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=(baseline_thinking_tokens if baseline_mode and baseline_thinking_tokens is not None else hyperparameters["cot_length"]),
                min_new_tokens=(baseline_thinking_tokens if baseline_mode and baseline_thinking_tokens is not None else hyperparameters["cot_length"]),
                do_sample=True,
                temperature=(baseline_temperature if baseline_mode and baseline_temperature is not None else hyperparameters["temperature"]),
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode CoT
        cot_texts = tokenizer.batch_decode(
            cot_outputs[:, tokenized_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 2. Generate answers using critic model (deterministic)
        # Honor Markovian flag: include question context when markovian=False
        include_question_in_eval = not hyperparameters.get("markovian", True)
        if baseline_mode:
            answer_prompts = [
                construct_baseline_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                    reasoning=r,
                    max_thinking_tokens=baseline_thinking_tokens,
                )
                for q, r in zip(questions, cot_texts)
            ]
        else:
            answer_prompts = [
                construct_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                    reasoning=r,
                    include_question=include_question_in_eval,
                )
                for q, r in zip(questions, cot_texts)
            ]
        
        tokenized_answer_inputs = tokenizer(
            answer_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            answer_outputs = critic_model.generate(
                input_ids=tokenized_answer_inputs.input_ids,
                attention_mask=tokenized_answer_inputs.attention_mask,
                max_new_tokens=10,  # Changed from 15 to 10 to match training
                do_sample=False,    # Deterministic
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and extract answers
        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, tokenized_answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        extracted_answers = [extract_answer(ans) for ans in generated_answers]
        
        # Check correctness
        for q, a, cot, gen_a, ext_a in zip(questions, answers, cot_texts, generated_answers, extracted_answers):
            correct_answer = extract_answer(a)
            is_correct = (ext_a == correct_answer)
            
            result = {
                "question": q,
                "correct_answer": correct_answer,
                "chain_of_thought": cot,
                "generated_answer": gen_a,
                "extracted_answer": ext_a,
                "is_correct": is_correct,
            }
            all_results.append(result)
            
            if is_correct:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, all_results


# ============================================================================
# Save/Plotting Functions
# ============================================================================

def plot_accuracy_over_batches(results_jsonl_path: str, save_path: str):
    """Plot accuracy vs. batch index from accumulated JSONL results and save one combined image.
    
    This generates a single plot across all recorded evaluations without smoothing.
    """
    if not os.path.exists(results_jsonl_path):
        return
    batch_to_entry = {}
    with open(results_jsonl_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            batch_idx = entry.get("batch_index")
            acc = entry.get("accuracy")
            if batch_idx is None or acc is None:
                continue
            # Keep the latest entry per batch index
            batch_to_entry[batch_idx] = acc
    if not batch_to_entry:
        return
    batch_indices = sorted(batch_to_entry.keys())
    accuracies = [batch_to_entry[i] for i in batch_indices]
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, accuracies, marker='o', linestyle='-', color='tab:blue')
    plt.title('GSM8K Accuracy vs Training Batch')
    plt.xlabel('Training Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(model_dir, checkpoint_path, model_type, accuracy, results, num_samples, batch_index_override=None):
    """Save results to file and generate plots for GSM8K.
    
    Args:
        model_dir: Directory where outputs should be written. Prefer a run directory (e.g., results/gsm8k/<timestamp>).
        checkpoint_path: Optional path to checkpoint file for inferring batch index.
        model_type: Model type string used in filenames.
        accuracy: Final accuracy value.
        results: Per-example evaluation results.
        num_samples: Optional number of evaluated samples.
        batch_index_override: If provided, force batch index in filenames/metadata (e.g., 0 for baseline).
    """
    # Create results entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": None,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,  # This is the critic model type
        "num_samples": num_samples,
        "detailed_results": results
    }
    
    # Determine batch index: prefer explicit override, then infer from checkpoint path
    if batch_index_override is not None:
        entry["batch_index"] = int(batch_index_override)
    elif checkpoint_path:
        basename = os.path.basename(checkpoint_path)
        # Support both new and old filename formats
        match = re.search(r'model_batch_(\d+)\.pt$', basename)
        if not match:
            match = re.search(r'model_(\d+)_', basename)
        if match:
            entry["batch_index"] = int(match.group(1))
    
    # Include model type in filenames
    model_type_suffix = f"_{model_type}"
    
    # Save JSONL results with model type in filename
    results_file = os.path.join(model_dir, f"gsm8k_results{model_type_suffix}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    
    # Update accuracy-over-batches plot in the run directory
    accuracy_plot_path = os.path.join(model_dir, "gsm8k_accuracy_over_batches.png")
    plot_accuracy_over_batches(results_file, accuracy_plot_path)
    print(f"Updated GSM8K accuracy plot at {accuracy_plot_path}")
    
    return results_file


def save_results_mmlu(output_dir, model_path, model_type, accuracy, results, total, subject=None, batch_index_override=None):
    """Save MMLU evaluation results to JSONL file.
    
    Args:
        output_dir: Directory where outputs should be written
        model_path: Path to the model checkpoint
        model_type: Model type string
        accuracy: Final accuracy value
        results: Per-example evaluation results
        total: Total number of examples evaluated
        subject: Optional MMLU subject filter
        batch_index_override: Optional explicit batch index
    """
    # Create results entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": None,
        "accuracy": accuracy,
        "model_path": model_path,
        "model_type": model_type,
        "subject": subject,
        "total_examples": total,
        "results": results
    }
    
    # Determine batch index from checkpoint path or use override
    if batch_index_override is not None:
        entry["batch_index"] = int(batch_index_override)
    elif model_path:
        basename = os.path.basename(model_path)
        match = re.search(r'adapter_(\d+)', basename)
        if match:
            entry["batch_index"] = int(match.group(1))
    
    # Save results to JSONL
    results_file = os.path.join(output_dir, f"mmlu_results_{model_type}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    
    return results_file

