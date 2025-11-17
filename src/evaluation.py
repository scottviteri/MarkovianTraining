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


def extract_answer_simple(answer: str):
    """Extract numerical answer using simple heuristics (original version).
    
    Heuristics:
    1) If '=' exists, extract first number after '='
    2) Else extract first number anywhere in text
    
    This is the original simple version kept for backward compatibility.
    """
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


def extract_answer_with_anchor(answer: str):
    """Extract numerical answer with 'Answer:' anchor priority (sophisticated version).
    
    Heuristics (in order):
    1) If an 'Answer' anchor exists (e.g., 'Answer:' or 'answer'), extract first integer after it
    2) Else if an equals sign exists, extract first integer after '='
    3) Else extract first integer anywhere in text
    
    This version prioritizes the 'Answer:' label which models are trained to use.
    """
    try:
        text = (answer or "").strip()
        # Normalize
        text = text.replace(",", "")
        
        # 1) Anchor on 'Answer' label (case-insensitive), optional colon
        m = re.search(r"(?i)answer\s*:?[\s\-]*", text)
        if m:
            after = text[m.end():]
            m_num = re.search(r"-?\d+", after)
            if m_num:
                return int(m_num.group(0))
        
        # 2) After equals sign
        if "=" in text:
            after_eq = text.split("=", 1)[1]
            m_num = re.search(r"-?\d+", after_eq)
            if m_num:
                return int(m_num.group(0))
        
        # 3) First integer anywhere
        m_num = re.search(r"-?\d+", text)
        if m_num:
            return int(m_num.group(0))
        
        return "[invalid]"
    except Exception:
        return "[invalid]"


def extract_answer_with_llm(answer: str, answer_format: str = "numeric"):
    """Extract answer using Claude Haiku as gold-standard extractor.
    
    This uses an LLM to extract the answer from generated text, serving as a
    gold-standard comparison point for heuristic methods.
    
    Args:
        answer: Generated answer text to extract from
        answer_format: Expected answer format:
            - "numeric": Extract a number
            - "A-D": Extract a letter choice A, B, C, or D
            - "A-E": Extract a letter choice A, B, C, D, or E
            
    Returns:
        Extracted answer or "[invalid]"
        
    Note:
        Requires ANTHROPIC_API_KEY environment variable to be set.
        Falls back to "[invalid]" if API call fails.
    """
    import os
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Fall back to simple extraction if no API key
        return extract_answer_simple(answer) if answer_format == "numeric" else "[invalid]"
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Construct prompt based on answer format
        if answer_format == "numeric":
            system_prompt = """You are an expert at extracting numerical answers from text.
Your task is to identify what numerical answer was provided in the text.
Output ONLY the number, nothing else. If no clear answer is present, output: [invalid]"""
            user_prompt = f"""Extract the numerical answer from this text:

{answer}

Output only the number (e.g., "42" or "-17"). If no clear numerical answer is present, output: [invalid]"""
        
        elif answer_format in ["A-D", "A-E"]:
            max_letter = "D" if answer_format == "A-D" else "E"
            system_prompt = f"""You are an expert at extracting multiple choice answers from text.
Your task is to identify which letter choice (A through {max_letter}) was selected.
Output ONLY the letter, nothing else. If no clear choice is present, output: [invalid]"""
            user_prompt = f"""Extract the letter choice from this text:

{answer}

Output only the letter ({answer_format}). If no clear choice is present, output: [invalid]"""
        
        else:
            raise ValueError(f"Unknown answer_format: {answer_format}")
        
        # Call Claude Haiku
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=20,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        extracted = message.content[0].text.strip()
        
        # Post-process: convert to int for numeric, validate for MCQ
        if answer_format == "numeric":
            if extracted == "[invalid]":
                return extracted
            try:
                # Try to extract just the number if LLM added extra text
                match = re.search(r"-?\d+", extracted)
                if match:
                    return int(match.group(0))
                else:
                    return "[invalid]"
            except:
                return "[invalid]"
        else:
            # For MCQ, validate it's in the expected range
            extracted_upper = extracted.upper()
            if answer_format == "A-D" and extracted_upper in ["A", "B", "C", "D"]:
                return extracted_upper
            elif answer_format == "A-E" and extracted_upper in ["A", "B", "C", "D", "E"]:
                return extracted_upper
            else:
                return "[invalid]" if extracted == "[invalid]" else "X"
    
    except Exception as e:
        # On any error, fall back to simple extraction
        print(f"Warning: LLM extraction failed ({str(e)}), falling back to simple method")
        if answer_format == "numeric":
            return extract_answer_simple(answer)
        else:
            return "[invalid]"


def extract_answer(answer: str, method: str = "simple", answer_format: str = "numeric"):
    """Configurable numerical answer extraction.
    
    Args:
        answer: Generated answer text to extract from
        method: Extraction method to use:
            - "simple": Simple heuristics (check '=', then first number)
            - "anchor": Check 'Answer:' label first, then '=', then first number
            - "llm": Use Claude Haiku as gold-standard extractor
        answer_format: For LLM method only - "numeric", "A-D", or "A-E"
            
    Returns:
        Extracted integer or "[invalid]"
    """
    if method == "simple":
        return extract_answer_simple(answer)
    elif method == "anchor":
        return extract_answer_with_anchor(answer)
    elif method == "llm":
        return extract_answer_with_llm(answer, answer_format=answer_format)
    else:
        raise ValueError(f"Unknown extraction method: {method}. Must be 'simple', 'anchor', or 'llm'")


def compare_extraction_methods(
    actor_model,
    critic_model,
    tokenizer,
    device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    methods: List[str] = None,
    batch_size: int = 16,
    num_samples: int = None,
    answer_format: str = "numeric"
) -> Dict[str, Tuple[float, List[Dict[str, Any]]]]:
    """Compare different answer extraction methods on the same evaluation run.
    
    This runs evaluation once with the model generating answers, then extracts
    results using different methods to compare their performance.
    
    Args:
        actor_model: Actor model for generating CoT
        critic_model: Critic model for generating answers
        tokenizer: Tokenizer
        device: Device
        test_data: List of (question, answer) tuples
        hyperparameters: Hyperparameters dict
        methods: List of extraction methods to compare (default: ["simple", "anchor", "llm"])
        batch_size: Batch size for evaluation
        num_samples: Optional limit on number of samples
        answer_format: Answer format for LLM extraction - "numeric", "A-D", or "A-E"
        
    Returns:
        Dictionary mapping method name to (accuracy, detailed_results) tuple
        
    Note:
        If "llm" is in methods, requires ANTHROPIC_API_KEY environment variable.
    """
    from utils import colored_print, Colors
    import os
    
    if methods is None:
        # Include LLM if API key is available
        methods = ["simple", "anchor"]
        if os.getenv("ANTHROPIC_API_KEY"):
            methods.append("llm")
            colored_print("LLM Method", "ANTHROPIC_API_KEY found - including LLM gold-standard comparison", Colors.CYAN)
        else:
            colored_print("LLM Method", "ANTHROPIC_API_KEY not found - skipping LLM comparison", Colors.YELLOW)
    
    # Limit samples if requested
    if num_samples and num_samples < len(test_data):
        test_data = test_data[:num_samples]
    
    colored_print("Comparison", f"Comparing extraction methods: {', '.join(methods)}", Colors.CYAN)
    
    # Run evaluation once to get generated answers
    # We'll use the generic function but extract with the first method
    accuracy_first, results_first, _ = evaluate_model_on_numeric(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size,
        answer_extraction_method=methods[0]
    )
    
    # Now re-extract and re-score with each method
    comparison_results = {}
    
    for method in methods:
        correct = 0
        method_results = []
        
        colored_print(f"Processing", f"Re-extracting with method: {method}", Colors.CYAN)
        
        # For LLM method, we may want to batch or rate limit
        import time
        
        for idx, result in enumerate(results_first):
            # Re-extract using this method
            generated_answer = result["generated_answer"]
            gold_answer = result["gold"]
            
            # Extract with this method
            extracted = extract_answer(generated_answer, method=method, answer_format=answer_format)
            gold_extracted = extract_answer(gold_answer, method=method, answer_format=answer_format)
            
            # Rate limiting for LLM calls (50 requests per second for Haiku)
            if method == "llm" and idx > 0 and idx % 50 == 0:
                time.sleep(1.1)  # Rate limit: max 50/sec
                colored_print("Rate Limit", f"Processed {idx}/{len(results_first)} with LLM", Colors.CYAN)
            
            is_correct = (extracted == gold_extracted)
            if is_correct:
                correct += 1
            
            # Create result entry for this method
            method_result = {
                "question": result["question"],
                "reasoning": result["reasoning"],
                "generated_answer": generated_answer,
                "predicted": extracted,
                "gold": gold_answer,
                "correct": is_correct,
                "extraction_method": method
            }
            method_results.append(method_result)
        
        accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0
        comparison_results[method] = (accuracy, method_results)
        
        colored_print(f"Method: {method}", f"Accuracy: {accuracy:.2%}", Colors.GREEN if accuracy > 0.5 else Colors.YELLOW)
    
    # Print comparison summary with agreement analysis
    print("\n" + "=" * 70)
    colored_print("Comparison Summary", "Extraction Method Comparison", Colors.BOLD)
    print("=" * 70)
    
    # Show accuracies
    for method in methods:
        accuracy, _ = comparison_results[method]
        marker = "📊" if method == "llm" else "🔧"
        print(f"{marker} {method:12s}: {accuracy:.2%}")
    
    # If LLM is included, show agreement with LLM (gold standard)
    if "llm" in methods and "llm" in comparison_results:
        print("\n" + "-" * 70)
        colored_print("Agreement with LLM Gold Standard", "", Colors.BOLD)
        print("-" * 70)
        
        llm_results = comparison_results["llm"][1]
        llm_predictions = [r["predicted"] for r in llm_results]
        
        for method in methods:
            if method == "llm":
                continue
            
            method_results = comparison_results[method][1]
            method_predictions = [r["predicted"] for r in method_results]
            
            # Calculate agreement
            agreements = sum(1 for m, l in zip(method_predictions, llm_predictions) if m == l)
            agreement_rate = agreements / len(method_predictions) if method_predictions else 0.0
            
            # Calculate where they differ
            disagreements = [(i, m, l) for i, (m, l) in enumerate(zip(method_predictions, llm_predictions)) if m != l]
            
            print(f"{method:12s}: {agreement_rate:.2%} agreement ({len(disagreements)} disagreements)")
            
            # Show a few examples of disagreements
            if disagreements and len(disagreements) <= 5:
                for i, method_pred, llm_pred in disagreements[:3]:
                    print(f"  Example: {method} extracted '{method_pred}' vs LLM '{llm_pred}'")
    
    print("=" * 70 + "\n")
    
    return comparison_results


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
    batch_size: int = 16,
    answer_extraction_method: str = "simple"
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate numeric QA tasks (GSM8K, SVAMP, MATH).
    
    Pipeline:
    1. extract_answer: str -> Union[int, str]  (extracts first number or "[invalid]")
    2. compare_normalized: Union[int, str], str -> bool  (normalizes and compares)
    
    Args:
        answer_extraction_method: Method for extracting numeric answers:
            - "simple": Check '=' then find first number (default, backward compatible)
            - "anchor": Check 'Answer:' label first, then '=', then first number
    """
    # Get method from hyperparameters if not explicitly provided
    if answer_extraction_method == "simple" and "answer_extraction_method" in hyperparameters:
        answer_extraction_method = hyperparameters["answer_extraction_method"]
    
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
    
    # Create extraction function with configured method
    def extract_with_method(answer: str):
        return extract_answer(answer, method=answer_extraction_method)
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters,
        answer_extractor_fn=extract_with_method,  # str -> Union[int, str]
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
    answer_extraction_method="simple",
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
        answer_extraction_method: Method for extracting numeric answers:
            - "simple": Check '=' then find first number (default, backward compatible)
            - "anchor": Check 'Answer:' label first, then '=', then first number
    """
    from utils import construct_baseline_prompts
    
    # Get method from hyperparameters if not explicitly provided
    if answer_extraction_method == "simple" and "answer_extraction_method" in hyperparameters:
        answer_extraction_method = hyperparameters["answer_extraction_method"]
    
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
        extracted_answers = [extract_answer(ans, method=answer_extraction_method) for ans in generated_answers]
        
        # Check correctness
        for q, a, cot, gen_a, ext_a in zip(questions, answers, cot_texts, generated_answers, extracted_answers):
            correct_answer = extract_answer(a, method=answer_extraction_method)
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


# ============================================================================
# Batch Evaluation Functions
# ============================================================================

def find_checkpoints(results_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint directories in results_dir.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of (batch_index, checkpoint_path) tuples, sorted by batch_index
    """
    import glob
    
    checkpoint_dirs = glob.glob(os.path.join(results_dir, "adapter_*"))
    
    # Extract batch indices from directory names
    checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        match = re.search(r"adapter_(\d+)", os.path.basename(ckpt_dir))
        if match:
            batch_idx = int(match.group(1))
            checkpoints.append((batch_idx, ckpt_dir))
    
    # Sort by batch index
    checkpoints.sort(key=lambda x: x[0])
    
    return checkpoints


def backup_existing_results(results_dir: str) -> List[str]:
    """Backup existing evaluation result files.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of backup file paths created
    """
    import glob
    import shutil
    from utils import colored_print, Colors
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_files = glob.glob(os.path.join(results_dir, "*_results_*.jsonl"))
    
    backups = []
    for result_file in result_files:
        base_name = os.path.basename(result_file)
        
        # Skip files that are already backups
        if "_backup_" in base_name:
            colored_print("Backup", f"Skipping already-backed-up file: {base_name}", Colors.CYAN)
            continue
        
        # Create backup filename
        backup_name = base_name.replace(".jsonl", f"_backup_{timestamp}.jsonl")
        backup_path = os.path.join(results_dir, backup_name)
        
        # Move file to backup
        shutil.move(result_file, backup_path)
        backups.append(backup_path)
        colored_print("Backup", f"Created backup: {backup_name}", Colors.YELLOW)
    
    return backups


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Unified CLI for all evaluation tasks."""
    import argparse
    from datasets import load_dataset
    
    parser = argparse.ArgumentParser(
        description="Unified evaluation CLI for all tasks (GSM8K, MMLU, ARC, SVAMP, AQuA, MathQA, Arithmetic)"
    )
    
    # Core arguments
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic", "arithmetic-negative"],
        help="Task to evaluate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model weights (default: latest result or base model)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"],
        default=None,
        help="Model type (default: infer from model path)"
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use base model without adapters"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: 1.5x training batch size)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every nth example"
    )
    
    # Generation parameters
    parser.add_argument(
        "--cot_length",
        type=int,
        default=None,
        help="Override CoT length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override temperature"
    )
    
    # Answer extraction
    parser.add_argument(
        "--answer_extraction_method",
        type=str,
        choices=["simple", "anchor", "llm"],
        default="simple",
        help="Answer extraction method for numeric tasks (default: simple)"
    )
    
    # Checkpoint selection
    parser.add_argument(
        "--training_index",
        type=int,
        default=None,
        help="Specific training index to evaluate"
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints in the directory"
    )
    parser.add_argument(
        "--all_adapters",
        action="store_true",
        help="Evaluate all adapter_* directories in run directory"
    )
    
    # Baseline mode
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use standard baseline prompting"
    )
    parser.add_argument(
        "--baseline_thinking_tokens",
        type=int,
        default=None,
        help="Max tokens for baseline thinking stage"
    )
    parser.add_argument(
        "--baseline_temperature",
        type=float,
        default=None,
        help="Temperature for baseline thinking generation"
    )
    
    # Task-specific arguments
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing adapter_* folders (for --all_adapters)"
    )
    parser.add_argument(
        "--arc_subset",
        type=str,
        choices=["ARC-Challenge", "ARC-Easy"],
        default=None,
        help="ARC subset to use (default: from env ARC_SUBSET or ARC-Challenge)"
    )
    parser.add_argument(
        "--mmlu_subject",
        type=str,
        default=None,
        help="Specific MMLU subject to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Import utilities
    from utils import (
        find_latest_result,
        get_hyperparameters_from_log,
        get_model_paths_and_type,
        load_model_for_evaluation,
        load_svamp_dataset,
        load_aqua_dataset,
        load_arc_dataset,
        load_mathqa_dataset,
        generate_question_answer_batches,
        load_gsm8k_dataset,
    )
    
    # Determine model paths
    if args.all_adapters:
        # Find all adapter directories in run_dir
        run_dir = args.run_dir or args.model_path or find_latest_result()
        if not run_dir:
            raise FileNotFoundError("No results directory found")
        if not os.path.isdir(run_dir):
            run_dir = os.path.dirname(run_dir)
        
        checkpoints = find_checkpoints(run_dir)
        if not checkpoints:
            print(f"No adapter directories found in {run_dir}")
            return
        
        model_paths = [ckpt_path for _, ckpt_path in checkpoints]
        
        # Infer model type from run_dir log
        try:
            hyperparameters_base = get_hyperparameters_from_log(run_dir, default_task=args.task_type)
            inferred_model_type = hyperparameters_base.get("model_type", "mistral")
        except Exception:
            inferred_model_type = "mistral"
        
        if args.model_type is None:
            args.model_type = inferred_model_type
            print(f"Inferred model type: {args.model_type}")
    elif not args.use_base_model:
        model_paths, inferred_model_type = get_model_paths_and_type(
            args.model_path, args.training_index, args.all_checkpoints
        )
        if args.model_type is None:
            args.model_type = inferred_model_type
            print(f"Inferred model type: {args.model_type}")
    else:
        model_paths = [None]
        args.model_type = args.model_type or "mistral"
    
    # Process each checkpoint
    for checkpoint_path in model_paths:
        if checkpoint_path:
            print(f"\nEvaluating checkpoint: {checkpoint_path}")
            run_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
            # If adapter dir, its parent is the run directory
            if os.path.basename(run_dir).startswith("adapter_"):
                run_dir = os.path.dirname(run_dir)
            
            hyperparameters = get_hyperparameters_from_log(run_dir, default_task=args.task_type)
            hyperparameters["task_type"] = args.task_type
            
            # Override hyperparameters if provided
            if args.cot_length is not None:
                hyperparameters["cot_length"] = args.cot_length
            if args.temperature is not None:
                hyperparameters["temperature"] = args.temperature
            if args.answer_extraction_method != "simple":
                hyperparameters["answer_extraction_method"] = args.answer_extraction_method
            if args.mmlu_subject:
                hyperparameters["mmlu_subject"] = args.mmlu_subject
                
            # Extract batch index for results
            if checkpoint_path:
                basename = os.path.basename(checkpoint_path)
                match = re.search(r'adapter_(\d+)', basename)
                if match:
                    batch_index = int(match.group(1))
                else:
                    batch_index = None
            else:
                batch_index = None
        else:
            # Base model evaluation
            hyperparameters = {
                "model_type": args.model_type,
                "task_type": args.task_type,
                "cot_length": args.cot_length or 150,
                "temperature": args.temperature or 1.0,
                "batch_size": 12,
                "markovian": True,
            }
            if args.answer_extraction_method != "simple":
                hyperparameters["answer_extraction_method"] = args.answer_extraction_method
            if args.mmlu_subject:
                hyperparameters["mmlu_subject"] = args.mmlu_subject
            batch_index = None
        
        # Load models
        actor_model, critic_model, tokenizer, device = load_model_for_evaluation(
            checkpoint_path,
            args.use_base_model,
            args.model_type
        )
        
        # Load dataset based on task type
        if args.task_type == "gsm8k":
            test_data = list(load_gsm8k_dataset(split="test"))
        elif args.task_type == "mmlu":
            subject = hyperparameters.get("mmlu_subject", None)
            split = hyperparameters.get("mmlu_split", "validation")
            ds = load_dataset("cais/mmlu", subject if subject else "all")
            split_name = "test" if split == "test" else "validation"
            d = ds[split_name]
            
            test_data = []
            for ex in d:
                stem = ex["question"]
                choices = ex["choices"] if "choices" in ex else ex.get("options", [])
                if not choices or len(choices) < 4:
                    continue
                options_text = "\n".join([
                    f"A. {choices[0]}",
                    f"B. {choices[1]}",
                    f"C. {choices[2]}",
                    f"D. {choices[3]}",
                ])
                question_text = f"{stem}\n\nOptions:\n{options_text}"
                if "answer" in ex and isinstance(ex["answer"], int):
                    correct_letter = ["A", "B", "C", "D"][ex["answer"]]
                else:
                    correct_letter = str(ex.get("answer", "")).strip().upper()
                    if correct_letter not in ["A", "B", "C", "D"]:
                        continue
                test_data.append((question_text, correct_letter))
        elif args.task_type == "arc":
            subset = args.arc_subset or os.getenv("ARC_SUBSET", "ARC-Challenge")
            test_data = list(load_arc_dataset(split="validation", subset=subset))
        elif args.task_type == "svamp":
            test_data = list(load_svamp_dataset(split="test"))
        elif args.task_type == "aqua":
            test_data = list(load_aqua_dataset(split="test"))
        elif args.task_type == "mathqa":
            test_data = list(load_mathqa_dataset(split="test"))
            if not test_data:
                raise FileNotFoundError("No MathQA test data found. Set MATHQA_PATH or use HuggingFace dataset.")
        elif args.task_type in ["arithmetic", "arithmetic-negative"]:
            batch_size_gen = args.num_samples or 200
            batch = next(
                generate_question_answer_batches(
                    num_batches=1,
                    batch_size=batch_size_gen,
                    task_type=args.task_type,
                    tokenizer=None,
                    hyperparameters={},
                )
            )
            test_data = batch
        else:
            raise ValueError(f"Unsupported task type: {args.task_type}")
        
        # Apply stride if specified
        if args.stride > 1:
            test_data = test_data[::args.stride]
            print(f"Using stride={args.stride}, evaluating on {len(test_data)} examples")
        
        # Determine eval batch size
        eval_bs = args.batch_size if args.batch_size is not None else get_default_eval_batch_size(
            hyperparameters.get("batch_size", 12)
        )
        
        # Run evaluation based on task type
        if args.task_type == "gsm8k":
            accuracy, results = evaluate_model_on_gsm8k(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                num_samples=args.num_samples,
                batch_size=eval_bs,
                baseline_mode=args.baseline,
                baseline_thinking_tokens=args.baseline_thinking_tokens,
                baseline_temperature=args.baseline_temperature,
                answer_extraction_method=args.answer_extraction_method,
            )
        elif args.task_type == "mmlu":
            accuracy, results, accuracy_wb = evaluate_model_on_mmlu(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("MMLU Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "arc":
            accuracy, results, accuracy_wb = evaluate_model_on_arc(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("ARC Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "aqua":
            accuracy, results, accuracy_wb = evaluate_model_on_aqua(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("AQuA Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "mathqa":
            accuracy, results, accuracy_wb = evaluate_model_on_mathqa(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("MathQA Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type in ["svamp", "arithmetic", "arithmetic-negative"]:
            accuracy, results, _ = evaluate_model_on_numeric(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                answer_extraction_method=args.answer_extraction_method,
            )
        else:
            raise ValueError(f"Unsupported task type: {args.task_type}")
        
        # Print results
        colored_print(f"{args.task_type.upper()} Accuracy", f"{accuracy:.2%}", Colors.GREEN if accuracy > 0.5 else Colors.YELLOW)
        
        # Save results
        model_dir = os.path.dirname(checkpoint_path) if checkpoint_path else f"results/{args.task_type}"
        if checkpoint_path and os.path.isdir(checkpoint_path):
            # If checkpoint_path is a directory (adapter_*), use its parent
            model_dir = os.path.dirname(checkpoint_path)
        os.makedirs(model_dir, exist_ok=True)
        
        if args.task_type == "gsm8k":
            results_file = save_results(
                model_dir, checkpoint_path, args.model_type,
                accuracy, results, args.num_samples,
                batch_index_override=(0 if args.baseline else batch_index)
            )
        elif args.task_type == "mmlu":
            results_file = save_results_mmlu(
                model_dir, checkpoint_path, args.model_type,
                accuracy, results, len(test_data),
                subject=args.mmlu_subject,
                batch_index_override=(0 if args.baseline else batch_index)
            )
        else:
            # Generic save for other tasks
            entry = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "batch_index": (0 if args.baseline else batch_index),
                "accuracy": accuracy,
                "model_path": checkpoint_path,
                "model_type": args.model_type,
                "task_type": args.task_type,
                "num_samples": args.num_samples or len(test_data),
                "detailed_results": results
            }
            results_file = os.path.join(model_dir, f"{args.task_type}_results_{args.model_type}.jsonl")
            with open(results_file, "a") as f:
                json.dump(entry, f)
                f.write("\n")
        
        colored_print("Results", f"Appended to {results_file}", Colors.CYAN)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        exit(0)

