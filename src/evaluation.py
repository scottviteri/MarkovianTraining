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
import glob
import shutil
import datetime
import random
import filecmp
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import construct_prompts, colored_print, Colors


# Track which results files have already been reset during the current process
_fresh_results_files: set[str] = set()


def _get_latest_backup_path(results_file: str) -> Optional[str]:
    """Return the most recent backup file for a results file, if any."""
    directory = os.path.dirname(results_file)
    base_name = os.path.basename(results_file)
    pattern = os.path.join(directory, base_name.replace(".jsonl", "_backup_*.jsonl"))
    backups = glob.glob(pattern)
    if not backups:
        return None
    return max(backups, key=os.path.getmtime)


def _maybe_backup_results_file(results_file: str) -> Optional[str]:
    """Create a backup of an existing results file if needed.
    
    A new backup is written only when the current contents differ from the most
    recent backup. The function returns the path to the backup that was created,
    or None if no backup was required.
    """
    if not os.path.exists(results_file) or os.path.getsize(results_file) == 0:
        return None
    
    latest_backup = _get_latest_backup_path(results_file)
    if latest_backup and filecmp.cmp(results_file, latest_backup, shallow=False):
        # Current contents match the latest backup; no need to duplicate it.
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = os.path.dirname(results_file)
    base_name = os.path.basename(results_file)
    backup_name = base_name.replace(".jsonl", f"_backup_{timestamp}.jsonl")
    backup_path = os.path.join(directory, backup_name)
    shutil.copy2(results_file, backup_path)
    return backup_path


def _ensure_fresh_results_file(results_file: str):
    """Truncate the results file once per process, creating backups when needed."""
    if results_file in _fresh_results_files:
        return
    
    if os.path.exists(results_file):
        backup_path = _maybe_backup_results_file(results_file)
        if backup_path:
            colored_print("Backup", f"Created backup: {os.path.basename(backup_path)}", Colors.YELLOW)
        # Start with a fresh file for the new evaluation run
        open(results_file, "w").close()
    _fresh_results_files.add(results_file)


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


def get_answer_format_for_task(task_type: str) -> Optional[str]:
    """Return answer format string for Haiku extraction."""
    numeric_tasks = {"gsm8k", "svamp", "math", "arithmetic"}
    mcq_ad = {"mmlu", "arc"}
    mcq_ae = {"aqua", "mathqa"}
    
    if task_type in numeric_tasks:
        return "numeric"
    if task_type in mcq_ad:
        return "A-D"
    if task_type in mcq_ae:
        return "A-E"
    return None


def compute_haiku_accuracy(
    results: List[Dict[str, Any]],
    task_type: str,
    answer_format: str,
) -> Optional[float]:
    """Re-compute accuracy using Haiku extraction."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        colored_print("Haiku Metric", "ANTHROPIC_API_KEY not set; skipping Haiku metric", Colors.YELLOW)
        return None
    
    total = 0
    correct = 0
    
    for entry in results:
        generated_text = entry.get("generated_answer") or entry.get("answer") or ""
        gold_raw = entry.get("answer") or entry.get("gold")
        if gold_raw is None:
            continue
        
        haiku_pred = extract_answer(generated_text, method="llm", answer_format=answer_format)
        
        if answer_format == "numeric":
            gold_value = extract_answer(str(gold_raw), method="anchor")
            haiku_gold = gold_value
        else:
            haiku_gold = str(gold_raw).strip().upper()[:1]
        
        total += 1
        if haiku_pred == haiku_gold:
            correct += 1
    
    if total == 0:
        return None
    return correct / total


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
    max_answer_tokens: int = 10,
    enable_haiku_metric: bool = True,
    answer_format: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Generic evaluation function for all task types using critic model for answer generation.
    
    This follows the Markovian framework:
    1. Actor generates CoT at training temperature
    2. Critic generates answer deterministically (temp 0) after "Answer:"
    3. Extract and compare answers using task-specific functions
    
    Model Selection (Actor vs Critic):
    - Standard Markovian (actor_reward_weight = 0): 
      * Actor generates CoT reasoning
      * Critic generates final answer (frozen, provides stable grading)
      * Evaluation: CoT quality drives reward signal
    - Actor-only mode (actor_reward_weight > 0):
      * Actor generates both CoT and answer
      * Actor receives reward for answer correctness
      * Evaluation: Actor must learn complete reasoning + answering
    
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
        enable_haiku_metric: If True and ANTHROPIC_API_KEY is set, compute Haiku extraction metric
        answer_format: Answer format for Haiku extraction ("numeric", "A-D", "A-E", or None to infer)
        
    Returns:
        tuple: (accuracy: float, detailed_results: List[Dict], accuracy_wb: Optional[float], 
                haiku_metrics: Optional[Dict])
        - accuracy: Primary accuracy metric
        - detailed_results: Per-example results with predictions and correctness
        - accuracy_wb: Word-boundary accuracy (MCQ tasks) or None
        - haiku_metrics: Dict with {"accuracy": float, "cost_usd": float, "num_calls": int} or None
    """
    # Limit number of samples if specified (deterministic)
    if num_samples and num_samples < len(test_data):
        test_data = test_data[:num_samples]
    
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
    
    # Haiku extraction metric (optional, requires ANTHROPIC_API_KEY)
    haiku_metrics = None
    if enable_haiku_metric and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import time
            colored_print("Haiku Metric", f"Running Claude Haiku extraction on {len(results)} samples...", Colors.CYAN)
            
            # Infer answer format if not provided
            if answer_format is None:
                # Try to infer from task_name
                if "numeric" in task_name.lower() or task_name.upper() in ["GSM8K", "SVAMP", "MATH"]:
                    answer_format = "numeric"
                elif any(x in task_name.upper() for x in ["MMLU", "ARC"]):
                    answer_format = "A-D"
                elif any(x in task_name.upper() for x in ["AQUA", "MATHQA"]):
                    answer_format = "A-E"
                else:
                    answer_format = "numeric"  # Default fallback
            
            haiku_correct = 0
            haiku_calls = 0
            start_time = time.time()
            
            for idx, result in enumerate(results):
                generated_text = result.get("generated_answer", "")
                gold_answer = result.get("gold", "")
                
                # Extract with Haiku
                haiku_pred = extract_answer(generated_text, method="llm", answer_format=answer_format)
                haiku_calls += 1
                
                # Extract gold answer with same method for consistency
                if answer_format == "numeric":
                    haiku_gold = extract_answer(str(gold_answer), method="anchor")
                else:
                    haiku_gold = str(gold_answer).strip().upper()[:1] if gold_answer else "X"
                
                # Compare
                if haiku_pred == haiku_gold:
                    haiku_correct += 1
                
                # Rate limiting: max 50 requests per second for Haiku
                if (idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    if elapsed < 1.0:
                        time.sleep(1.0 - elapsed)
                    start_time = time.time()
            
            haiku_accuracy = haiku_correct / len(results) if results else 0.0
            # Cost: ~$0.0001 per call for Haiku (approximate)
            haiku_cost_usd = haiku_calls * 0.0001
            
            haiku_metrics = {
                "accuracy": haiku_accuracy,
                "cost_usd": haiku_cost_usd,
                "num_calls": haiku_calls,
            }
            
            colored_print(
                "Haiku Metric", 
                f"Accuracy: {haiku_accuracy:.2%} | Cost: ${haiku_cost_usd:.4f} ({haiku_calls} calls)", 
                Colors.GREEN if haiku_accuracy > 0.5 else Colors.YELLOW
            )
            
        except Exception as e:
            colored_print("Haiku Metric", f"Failed: {str(e)}", Colors.YELLOW)
            haiku_metrics = None
    
    return accuracy, results, accuracy_wb, haiku_metrics


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
    task_name: str = "MCQ",
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Generic MCQ evaluation for any number of choices.
    
    Args:
        num_choices: 4 for A-D (MMLU, ARC), 5 for A-E (AQuA, MathQA)
        task_name: Name for progress bar
        enable_haiku_metric: If True and ANTHROPIC_API_KEY is set, compute Haiku extraction metric
    
    Returns:
        tuple: (accuracy_wb: float, results: List[Dict], accuracy_legacy: Optional[float], 
                haiku_metrics: Optional[Dict])
        - accuracy_wb: Word boundary extraction (PRIMARY, correct method)
        - results: Detailed results with both extraction methods
        - accuracy_legacy: Legacy extraction without word boundaries (for backward compatibility)
        - haiku_metrics: Haiku extraction accuracy and cost tracking
    """
    choice_letter = chr(64 + num_choices)  # D for 4, E for 5
    answer_format = "A-D" if num_choices == 4 else "A-E"
    
    def extract_letter_wb(text: str) -> str:
        """Extract first letter with word boundaries (PRIMARY method)."""
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
    
    def extract_letter_legacy(text: str) -> str:
        """Extract first letter without word boundaries (LEGACY, kept for backward compatibility)."""
        pattern = f"[A-{choice_letter}]"
        matches = re.findall(pattern, text.upper())
        return matches[0] if matches else "X"
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, extract_letter_wb,
        answer_extractor_fn_wb=extract_letter_legacy,
        batch_size=batch_size, num_samples=num_samples,
        task_name=task_name, max_answer_tokens=10,
        enable_haiku_metric=enable_haiku_metric,
        answer_format=answer_format,
    )


def evaluate_model_on_mmlu(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: int = 500,
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Evaluate MMLU - 4-choice MCQ (A-D).
    
    Returns word-boundary accuracy (primary), legacy accuracy, and Haiku metrics.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=4, task_name="MMLU", enable_haiku_metric=enable_haiku_metric,
    )


def evaluate_model_on_arc(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None,
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Evaluate ARC - 4-choice MCQ (A-D).
    
    Returns word-boundary accuracy (primary), legacy accuracy, and Haiku metrics.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=4, task_name="ARC", enable_haiku_metric=enable_haiku_metric,
    )


def evaluate_model_on_aqua(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None,
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Evaluate AQuA - 5-choice MCQ (A-E).
    
    Returns word-boundary accuracy (primary), legacy accuracy, and Haiku metrics.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=5, task_name="AQuA", enable_haiku_metric=enable_haiku_metric,
    )


def evaluate_model_on_mathqa(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    num_samples: Optional[int] = None,
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Evaluate MathQA - 5-choice MCQ (A-E).
    
    Returns word-boundary accuracy (primary), legacy accuracy, and Haiku metrics.
    """
    return evaluate_model_on_mcq(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, batch_size=batch_size, num_samples=num_samples,
        num_choices=5, task_name="MathQA", enable_haiku_metric=enable_haiku_metric,
    )


def evaluate_model_on_numeric(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    hyperparameters: Dict[str, Any],
    batch_size: int = 16,
    answer_extraction_method: str = "anchor",
    num_samples: Optional[int] = None,
    max_answer_tokens: int = 16,
    enable_haiku_metric: bool = True,
) -> Tuple[float, List[Dict[str, Any]], Optional[float], Optional[Dict[str, Any]]]:
    """Evaluate numeric QA tasks (GSM8K, SVAMP, MATH).
    
    Pipeline:
    1. extract_answer: str -> Union[int, str]  (extracts first number or "[invalid]")
    2. compare_normalized: Union[int, str], str -> bool  (normalizes and compares)
    
    Args:
        answer_extraction_method: Method for extracting numeric answers:
            - "anchor": Check 'Answer:' label first, then '=', then first number (DEFAULT, recommended)
            - "simple": Check '=' then find first number (legacy, backward compatible)
            - "llm": Use Claude Haiku as gold-standard extractor (requires ANTHROPIC_API_KEY)
        enable_haiku_metric: If True and ANTHROPIC_API_KEY is set, compute Haiku extraction metric
    """
    # Get method from hyperparameters if not explicitly provided
    if answer_extraction_method == "anchor" and "answer_extraction_method" in hyperparameters:
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
        actor_model,
        critic_model,
        tokenizer,
        device,
        test_data,
        hyperparameters,
        answer_extractor_fn=extract_with_method,  # str -> Union[int, str]
        answer_comparator_fn=compare_normalized,  # Union[int, str], str -> bool
        batch_size=batch_size,
        num_samples=num_samples,
        task_name="Numeric",
        max_answer_tokens=max_answer_tokens,
        enable_haiku_metric=enable_haiku_metric,
        answer_format="numeric",
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
    answer_extraction_method="anchor",
    enable_haiku_metric=True,
):
    """Evaluate GSM8K using the unified numeric pipeline.
    
    Args:
        answer_extraction_method: "anchor" (default), "simple" (legacy), or "llm"
        enable_haiku_metric: If True and ANTHROPIC_API_KEY is set, compute Haiku extraction metric
    
    Returns:
        tuple: (accuracy: float, results: List[Dict], haiku_metrics: Optional[Dict])
        Note: Returns 3-tuple (not 4) for backward compatibility since GSM8K doesn't use word boundary
    """
    # Determine default eval batch size: floor(1.5x) of training batch size (defaults to 12 if absent)
    if batch_size is None:
        try:
            batch_size = max(1, int(hyperparameters.get("batch_size", 8) * 1.5))
        except Exception:
            batch_size = 12

    eval_data = test_data[:num_samples] if num_samples else test_data

    accuracy, results, _, haiku_metrics = evaluate_model_on_numeric(
        actor_model,
        critic_model,
        tokenizer,
        device,
        eval_data,
        hyperparameters,
        batch_size=batch_size,
        answer_extraction_method=answer_extraction_method,
        max_answer_tokens=10,
        num_samples=None,
        enable_haiku_metric=enable_haiku_metric,
    )
    return accuracy, results, haiku_metrics


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


def save_task_results(
    task_type: str,
    output_dir: str,
    model_type: str,
    accuracy: float,
    results: List[Dict[str, Any]],
    num_examples: int,
    checkpoint_path: Optional[str] = None,
    batch_index: Optional[int] = None,
    subject: Optional[str] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist evaluation results for any task and update task-specific artifacts."""
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "task_type": task_type,
        "batch_index": batch_index,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,
        "num_examples": num_examples,
        "detailed_results": results,
    }
    
    # Backward compatibility for older tooling that expects num_samples/total_examples keys
    entry["num_samples"] = num_examples
    entry["total_examples"] = num_examples
    
    if subject:
        entry["subject"] = subject
    if extra_metrics:
        entry.update(extra_metrics)
    
    results_file = os.path.join(output_dir, f"{task_type}_results_{model_type}.jsonl")
    _ensure_fresh_results_file(results_file)
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    
    accuracy_plot_path: Optional[str] = None
    if task_type == "gsm8k":
        accuracy_plot_path = os.path.join(output_dir, "gsm8k_accuracy_over_batches.png")
        plot_accuracy_over_batches(results_file, accuracy_plot_path)
        print(f"Updated GSM8K accuracy plot at {accuracy_plot_path}")
    
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
        choices=["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"],
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
    parser.add_argument(
        "--answer_prompt_variant",
        type=str,
        choices=["default", "letter", "letter_strict"],
        default="default",
        help="Modify answer prompt formatting (MCQ tasks only)"
    )
    parser.add_argument(
        "--haiku_metric",
        action="store_true",
        help="Compute an additional accuracy metric using Claude Haiku extraction (requires ANTHROPIC_API_KEY)"
    )
    
    # Checkpoint selection
    parser.add_argument(
        "--training_index",
        type=int,
        default=None,
        help="Specific training index to evaluate"
    )
    parser.add_argument(
        "--all_adapters",
        action="store_true",
        help="Evaluate each LoRA adapter_* directory inside --run_dir or --model_path"
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
        load_model_for_evaluation,
        load_svamp_dataset,
        load_aqua_dataset,
        load_arc_dataset,
        load_mmlu_dataset,
        load_mathqa_dataset,
        load_math_dataset,
        generate_question_answer_batches,
        load_gsm8k_dataset,
        load_arithmetic_dataset,
    )
    
    def resolve_run_dir(path: Optional[str]) -> Optional[str]:
        if path and os.path.isdir(path) and os.path.basename(path).startswith("adapter_"):
            return os.path.dirname(path)
        return path
    
    def infer_model_type(run_dir: str) -> str:
        try:
            hyperparameters_base = get_hyperparameters_from_log(run_dir, default_task=args.task_type)
            return hyperparameters_base.get("model_type", "mistral")
        except Exception:
            return "mistral"
    
    model_paths: List[Optional[str]] = []
    
    if args.use_base_model:
        model_paths = [None]
        args.model_type = args.model_type or "mistral"
    else:
        if args.all_adapters:
            run_dir = resolve_run_dir(args.run_dir or args.model_path or find_latest_result(args.task_type))
            if not run_dir:
                raise FileNotFoundError(f"No results directory found for task {args.task_type}")
            checkpoints = find_checkpoints(run_dir)
            if not checkpoints:
                raise FileNotFoundError(f"No adapter directories found in {run_dir}")
            model_paths = [ckpt_path for _, ckpt_path in checkpoints]
            if args.model_type is None:
                args.model_type = infer_model_type(run_dir)
                print(f"Inferred model type: {args.model_type}")
        else:
            candidate_path = args.model_path or args.run_dir or find_latest_result(args.task_type)
            if not candidate_path:
                raise FileNotFoundError("No model_path provided and no results directory found")
            
            if os.path.isdir(candidate_path) and os.path.basename(candidate_path).startswith("adapter_"):
                model_paths = [candidate_path]
                run_dir = os.path.dirname(candidate_path)
            else:
                run_dir = resolve_run_dir(candidate_path)
                if not run_dir or not os.path.isdir(run_dir):
                    raise FileNotFoundError(f"Run directory not found: {candidate_path}")
                checkpoints = find_checkpoints(run_dir)
                if not checkpoints:
                    raise FileNotFoundError(f"No adapter directories found in {run_dir}")
                if args.training_index is not None:
                    matches = [ckpt_path for idx, ckpt_path in checkpoints if idx == args.training_index]
                    if not matches:
                        raise FileNotFoundError(f"No adapter_{args.training_index} found in {run_dir}")
                    model_paths = [matches[0]]
                else:
                    model_paths = [checkpoints[-1][1]]
            
            if args.model_type is None:
                args.model_type = infer_model_type(resolve_run_dir(model_paths[0]) if model_paths[0] else run_dir)
                print(f"Inferred model type: {args.model_type}")
    
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
            hyperparameters["answer_prompt_variant"] = args.answer_prompt_variant
            
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
                "answer_prompt_variant": args.answer_prompt_variant,
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
        
        def load_cli_dataset(task_type: str) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
            meta: Dict[str, Any] = {}
            loaders = {
                "gsm8k": lambda: list(load_gsm8k_dataset(split="test")),
                "mmlu": lambda: list(load_mmlu_dataset(
                    split=hyperparameters.get("mmlu_split", "validation"),
                    subject=meta.setdefault("subject", hyperparameters.get("mmlu_subject")),
                )),
                "arc": lambda: list(load_arc_dataset(
                    split="validation",
                    subset=meta.setdefault("subset", args.arc_subset or os.getenv("ARC_SUBSET", "ARC-Challenge")),
                )),
                "svamp": lambda: list(load_svamp_dataset(split="test")),
                "aqua": lambda: list(load_aqua_dataset(split="test")),
                "mathqa": lambda: list(load_mathqa_dataset(split="test")),
                "math": lambda: list(load_math_dataset(split="test")),
                "arithmetic": lambda: list(load_arithmetic_dataset(chunk_size=args.num_samples or 200, split="test")),
            }
            if task_type not in loaders:
                raise ValueError(f"Unsupported task type: {task_type}")
            data = loaders[task_type]()
            if not data:
                raise FileNotFoundError(f"No data found for task {task_type}")
            return data, meta
        
        test_data, dataset_meta = load_cli_dataset(args.task_type)
        mmlu_subject: Optional[str] = dataset_meta.get("subject")
        arc_subset: Optional[str] = dataset_meta.get("subset")
        
        # Apply stride if specified
        if args.stride > 1:
            test_data = test_data[::args.stride]
            print(f"Using stride={args.stride}, evaluating on {len(test_data)} examples")
        
        # Determine eval batch size
        eval_bs = args.batch_size if args.batch_size is not None else get_default_eval_batch_size(
            hyperparameters.get("batch_size", 12)
        )
        
        # Run evaluation based on task type
        accuracy_wb: Optional[float] = None
        haiku_metrics: Optional[Dict[str, Any]] = None
        
        if args.task_type == "gsm8k":
            accuracy, results, haiku_metrics = evaluate_model_on_gsm8k(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                num_samples=args.num_samples,
                batch_size=eval_bs,
                answer_extraction_method=args.answer_extraction_method,
            )
        elif args.task_type == "mmlu":
            accuracy, results, accuracy_wb, haiku_metrics = evaluate_model_on_mmlu(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("MMLU Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "arc":
            accuracy, results, accuracy_wb, haiku_metrics = evaluate_model_on_arc(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("ARC Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "aqua":
            accuracy, results, accuracy_wb, haiku_metrics = evaluate_model_on_aqua(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("AQuA Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type == "mathqa":
            accuracy, results, accuracy_wb, haiku_metrics = evaluate_model_on_mathqa(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                num_samples=args.num_samples,
            )
            if accuracy_wb is not None:
                colored_print("MathQA Word Boundary", f"Accuracy (word boundary): {accuracy_wb:.2%}", Colors.CYAN)
        elif args.task_type in ["svamp", "arithmetic"]:
            accuracy, results, _, haiku_metrics = evaluate_model_on_numeric(
                actor_model, critic_model, tokenizer, device,
                test_data, hyperparameters,
                batch_size=eval_bs,
                answer_extraction_method=args.answer_extraction_method,
                num_samples=args.num_samples,
            )
        else:
            raise ValueError(f"Unsupported task type: {args.task_type}")
        
        # Display Haiku metrics if available
        if haiku_metrics is not None:
            colored_print("Haiku Metric", f"Accuracy: {haiku_metrics['accuracy']:.2%} | Cost: ${haiku_metrics['cost_usd']:.4f}", Colors.CYAN)
        
        # Print results
        colored_print(f"{args.task_type.upper()} Accuracy", f"{accuracy:.2%}", Colors.GREEN if accuracy > 0.5 else Colors.YELLOW)
        
        # Save results
        model_dir = os.path.dirname(checkpoint_path) if checkpoint_path else f"results/{args.task_type}"
        if checkpoint_path and os.path.isdir(checkpoint_path):
            model_dir = os.path.dirname(checkpoint_path)
        os.makedirs(model_dir, exist_ok=True)
        
        extra_metrics = {}
        if accuracy_wb is not None:
            extra_metrics["accuracy_word_boundary"] = accuracy_wb
        if args.task_type == "arc" and arc_subset:
            extra_metrics["subset"] = arc_subset
        if haiku_metrics is not None:
            extra_metrics["haiku_accuracy"] = haiku_metrics["accuracy"]
            extra_metrics["haiku_cost_usd"] = haiku_metrics["cost_usd"]
            extra_metrics["haiku_num_calls"] = haiku_metrics["num_calls"]
        if args.answer_prompt_variant != "default":
            extra_metrics["answer_prompt_variant"] = args.answer_prompt_variant
        
        results_file = save_task_results(
            task_type=args.task_type,
            output_dir=model_dir,
            model_type=args.model_type,
            accuracy=accuracy,
            results=results,
            num_examples=len(test_data),
            checkpoint_path=checkpoint_path,
            batch_index=batch_index,
            subject=mmlu_subject if args.task_type == "mmlu" else None,
            extra_metrics=extra_metrics or None,
        )
        
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

