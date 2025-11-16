#!/usr/bin/env python3
"""
Test evaluation formatting for all datasets.
Uses 10 samples per dataset to verify we can get at least one correct answer.
This ensures extraction and comparison logic is working properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from typing import Dict, Any, List, Tuple
from utils import (
    load_gsm8k_dataset,
    load_mmlu_dataset,
    load_aqua_dataset,
    load_svamp_dataset,
    load_arc_dataset,
    load_mathqa_dataset,
    load_model,
)
from train import (
    evaluate_model,
    evaluate_model_on_mmlu,
    evaluate_model_on_aqua,
    evaluate_model_on_numeric,
    Colors,
)
from utils import colored_print


def test_dataset_evaluation(
    task_type: str,
    dataset_name: str,
    load_fn,
    eval_fn,
    num_samples: int = 10,
    **load_kwargs
) -> Dict[str, Any]:
    """Test evaluation on a small sample of a dataset.
    
    Args:
        task_type: Task type identifier
        dataset_name: Human-readable dataset name
        load_fn: Function to load dataset
        eval_fn: Evaluation function
        num_samples: Number of samples to test
        **load_kwargs: Additional arguments for load_fn
    
    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    colored_print(f"Testing {dataset_name}", f"Using {num_samples} samples", Colors.CYAN)
    print('='*60)
    
    try:
        # Load test data
        test_data = []
        data_iter = load_fn(split="test", **load_kwargs)
        for i, item in enumerate(data_iter):
            if i >= num_samples:
                break
            test_data.append(item)
        
        if len(test_data) == 0:
            colored_print("ERROR", f"No data loaded for {dataset_name}", Colors.RED)
            return {"error": "No data loaded", "accuracy": None}
        
        print(f"Loaded {len(test_data)} samples")
        
        # Show first example
        q, a = test_data[0]
        print(f"\nExample question: {q[:100]}...")
        print(f"Example answer: {a}")
        
        # Create minimal hyperparameters
        hyperparameters = {
            "task_type": task_type,
            "model_type": "llama",
            "cot_length": 50,
            "temperature": 1.0,
            "question_length": 512,
            "markovian": True,
        }
        
        # Load model (use base model for quick testing)
        colored_print("Loading model", "Using Llama for testing", Colors.YELLOW)
        
        actor_model, critic_model, tokenizer, device = load_model(
            model_type="llama",
            hyperparameters=hyperparameters
        )
        
        # Run evaluation
        colored_print("Running evaluation", "This may take a minute...", Colors.YELLOW)
        accuracy, results = eval_fn(
            actor_model,
            critic_model,
            tokenizer,
            device,
            test_data,
            hyperparameters,
            batch_size=2
        )
        
        # Analyze results
        correct_count = sum(1 for r in results if r["is_correct"])
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
        
        # Show a few examples
        print(f"\nFirst 3 results:")
        for i, r in enumerate(results[:3]):
            status = "✓" if r["is_correct"] else "✗"
            print(f"\n  {status} Example {i+1}:")
            print(f"    Question: {r['question'][:80]}...")
            
            # Handle different key names for generated answer
            gen_ans = r.get('generated_answer', r.get('reasoning', ''))
            print(f"    Generated: {gen_ans[:80]}...")
            
            # Handle different key names for predicted/extracted answer
            pred = r.get('predicted', r.get('extracted_answer', 'N/A'))
            print(f"    Predicted: {pred}")
            
            # Handle different key names for gold answer
            gold = r.get('gold', r.get('correct_answer', r.get('answer', 'N/A')))
            print(f"    Gold: {gold}")
            print(f"    Correct: {r['is_correct']}")
        
        # Check for zero accuracy issue
        if accuracy == 0.0:
            colored_print("WARNING", f"Got 0% accuracy - possible formatting issue!", Colors.RED)
            status = "FAIL"
        else:
            colored_print("SUCCESS", f"Got {accuracy:.1%} accuracy - formatting appears correct!", Colors.GREEN)
            status = "PASS"
        
        return {
            "dataset": dataset_name,
            "status": status,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(results),
            "error": None
        }
        
    except Exception as e:
        colored_print("ERROR", f"Failed to test {dataset_name}: {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "status": "ERROR",
            "accuracy": None,
            "error": str(e)
        }


def main():
    """Run evaluation format tests on all datasets."""
    
    colored_print("Evaluation Format Test", "Testing all datasets with 10 samples each", Colors.BOLD)
    
    results = []
    
    # Test GSM8K
    results.append(test_dataset_evaluation(
        task_type="gsm8k",
        dataset_name="GSM8K",
        load_fn=load_gsm8k_dataset,
        eval_fn=evaluate_model,
        num_samples=10
    ))
    
    # Test SVAMP
    results.append(test_dataset_evaluation(
        task_type="svamp",
        dataset_name="SVAMP",
        load_fn=load_svamp_dataset,
        eval_fn=evaluate_model_on_numeric,
        num_samples=10
    ))
    
    # Test MMLU (pick one subject)
    results.append(test_dataset_evaluation(
        task_type="mmlu",
        dataset_name="MMLU (Abstract Algebra)",
        load_fn=load_mmlu_dataset,
        eval_fn=evaluate_model_on_mmlu,
        num_samples=10,
        subject="abstract_algebra"
    ))
    
    # Test AQuA
    results.append(test_dataset_evaluation(
        task_type="aqua",
        dataset_name="AQuA",
        load_fn=load_aqua_dataset,
        eval_fn=evaluate_model_on_aqua,
        num_samples=10
    ))
    
    # Test MathQA
    results.append(test_dataset_evaluation(
        task_type="mathqa",
        dataset_name="MathQA",
        load_fn=load_mathqa_dataset,
        eval_fn=evaluate_model,  # Uses same as GSM8K (handles both numeric and MCQ)
        num_samples=10
    ))
    
    # Test ARC
    results.append(test_dataset_evaluation(
        task_type="arc",
        dataset_name="ARC-Challenge",
        load_fn=load_arc_dataset,
        eval_fn=evaluate_model_on_aqua,  # Uses same multiple choice eval
        num_samples=10,
        subset="ARC-Challenge"
    ))
    
    # Print summary
    print(f"\n\n{'='*60}")
    colored_print("SUMMARY", "Evaluation Format Test Results", Colors.BOLD)
    print('='*60)
    
    for r in results:
        if r["status"] == "PASS":
            color = Colors.GREEN
            symbol = "✓"
        elif r["status"] == "FAIL":
            color = Colors.RED
            symbol = "✗"
        else:
            color = Colors.YELLOW
            symbol = "?"
        
        if r["accuracy"] is not None:
            colored_print(
                symbol,
                f"{r['dataset']:30s} {r['status']:8s} ({r['correct']}/{r['total']} = {r['accuracy']:.1%})",
                color
            )
        else:
            colored_print(symbol, f"{r['dataset']:30s} {r['status']:8s} (ERROR: {r['error']})", color)
    
    # Final verdict
    all_pass = all(r["status"] == "PASS" for r in results)
    any_fail = any(r["status"] == "FAIL" for r in results)
    
    print("\n" + "="*60)
    if all_pass:
        colored_print("VERDICT", "All datasets PASSED - formatting looks good!", Colors.GREEN)
        return 0
    elif any_fail:
        colored_print("VERDICT", "Some datasets FAILED with 0% accuracy - check formatting!", Colors.RED)
        return 1
    else:
        colored_print("VERDICT", "Some tests had errors - check logs above", Colors.YELLOW)
        return 2


if __name__ == "__main__":
    sys.exit(main())

