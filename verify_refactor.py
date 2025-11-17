#!/usr/bin/env python3
"""
Verification script to ensure evaluation refactor doesn't change scores.

This script re-evaluates existing checkpoints with the new unified evaluation code
and compares results to ensure consistency (for non-buggy tasks).
"""

import sys
sys.path.insert(0, 'src')

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from evaluation import (
    evaluate_model_on_gsm8k,
    evaluate_model_on_mmlu,
    evaluate_model_on_aqua,
    evaluate_model_on_numeric,
    get_default_eval_batch_size,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    load_gsm8k_dataset,
    load_svamp_dataset,
    load_mmlu_dataset,
    load_aqua_dataset,
    colored_print,
    Colors,
)


def get_model_path(model_type: str) -> str:
    """Get HuggingFace model path."""
    model_paths = {
        "llama": "meta-llama/Llama-3.1-8B",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "phi-4": "microsoft/phi-4",
        "qwen3": "Qwen/Qwen2.5-7B",
        "gemma-3": "google/gemma-2-9b",
    }
    return model_paths.get(model_type, model_type)


def load_checkpoint(checkpoint_dir: str, hyperparameters: Dict, device: torch.device):
    """Load model and tokenizer with checkpoint."""
    model_type = hyperparameters["model_type"]
    model_path = get_model_path(model_type)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load adapter
    actor_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    actor_model.eval()
    
    # Critic is just base model (frozen)
    critic_model = base_model
    
    return actor_model, critic_model, tokenizer


def read_old_results(results_file: str, batch_index: int) -> Tuple[float, int]:
    """Read old evaluation results for comparison."""
    if not os.path.exists(results_file):
        return None, None
    
    with open(results_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('batch_index') == batch_index:
                    return entry.get('accuracy'), entry.get('total_examples')
            except:
                continue
    
    return None, None


def verify_checkpoint(
    checkpoint_dir: str,
    task_type: str,
    hyperparameters: Dict,
    device: torch.device,
    batch_index: int,
    stride: int = 10,
) -> Dict:
    """Verify one checkpoint's evaluation."""
    colored_print("Verify", f"Evaluating {checkpoint_dir}", Colors.CYAN)
    
    # Load models
    actor_model, critic_model, tokenizer = load_checkpoint(
        checkpoint_dir, hyperparameters, device
    )
    
    # Load test data
    if task_type == "mmlu":
        test_data = list(load_mmlu_dataset(split="test", subject=None))
        # Apply stride for faster verification
        test_data = test_data[::stride]
        accuracy, results, accuracy_wb = evaluate_model_on_mmlu(
            actor_model, critic_model, tokenizer, device, test_data,
            hyperparameters, batch_size=16, num_samples=len(test_data)
        )
    elif task_type == "aqua":
        test_data = list(load_aqua_dataset(split="test"))
        test_data = test_data[::stride]
        accuracy, results, accuracy_wb = evaluate_model_on_aqua(
            actor_model, critic_model, tokenizer, device, test_data,
            hyperparameters, batch_size=16
        )
    elif task_type == "svamp":
        test_data = list(load_svamp_dataset(split="test"))
        test_data = test_data[::stride]
        accuracy, results, accuracy_wb = evaluate_model_on_numeric(
            actor_model, critic_model, tokenizer, device, test_data,
            hyperparameters, batch_size=16
        )
    elif task_type == "gsm8k":
        test_data = list(load_gsm8k_dataset(split="test"))
        test_data = test_data[::stride]
        accuracy, results = evaluate_model_on_gsm8k(
            actor_model, critic_model, tokenizer, device, test_data,
            hyperparameters, batch_size=16, num_samples=len(test_data)
        )
        accuracy_wb = None
    else:
        colored_print("Error", f"Unsupported task: {task_type}", Colors.RED)
        return None
    
    # Cleanup
    del actor_model
    del critic_model
    torch.cuda.empty_cache()
    
    return {
        "batch_index": batch_index,
        "new_accuracy": accuracy,
        "new_accuracy_wb": accuracy_wb,
        "num_examples": len(test_data),
    }


def main():
    """Main verification routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify evaluation refactor doesn't change scores")
    parser.add_argument("--stride", type=int, default=10, 
                        help="Evaluate every Nth example for faster testing (default: 10)")
    args = parser.parse_args()
    
    # Test checkpoints to verify
    test_configs = [
        {
            "name": "MMLU batch 1000",
            "checkpoint": "results/mmlu/20251116_191617/adapter_1000",
            "task": "mmlu",
            "results_file": "results/mmlu/20251116_191617/mmlu_results_llama.jsonl",
            "batch_index": 1000,
        },
        {
            "name": "AQuA batch 1000",
            "checkpoint": "results/aqua/20251116_193803/adapter_1000",
            "task": "aqua",
            "results_file": "results/aqua/20251116_193803/aqua_results_llama.jsonl",
            "batch_index": 1000,
        },
        {
            "name": "SVAMP batch 1000",
            "checkpoint": "results/svamp/20251116_063242/adapter_1000",
            "task": "svamp",
            "results_file": "results/svamp/20251116_063242/svamp_results_llama.jsonl",
            "batch_index": 1000,
        },
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colored_print("Device", f"Using {device}", Colors.CYAN)
    
    results_summary = []
    
    for config in test_configs:
        checkpoint_dir = config["checkpoint"]
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_dir):
            colored_print("Skip", f"{config['name']}: Checkpoint not found", Colors.YELLOW)
            continue
        
        # Load hyperparameters
        run_dir = os.path.dirname(checkpoint_dir)
        log_file = os.path.join(run_dir, "log.jsonl")
        if not os.path.exists(log_file):
            colored_print("Skip", f"{config['name']}: Log file not found", Colors.YELLOW)
            continue
        
        with open(log_file, 'r') as f:
            hyperparameters = json.loads(f.readline())
        
        # Read old results
        old_accuracy, old_num_examples = read_old_results(
            config["results_file"], config["batch_index"]
        )
        
        # Run new evaluation
        result = verify_checkpoint(
            checkpoint_dir,
            config["task"],
            hyperparameters,
            device,
            config["batch_index"],
            stride=args.stride,
        )
        
        if result is None:
            continue
        
        # Compare results
        result["name"] = config["name"]
        result["old_accuracy"] = old_accuracy
        result["old_num_examples"] = old_num_examples
        results_summary.append(result)
        
        # Print comparison
        colored_print("Result", f"{config['name']}", Colors.BOLD)
        if old_accuracy is not None:
            print(f"  Old accuracy: {old_accuracy:.4f} ({old_num_examples} examples)")
        print(f"  New accuracy: {result['new_accuracy']:.4f} ({result['num_examples']} examples)")
        if result['new_accuracy_wb'] is not None:
            print(f"  New WB accuracy: {result['new_accuracy_wb']:.4f}")
        
        if old_accuracy is not None:
            diff = abs(result['new_accuracy'] - old_accuracy)
            if diff < 0.01:  # Within 1%
                colored_print("Status", "✓ Results match!", Colors.GREEN)
            else:
                colored_print("Status", f"⚠ Difference: {diff:.4f}", Colors.YELLOW)
        print()
    
    # Summary
    colored_print("Summary", "Verification Complete", Colors.BOLD)
    print("\nResults:")
    print("-" * 80)
    for result in results_summary:
        status = "✓" if (result['old_accuracy'] is None or 
                        abs(result['new_accuracy'] - result['old_accuracy']) < 0.01) else "⚠"
        print(f"{status} {result['name']}: {result['new_accuracy']:.2%}")
    print("-" * 80)
    
    colored_print("Note", "Expected: MMLU, AQuA, SVAMP should match old results", Colors.CYAN)
    colored_print("Note", "MathQA and ARC scores WILL change (bug fixes)", Colors.YELLOW)


if __name__ == "__main__":
    main()

