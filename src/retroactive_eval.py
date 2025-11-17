#!/usr/bin/env python3
"""
Retroactive Checkpoint Evaluation Script

Evaluates saved model checkpoints and generates dual-metric evaluation logs,
enabling decoupled evaluation from training.

Usage:
    python src/retroactive_eval.py --results_dir results/svamp/20251116_063242 --stride 1
"""

import argparse
import json
import os
import re
import glob
import datetime
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm

# Import from train.py and utils.py
from evaluation import (
    evaluate_model_on_gsm8k,
    evaluate_model_on_mmlu,
    evaluate_model_on_arc,
    evaluate_model_on_aqua,
    evaluate_model_on_mathqa,
    evaluate_model_on_numeric,
    get_default_eval_batch_size,
)
from utils import (
    load_gsm8k_dataset,
    load_svamp_dataset,
    load_mmlu_dataset,
    load_aqua_dataset,
    load_arc_dataset,
    load_mathqa_dataset,
    load_math_dataset,
    generate_arithmetic_pairs,
    colored_print,
    Colors,
)


def get_model_path(model_type: str) -> str:
    """Get HuggingFace model path from model type.
    
    Args:
        model_type: Model type string
        
    Returns:
        HuggingFace model path
    """
    model_paths = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "gpt2": "openai-community/gpt2",
        "tinystories": "roneneldan/TinyStories",
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "phi-4": "microsoft/phi-4",
        "qwen3": "Qwen/Qwen3-4B",
        "gemma-3": "google/gemma-3-12b-it",
        "gemma-3-small": "google/gemma-3-1b-it",
    }
    
    if model_type not in model_paths:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_paths[model_type]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retroactively evaluate saved checkpoints with dual metrics"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory containing checkpoints (e.g., results/svamp/20251116_063242)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every Nth test example (default: 1, use all test data)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Evaluation batch size (default: auto-determined from training batch size)",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline (batch 0) evaluation of untrained model",
    )
    return parser.parse_args()


def find_checkpoints(results_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint directories in results_dir.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of (batch_index, checkpoint_path) tuples, sorted by batch_index
    """
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


def load_hyperparameters(results_dir: str) -> Dict[str, Any]:
    """Load hyperparameters from log.jsonl first line.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary of hyperparameters
    """
    log_file = os.path.join(results_dir, "log.jsonl")
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"No log.jsonl found in {results_dir}")
    
    with open(log_file, "r") as f:
        first_line = f.readline()
        hyperparameters = json.loads(first_line)
    
    return hyperparameters


def load_model_and_tokenizer(hyperparameters: Dict[str, Any], device: torch.device):
    """Load base model and tokenizer from hyperparameters.
    
    Args:
        hyperparameters: Dictionary of hyperparameters
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    model_type = hyperparameters["model_type"]
    model_path = get_model_path(model_type)
    
    colored_print("Model", f"Loading base model: {model_type}", Colors.CYAN)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    return model, tokenizer


def load_checkpoint_adapter(model, checkpoint_path: str, device: torch.device):
    """Load LoRA adapter from checkpoint directory.
    
    Args:
        model: Base model to attach adapter to
        checkpoint_path: Path to checkpoint directory
        device: Device to load adapter on
        
    Returns:
        Model with adapter loaded
    """
    from peft import PeftModel
    
    adapter_path = checkpoint_path
    colored_print("Checkpoint", f"Loading adapter from: {os.path.basename(checkpoint_path)}", Colors.CYAN)
    
    # Load the adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device)
    model.eval()
    
    return model


def get_test_dataset(task_type: str, hyperparameters: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Get test dataset for the given task type.
    
    Args:
        task_type: Type of task (gsm8k, svamp, mmlu, etc.)
        hyperparameters: Dictionary of hyperparameters
        
    Returns:
        List of (question, answer) tuples
    """
    if task_type == "gsm8k":
        return list(load_gsm8k_dataset(split="test"))
    elif task_type == "svamp":
        return list(load_svamp_dataset(split="test"))
    elif task_type == "mmlu":
        subject = hyperparameters.get("mmlu_subject", None)
        return list(load_mmlu_dataset(split="test", subject=subject))
    elif task_type == "aqua":
        return list(load_aqua_dataset(split="test"))
    elif task_type == "arc":
        return list(load_arc_dataset(split="test"))
    elif task_type == "mathqa":
        return list(load_mathqa_dataset(split="test"))
    elif task_type == "math":
        return list(load_math_dataset(split="test"))
    elif task_type.startswith("arithmetic"):
        # Arithmetic tasks generate their own test data
        colored_print("Info", f"Generating arithmetic test data for '{task_type}'", Colors.CYAN)
        test_pairs = generate_arithmetic_pairs(task_type=task_type, num_examples=1000)
        return test_pairs
    elif task_type.startswith("wiki_"):
        # Wiki tasks don't have standard test evaluation
        colored_print("Warning", f"Task type '{task_type}' doesn't support standard evaluation", Colors.YELLOW)
        return []
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def evaluate_checkpoint(
    actor_model: nn.Module,
    critic_model: nn.Module,
    tokenizer,
    device: torch.device,
    test_data: List[Tuple[str, str]],
    task_type: str,
    hyperparameters: Dict[str, Any],
    batch_size: int,
) -> Tuple[float, List[Dict[str, Any]], Optional[float]]:
    """Evaluate a checkpoint on the test dataset.
    
    Args:
        actor_model: Model to evaluate
        critic_model: Critic model (frozen)
        tokenizer: Tokenizer
        device: Device
        test_data: List of (question, answer) tuples
        task_type: Type of task
        hyperparameters: Hyperparameters
        batch_size: Evaluation batch size
        
    Returns:
        Tuple of (accuracy, results, accuracy_wb)
    """
    with torch.no_grad():
        actor_model.eval()
        
        if task_type == "gsm8k":
            # GSM8K uses its own evaluation function
            accuracy, results = evaluate_model_on_gsm8k(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
            )
            return accuracy, results, None
        
        elif task_type == "mmlu":
            # MMLU is MCQ - returns 3-tuple with word boundary metric
            return evaluate_model_on_mmlu(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
                num_samples=len(test_data),  # Evaluate all
            )
        
        elif task_type == "aqua":
            # AQuA is MCQ - returns 3-tuple with word boundary metric
            return evaluate_model_on_aqua(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
            )
        
        elif task_type in ["svamp", "math"] or task_type.startswith("arithmetic"):
            # Numeric tasks - returns 3-tuple with accuracy_wb=None
            return evaluate_model_on_numeric(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
            )
        
        elif task_type == "arc":
            # ARC is 4-choice MCQ (A-D)
            return evaluate_model_on_arc(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
                num_samples=len(test_data),
            )
        
        elif task_type == "mathqa":
            # MathQA is 5-choice MCQ (A-E)
            return evaluate_model_on_mathqa(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                hyperparameters,
                batch_size=batch_size,
                num_samples=len(test_data),
            )
        
        else:
            raise ValueError(f"Unsupported task type for evaluation: {task_type}")


def save_evaluation_results(
    results_dir: str,
    task_type: str,
    model_type: str,
    batch_index: int,
    accuracy: float,
    accuracy_wb: Optional[float],
    results: List[Dict[str, Any]],
    checkpoint_path: str,
):
    """Save evaluation results to JSONL file.
    
    Args:
        results_dir: Path to results directory
        task_type: Type of task
        model_type: Type of model
        batch_index: Batch index of checkpoint
        accuracy: Main accuracy metric
        accuracy_wb: Word boundary accuracy (or None)
        results: Detailed per-question results
        checkpoint_path: Path to checkpoint
    """
    results_file = os.path.join(results_dir, f"{task_type}_results_{model_type}.jsonl")
    
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": batch_index,
        "accuracy": accuracy,
        "accuracy_wb": accuracy_wb,
        "model_path": checkpoint_path,
        "model_type": model_type,
        "total_examples": len(results),
        "results": results,
    }
    
    # Append to file
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")


def regenerate_plots(results_dir: str, task_type: str):
    """Regenerate combined metrics plot.
    
    Args:
        results_dir: Path to results directory
        task_type: Type of task
    """
    import subprocess
    
    log_file = os.path.join(results_dir, "log.jsonl")
    output_file = os.path.join(results_dir, f"combined_metrics_{task_type}.png")
    
    colored_print("Plotting", f"Regenerating combined metrics plot...", Colors.CYAN)
    
    try:
        cmd = [
            "python", "src/plot_training_metrics.py",
            "--files", log_file,
            "--output", output_file,
            "--plot_summary",
        ]
        subprocess.run(cmd, check=True, cwd="/root/MarkovianTraining")
        colored_print("Plotting", f"Plot saved to: {os.path.basename(output_file)}", Colors.GREEN)
    except Exception as e:
        colored_print("Plotting", f"Failed to regenerate plot: {e}", Colors.RED)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        colored_print("Error", f"Results directory not found: {args.results_dir}", Colors.RED)
        return
    
    colored_print("Retroactive Evaluation", f"Processing: {args.results_dir}", Colors.BOLD)
    
    # Load hyperparameters
    try:
        hyperparameters = load_hyperparameters(args.results_dir)
        task_type = hyperparameters["task_type"]
        model_type = hyperparameters["model_type"]
        colored_print("Config", f"Task: {task_type}, Model: {model_type}", Colors.CYAN)
    except Exception as e:
        colored_print("Error", f"Failed to load hyperparameters: {e}", Colors.RED)
        return
    
    # Find checkpoints
    checkpoints = find_checkpoints(args.results_dir)
    if not checkpoints:
        colored_print("Error", "No checkpoints found", Colors.RED)
        return
    
    colored_print("Checkpoints", f"Found {len(checkpoints)} checkpoints to evaluate", Colors.GREEN)
    
    # Backup existing results
    backups = backup_existing_results(args.results_dir)
    if backups:
        colored_print("Backup", f"Backed up {len(backups)} existing result files", Colors.GREEN)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colored_print("Device", f"Using device: {device}", Colors.CYAN)
    
    # Load test dataset
    colored_print("Dataset", f"Loading test dataset for {task_type}...", Colors.CYAN)
    test_data = get_test_dataset(task_type, hyperparameters)
    if not test_data:
        colored_print("Error", "No test data available", Colors.RED)
        return
    
    # Apply stride to test data
    if args.stride > 1:
        test_data = test_data[::args.stride]
        colored_print("Dataset", f"Using every {args.stride}th example: {len(test_data)} test examples", Colors.GREEN)
    else:
        colored_print("Dataset", f"Loaded {len(test_data)} test examples", Colors.GREEN)
    
    # Determine batch size
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = get_default_eval_batch_size(hyperparameters.get("batch_size", 16))
    colored_print("Config", f"Evaluation batch size: {batch_size}", Colors.CYAN)
    
    # Load critic model (frozen, reuse for all checkpoints)
    colored_print("Critic", "Loading critic model (frozen)...", Colors.CYAN)
    critic_model, tokenizer = load_model_and_tokenizer(hyperparameters, device)
    critic_model.eval()
    for param in critic_model.parameters():
        param.requires_grad = False
    
    # Evaluate each checkpoint
    results_summary = []
    
    # Check if baseline already exists in results file
    results_file = os.path.join(args.results_dir, f"{task_type}_results_{model_type}.jsonl")
    baseline_exists = False
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('batch_index') == 0:
                        baseline_exists = True
                        colored_print("Baseline", "Batch 0 already exists in results, skipping baseline evaluation", Colors.CYAN)
                        break
                except:
                    continue
    
    # Evaluate baseline (batch 0 - no training) if not skipped and doesn't exist
    if not args.skip_baseline and not baseline_exists:
        colored_print("Baseline", "Evaluating untrained base model (batch 0)...", Colors.BOLD)
        try:
            base_model_baseline, _ = load_model_and_tokenizer(hyperparameters, device)
            base_model_baseline.eval()
            
            with torch.no_grad():
                accuracy, results, accuracy_wb = evaluate_checkpoint(
                    base_model_baseline,
                    critic_model,
                    tokenizer,
                    device,
                    test_data,
                    task_type,
                    hyperparameters,
                    batch_size,
                )
            
            # Save baseline results
            save_evaluation_results(
                args.results_dir,
                task_type,
                model_type,
                0,  # batch_index = 0 for baseline
                accuracy,
                accuracy_wb,
                results,
                "baseline (no training)",
            )
            
            # Track for summary
            results_summary.append((0, accuracy, accuracy_wb))
            
            # Log result
            if accuracy_wb is not None:
                colored_print(
                    "Batch 0 (Baseline)",
                    f"Accuracy: {accuracy:.2%} | WB Accuracy: {accuracy_wb:.2%}",
                    Colors.GREEN
                )
            else:
                colored_print(
                    "Batch 0 (Baseline)",
                    f"Accuracy: {accuracy:.2%}",
                    Colors.GREEN
                )
            
            # Cleanup
            del base_model_baseline
            torch.cuda.empty_cache()
            
        except Exception as e:
            colored_print("Batch 0 (Baseline)", f"Evaluation failed: {e}", Colors.RED)
    
    for batch_idx, checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        try:
            # Load fresh base model for this checkpoint to avoid adapter stacking
            base_model, _ = load_model_and_tokenizer(hyperparameters, device)
            
            # Load adapter for this checkpoint
            actor_model = load_checkpoint_adapter(base_model, checkpoint_path, device)
            
            # Run evaluation
            accuracy, results, accuracy_wb = evaluate_checkpoint(
                actor_model,
                critic_model,
                tokenizer,
                device,
                test_data,
                task_type,
                hyperparameters,
                batch_size,
            )
            
            # Save results
            save_evaluation_results(
                args.results_dir,
                task_type,
                model_type,
                batch_idx,
                accuracy,
                accuracy_wb,
                results,
                checkpoint_path,
            )
            
            # Track for summary
            results_summary.append((batch_idx, accuracy, accuracy_wb))
            
            # Log result
            if accuracy_wb is not None:
                colored_print(
                    f"Batch {batch_idx}",
                    f"Accuracy: {accuracy:.2%} | WB Accuracy: {accuracy_wb:.2%}",
                    Colors.GREEN
                )
            else:
                colored_print(
                    f"Batch {batch_idx}",
                    f"Accuracy: {accuracy:.2%}",
                    Colors.GREEN
                )
            
            # Unload adapter and base model to free memory
            del actor_model
            del base_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            colored_print(f"Batch {batch_idx}", f"Evaluation failed: {e}", Colors.RED)
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    colored_print("Summary", f"Completed evaluation of {len(results_summary)} checkpoints", Colors.BOLD)
    colored_print("Test Set", f"Evaluated on {len(test_data)} test examples", Colors.CYAN)
    print("=" * 80)
    
    if results_summary:
        print(f"\n{'Batch':<10} {'Accuracy':<12} {'WB Accuracy':<12}")
        print("-" * 40)
        for batch_idx, accuracy, accuracy_wb in results_summary:
            if accuracy_wb is not None:
                print(f"{batch_idx:<10} {accuracy:.2%}        {accuracy_wb:.2%}")
            else:
                print(f"{batch_idx:<10} {accuracy:.2%}        N/A")
    
    # Regenerate plots
    regenerate_plots(args.results_dir, task_type)
    
    colored_print("Done", f"Results saved to: {task_type}_results_{model_type}.jsonl", Colors.GREEN)


if __name__ == "__main__":
    main()

