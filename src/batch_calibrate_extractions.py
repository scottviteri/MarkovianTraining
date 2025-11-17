#!/usr/bin/env python3
"""
Batch Calibration of Answer Extraction Methods

This script runs calibration comparisons across multiple training checkpoints
for different datasets, comparing simple, anchor, and LLM extraction methods.

Usage:
    python batch_calibrate_extractions.py --num_samples 50
    python batch_calibrate_extractions.py --datasets gsm8k svamp --num_samples 30
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    load_model_for_evaluation,
    load_gsm8k_dataset,
    load_svamp_dataset,
    load_arc_dataset,
    load_aqua_dataset,
    colored_print,
    Colors,
)
from evaluation import compare_extraction_methods


def find_all_checkpoints(results_dir: str, dataset_name: str):
    """Find all checkpoint directories for a given dataset."""
    dataset_dir = os.path.join(results_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        return []
    
    checkpoints = []
    # Look in timestamped subdirectories
    for subdir in sorted(os.listdir(dataset_dir)):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            # Find adapter_* directories
            for item in sorted(os.listdir(subdir_path)):
                if item.startswith("adapter_") and os.path.isdir(os.path.join(subdir_path, item)):
                    checkpoint_path = os.path.join(subdir_path, item)
                    # Extract step number
                    try:
                        step = int(item.split("_")[1])
                        checkpoints.append((step, checkpoint_path, subdir))
                    except (IndexError, ValueError):
                        continue
    
    return sorted(checkpoints)


def load_dataset_samples(dataset_name: str, num_samples: int):
    """Load test samples for a given dataset."""
    if dataset_name == "gsm8k":
        test_data = []
        for i, qa in enumerate(load_gsm8k_dataset(split="test")):
            if i >= num_samples:
                break
            test_data.append(qa)
        return test_data, "numeric"
    
    elif dataset_name == "svamp":
        test_data = []
        for i, qa in enumerate(load_svamp_dataset(split="test")):
            if i >= num_samples:
                break
            test_data.append(qa)
        return test_data, "numeric"
    
    elif dataset_name == "arc":
        test_data = []
        for i, qa in enumerate(load_arc_dataset(split="test", subset="ARC-Challenge")):
            if i >= num_samples:
                break
            test_data.append(qa)
        return test_data, "A-D"
    
    elif dataset_name == "aqua":
        test_data = []
        for i, qa in enumerate(load_aqua_dataset(split="test")):
            if i >= num_samples:
                break
            test_data.append(qa)
        return test_data, "A-E"
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_calibration_for_checkpoint(
    checkpoint_path: str,
    dataset_name: str,
    test_data: list,
    answer_format: str,
    model_type: str,
    batch_size: int,
    methods: list
):
    """Run calibration comparison for a single checkpoint."""
    try:
        # Load model
        colored_print("Loading", f"Loading checkpoint: {checkpoint_path}", Colors.CYAN)
        actor_model, critic_model, tokenizer, device = load_model_for_evaluation(
            model_path=checkpoint_path,
            model_type=model_type
        )
        
        # Setup hyperparameters
        hyperparameters = {
            "task_type": dataset_name,
            "model_type": model_type,
            "cot_length": 100,
            "temperature": 1.0,
            "batch_size": batch_size,
        }
        
        # Run comparison
        results = compare_extraction_methods(
            actor_model, critic_model, tokenizer, device,
            test_data, hyperparameters,
            methods=methods,
            batch_size=batch_size,
            num_samples=len(test_data),
            answer_format=answer_format
        )
        
        # Extract summary statistics
        summary = {
            method: {
                "accuracy": acc,
                "num_samples": len(test_data),
                "correct": int(acc * len(test_data))
            }
            for method, (acc, _) in results.items()
        }
        
        colored_print("Success", f"Checkpoint completed: {checkpoint_path}", Colors.GREEN)
        return summary
        
    except Exception as e:
        colored_print("Error", f"Failed for {checkpoint_path}: {str(e)}", Colors.RED)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch calibration of answer extraction methods across checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing training results"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "svamp", "arc"],
        choices=["gsm8k", "svamp", "arc", "aqua"],
        help="Datasets to calibrate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of test samples per checkpoint"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["mistral", "llama", "phi"],
        help="Model type"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Extraction methods to compare (default: all available)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for results (default: batch_calibration_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--max_checkpoints_per_run",
        type=int,
        default=None,
        help="Maximum number of checkpoints to process per training run"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if os.getenv("ANTHROPIC_API_KEY"):
        colored_print("API Key", "ANTHROPIC_API_KEY found - will include LLM gold-standard", Colors.GREEN)
    else:
        colored_print("Warning", "ANTHROPIC_API_KEY not set - LLM comparison will be skipped", Colors.YELLOW)
        print("To use LLM comparison: export ANTHROPIC_API_KEY=your-key-here\n")
    
    # Setup output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"batch_calibration_{timestamp}.json"
    
    all_results = {}
    
    # Process each dataset
    for dataset_name in args.datasets:
        colored_print("Dataset", f"Processing dataset: {dataset_name.upper()}", Colors.BOLD)
        print("=" * 70)
        
        # Find all checkpoints
        checkpoints = find_all_checkpoints(args.results_dir, dataset_name)
        if not checkpoints:
            colored_print("Warning", f"No checkpoints found for {dataset_name}", Colors.YELLOW)
            continue
        
        colored_print("Found", f"{len(checkpoints)} checkpoints for {dataset_name}", Colors.CYAN)
        
        # Limit number of checkpoints if requested
        if args.max_checkpoints_per_run:
            # Sample evenly across the training run
            step_size = max(1, len(checkpoints) // args.max_checkpoints_per_run)
            checkpoints = checkpoints[::step_size][:args.max_checkpoints_per_run]
            colored_print("Sampling", f"Processing {len(checkpoints)} checkpoints", Colors.CYAN)
        
        # Load test data once for this dataset
        colored_print("Loading", f"Loading {args.num_samples} test samples for {dataset_name}...", Colors.CYAN)
        test_data, answer_format = load_dataset_samples(dataset_name, args.num_samples)
        colored_print("Loaded", f"Loaded {len(test_data)} samples", Colors.GREEN)
        
        # Process each checkpoint
        dataset_results = {}
        for step, checkpoint_path, run_id in checkpoints:
            print(f"\n{'-' * 70}")
            colored_print("Processing", f"Step {step} from run {run_id}", Colors.BOLD)
            print(f"{'-' * 70}")
            
            result = run_calibration_for_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_name=dataset_name,
                test_data=test_data,
                answer_format=answer_format,
                model_type=args.model_type,
                batch_size=args.batch_size,
                methods=args.methods
            )
            
            if result:
                if run_id not in dataset_results:
                    dataset_results[run_id] = {}
                dataset_results[run_id][step] = result
                
                # Print summary
                print("\nResults for this checkpoint:")
                for method, stats in result.items():
                    print(f"  {method:10s}: {stats['accuracy']*100:5.1f}% ({stats['correct']}/{stats['num_samples']})")
        
        all_results[dataset_name] = dataset_results
        
        print("\n" + "=" * 70 + "\n")
    
    # Save all results
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    colored_print("Complete", f"All results saved to {args.output_file}", Colors.GREEN)
    
    # Print summary
    print("\n" + "=" * 70)
    colored_print("Summary", "Batch Calibration Complete", Colors.BOLD)
    print("=" * 70)
    for dataset_name, dataset_results in all_results.items():
        total_checkpoints = sum(len(run_results) for run_results in dataset_results.values())
        print(f"\n{dataset_name.upper()}:")
        print(f"  Training runs: {len(dataset_results)}")
        print(f"  Checkpoints processed: {total_checkpoints}")
        print(f"  Samples per checkpoint: {args.num_samples}")


if __name__ == "__main__":
    main()

