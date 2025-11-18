#!/usr/bin/env python3
"""
Summarize evaluation results from all result files.
"""

import json
import glob
import os
from pathlib import Path

def summarize_results():
    results_dir = "/root/MarkovianTraining/results"
    
    print("=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Find all results files
    result_files = glob.glob(f"{results_dir}/**/*_results_*.jsonl", recursive=True)
    
    # Skip backup files
    result_files = [f for f in result_files if "_backup_" not in f]
    
    # Group by task
    by_task = {}
    for result_file in result_files:
        task = Path(result_file).parts[-3]  # Get task from path
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(result_file)
    
    for task in sorted(by_task.keys()):
        print(f"\n{task.upper()}")
        print("-" * 80)
        
        for result_file in sorted(by_task[task]):
            run_name = Path(result_file).parts[-2]
            
            # Read last line
            try:
                with open(result_file, 'r') as f:
                    lines = [line for line in f if line.strip()]
                    if lines:
                        last_result = json.loads(lines[-1])
                        
                        acc = last_result.get("accuracy", 0)
                        acc_wb = last_result.get("accuracy_word_boundary")
                        haiku_acc = last_result.get("haiku_accuracy")
                        batch_idx = last_result.get("batch_index", "?")
                        num_examples = last_result.get("num_examples", "?")
                        
                        # Build output
                        output = f"  {run_name}/adapter_{batch_idx}: {acc:.2%} (n={num_examples})"
                        
                        if acc_wb is not None:
                            output += f" | WB: {acc_wb:.2%}"
                        if haiku_acc is not None:
                            output += f" | Haiku: {haiku_acc:.2%}"
                        
                        print(output)
            except Exception as e:
                print(f"  {run_name}: Error reading file - {e}")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    summarize_results()

