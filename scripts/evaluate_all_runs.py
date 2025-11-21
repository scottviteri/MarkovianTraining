#!/usr/bin/env python3
"""
Comprehensive evaluation script for all runs in results folder.

This script:
1. Finds all runs with adapter checkpoints
2. Evaluates the latest checkpoint from each run
3. Compiles a summary report with all metrics
"""

import os
import sys
import glob
import re
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(SRC_DIR))


def find_runs_with_checkpoints(results_dir: Path = RESULTS_DIR) -> List[Tuple[str, str, str]]:
    """
    Find all runs with adapter checkpoints.
    
    Returns:
        List of (task_type, run_dir, latest_checkpoint) tuples
    """
    runs = []
    
    # Pattern: results/{task}/{timestamp}/adapter_*
    for task_dir in Path(results_dir).iterdir():
        if not task_dir.is_dir():
            continue
            
        task_type = task_dir.name
        
        # Skip non-task directories
        if task_type in ["keep.txt"]:
            continue
            
        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Find adapter checkpoints in this run
            adapter_dirs = list(run_dir.glob("adapter_*"))
            if not adapter_dirs:
                continue
            
            # Extract batch indices and find latest
            checkpoints = []
            for adapter_dir in adapter_dirs:
                match = re.search(r'adapter_(\d+)', adapter_dir.name)
                if match:
                    batch_idx = int(match.group(1))
                    checkpoints.append((batch_idx, str(adapter_dir)))
            
            if checkpoints:
                checkpoints.sort(key=lambda x: x[0])
                latest_batch, latest_checkpoint = checkpoints[-1]
                runs.append((task_type, str(run_dir), latest_checkpoint))
                print(f"Found {task_type} run: {run_dir.name} (latest: adapter_{latest_batch})")
    
    return runs


def evaluate_checkpoint(task_type: str, checkpoint_path: str) -> Dict:
    """
    Run evaluation on a checkpoint.
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"Task: {task_type}")
    print(f"{'='*80}\n")
    
    # Build evaluation command
    cmd = [
        sys.executable,
        "-m", "evaluation",
        "--task_type", task_type,
        "--model_path", checkpoint_path,
    ]
    
    # Task-specific parameters
    if task_type == "mmlu":
        cmd.extend(["--num_samples", "500"])
    
    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)}
        )
        
        # Parse output for accuracy
        output = result.stdout + result.stderr
        
        # Look for accuracy in output
        accuracy = None
        haiku_accuracy = None
        
        for line in output.split('\n'):
            if "Accuracy:" in line and "%" in line:
                # Extract percentage
                match = re.search(r'(\d+\.\d+)%', line)
                if match and accuracy is None:
                    accuracy = float(match.group(1)) / 100
            
            if "haiku" in line.lower() and "accuracy" in line.lower() and "%" in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    haiku_accuracy = float(match.group(1)) / 100
        
        # Also check adapter-level metadata/results if available
        run_dir = os.path.dirname(checkpoint_path) if checkpoint_path else None
        adapter_dir = Path(checkpoint_path) if checkpoint_path and os.path.isdir(checkpoint_path) else None

        def maybe_extract_from_metadata(meta_path: str):
            nonlocal accuracy, haiku_accuracy
            try:
                with open(meta_path, "r") as f:
                    data = json.load(f)
                if accuracy is None:
                    accuracy = data.get("accuracy")
                if haiku_accuracy is None:
                    haiku = data.get("haiku_metrics")
                    if isinstance(haiku, dict):
                        haiku_accuracy = haiku.get("accuracy")
            except Exception:
                pass

        if adapter_dir:
            metadata_files = sorted(adapter_dir.glob("eval_metadata*.json"))
            for meta_file in metadata_files:
                maybe_extract_from_metadata(str(meta_file))
                if accuracy is not None:
                    break

        # Fallback to legacy shared results file if it exists
        if run_dir and accuracy is None:
            legacy_file = os.path.join(run_dir, f"{task_type}_results_llama.jsonl")
            if os.path.exists(legacy_file):
                try:
                    with open(legacy_file, "r") as f:
                        lines = f.readlines()
                    if lines:
                        last_result = json.loads(lines[-1])
                        if accuracy is None:
                            accuracy = last_result.get("accuracy")
                        if haiku_accuracy is None:
                            haiku_accuracy = last_result.get("haiku_accuracy")
                except Exception:
                    pass
        
        return {
            "success": result.returncode == 0,
            "accuracy": accuracy,
            "haiku_accuracy": haiku_accuracy,
            "output": output,
            "error": result.stderr if result.returncode != 0 else None,
        }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "accuracy": None,
            "error": "Evaluation timed out (30 minutes)",
        }
    except Exception as e:
        return {
            "success": False,
            "accuracy": None,
            "error": str(e),
        }


def main():
    """Main evaluation loop."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION OF ALL RUNS")
    print("="*80)
    print()
    
    # Find all runs
    runs = find_runs_with_checkpoints(RESULTS_DIR)
    
    if not runs:
        print("No runs with checkpoints found!")
        return
    
    print(f"\nFound {len(runs)} runs to evaluate\n")
    
    # Evaluate each run
    results = []
    for task_type, run_dir, checkpoint_path in runs:
        result = evaluate_checkpoint(task_type, checkpoint_path)
        
        results.append({
            "task_type": task_type,
            "run_dir": run_dir,
            "checkpoint": checkpoint_path,
            **result
        })
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    # Group by task type
    by_task = {}
    for r in results:
        task = r["task_type"]
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(r)
    
    for task_type in sorted(by_task.keys()):
        print(f"\n{task_type.upper()}")
        print("-" * 80)
        
        for r in by_task[task_type]:
            run_name = os.path.basename(r["run_dir"])
            checkpoint_name = os.path.basename(r["checkpoint"])
            
            if r["success"]:
                acc_str = f"{r['accuracy']:.2%}" if r['accuracy'] is not None else "N/A"
                
                extra = []
                if r.get("haiku_accuracy") is not None:
                    extra.append(f"Haiku: {r['haiku_accuracy']:.2%}")
                
                extra_str = " | " + " | ".join(extra) if extra else ""
                
                print(f"  {run_name}/{checkpoint_name}: {acc_str}{extra_str}")
            else:
                print(f"  {run_name}/{checkpoint_name}: FAILED - {r.get('error', 'Unknown error')}")
    
    # Save detailed results
    output_file = REPO_ROOT / "evaluation_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Print statistics
    successful = sum(1 for r in results if r["success"])
    print(f"\n{'='*80}")
    print(f"Evaluated: {len(results)} runs")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

