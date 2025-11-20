import os
import glob
import json
import argparse
import subprocess
import pandas as pd
from collections import defaultdict

def parse_run_dir(run_dir):
    """
    Extract dataset and method from run directory name.
    Expected format: {dataset}_{method}_{timestamp}
    """
    basename = os.path.basename(run_dir)
    parent = os.path.basename(os.path.dirname(run_dir))
    
    # Verify dataset matches parent
    if not basename.startswith(parent + "_"):
        return None, None
    
    # Extract method
    # Format: dataset_Method_YYYYMMDD_HHMMSS
    # We assume timestamp is the last two parts (date_time)
    parts = basename.split('_')
    if len(parts) < 4:
        return None, None
    
    # Method is everything between dataset and timestamp
    # parts[0] is dataset (or parts[0...k] if dataset has underscores?)
    # Usually dataset name in folder matches dataset name in prefix.
    # gsm8k -> gsm8k_...
    # wiki_continuation -> wiki_continuation_...
    
    # Find where the timestamp starts (last 2 parts)
    timestamp_start_idx = len(parts) - 2
    
    # Dataset part is likely the parent directory name
    dataset_name = parent
    
    # Check if prefix matches
    prefix = dataset_name + "_"
    if not basename.startswith(prefix):
        return None, None
        
    method_part = basename[len(prefix):].split('_')[:-2]
    method = "_".join(method_part)
    
    return dataset_name, method

def get_baseline_score(dataset, model_type, args, project_root, force_skip_eval=False):
    """
    Get baseline (base model) score for the dataset.
    Runs evaluation if not present in logs.
    """
    results_file = os.path.join(project_root, "results", dataset, f"{dataset}_results_{model_type}.jsonl")
    
    # Check if already exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Look for entry with no model_path (base model)
                    if data.get("model_path") is None and data.get("model_type") == model_type:
                        return data.get("accuracy", 0.0)
                except:
                    continue

    # Run evaluation
    if not args.dry_run and not args.skip_eval and not force_skip_eval:
        print(f"Evaluating baseline for {dataset}...")
        eval_script = os.path.join(project_root, "src", "evaluation.py")
        cmd = [
            "python", eval_script,
            "--task_type", dataset,
            "--model_type", model_type,
            "--use_base_model"
        ]
        if args.num_samples:
            cmd.extend(["--num_samples", str(args.num_samples)])
        if args.stride:
            cmd.extend(["--stride", str(args.stride)])
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
            
        try:
            subprocess.run(cmd, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating baseline for {dataset}: {e}")
            return 0.0
            
        # Read result
        if os.path.exists(results_file):
             with open(results_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    data = json.loads(line)
                    if data.get("model_path") is None:
                        return data.get("accuracy", 0.0)
    
    return 0.0

def get_max_adapter_score(run_dir, dataset, model_type, args, project_root, force_skip_eval=False):
    """
    Evaluate all adapters in a run directory and return the max accuracy.
    """
    results_file = os.path.join(run_dir, f"{dataset}_results_{model_type}.jsonl")
    
    # Check for adapters
    adapters = glob.glob(os.path.join(run_dir, "adapter_*"))
    if not adapters:
        if not force_skip_eval:
            print(f"No adapters found in {run_dir}")
        return 0.0

    # Run evaluation
    if not args.dry_run and not args.skip_eval and not force_skip_eval:
        print(f"Evaluating adapters in {os.path.basename(run_dir)}...")
        eval_script = os.path.join(project_root, "src", "evaluation.py")
        cmd = [
            "python", eval_script,
            "--task_type", dataset,
            "--run_dir", run_dir,
            "--all_adapters",
            "--model_type", model_type
        ]
        if args.num_samples:
            cmd.extend(["--num_samples", str(args.num_samples)])
        if args.stride:
            cmd.extend(["--stride", str(args.stride)])
        if args.batch_size:
            cmd.extend(["--batch_size", str(args.batch_size)])
            
        try:
            subprocess.run(cmd, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating adapters for {run_dir}: {e}")
            # Continue to try reading whatever exists
    
    # Read results
    max_acc = -1.0  # Start with -1 so even 0.0 accuracy is captured
    best_run = None
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Filter for this specific run? The file is in the run dir, so it should be specific.
                    if data.get("accuracy") is not None:
                        acc = data.get("accuracy")
                        if acc > max_acc:
                            max_acc = acc
                            best_run = data
                except:
                    continue
    
    # If max_acc is still -1, it means we found no valid entries or all were None
    # If max_acc is 0.0, that's a valid score and we should have a best_run
    if max_acc == -1.0:
        max_acc = 0.0
    
    if best_run and not args.dry_run:
        # Save best run info to a separate JSON file for easy retrieval
        best_info_path = os.path.join(run_dir, "best_adapter.json")
        with open(best_info_path, "w") as f:
            json.dump(best_run, f, indent=2)
        print(f"  - Best adapter: {best_run.get('model_path')} (Accuracy: {max_acc:.2%})")
        print(f"  - Saved best adapter info to {best_info_path}")

    return max_acc

def main():
    parser = argparse.ArgumentParser(description="Compile sweep results into a table")
    parser.add_argument("--num_samples", type=int, help="Number of samples for evaluation")
    parser.add_argument("--stride", type=int, default=1, help="Stride for evaluation")
    parser.add_argument("--model_type", type=str, default="llama", help="Model type")
    parser.add_argument("--dry_run", action="store_true", help="Don't run actual evaluations or save files")
    parser.add_argument("--skip_eval", action="store_true", help="Skip running evaluations (only process existing logs)")
    parser.add_argument("--task_type", type=str, default=None, help="Only process a specific task/dataset (e.g. gsm8k)")
    parser.add_argument("--method", type=str, default=None, help="Only process a specific method/hyperparameter (e.g. PPO, EI, Markovian)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    args = parser.parse_args()

    # Supported tasks in evaluation.py
    SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
    
    # Data structure: table[dataset][method] = score
    table = defaultdict(lambda: defaultdict(float))
    
    # 1. Scan results directory
    # Robustly find results directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
    results_root = os.path.join(project_root, "results")
    
    if not os.path.exists(results_root):
        # Fallback to current directory if running from root
        if os.path.exists("results"):
            results_root = "results"
            project_root = os.getcwd()
        else:
            print(f"Error: Could not find results directory at {results_root} or ./results")
            return

    run_dirs = []
    for task in os.listdir(results_root):
        # Filter by task_type if specified
        if args.task_type and task != args.task_type:
            continue

        task_dir = os.path.join(results_root, task)
        if not os.path.isdir(task_dir):
            continue
            
        if task not in SUPPORTED_TASKS:
            print(f"Skipping unsupported task: {task}")
            continue
            
        # Find run directories
        for item in os.listdir(task_dir):
            path = os.path.join(task_dir, item)
            if os.path.isdir(path):
                dataset, method_name = parse_run_dir(path)
                if dataset and method_name:
                    # Filter by method if specified
                    if args.method and method_name != args.method:
                        continue
                        
                    run_dirs.append((dataset, method_name, path))

    # 2. Process Baselines (Batch 0)
    datasets = set(d[0] for d in run_dirs)
    print(f"Found datasets: {datasets}")
    
    for dataset in datasets:
        # Determine if we should evaluate this baseline
        should_evaluate = True
        if args.task_type and dataset != args.task_type:
            should_evaluate = False
            
        score = get_baseline_score(dataset, args.model_type, args, project_root, force_skip_eval=not should_evaluate)
        table[dataset]["Baseline"] = score
        
    # 3. Process Runs
    for dataset, method, path in run_dirs:
        # Determine if we should evaluate this run
        should_evaluate = True
        if args.task_type and dataset != args.task_type:
            should_evaluate = False
        if args.method and method != args.method:
            should_evaluate = False
            
        score = get_max_adapter_score(path, dataset, args.model_type, args, project_root, force_skip_eval=not should_evaluate)
        table[dataset][method] = score

    # 4. Generate Table
    df = pd.DataFrame.from_dict(table, orient='index')
    
    # Add Mean Row
    if not df.empty:
        df.loc['Mean'] = df.mean()
        
    print("\n" + "="*50)
    print("Sweep Results Table")
    print("="*50)
    print(df.to_markdown())
    
    # Save to CSV
    df.to_csv("sweep_results_table.csv")
    print("\nSaved to sweep_results_table.csv")

if __name__ == "__main__":
    main()

