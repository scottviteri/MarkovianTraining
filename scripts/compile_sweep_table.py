import os
import glob
import json
import argparse
import subprocess
import math
import random
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

def get_wiki_max_smoothed_logprob(run_dir, window_size=200):
    """
    For wiki tasks, we use the 'Actor Answer Log Probs' from the log file.
    This metric corresponds to the log probability of the answer (continuation)
    given the reasoning (and question/context if non-Markovian).
    
    As requested, we smooth this metric using a rolling window and take the maximum.
    """
    log_file = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_file):
        return 0.0
        
    log_probs = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We want "Actor Answer Log Probs" which is P(Answer | Context + Reasoning)
                    # or P(Answer | Reasoning) depending on Markovian setting.
                    # This matches the user's request for "log prob of the subsequent text given the reasoning"
                    if "Training Metrics" in data and "Actor Answer Log Probs" in data["Training Metrics"]:
                        val = data["Training Metrics"]["Actor Answer Log Probs"]
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            log_probs.append(val)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return 0.0
        
    if not log_probs:
        return 0.0
        
    # Create a pandas Series to easily calculate rolling mean
    s = pd.Series(log_probs)
    # Smooth with window size 200 (or length of data if smaller)
    smoothed = s.rolling(window=min(window_size, len(s)), min_periods=1).mean()
    
    # Return the maximum smoothed value
    return smoothed.max()


def get_wiki_baseline_score(dataset, model_type, project_root):
    """
    For wiki tasks baseline, we use the average of the first 100 'Critic Answer Log Probs'.
    The critic (frozen model) provides a baseline for the answer log probability.
    We pick the longest run (most logged lines) to extract this baseline from,
    as it's likely to be the most complete/reliable record.
    """
    results_dir = os.path.join(project_root, "results", dataset)
    if not os.path.exists(results_dir):
        return 0.0
        
    # Find all run directories
    runs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not runs:
        return 0.0
        
    # Find the longest run (most lines in log.jsonl)
    best_run_dir = None
    max_lines = -1

    for run in runs:
        run_path = os.path.join(results_dir, run)
        log_path = os.path.join(run_path, "log.jsonl")
        if os.path.exists(log_path):
            try:
                # Count lines efficiently
                with open(log_path, 'rb') as f:
                    lines = sum(1 for _ in f)
                if lines > max_lines:
                    max_lines = lines
                    best_run_dir = run_path
            except Exception:
                continue
    
    if not best_run_dir:
        return 0.0
        
    log_file = os.path.join(best_run_dir, "log.jsonl")
    
    critic_log_probs = []
    try:
        with open(log_file, 'r') as f:
            count = 0
            for line in f:
                if count >= 100:
                    break
                try:
                    data = json.loads(line)
                    if "Training Metrics" in data and "Critic Answer Log Probs" in data["Training Metrics"]:
                        val = data["Training Metrics"]["Critic Answer Log Probs"]
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            critic_log_probs.append(val)
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return 0.0
        
    if not critic_log_probs:
        return 0.0
        
    return sum(critic_log_probs) / len(critic_log_probs)


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

    # Check if best_adapter.json already exists (skip eval if so)
    best_adapter_path = os.path.join(run_dir, "best_adapter.json")
    if os.path.exists(best_adapter_path) and not force_skip_eval:
        try:
            with open(best_adapter_path, 'r') as f:
                data = json.load(f)
                if "accuracy" in data:
                    # If we are just compiling the table (not specifically asked to re-run),
                    # we can use the cached result.
                    # However, if the user specifically omitted --skip_eval, they might WANT to re-run?
                    # But for the distributed use case, we definitely want to skip.
                    # Let's assume presence of best_adapter.json means "done" unless explicit re-run logic is added.
                    print(f"Found existing best_adapter.json in {os.path.basename(run_dir)}, skipping eval.")
                    
                    # Ensure we still sync if requested (in case it was computed but not synced)
                    if args.s3_sync and not args.dry_run:
                        # ... sync logic ...
                        rel_path = os.path.relpath(run_dir, project_root)
                        s3_dest = f"{args.s3_sync}/{rel_path}"
                        # Check existence first (optimization)
                        try:
                             res = subprocess.run(["aws", "s3", "ls", s3_dest + "/"], capture_output=True, text=True)
                             if not res.stdout.strip():
                                 # Not in S3, but we have it locally. Sync it.
                                 print(f"Syncing cached results for {run_dir} to {s3_dest}...")
                                 subprocess.run([
                                    "aws", "s3", "sync", run_dir, s3_dest,
                                    "--exclude", "*",
                                    "--include", "best_adapter.json",
                                    "--include", "*_results_*.jsonl"
                                 ], check=True)
                        except Exception as e:
                            print(f"Error checking/syncing S3: {e}")

                    return data["accuracy"]
        except Exception as e:
            print(f"Error reading {best_adapter_path}: {e}, will re-evaluate.")

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

    # Sync to S3 if requested
    if args.s3_sync and not args.dry_run:
        rel_path = os.path.relpath(run_dir, project_root)
        s3_dest = f"{args.s3_sync}/{rel_path}"
        
        # 1. Check if this run exists on S3 to avoid uploading local-only experiments
        # We only want to update existing "representatives"
        try:
            # ls the directory. If it returns output, it exists.
            result = subprocess.run(
                ["aws", "s3", "ls", s3_dest + "/"], 
                capture_output=True, 
                text=True
            )
            if not result.stdout.strip():
                print(f"Skipping S3 sync for {run_dir} (not found in S3 bucket)")
                return max_acc
        except Exception as e:
            print(f"Error checking S3 existence: {e}")
            return max_acc

        print(f"Syncing results for {run_dir} to {s3_dest}...")
        try:
            # 2. Selective sync: ONLY upload the result files we just generated/modified
            # Exclude everything, then include specific JSON/JSONL files
            subprocess.run(
                [
                    "aws", "s3", "sync", run_dir, s3_dest,
                    "--exclude", "*",
                    "--include", "best_adapter.json",
                    "--include", "*_results_*.jsonl"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error syncing to S3: {e}")

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
    parser.add_argument("--shuffle", action="store_true", help="Process run directories in random order (useful for distributed workers)")
    parser.add_argument("--s3_sync", type=str, default=None, help="S3 bucket URL (e.g. s3://my-bucket) to sync results to after processing each run")
    args = parser.parse_args()

    # Supported tasks in evaluation.py
    SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
    WIKI_TASKS = ["wiki_continuation", "wiki_compression"]
    
    # Data structure: table[dataset][method] = score
    table = defaultdict(lambda: defaultdict(float))
    wiki_table = defaultdict(lambda: defaultdict(float))
    
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
            
        if task not in SUPPORTED_TASKS and task not in WIKI_TASKS:
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

    # Shuffle run directories if requested (for distributed processing)
    if args.shuffle:
        random.shuffle(run_dirs)

    # 2. Process Baselines (Batch 0)
    datasets = set(d[0] for d in run_dirs)
    print(f"Found datasets: {datasets}")
    
    for dataset in datasets:
        if dataset in WIKI_TASKS:
            # Wiki baseline calculation
            score = get_wiki_baseline_score(dataset, args.model_type, project_root)
            wiki_table[dataset]["Baseline"] = score
        else:
            # Standard task baseline evaluation
            should_evaluate = True
            if args.task_type and dataset != args.task_type:
                should_evaluate = False
                
            score = get_baseline_score(dataset, args.model_type, args, project_root, force_skip_eval=not should_evaluate)
            table[dataset]["Baseline"] = score
        
    # 3. Process Runs
    for dataset, method, path in run_dirs:
        if dataset in WIKI_TASKS:
            # Wiki score calculation (max smoothed logprob)
            score = get_wiki_max_smoothed_logprob(path)
            wiki_table[dataset][method] = score
        else:
            # Standard task evaluation
            should_evaluate = True
            if args.task_type and dataset != args.task_type:
                should_evaluate = False
            if args.method and method != args.method:
                should_evaluate = False
                
            score = get_max_adapter_score(path, dataset, args.model_type, args, project_root, force_skip_eval=not should_evaluate)
            table[dataset]["Baseline"] = table[dataset]["Baseline"] # ensure baseline col exists
            table[dataset][method] = score

    # 4. Generate Table
    df = pd.DataFrame.from_dict(table, orient='index')
    
    # Add Mean Row for QA tasks
    if not df.empty:
        df.loc['Mean'] = df.mean()
        
    print("\n" + "="*50)
    print("Sweep Results Table (QA Tasks - Accuracy)")
    print("="*50)
    print(df.to_markdown())
    
    # 5. Generate Wiki Table
    if wiki_table:
        df_wiki = pd.DataFrame.from_dict(wiki_table, orient='index')
        print("\n" + "="*50)
        print("Sweep Results Table (Wiki Tasks - Max Smoothed LogProb)")
        print("="*50)
        print(df_wiki.to_markdown())
        
        # Combine for CSV saving (if desired, though formats differ)
        # For now, maybe save separate CSVs or append with a clear separator/index
        df_wiki.to_csv("sweep_results_wiki.csv")
        print("\nSaved wiki results to sweep_results_wiki.csv")
    
    # Save QA results to CSV
    df.to_csv("sweep_results_table.csv")
    print("Saved QA results to sweep_results_table.csv")

if __name__ == "__main__":
    main()
