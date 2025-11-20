import os
import glob
import json
import argparse
import subprocess
import math
import random
import datetime
import re
import pandas as pd
from collections import defaultdict


S3_BUCKET = os.environ.get("SWEEP_S3_BUCKET", "s3://scottviteri")
S3_BUCKET = S3_BUCKET.rstrip("/") if S3_BUCKET else None
METADATA_PATTERN = re.compile(r"eval_metadata(?:_stride(\d+))?\.json$")
_S3_WARNING_PRINTED = False

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

def list_adapter_dirs(run_dir):
    adapters = glob.glob(os.path.join(run_dir, "adapter_*"))
    return sorted(
        adapters,
        key=lambda path: int(os.path.basename(path).split("_")[-1])
        if os.path.basename(path).split("_")[-1].isdigit()
        else os.path.basename(path),
    )


def metadata_filename_for_stride(stride):
    stride = stride or 1
    return f"eval_metadata_stride{stride}.json"


def adapter_metadata_path(adapter_dir, stride):
    return os.path.join(adapter_dir, metadata_filename_for_stride(stride))


def parse_stride_from_metadata(data, path):
    stride = (
        data.get("evaluation", {}).get("stride")
        if isinstance(data.get("evaluation"), dict)
        else None
    )
    if stride is not None:
        return int(stride)
    match = METADATA_PATTERN.match(os.path.basename(path))
    if match and match.group(1):
        return int(match.group(1))
    return 1


def load_adapter_metadata_entries(adapter_dir):
    entries = []
    pattern = os.path.join(adapter_dir, "eval_metadata*.json")
    for metadata_file in glob.glob(pattern):
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading metadata {metadata_file}: {e}")
            continue
        stride = parse_stride_from_metadata(data, metadata_file)
        data["_metadata_path"] = metadata_file
        data["_adapter_dir"] = adapter_dir
        data["_stride"] = stride
        if "adapter_name" not in data:
            data["adapter_name"] = os.path.basename(adapter_dir)
        entries.append(data)
    return entries


def select_metadata_for_stride(entries, required_stride):
    if not entries:
        return None
    def sort_key(entry):
        stride = entry.get("_stride", float("inf"))
        acc = entry.get("accuracy")
        acc_value = acc if isinstance(acc, (int, float)) else float("-inf")
        return (stride, -acc_value)

    entries = sorted(entries, key=sort_key)
    for entry in entries:
        stride = entry.get("_stride", 1)
        if required_stride is None or stride <= required_stride:
            return entry
    return None


def collect_adapter_metadata(adapter_dirs, required_stride):
    metadata_entries = []
    missing_adapters = []
    for adapter_dir in adapter_dirs:
        entries = load_adapter_metadata_entries(adapter_dir)
        metadata = select_metadata_for_stride(entries, required_stride)
        if metadata:
            metadata_entries.append(metadata)
        else:
            missing_adapters.append(adapter_dir)
    return metadata_entries, missing_adapters


def safe_relpath(path, base_dir):
    if not base_dir:
        return path
    try:
        base_abs = os.path.abspath(base_dir)
        path_abs = os.path.abspath(path)
        common = os.path.commonpath([path_abs, base_abs])
        if common == base_abs:
            return os.path.relpath(path_abs, base_abs)
    except ValueError:
        pass
    return path


def sync_run_dir(run_dir, project_root):
    global _S3_WARNING_PRINTED
    if S3_BUCKET is None:
        if not _S3_WARNING_PRINTED:
            print("Warning: SWEEP_S3_BUCKET not set; skipping S3 sync.")
            _S3_WARNING_PRINTED = True
        return

    rel_path = safe_relpath(run_dir, project_root).replace("\\", "/")
    s3_dest = f"{S3_BUCKET}/{rel_path}"
    if s3_dest.startswith("s3:/") and not s3_dest.startswith("s3://"):
        s3_dest = s3_dest.replace("s3:/", "s3://", 1)
    include_args = [
        "--exclude", "*",
        "--include", "best_adapter.json",
        "--include", "*_results_*.jsonl",
        "--include", "adapter_*/eval_metadata*.json",
    ]
    print(f"Syncing {run_dir} -> {s3_dest}")
    try:
        subprocess.run(
            ["aws", "s3", "sync", run_dir, s3_dest, *include_args],
            check=True,
        )
    except FileNotFoundError:
        if not _S3_WARNING_PRINTED:
            print("Error: aws CLI not found; cannot sync to S3.")
            _S3_WARNING_PRINTED = True
    except subprocess.CalledProcessError as e:
        print(f"Error syncing to S3: {e}")

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

    adapters = list_adapter_dirs(run_dir)
    if not adapters:
        if not force_skip_eval:
            print(f"No adapters found in {run_dir}")
        return 0.0

    required_stride = args.stride if args.stride else 1
    metadata_entries, missing_adapters = collect_adapter_metadata(adapters, required_stride)

    if missing_adapters and not args.dry_run and not args.skip_eval and not force_skip_eval:
        print(
            f"Evaluating adapters in {os.path.basename(run_dir)} "
            f"(missing metadata for {len(missing_adapters)} adapters)..."
        )
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
        metadata_entries, missing_adapters = collect_adapter_metadata(adapters, required_stride)
    elif missing_adapters:
        print(
            f"Warning: Missing/insufficient metadata for {len(missing_adapters)} adapters in {run_dir} "
            f"(evaluation skipped due to flags or higher-precision request)."
        )

    if not metadata_entries:
        print(f"No adapter metadata available in {run_dir}.")
        return 0.0

    def metadata_accuracy(meta):
        acc = meta.get("accuracy")
        return acc if isinstance(acc, (int, float)) else None

    best_metadata = None
    best_acc_value = float("-inf")
    for meta in metadata_entries:
        acc = metadata_accuracy(meta)
        if acc is None:
            continue
        if acc > best_acc_value:
            best_acc_value = acc
            best_metadata = meta

    if best_metadata is None:
        best_metadata = metadata_entries[0]
        best_acc_value = metadata_accuracy(best_metadata) or 0.0

    if not args.dry_run:
        metadata_copy = {
            k: v for k, v in best_metadata.items() if not k.startswith("_")
        }
        rel_metadata_path = safe_relpath(best_metadata["_metadata_path"], run_dir)
        best_info = {
            "adapter": metadata_copy.get("adapter_name"),
            "accuracy": best_acc_value,
            "model_path": metadata_copy.get("model_path"),
            "model_type": metadata_copy.get("model_type"),
            "task_type": metadata_copy.get("task_type", dataset),
            "stride": best_metadata.get("_stride", metadata_copy.get("evaluation", {}).get("stride")),
            "num_examples": metadata_copy.get("num_examples"),
            "batch_index": metadata_copy.get("batch_index"),
            "metadata_file": rel_metadata_path,
            "metadata": metadata_copy,
            "generated_at": datetime.datetime.now().isoformat(),
        }

        best_info_path = os.path.join(run_dir, "best_adapter.json")
        with open(best_info_path, "w") as f:
            json.dump(best_info, f, indent=2)
        print(
            f"  - Best adapter: {best_info.get('adapter')} "
            f"(Accuracy: {best_acc_value:.2%})"
        )
        print(f"  - Saved best adapter info to {best_info_path}")
        sync_run_dir(run_dir, project_root)

    return best_acc_value

def main():
    parser = argparse.ArgumentParser(description="Compile sweep results into a table")
    parser.add_argument("--num_samples", type=int, help="Number of samples for evaluation")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every nth example (1 = full test set)")
    parser.add_argument("--model_type", type=str, default="llama", help="Model type")
    parser.add_argument("--dry_run", action="store_true", help="Simulate the sweep without writing files or syncing to S3")
    parser.add_argument("--skip_eval", action="store_true", help="Do not launch new evaluations; rely solely on existing metadata")
    parser.add_argument("--task_type", type=str, default=None, help="Only process a specific task/dataset (e.g. gsm8k)")
    parser.add_argument("--method", type=str, default=None, help="Only process a specific method/hyperparameter (e.g. PPO, EI, Markovian)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--run_dir", type=str, nargs="+", default=None, help="Specific run directory (or directories) to process. Overrides automatic scanning.")
    parser.add_argument("--shuffle", action="store_true", help="Process run directories in random order (useful for distributed workers)")
    parser.add_argument("--reverse", action="store_true", help="Process run directories in reverse order (useful for simple 2-machine parallelism)")
    args = parser.parse_args()

    # Supported tasks in evaluation.py
    SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
    WIKI_TASKS = ["wiki_continuation", "wiki_compression"]
    
    # Data structure: table[dataset][method] = score
    table = defaultdict(lambda: defaultdict(float))
    wiki_table = defaultdict(lambda: defaultdict(float))
    
    # 1. Identify runs to process
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
    
    run_dirs = []
    
    if args.run_dir:
        # User provided specific directories
        for r in args.run_dir:
            # Handle relative paths
            full_path = os.path.abspath(r)
            if not os.path.isdir(full_path):
                print(f"Warning: {r} is not a directory, skipping.")
                continue
                
            dataset, method_name = parse_run_dir(full_path)
            if not dataset or not method_name:
                print(f"Warning: Could not parse dataset/method from {r}, skipping.")
                continue
                
            run_dirs.append((dataset, method_name, full_path))
    else:
        # Automatic scanning of results directory
        results_root = os.path.join(project_root, "results")
        
        if not os.path.exists(results_root):
            # Fallback to current directory if running from root
            if os.path.exists("results"):
                results_root = "results"
                project_root = os.getcwd()
            else:
                print(f"Error: Could not find results directory at {results_root} or ./results")
                return

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

    # Shuffle or reverse run directories if requested
    if args.shuffle:
        random.shuffle(run_dirs)
    elif args.reverse:
        run_dirs.reverse()

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
