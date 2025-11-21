import os
import glob
import json
import argparse
import subprocess
import math
import random
import datetime
import re
import sys
from collections import defaultdict

import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.utils import load_model_for_evaluation
from src.evaluation import compute_wiki_logprob, write_adapter_metadata


DEFAULT_S3_BUCKET = os.environ.get("SWEEP_S3_BUCKET", "s3://scottviteri")
METADATA_PATTERN = re.compile(r"eval_metadata(?:_stride(\d+))?\.json$")
_S3_WARNING_PRINTED = False
WIKI_DEFAULT_NUM_SAMPLES = 256
WIKI_DEFAULT_START_INDEX = 10000
WIKI_DEFAULT_QUESTION_LENGTH = 512
WIKI_DEFAULT_TARGET_LENGTH = 128

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



def read_run_hyperparameters(run_dir, default_task=None):
    log_file = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "r") as f:
            first_line = f.readline()
            if not first_line:
                return None
            data = json.loads(first_line)
            if default_task:
                data.setdefault("task_type", default_task)
            return data
    except Exception:
        return None


def find_wiki_metadata_entry(adapter_dir, dataset, required_stride=1):
    entries, _ = collect_adapter_metadata([adapter_dir], required_stride)
    for entry in entries:
        if entry.get("task_type") != dataset:
            continue
        if entry.get("metric") != "wiki_log_prob":
            continue
        return entry
    return None


def evaluate_wiki_adapter(adapter_dir, run_dir, dataset, model_type, args, project_root, enable_s3=False, s3_bucket=None):
    required_stride = args.stride if args.stride else 1
    existing = find_wiki_metadata_entry(adapter_dir, dataset, required_stride=required_stride)
    if existing:
        return existing.get("accuracy", existing.get("average_log_prob", 0.0))
    hyper = read_run_hyperparameters(run_dir, default_task=dataset)
    if hyper is None:
        hyper = {
            "task_type": dataset,
            "question_length": WIKI_DEFAULT_QUESTION_LENGTH,
            "target_length": WIKI_DEFAULT_TARGET_LENGTH,
        }
    try:
        actor_model, _, tokenizer, device = load_model_for_evaluation(
            model_path=adapter_dir,
            model_type=model_type,
        )
        mean_log_prob, eval_meta = compute_wiki_logprob(
            actor_model,
            tokenizer,
            device,
            hyper,
            num_samples=WIKI_DEFAULT_NUM_SAMPLES,
            stride=required_stride,
            start_index=WIKI_DEFAULT_START_INDEX,
            question_length=hyper.get("question_length"),
            target_length=hyper.get("target_length"),
        )
    finally:
        del actor_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = {
        "adapter_name": os.path.basename(adapter_dir),
        "task_type": dataset,
        "model_type": model_type,
        "model_path": adapter_dir,
        "accuracy": mean_log_prob,
        "metric": "wiki_log_prob",
        "num_examples": eval_meta["num_samples"],
        "timestamp": datetime.datetime.now().isoformat(),
        "evaluation": eval_meta,
    }
    write_adapter_metadata(adapter_dir, metadata, eval_meta["stride"])
    sync_run_dir(run_dir, project_root, enable_s3, s3_bucket)
    return mean_log_prob


def get_wiki_run_score(run_dir, dataset, model_type, args, project_root, enable_s3=False, s3_bucket=None):
    adapters = list_adapter_dirs(run_dir)
    if not adapters:
        return 0.0
    best_score = float("-inf")
    for adapter_dir in adapters:
        score = evaluate_wiki_adapter(adapter_dir, run_dir, dataset, model_type, args, project_root, enable_s3, s3_bucket)
        if score is not None and score > best_score:
            best_score = score
    if best_score == float("-inf"):
        return 0.0
    return best_score


def detect_model_type_from_metadata(metadata_entries):
    for meta in metadata_entries:
        mt = meta.get("model_type")
        if mt:
            return mt
    return None


def detect_model_type_from_log(run_dir):
    if not run_dir:
        return None
    log_path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                return data.get("model_type")
    except Exception as e:
        print(f"Warning: failed to parse {log_path}: {e}")
    return None


DEFAULT_MODEL_TYPE = os.environ.get("SWEEP_DEFAULT_MODEL_TYPE", "llama")


def detect_model_type(run_dir, metadata_entries):
    mt = detect_model_type_from_metadata(metadata_entries)
    if mt:
        return mt
    mt = detect_model_type_from_log(run_dir)
    if mt:
        return mt
    return DEFAULT_MODEL_TYPE


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


def sync_run_dir(run_dir, project_root, enable_sync=False, bucket=None):
    global _S3_WARNING_PRINTED
    if not enable_sync:
        return

    bucket = bucket or DEFAULT_S3_BUCKET
    if not bucket:
        if not _S3_WARNING_PRINTED:
            print("Warning: S3 sync requested but no bucket configured.")
            _S3_WARNING_PRINTED = True
        return

    rel_path = safe_relpath(run_dir, project_root).replace("\\", "/")
    s3_dest = f"{bucket.rstrip('/')}/{rel_path}"
    if s3_dest.startswith("s3:/") and not s3_dest.startswith("s3://"):
        s3_dest = s3_dest.replace("s3:/", "s3://", 1)
    include_args = [
        "--exclude", "*",
        "--include", "best_adapter.json",
        "--include", "*_results_*.jsonl",
        "--include", "adapter_*/eval_metadata*.json",
        "--include", "wiki_baseline_*.json",
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

def get_wiki_baseline_score(dataset, model_type, args, project_root, enable_s3=False, s3_bucket=None):
    results_dir = os.path.join(project_root, "results", dataset)
    baseline_dir = os.path.join(results_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    existing = find_wiki_metadata_entry(baseline_dir, dataset, required_stride=args.stride or 1)
    if existing:
        return existing.get("accuracy", existing.get("average_log_prob", 0.0))
    try:
        base_model, _, tokenizer, device = load_model_for_evaluation(
            use_base_model=True,
            model_type=model_type,
        )
        hyper = {
            "task_type": dataset,
            "question_length": WIKI_DEFAULT_QUESTION_LENGTH,
            "target_length": WIKI_DEFAULT_TARGET_LENGTH,
        }
        mean_log_prob, eval_meta = compute_wiki_logprob(
            base_model,
            tokenizer,
            device,
            hyper,
            num_samples=WIKI_DEFAULT_NUM_SAMPLES,
            stride=args.stride or 1,
            start_index=WIKI_DEFAULT_START_INDEX,
        )
    finally:
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = {
        "adapter_name": "baseline",
        "task_type": dataset,
        "model_type": model_type,
        "model_path": None,
        "accuracy": mean_log_prob,
        "metric": "wiki_log_prob",
        "num_examples": eval_meta["num_samples"],
        "timestamp": datetime.datetime.now().isoformat(),
        "evaluation": eval_meta,
    }
    write_adapter_metadata(baseline_dir, metadata, eval_meta["stride"])
    sync_run_dir(baseline_dir, project_root, enable_s3, s3_bucket)
    return mean_log_prob


def get_baseline_score(
    dataset,
    run_dir,
    metadata_entries,
    args,
    project_root,
    force_skip_eval=False,
    enable_s3=False,
    s3_bucket=None,
):
    """
    Get baseline (base model) score for the dataset.
    Runs evaluation if not present in logs.
    """
    model_type = detect_model_type(run_dir, metadata_entries)
    if not model_type:
        print(f"Warning: could not determine model type for baseline {dataset}; skipping.")
        return 0.0

    base_dir = run_dir or os.path.join(project_root, "results", dataset)
    os.makedirs(base_dir, exist_ok=True)
    results_file = os.path.join(base_dir, f"{dataset}_results_{model_type}.jsonl")
    
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
        if enable_s3 and s3_bucket:
            cmd.append("--sync_metadata")
            cmd.extend(["--s3_bucket", s3_bucket])
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

def get_max_adapter_score(
    run_dir,
    dataset,
    args,
    project_root,
    model_type_hint,
    metadata_entries_hint,
    missing_adapters_hint,
    enable_s3=False,
    s3_bucket=None,
    force_skip_eval=False,
):
    """
    Evaluate all adapters in a run directory and return the max accuracy.
    """
    required_stride = args.stride if args.stride else 1
    metadata_entries = list(metadata_entries_hint) if metadata_entries_hint else []
    missing_adapters = list(missing_adapters_hint) if missing_adapters_hint else []
    if not metadata_entries:
        metadata_entries, missing_adapters = collect_adapter_metadata(list_adapter_dirs(run_dir), required_stride)

    model_type = model_type_hint or detect_model_type(run_dir, metadata_entries)
    results_file = os.path.join(run_dir, f"{dataset}_results_{model_type}.jsonl")
    
    adapters = list_adapter_dirs(run_dir)
    if not adapters:
        if not force_skip_eval:
            print(f"No adapters found in {run_dir}")
            return 0.0

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
        if enable_s3 and s3_bucket:
            cmd.append("--sync_metadata")
            cmd.extend(["--s3_bucket", s3_bucket])
        
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
        sync_run_dir(run_dir, project_root, enable_s3, s3_bucket)

    return best_acc_value

def main():
    parser = argparse.ArgumentParser(description="Compile sweep results into a table")
    parser.add_argument("--num_samples", type=int, help="Number of samples for evaluation")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every nth example (1 = full test set)")
    parser.add_argument("--dry_run", action="store_true", help="Simulate the sweep without writing files or syncing to S3")
    parser.add_argument("--skip_eval", action="store_true", help="Do not launch new evaluations; rely solely on existing metadata")
    parser.add_argument("--task_type", type=str, default=None, help="Only process a specific task/dataset (e.g. gsm8k)")
    parser.add_argument("--method", type=str, default=None, help="Only process a specific method/hyperparameter (e.g. PPO, EI, Markovian)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--run_dir", type=str, nargs="+", default=None, help="Specific run directory (or directories) to process. Overrides automatic scanning.")
    parser.add_argument("--shuffle", action="store_true", help="Process run directories in random order (useful for distributed workers)")
    parser.add_argument("--reverse", action="store_true", help="Process run directories in reverse order (useful for simple 2-machine parallelism)")
    parser.add_argument("--s3_sync", action="store_true", help="Upload results/metadata to S3 (requires bucket)")
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket to use (overrides SWEEP_S3_BUCKET)")
    args = parser.parse_args()

    resolved_s3_bucket = args.s3_bucket or DEFAULT_S3_BUCKET
    if resolved_s3_bucket:
        resolved_s3_bucket = resolved_s3_bucket.rstrip("/")
    enable_s3_sync = bool(args.s3_sync and resolved_s3_bucket and not args.dry_run)
    if args.s3_sync and not resolved_s3_bucket:
        print("Warning: --s3_sync enabled but no bucket configured; skipping S3 uploads.")

    # Supported tasks in evaluation.py
    SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
    WIKI_TASKS = ["wiki_continuation", "wiki_compression"]
    
    # Data structure: table[dataset][method] = score
    table = defaultdict(lambda: defaultdict(float))
    wiki_table = defaultdict(lambda: defaultdict(float))
    
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
        dataset_runs = [rd for rd in run_dirs if rd[0] == dataset]
        sample_metadata = []
        for _, _, path in dataset_runs:
            adapters = list_adapter_dirs(path)
            metadata_entries, _ = collect_adapter_metadata(adapters, args.stride if args.stride else 1)
            sample_metadata.extend(metadata_entries)
            if sample_metadata:
                break

        model_type = detect_model_type(dataset_runs[0][2] if dataset_runs else None, sample_metadata)

        inferred_model_type = model_type or args.model_type
        if dataset in WIKI_TASKS:
            score = get_wiki_baseline_score(dataset, inferred_model_type, args, project_root, enable_s3=enable_s3_sync, s3_bucket=resolved_s3_bucket)
            wiki_table[dataset]["Baseline"] = score
        else:
            should_evaluate = dataset_runs and (not args.task_type or dataset == args.task_type)
            run_dir_for_baseline = dataset_runs[0][2] if dataset_runs else os.path.join(project_root, "results", dataset)
            score = get_baseline_score(
                dataset,
                run_dir_for_baseline,
                sample_metadata,
                args,
                project_root,
                force_skip_eval=not should_evaluate,
                enable_s3=enable_s3_sync,
                s3_bucket=resolved_s3_bucket,
            )
        table[dataset]["Baseline"] = score
        
    # 3. Process Runs
    for dataset, method, path in run_dirs:
        metadata_entries, missing_adapters = collect_adapter_metadata(list_adapter_dirs(path), args.stride if args.stride else 1)
        model_type = detect_model_type(path, metadata_entries)

        if dataset in WIKI_TASKS:
            score = get_wiki_run_score(path, dataset, model_type or args.model_type, args, project_root, enable_s3=enable_s3_sync, s3_bucket=resolved_s3_bucket)
            wiki_table[dataset][method] = score
        else:
            should_evaluate = True
            if args.task_type and dataset != args.task_type:
                should_evaluate = False
            if args.method and method != args.method:
                should_evaluate = False

            score = get_max_adapter_score(
                path,
                dataset,
                args,
                project_root,
                model_type,
                metadata_entries,
                missing_adapters,
                enable_s3=enable_s3_sync,
                s3_bucket=resolved_s3_bucket,
                force_skip_eval=not should_evaluate,
            )
            table[dataset].setdefault("Baseline", table[dataset].get("Baseline", 0.0))
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
