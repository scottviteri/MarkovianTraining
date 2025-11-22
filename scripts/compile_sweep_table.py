import os
import glob
import json
import argparse
import subprocess
import random
import datetime
import re
from collections import defaultdict
from typing import Dict, Any, Optional

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_S3_BUCKET = os.environ.get("SWEEP_S3_BUCKET", "s3://scottviteri")
METADATA_PATTERN = re.compile(r"eval_metadata(?:_stride(\d+))?\.json$")
_S3_WARNING_PRINTED = False
DEFAULT_WIKI_NUM_SAMPLES = 1024
SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
WIKI_TASKS = ["wiki_continuation", "wiki_compression"]
# Ground-truth test/validation set sizes pulled from the scripted dataset loaders.
# These reflect the exact number of evaluation examples consumed by src/evaluation.py
# when --num_samples is omitted, as of 2025-11-22.
TASK_TEST_SET_SIZES = {
    "gsm8k": 1319,       # openai/gsm8k (test split)
    "mmlu": 1531,        # cais/mmlu (validation split)
    "arc": 294,          # ai2_arc ARC-Challenge (validation split, filtered to A-D choices)
    "svamp": 300,        # SVAMP test split
    "aqua": 254,         # AQuA-RAT test split
    "mathqa": 2985,      # MathQA test split
    "arithmetic": 200,   # synthetic evaluation set generated in evaluation.py (chunk_size default)
}


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


def _s3_uri_for_path(path, project_root, bucket):
    rel_path = safe_relpath(path, project_root).replace("\\", "/")
    uri = f"{bucket.rstrip('/')}/{rel_path}"
    if uri.startswith("s3:/") and not uri.startswith("s3://"):
        uri = uri.replace("s3:/", "s3://", 1)
    return uri


def upload_adapter_metadata(adapter_dir, project_root, bucket=None):
    """Upload metadata files from an adapter directory to S3."""
    bucket = bucket or DEFAULT_S3_BUCKET
    if not bucket: return
    s3_dest = _s3_uri_for_path(adapter_dir, project_root, bucket)
    
    include_args = [
        "--exclude", "*",
        "--include", "eval_metadata*.json",
        "--include", "eval_results*.jsonl"
    ]
    
    print(f"Uploading metadata for {os.path.basename(adapter_dir)} to S3...")
    try:
        subprocess.run(
            ["aws", "s3", "sync", adapter_dir, s3_dest, *include_args],
            check=True
        )
    except Exception as e:
        print(f"Warning: failed to upload metadata to {s3_dest}: {e}")


def list_s3_runs(dataset, s3_results_prefix, method_filter=None):
    """
    Discover runs on S3.
    Returns list of (dataset, method, s3_run_path) tuples.
    """
    if not s3_results_prefix:
        return []

    prefix = s3_results_prefix.rstrip("/")
    s3_path = f"{prefix}/{dataset}/"
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    discovered = []
    method_filter_lower = method_filter.lower() if method_filter else None

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("PRE "):
            continue
        run_name = line[4:].strip("/")
        if not run_name:
            continue
            
        # Parse method from run name {dataset}_{method}_{timestamp}
        # Simple heuristic: matches parse_run_dir logic
        if run_name == "baseline":
             discovered.append((dataset, "baseline", run_name))
             continue

        if not run_name.startswith(dataset + "_"):
            continue
            
        parts = run_name.split('_')
        if len(parts) < 3:
            continue
            
        # Method is middle part(s)
        method_part = run_name[len(dataset)+1:].split('_')[:-2]
        method = "_".join(method_part)
        
        if method_filter_lower and method_filter_lower not in method.lower():
            continue
            
        discovered.append((dataset, method, run_name))
        
    return discovered


def list_s3_adapters(dataset, run_name, s3_results_prefix):
    """
    List adapter directories for a run on S3.
    Returns list of adapter names (e.g. 'adapter_50').
    """
    prefix = s3_results_prefix.rstrip("/")
    s3_path = f"{prefix}/{dataset}/{run_name}/"
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    adapters = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("PRE adapter_"):
            continue
        adapter_name = line[4:].strip("/")
        adapters.append(adapter_name)
    return sorted(adapters, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x)


def check_s3_metadata(dataset, run_name, adapter_name, s3_results_prefix):
    """
    Check if evaluation metadata already exists on S3 for this adapter.
    Returns (has_metadata, metadata_content_if_available)
    """
    prefix = s3_results_prefix.rstrip("/")
    if adapter_name:
        s3_adapter_path = f"{prefix}/{dataset}/{run_name}/{adapter_name}/"
    else:
        s3_adapter_path = f"{prefix}/{dataset}/{run_name}/"
    
    # List files in adapter dir
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_adapter_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False, None

    has_metadata = False
    for line in result.stdout.splitlines():
        if "eval_metadata" in line and line.endswith(".json"):
            has_metadata = True
            break
            
    return has_metadata, None


def download_adapter_metadata(dataset, run_name, adapter_name, project_root, s3_results_prefix):
    """Download just metadata files for an adapter."""
    prefix = s3_results_prefix.rstrip("/")
    if adapter_name:
        s3_path = f"{prefix}/{dataset}/{run_name}/{adapter_name}/"
    else:
        s3_path = f"{prefix}/{dataset}/{run_name}/"
    local_path = os.path.join(project_root, "results", dataset, run_name, adapter_name)
    os.makedirs(local_path, exist_ok=True)
    
    include_args = [
        "--exclude", "*",
        "--include", "eval_metadata*.json",
        "--include", "eval_results*.jsonl"
    ]
    
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, local_path, *include_args],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Warning: failed to download metadata from {s3_path}: {e}")


def download_adapter_weights(dataset, run_name, adapter_name, project_root, s3_results_prefix):
    """Download weights for a specific adapter."""
    prefix = s3_results_prefix.rstrip("/")
    if adapter_name:
        s3_path = f"{prefix}/{dataset}/{run_name}/{adapter_name}/"
    else:
        s3_path = f"{prefix}/{dataset}/{run_name}/"
    local_path = os.path.join(project_root, "results", dataset, run_name, adapter_name)
    
    print(f"Syncing weights for {adapter_name} from S3...")
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, local_path],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading weights from {s3_path}: {e}")
        return False


def load_local_metadata(adapter_dir):
    """Load best metadata from a local adapter directory."""
    pattern = os.path.join(adapter_dir, "eval_metadata*.json")
    files = glob.glob(pattern)
    if not files:
        return None
        
    # Pick best stride/most samples
    best_meta = None
    best_score = (-1, -1, -float('inf')) # (stride, num_samples, accuracy)
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            stride = data.get("evaluation", {}).get("stride", 1)
            num_samples = data.get("evaluation", {}).get("num_samples", 0)
            acc = data.get("accuracy", 0)
            
            score = (stride, num_samples, acc)
            # Prefer lower stride (more detailed), then higher samples, then higher accuracy
            # Actually for score comparison: 
            # We want minimal stride (1 is best), max samples.
            # Let's use simple heuristic: just take the one with most samples
            
            current_score = (
                -stride, 
                num_samples if num_samples is not None else 0,
                acc if isinstance(acc, (int, float)) else 0
            )
            
            if best_meta is None or current_score > best_score:
                best_score = current_score
                best_meta = data
                best_meta["_metadata_path"] = fpath
        except:
            continue
            
    return best_meta


def _parse_metadata_num_samples(entry: Dict[str, Any]) -> Optional[int]:
    evaluation_block = entry.get("evaluation") or {}
    value = evaluation_block.get("num_samples")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def get_model_type_from_s3(dataset, run_name, s3_results_prefix, project_root):
    """Always fetch log.jsonl to determine model type. S3 is ground truth."""
    prefix = s3_results_prefix.rstrip("/")
    s3_log = f"{prefix}/{dataset}/{run_name}/log.jsonl"
    local_run_dir = os.path.join(project_root, "results", dataset, run_name)
    local_log = os.path.join(local_run_dir, "log.jsonl")
    
    # Always fetch from S3 - S3 is ground truth
    os.makedirs(local_run_dir, exist_ok=True)
    try:
        subprocess.run(["aws", "s3", "cp", s3_log, local_log], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise FileNotFoundError(f"Could not find log.jsonl for run {run_name} at {s3_log} - required for model type inference. Ground truth is S3.")

    try:
        with open(local_log, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            if "model_type" not in data:
                raise ValueError(f"log.jsonl for {run_name} is missing 'model_type' field")
            return data["model_type"]
    except Exception as e:
        raise ValueError(f"Failed to parse log.jsonl for {run_name}: {e}")


def evaluate_adapter(dataset, run_name, adapter_name, project_root, args, s3_results_prefix, model_type):
    """
    Evaluate a specific adapter.
    1. Download weights
    2. Run eval
    3. Upload metadata
    """
    local_adapter_dir = os.path.join(project_root, "results", dataset, run_name, adapter_name)
    
    # 1. Download weights
    if not download_adapter_weights(dataset, run_name, adapter_name, project_root, s3_results_prefix):
        return None

    # 2. Run eval
    print(f"Evaluating {dataset}/{run_name}/{adapter_name}...")
    eval_script = os.path.join(project_root, "src", "evaluation.py")
    
    cmd = [
        "python", eval_script,
        "--task_type", dataset,
        "--model_path", local_adapter_dir,
        "--model_type", model_type,
    ]
    
    if args.force_eval:
        cmd.append("--force_eval")
    
    if args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    if args.stride:
        cmd.extend(["--stride", str(args.stride)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
        
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return None
        
    # 3. Sync metadata back
    upload_adapter_metadata(local_adapter_dir, project_root, args.s3_bucket)
    
    return load_local_metadata(local_adapter_dir)


def main():
    parser = argparse.ArgumentParser(description="Compile sweep results from S3")
    parser.add_argument("--task_type", type=str, help="Limit to specific task")
    parser.add_argument("--method", type=str, help="Limit to specific method")
    parser.add_argument("--column", type=str, help="Alias for method")
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket")
    parser.add_argument("--dry_run", action="store_true", help="Don't actually run evals")
    parser.add_argument("--force_eval", action="store_true", help="Re-run existing evals")
    parser.add_argument("--num_samples", type=int, help="Num samples for eval")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int)
    
    args = parser.parse_args()
    
    method_filter = args.column or args.method
    bucket = args.s3_bucket or DEFAULT_S3_BUCKET
    s3_results_prefix = f"{bucket.rstrip('/')}/results"
    project_root = PROJECT_ROOT
    
    # Determine tasks
    if args.task_type:
        tasks = [args.task_type]
    else:
        tasks = sorted(set(SUPPORTED_TASKS + WIKI_TASKS))
        
    # Shuffle tasks to reduce contention between workers
    random.shuffle(tasks)
        
    results_table = defaultdict(lambda: defaultdict(lambda: -float('inf')))
    
    for task in tasks:
        print(f"\nScanning {task}...")
        is_wiki_task = task in WIKI_TASKS
        desired_task_samples = args.num_samples
        if desired_task_samples is None:
            if is_wiki_task:
                desired_task_samples = DEFAULT_WIKI_NUM_SAMPLES
            else:
                desired_task_samples = TASK_TEST_SET_SIZES.get(task)
        
        # 1. Check for baseline
        baseline_dir = f"{s3_results_prefix}/{task}/baseline/"
        try:
            # Check if eval_metadata exists in baseline dir
            result = subprocess.run(
                ["aws", "s3", "ls", baseline_dir],
                capture_output=True,
                text=True,
                check=True
            )
            has_baseline = False
            for line in result.stdout.splitlines():
                if "eval_metadata" in line and line.endswith(".json"):
                    has_baseline = True
                    break
            
            if has_baseline:
                print(f"  Checking run: baseline (baseline)")
                # Download baseline metadata
                local_baseline_dir = os.path.join(project_root, "results", task, "baseline")
                os.makedirs(local_baseline_dir, exist_ok=True)
                
                # Use download_adapter_metadata but adapt paths
                s3_baseline_path = baseline_dir
                include_args = [
                    "--exclude", "*",
                    "--include", "eval_metadata*.json"
                ]
                subprocess.run(
                    ["aws", "s3", "sync", s3_baseline_path, local_baseline_dir, *include_args],
                    check=True,
                    capture_output=True
                )
                
                meta = load_local_metadata(local_baseline_dir)
                if meta:
                    acc = meta.get("accuracy", 0)
                    results_table[task]["baseline"] = acc
        except subprocess.CalledProcessError:
            pass # No baseline found
            
        runs = list_s3_runs(task, s3_results_prefix, method_filter)
        
        # Also shuffle runs
        random.shuffle(runs)
        
        for dataset, method, run_name in runs:
            print(f"  Checking run: {run_name} ({method})")
            
            if method == "baseline":
                # For baseline, the run_name IS the adapter (kind of)
                # Check if baseline ITSELF has metadata
                has_meta, _ = check_s3_metadata(dataset, run_name, "", s3_results_prefix)
                
                meta = None
                if has_meta and not args.force_eval:
                     print(f"    Found metadata in {run_name} root")
                     download_adapter_metadata(dataset, run_name, "", project_root, s3_results_prefix)
                     local_path = os.path.join(project_root, "results", dataset, run_name)
                     meta = load_local_metadata(local_path)
                elif not args.dry_run:
                    # Run baseline eval
                    print(f"    Evaluating baseline (metadata missing)")
                    # Need to know model_type. Can't get from log.jsonl because it doesn't exist for baseline.
                    # Defaulting to llama if not found, or maybe we skip?
                    # Let's assume llama for now or try to find a way to specify it. 
                    # Actually, usually we know the model from context or we just use a default.
                    # For now let's hardcode 'llama' or look for a way to map it. 
                    # Given the repo structure, 'llama' seems to be the default base.
                    default_model = "llama" 
                    meta = evaluate_adapter(dataset, run_name, "", project_root, args, s3_results_prefix, default_model)

                if meta:
                    acc = meta.get("accuracy", 0)
                    results_table[dataset][method] = max(results_table[dataset][method], acc)
                continue

            adapters = list_s3_adapters(dataset, run_name, s3_results_prefix)
            if not adapters:
                continue
                
            model_type = get_model_type_from_s3(dataset, run_name, s3_results_prefix, project_root)
            
            best_acc = -float('inf')
            
            for adapter in adapters:
                # Check if done
                has_meta, _ = check_s3_metadata(dataset, run_name, adapter, s3_results_prefix)
                
                meta = None
                # Check for existing metadata
                if has_meta and not args.force_eval:
                    print(f"    Skipping {adapter} (metadata found on S3)")
                    # Download metadata only
                    download_adapter_metadata(dataset, run_name, adapter, project_root, s3_results_prefix)
                    local_path = os.path.join(project_root, "results", dataset, run_name, adapter)
                    meta = load_local_metadata(local_path)
                    
                    # If metadata has low sample count and we want more, force re-eval
                    if meta and desired_task_samples:
                        current_samples = _parse_metadata_num_samples(meta)
                        if current_samples is not None and current_samples < desired_task_samples:
                            print(f"    Re-evaluating {adapter} (low sample count: {current_samples} < {desired_task_samples})")
                            # Force eval by setting meta to None so we fall through to the eval block
                            meta = None 
                            
                if meta is None and not args.dry_run:
                    # Needs eval
                    print(f"    Evaluating {adapter} (metadata missing or insufficient samples)")
                    meta = evaluate_adapter(dataset, run_name, adapter, project_root, args, s3_results_prefix, model_type)
                
                if meta:
                    acc = meta.get("accuracy", 0)
                    num_samples = meta.get("evaluation", {}).get("num_samples", 0)
                    if num_samples and num_samples < 100:
                         print(f"    Warning: Low sample count ({num_samples}) for {run_name}/{adapter}")

                    if acc > best_acc:
                        best_acc = acc
                        
            # Record score
            if dataset in WIKI_TASKS:
                results_table[dataset][method] = max(results_table[dataset][method], best_acc)
            else:
                results_table[dataset][method] = max(results_table[dataset][method], best_acc)

    # Print tables
    print("\n" + "="*50)
    print("Results Table")
    print("="*50)
    df = pd.DataFrame.from_dict(results_table, orient='index')
    if not df.empty:
        print(df.to_markdown())
        df.to_csv("sweep_results_table.csv")

if __name__ == "__main__":
    main()
