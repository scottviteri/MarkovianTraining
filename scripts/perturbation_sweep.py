#!/usr/bin/env python3
import argparse
import os
import sys
import random
import subprocess
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import perturbation_analysis as pa

DEFAULT_S3_BUCKET = os.environ.get("SWEEP_S3_BUCKET", "s3://scottviteri")
SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "arithmetic"]

def ls_s3_dirs(s3_path):
    """List subdirectories in an S3 path."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []
    
    dirs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("PRE "):
            dirs.append(line[4:].strip("/"))
    return dirs

def list_s3_runs(dataset, bucket):
    prefix = f"{bucket.rstrip('/')}/results"
    s3_path = f"{prefix}/{dataset}/"
    return ls_s3_dirs(s3_path)

def find_role_run_s3(dataset, role, bucket):
    runs = list_s3_runs(dataset, bucket)
    candidates = [r for r in runs if role in r]
    
    if role == "Markovian":
        candidates = [r for r in candidates if "NonMarkovian" not in r]
        
    if not candidates:
        return None
    
    candidates.sort()
    latest_run_name = candidates[-1]
    
    # Return local path structure
    return os.path.join(PROJECT_ROOT, "results", dataset, latest_run_name)

def list_s3_adapter_indices(dataset, run_name, bucket):
    prefix = f"{bucket.rstrip('/')}/results"
    s3_path = f"{prefix}/{dataset}/{run_name}/"
    dirs = ls_s3_dirs(s3_path)
    
    indices = []
    for d in dirs:
        if d.startswith("adapter_"):
            parts = d.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                indices.append(int(parts[-1]))
    return sorted(indices)

def download_file(s3_path, local_path):
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    subprocess.run(["aws", "s3", "cp", s3_path, local_path], check=True)

def download_adapter_weights(dataset, run_name, adapter_idx, bucket):
    prefix = f"{bucket.rstrip('/')}/results"
    adapter_name = f"adapter_{adapter_idx}"
    s3_path = f"{prefix}/{dataset}/{run_name}/{adapter_name}/"
    local_path = os.path.join(PROJECT_ROOT, "results", dataset, run_name, adapter_name)
    
    print(f"Syncing weights for {run_name}/{adapter_name} from S3...")
    try:
        subprocess.run(
            ["aws", "s3", "sync", s3_path, local_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error downloading weights: {e}")

def build_tasks(datasets, perturbations, args):
    tasks = []
    bucket = args.s3_bucket
    
    for dataset in datasets:
        print(f"Scanning {dataset} on S3...")
        mark_run_path = find_role_run_s3(dataset, "Markovian", bucket)
        non_run_path = find_role_run_s3(dataset, "NonMarkovian", bucket)
        
        if not mark_run_path or not non_run_path:
            print(f"[Skip] Missing Markovian/NonMarkovian runs for {dataset}")
            continue
            
        mark_run_name = os.path.basename(mark_run_path)
        non_run_name = os.path.basename(non_run_path)
        
        # Download log files
        s3_prefix = f"{bucket.rstrip('/')}/results"
        download_file(f"{s3_prefix}/{dataset}/{mark_run_name}/log.jsonl", os.path.join(mark_run_path, "log.jsonl"))
        download_file(f"{s3_prefix}/{dataset}/{non_run_name}/log.jsonl", os.path.join(non_run_path, "log.jsonl"))
        
        mark_indices = list_s3_adapter_indices(dataset, mark_run_name, bucket)
        non_indices = list_s3_adapter_indices(dataset, non_run_name, bucket)
        
        common = sorted(set(mark_indices).intersection(non_indices))
        if not common:
            print(f"[Skip] No common adapter indices for {dataset}")
            continue
            
        mark_log = os.path.join(mark_run_path, "log.jsonl")
        non_log = os.path.join(non_run_path, "log.jsonl")
        
        for idx in common:
            # Check metadata using pa logic (which will sync metadata if needed)
            mark_adapter_dir = os.path.join(mark_run_path, f"adapter_{idx}")
            non_adapter_dir = os.path.join(non_run_path, f"adapter_{idx}")
            os.makedirs(mark_adapter_dir, exist_ok=True)
            os.makedirs(non_adapter_dir, exist_ok=True)
            
            for perturb in perturbations:
                metadata_key = pa.build_perturb_metadata_key(
                    task_type=dataset,
                    perturb_type=perturb,
                    metric="accuracy",
                    paired_role="NonMarkovian",
                    paired_adapter_index=idx,
                    markovian_run=mark_run_path,
                    non_markovian_run=non_run_path,
                )
                
                # This implicitly calls pull_adapter_metadata via pa
                mark_meta = pa.get_cached_metadata(mark_adapter_dir)
                non_meta = pa.get_cached_metadata(non_adapter_dir)
                
                already_done = (
                    pa.metadata_has_record(mark_meta, metadata_key, args.stride)
                    and pa.metadata_has_record(non_meta, metadata_key, args.stride)
                )
                if already_done and not args.force:
                    continue
                    
                tasks.append({
                    "dataset": dataset,
                    "perturb": perturb,
                    "adapter_index": idx,
                    "markovian_log": mark_log,
                    "non_markovian_log": non_log,
                    "mark_run_path": mark_run_path,
                    "non_run_path": non_run_path,
                    "mark_run_name": mark_run_name,
                    "non_run_name": non_run_name,
                    "stride": args.stride,
                })
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Sweep perturbation accuracy across datasets/adapters")
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_TASKS, default=SUPPORTED_TASKS, help="Datasets to process")
    parser.add_argument("--perturb", nargs="+", choices=list(pa.PERTURBATION_SETS.keys()), help="Perturbation types to run")
    parser.add_argument("--all", action="store_true", help="Run all perturbation types")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every nth example")
    parser.add_argument("--limit", type=int, help="Limit number of tasks processed")
    parser.add_argument("--dry_run", action="store_true", help="Print tasks without executing")
    parser.add_argument("--force", action="store_true", help="Recompute even if metadata exists")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffle")
    parser.add_argument("--s3_bucket", type=str, default=DEFAULT_S3_BUCKET, help="S3 bucket")
    args = parser.parse_args()

    perturbations = args.perturb
    if args.all or not perturbations:
        perturbations = list(pa.PERTURBATION_SETS.keys())

    tasks = build_tasks(args.datasets, perturbations, args)
    if not tasks:
        print("No tasks to process.")
        return

    # Always shuffle tasks to reduce contention across machines
    random.shuffle(tasks)

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Processing {len(tasks)} tasks...")

    for job in tasks:
        msg = f"[{job['dataset']}] adapter_{job['adapter_index']} perturb={job['perturb']}"
        if args.dry_run:
            print(f"[DRY] {msg}")
            continue
        print(msg)
        
        # Download adapter weights specifically
        download_adapter_weights(job['dataset'], job['mark_run_name'], job['adapter_index'], args.s3_bucket)
        download_adapter_weights(job['dataset'], job['non_run_name'], job['adapter_index'], args.s3_bucket)
        
        pa.run_qa_perturbation_accuracy(
            markovian_log_file=job["markovian_log"],
            non_markovian_log_file=job["non_markovian_log"],
            perturb_type=job["perturb"],
            task_type=job["dataset"],
            num_samples=None,
            batch_size=args.batch_size,
            evaluator="actor",
            markovian_adapter_index=job["adapter_index"],
            non_markovian_adapter_index=job["adapter_index"],
            stride=job["stride"],
            sync_s3=False,  # We handled sync manually
        )

if __name__ == "__main__":
    main()
