#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import perturbation_analysis as pa


SUPPORTED_TASKS = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]


def find_role_run(dataset: str, role: str) -> str:
    base_dir = os.path.join(PROJECT_ROOT, "results", dataset)
    if not os.path.isdir(base_dir):
        return None
    pattern = os.path.join(base_dir, f"*{role}*")
    candidates = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if role == "Markovian":
        candidates = [d for d in candidates if "NonMarkovian" not in os.path.basename(d)]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def list_adapter_indices(run_dir: str) -> list[int]:
    adapter_dirs = glob.glob(os.path.join(run_dir, "adapter_*"))
    indices = []
    for path in adapter_dirs:
        name = os.path.basename(path)
        parts = name.split("_")
        if parts[-1].isdigit():
            indices.append(int(parts[-1]))
    return sorted(indices)


def build_tasks(datasets, perturbations, args):
    tasks = []
    for dataset in datasets:
        mark_run = find_role_run(dataset, "Markovian")
        non_run = find_role_run(dataset, "NonMarkovian")
        if not mark_run or not non_run:
            print(f"[Skip] Missing Markovian/NonMarkovian runs for {dataset}")
            continue
        pa.sync_run_dir_from_s3(mark_run)
        pa.sync_run_dir_from_s3(non_run)
        mark_indices = list_adapter_indices(mark_run)
        non_indices = list_adapter_indices(non_run)
        common = sorted(set(mark_indices).intersection(non_indices))
        if not common:
            print(f"[Skip] No common adapter indices for {dataset}")
            continue
        mark_log = os.path.join(mark_run, "log.jsonl")
        non_log = os.path.join(non_run, "log.jsonl")
        for idx in common:
            mark_adapter_dir = os.path.join(mark_run, f"adapter_{idx}")
            non_adapter_dir = os.path.join(non_run, f"adapter_{idx}")
            if not (os.path.isdir(mark_adapter_dir) and os.path.isdir(non_adapter_dir)):
                continue
            for perturb in perturbations:
                metadata_key = pa.build_perturb_metadata_key(
                    task_type=dataset,
                    perturb_type=perturb,
                    metric="accuracy",
                    paired_role="NonMarkovian",
                    paired_adapter_index=idx,
                    markovian_run=mark_run,
                    non_markovian_run=non_run,
                )
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
                    "mark_run": mark_run,
                    "non_run": non_run,
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
    parser.add_argument("--shuffle", action="store_true", help="Randomize task order")
    parser.add_argument("--reverse", action="store_true", help="Reverse task order")
    parser.add_argument("--limit", type=int, help="Limit number of tasks processed")
    parser.add_argument("--dry_run", action="store_true", help="Print tasks without executing")
    parser.add_argument("--force", action="store_true", help="Recompute even if metadata exists")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffle")
    args = parser.parse_args()

    perturbations = args.perturb
    if args.all or not perturbations:
        perturbations = list(pa.PERTURBATION_SETS.keys())

    tasks = build_tasks(args.datasets, perturbations, args)
    if not tasks:
        print("No tasks to process.")
        return

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(tasks)
    elif args.reverse:
        tasks = list(reversed(tasks))

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Processing {len(tasks)} tasks...")

    for job in tasks:
        msg = f"[{job['dataset']}] adapter_{job['adapter_index']} perturb={job['perturb']}"
        if args.dry_run:
            print(f"[DRY] {msg}")
            continue
        print(msg)
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
        )


if __name__ == "__main__":
    main()

