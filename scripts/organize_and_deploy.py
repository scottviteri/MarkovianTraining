#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
import datetime
from pathlib import Path
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

# Configuration
LOCAL_LOGS = Path("/home/scottviteri/Projects/MarkovianTraining/ICLR2026Rebuttals/Logs")
REMOTE_RESULTS_DIR = "/root/MarkovianTraining/results"

# Host mapping (screen-position -> hostname)
# 1-1 left, 1-2 left, ... 
# Position: 1=left, 2=mid, 3=right, 4=riight
# Suffix: none for 1, "2" for 2, etc.
HOST_MAP = {
    "1-1": "left",   "1-2": "mid",   "1-3": "right",   "1-4": "riight",
    "2-1": "left2",  "2-2": "mid2",  "2-3": "right2",  "2-4": "riight2",
    "3-1": "left3",  "3-2": "mid3",  "3-3": "right3",  "3-4": "riight3",
    "4-1": "left4",  "4-2": "mid4",  "4-3": "right4",  "4-4": "riight4",
}

# Default destinations for pushing results when no targets are specified
TARGET_HOSTS = {
    "gsm8k": ["left3"],
    "svamp": ["left3"],
    "wiki_continuation": ["left3"],
    "arc": ["right3"],
    "arithmetic": ["right3"],
    "mmlu": ["right3"],
}


def get_target_hosts(dataset, specified_targets=None):
    """Return list of target hosts for a dataset."""
    if specified_targets:
        return specified_targets
    return TARGET_HOSTS.get(dataset, ["left3"])


def run_ssh_command(host, command, capture_output=True):
    """Run an SSH command on a remote host."""
    result = subprocess.run(
        ["ssh", host, command],
        capture_output=capture_output,
        text=True,
    )
    return result


def list_remote_runs(host, dataset):
    """List run directories on the remote host for a dataset."""
    remote_path = f"{REMOTE_RESULTS_DIR}/{dataset}"
    cmd = (
        f"bash -lc 'if [ -d \"{remote_path}\" ]; then "
        f"cd \"{remote_path}\" && ls -1; fi'"
    )
    result = run_ssh_command(host, cmd)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    runs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # Filter to timestamp-like directories (start with 20)
    runs = [r for r in runs if r.startswith("20")]
    return runs


def get_remote_log_params(host, dataset, run):
    """Fetch first-line JSON params of remote log."""
    log_path = f"{REMOTE_RESULTS_DIR}/{dataset}/{run}/log.jsonl"
    cmd = f"bash -lc 'head -n 1 \"{log_path}\"'"
    result = run_ssh_command(host, cmd)
    if result.returncode != 0:
        return None
    line = result.stdout.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def count_remote_adapters(host, dataset, run):
    """Count adapter_* directories for scoring."""
    run_path = f"{REMOTE_RESULTS_DIR}/{dataset}/{run}"
    cmd = (
        f"bash -lc 'cd \"{run_path}\" 2>/dev/null && "
        f"ls -1 adapter_* 2>/dev/null | wc -l'"
    )
    result = run_ssh_command(host, cmd)
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip() or "0")
    except ValueError:
        return 0


def select_best_remote_run(host, dataset, overrides):
    """Select best run on remote host matching overrides."""
    runs = list_remote_runs(host, dataset)
    best = None
    best_score = -1
    for run in runs:
        params = get_remote_log_params(host, dataset, run)
        if not params:
            continue
        if not check_params_dict(params, overrides):
            continue
        adapters = count_remote_adapters(host, dataset, run)
        try:
            dt = datetime.datetime.strptime(run, "%Y%m%d_%H%M%S")
            ts = dt.timestamp()
        except ValueError:
            ts = 0
        score = adapters * 1e10 + ts
        if score > best_score:
            best = run
            best_score = score
    return best


def check_params_dict(params, overrides):
    """Check params dict against overrides."""
    if not overrides:
        return True

    for k, v in overrides.items():
        if k == "r":
            parallel_flag = params.get("parallel", True)
            val = params.get(k)
            if parallel_flag:
                val = 0.0
            if val is None:
                val = 0.0
            if v > 0 and val <= 0:
                return False
            if v == 0 and val > 0:
                return False
            continue
        if k == "use_ei":
            val = params.get(k)
            if val is None:
                val = 0.0
            if v > 0 and val <= 0:
                return False
            if v == 0 and val > 0:
                return False
            continue
        if k == "parallel":
            val = params.get(k, True)
            if val != v:
                return False
            continue
        if k == "actor_reward_weight":
            val = params.get("actor_reward_weight", params.get("actor_weight"))
        else:
            val = params.get(k)

        if k == "actor_reward_weight" and val is None:
            val = params.get("actor_weight")

        if val is None:
            if v:
                return False
            continue
        if val != v:
            return False
    return True


def sanitize_role(role_name):
    """Sanitize role name for directory usage."""
    return role_name.replace(" ", "").replace("-", "")

def upload_run_to_s3(source_host, dataset, run, s3_run_name, s3_prefix):
    """Trigger aws s3 sync on source host to upload run to a named S3 folder."""
    source_path = f"{REMOTE_RESULTS_DIR}/{dataset}/{run}"
    s3_path = f"{s3_prefix}/{s3_run_name}"
    cmd = (
        f"bash -lc 'aws s3 sync \"{source_path}\" \"{s3_path}\" --delete'"
    )
    print(f"    Uploading {dataset}/{run} from {source_host} to {s3_path} ...")
    result = run_ssh_command(source_host, cmd)
    if result.returncode != 0:
        print(f"    ! Upload failed: {result.stderr.strip()}")
    return result.returncode == 0


def download_run_from_s3(target_host, dataset, s3_run_name, dest_run_name, s3_prefix):
    """Trigger aws s3 sync on target host to download run from S3 to a named destination."""
    dest_path = f"{REMOTE_RESULTS_DIR}/{dataset}/{dest_run_name}"
    s3_path = f"{s3_prefix}/{s3_run_name}"
    cmd = (
        f"bash -lc 'mkdir -p \"{dest_path}\" && "
        f"aws s3 sync \"{s3_path}\" \"{dest_path}\" --delete'"
    )
    print(f"    Downloading {s3_path} to {target_host}:{dest_path} ...")
    result = run_ssh_command(target_host, cmd)
    if result.returncode != 0:
        print(f"    ! Download failed: {result.stderr.strip()}")
    return result.returncode == 0


def handle_find_requests(requests):
    all_hosts = sorted(set(HOST_MAP.values()))
    if not all_hosts:
        print("No hosts configured in HOST_MAP; cannot search.")
        return

    for request in requests:
        if ":" not in request:
            print(f"Invalid --find entry '{request}'. Use dataset:VariantName.")
            continue
        dataset, variant = request.split(":", 1)
        dataset = dataset.strip()
        variant_key = variant.strip().lower()

        if dataset not in MATRIX:
            print(f"[{request}] Dataset '{dataset}' not recognized.")
            continue

        if variant_key not in COLUMN_NAME_TO_INDEX:
            print(f"[{request}] Variant '{variant}' not recognized.")
            continue

        idx = COLUMN_NAME_TO_INDEX[variant_key]
        overrides = COLUMNS[idx][1]

        print(f"[{dataset}:{variant}] Searching all hosts (in parallel)...")
        found_any = False

        def check_host(host):
            run = select_best_remote_run(host, dataset, overrides)
            return host, run

        with ThreadPoolExecutor(max_workers=min(8, len(all_hosts))) as executor:
            futures = [executor.submit(check_host, host) for host in all_hosts]
            for future in futures:
                host, run = future.result()
                if run:
                    print(f"    {host}: {run}")
                    found_any = True

        if not found_any:
            print("    No matching runs found on any host.")

# Default hyperparameters to check against
# Matrix definition
# Columns: Mkv, Non-Mkv, NoRew, PPO, Unnorm, EMA, EI
# Each cell is the "a-b" code from your notes
MATRIX = {
    "gsm8k": ["1-1", "3-1", "1-4", "1-2", "1-3", "3-2", "4-2"], # Mkv, NonMkv, NoRew, PPO, Unnorm, EMA(was NoPar), EI
    "svamp": ["1-3", "3-3", "1-3", "3-2", "2-4", "3-3", "4-4"],
    "wiki_continuation": ["1-4", "3-4", "3-4", "2-2", "3-3", "1-4", "4-3"], 
    "arc":   ["2-2", "4-2", "2-2", "2-2", "3-4", "4-1", "3-4"],
    "arithmetic": ["2-3", "2-3", "2-3", "3-3", "2-4", "2-4", "2-4"],
    "mmlu":  ["4-3", "3-4", "4-4", "3-4", "2-2", "2-1", "1-3"], 
}

# Column definitions (Hyperparameter overrides for each column)
COLUMNS = [
    ("Markovian", {"markovian": True}),
    ("Non-Markovian", {"markovian": False}),
    ("No Reward", {"actor_reward_weight": 0.0}),
    ("PPO", {"use_ppo": True}),
    ("Unnorm", {"normalize_loss": False}),
    ("EMA", {"parallel": False}), 
    ("EI", {"use_ei": 1.0}),
]

COLUMN_NAME_TO_INDEX = {name.lower(): idx for idx, (name, _) in enumerate(COLUMNS)}


def parse_column_filters(column_args):
    if not column_args:
        return None

    indexes = set()
    for entry in column_args:
        entry_clean = entry.strip().lower()
        if entry_clean.isdigit():
            idx = int(entry_clean)
            if idx < 0 or idx >= len(COLUMNS):
                raise ValueError(f"Column index {idx} out of range (0-{len(COLUMNS)-1})")
            indexes.add(idx)
            continue

        if entry_clean not in COLUMN_NAME_TO_INDEX:
            raise ValueError(f"Unknown column name: {entry}")

        indexes.add(COLUMN_NAME_TO_INDEX[entry_clean])

    return indexes

def get_run_score(run_path):
    """
    Calculate a score for a run to break ties.
    Score = (number of checkpoint directories) * 10^10 + timestamp
    """
    try:
        # Count adapter directories
        adapters = list(run_path.glob("adapter_*"))
        num_adapters = len(adapters)
        
        # Get timestamp from folder name
        timestamp_str = run_path.name
        # Format: YYYYMMDD_HHMMSS
        dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        ts = dt.timestamp()
        
        return num_adapters * 1e10 + ts
    except Exception:
        return 0

def check_params(log_path, overrides):
    """
    Check if the run at log_path matches the default params + overrides.
    """
    try:
        with open(log_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                return False
            params = json.loads(first_line)
        return check_params_dict(params, overrides)
    except Exception as e:
        print(f"Error checking params for {log_path}: {e}")
        return False

def process_local(selected_datasets, column_filters=None, skip_push=False, skip_pull=False, specified_targets=None):
    datasets_to_process = []
    for dataset in selected_datasets:
        if dataset not in MATRIX:
            print(f"! Dataset '{dataset}' not recognized, skipping")
            continue
        datasets_to_process.append(dataset)

    if not datasets_to_process:
        print("No valid datasets to process. Exiting.")
        return

    selected_any = False

    # Harvest (and optionally push) per dataset/column
    for dataset in datasets_to_process:
        hosts_codes = MATRIX[dataset]
        print(f"\n=== Processing {dataset} ===")

        for col_idx, (col_name, overrides) in enumerate(COLUMNS):
            if column_filters is not None and col_idx not in column_filters:
                continue

            host_code = hosts_codes[col_idx]
            hostname = HOST_MAP.get(host_code, host_code)

            print(f"  > Looking for {col_name} run on {hostname} ({host_code})...")

            if not skip_pull:
                cmd = [
                    "./pull_results.sh",
                    "--source", f"{hostname}:{dataset}",
                    "--all",
                    "--parallel", "8"
                ]
                print("    Pulling logs...")
                subprocess.run(cmd)
            else:
                print("    Skipping pull (using existing logs)...")

            local_search_path = LOCAL_LOGS / hostname / dataset
            if not local_search_path.exists():
                print(f"    ! No logs found locally for {hostname}:{dataset}")
                continue

            candidates = []
            for run_dir in local_search_path.iterdir():
                if not run_dir.is_dir():
                    continue
                log_file = run_dir / "log.jsonl"
                if not log_file.exists():
                    continue
                if check_params(log_file, overrides):
                    candidates.append(run_dir)

            if not candidates:
                print(f"    ! No matching runs found for {col_name}")
                continue

            best_run = max(candidates, key=get_run_score)
            print(f"    + Found {len(candidates)} matches. Best: {best_run.name}")
            selected_any = True

            if skip_push:
                continue

            target_hosts = list(get_target_hosts(dataset, specified_targets))
            timestamp = best_run.name
            for target_host in target_hosts:
                print(f"    > Pushing {dataset} ({col_name}) run {timestamp} to {target_host}...")
                cmd = [
                    "./push_results.sh",
                    "--source", f"{hostname}:{dataset}:{timestamp}",
                    "--target", target_host,
                    "--parallel", "8"
                ]
                subprocess.run(cmd)

    if not selected_any:
        print("No runs selected. Nothing to do.")


def process_s3(
    selected_datasets,
    column_filters=None,
    skip_upload=False,
    skip_download=False,
    s3_prefix="s3://scottviteri/dataset_hrmnsc",
    s3_parallel=4,
    specified_targets=None,
):
    uploaded_runs = set()
    downloaded_runs = set()
    selected_any = False
    upload_executor = ThreadPoolExecutor(max_workers=s3_parallel) if not skip_upload else None
    download_executor = ThreadPoolExecutor(max_workers=s3_parallel) if not skip_download else None
    upload_jobs = []
    download_jobs = []

    for dataset in selected_datasets:
        hosts_codes = MATRIX[dataset]
        print(f"\n=== Processing {dataset} (S3 mode) ===")

        for col_idx, (col_name, overrides) in enumerate(COLUMNS):
            if column_filters is not None and col_idx not in column_filters:
                continue

            host_code = hosts_codes[col_idx]
            source_host = HOST_MAP.get(host_code, host_code)

            print(f"  > Looking for {col_name} run on {source_host} ({host_code})...")

            best_run = select_best_remote_run(source_host, dataset, overrides)
            if not best_run:
                print(f"    ! No matching runs found for {col_name}")
                continue

            print(f"    + Selected run: {best_run}")
            selected_any = True

            target_hosts = list(get_target_hosts(dataset, specified_targets))
            
            # Construct descriptive run name
            role_slug = sanitize_role(col_name)
            new_run_name = f"{dataset}_{role_slug}_{best_run}"
            
            upload_key = (source_host, dataset, best_run)
            
            if not skip_upload and upload_key not in uploaded_runs:
                future = upload_executor.submit(
                    upload_run_to_s3, source_host, dataset, best_run, new_run_name, s3_prefix
                )
                upload_jobs.append((future, dataset, new_run_name, tuple(target_hosts), upload_key))
                uploaded_runs.add(upload_key)
            elif skip_upload:
                print("    Skipping upload (per flag)")
                # even when skipping upload, ensure we schedule downloads if allowed
                for target_host in target_hosts:
                    download_key = (target_host, dataset, new_run_name)
                    if not skip_download and download_key not in downloaded_runs:
                        future = download_executor.submit(
                            download_run_from_s3, target_host, dataset, new_run_name, new_run_name, s3_prefix
                        )
                        download_jobs.append((future, dataset, new_run_name, target_host))
                        downloaded_runs.add(download_key)

    if upload_executor:
        for future, dataset, new_run_name, target_hosts, upload_key in upload_jobs:
            success = future.result()
            if not success:
                continue
            if download_executor:
                for target_host in target_hosts:
                    download_key = (target_host, dataset, new_run_name)
                    if download_key in downloaded_runs:
                        continue
                    dl_future = download_executor.submit(
                        download_run_from_s3, target_host, dataset, new_run_name, new_run_name, s3_prefix
                    )
                    download_jobs.append((dl_future, dataset, new_run_name, target_host))
                    downloaded_runs.add(download_key)
        upload_executor.shutdown(wait=True)

    if download_executor:
        for future, dataset, new_run_name, target_host in download_jobs:
            future.result()
        download_executor.shutdown(wait=True)

    if not selected_any:
        print("No runs selected. Nothing to do.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize and deploy evaluation logs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Subset of datasets to process (default: all)",
        choices=list(MATRIX.keys()),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Subset of hyperparameter columns to process (by index or name)",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing to remote hosts (useful for dry-runs)",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip pulling logs from remote hosts (use existing local logs)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Explicit list of destination hosts to push/download into (defaults use dataset-specific hosts).",
    )
    parser.add_argument(
        "--use-s3",
        action="store_true",
        help="Use AWS S3 transfers (orchestrate aws s3 sync on remote hosts instead of local pull/push).",
    )
    parser.add_argument(
        "--s3-prefix",
        default="s3://scottviteri",
        help="S3 prefix to use for temporary storage (default: s3://scottviteri/{dataset}).",
    )
    parser.add_argument(
        "--s3-parallel",
        type=int,
        default=4,
        help="Maximum number of concurrent aws s3 sync operations (default: 4).",
    )
    parser.add_argument(
        "--find",
        nargs="+",
        help="Find host/location for dataset:variant pairs (e.g. gsm8k:No Parallel).",
    )

    args = parser.parse_args()

    # Handle find-only flow
    if args.find:
        handle_find_requests(args.find)
        sys.exit(0)

    # Combine dataset filters
    if args.datasets:
        selected_datasets = args.datasets
    else:
        selected_datasets = list(MATRIX.keys())

    # Combine column filters
    column_filters = parse_column_filters(args.columns) if args.columns else None

    try:
        if args.use_s3:
            process_s3(
                selected_datasets,
                column_filters=column_filters,
                skip_upload=args.skip_pull,
                skip_download=args.skip_push,
                s3_prefix=args.s3_prefix,
                s3_parallel=max(1, args.s3_parallel),
                specified_targets=args.targets,
            )
        else:
            process_local(
                selected_datasets,
                column_filters=column_filters,
                skip_push=args.skip_push,
                skip_pull=args.skip_pull,
                specified_targets=args.targets,
            )
    except ValueError as exc:
        print(f"Error: {exc}")
