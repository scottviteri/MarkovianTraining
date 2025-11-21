import argparse
import os
import json
import re
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter
from evaluation import calculate_answer_log_probs
from typing import Optional
from utils import find_latest_result, print_debug_info
from utils import (
    get_text_with_token_length,
    load_gsm8k_dataset,
    load_math_dataset,
    load_mmlu_dataset,
    load_svamp_dataset,
    load_aqua_dataset,
    load_mathqa_dataset,
    load_arc_dataset,
    load_arithmetic_dataset,
    load_model_for_evaluation
)
from evaluation import (
    evaluate_model_on_gsm8k,
    evaluate_model_on_mmlu,
    evaluate_model_on_arc,
    evaluate_model_on_aqua,
    evaluate_model_on_mathqa,
    evaluate_model_on_numeric,
    load_wiki_pairs,
    generate_actor_reasoning,
)
from tqdm import tqdm
import string
from pathlib import Path
from peft import PeftModel
import glob
import hashlib
import datetime
import shutil
import subprocess


def load_model_with_adapters(log_file_path, model_type, hyperparameters, adapter_index=None):
    """
    Load a model with its trained adapters if they exist.
    
    Args:
        log_file_path: Path to the log file
        model_type: Type of model to load
        hyperparameters: Hyperparameters for the model
        
    Returns:
        tuple: (actor_model, frozen_model, tokenizer, device)
    """
    # Look for adapter directories in the same directory as the log file
    log_dir = os.path.dirname(log_file_path)
    adapter_pattern = os.path.join(log_dir, "adapter_*")
    adapter_dirs = glob.glob(adapter_pattern)
    
    adapter_to_load = None
    
    if adapter_dirs:
        # If a specific adapter index is requested, try to use it first
        if adapter_index is not None:
            requested = os.path.join(log_dir, f"adapter_{adapter_index}")
            if os.path.isdir(requested):
                adapter_to_load = requested
                print(f"Loading requested adapter: {adapter_to_load}")
            else:
                print(f"Requested adapter adapter_{adapter_index} not found in {log_dir}. Falling back to latest available.")
        
        if adapter_to_load is None:
            # Sort by batch number to get the latest adapter
            def get_batch_number(adapter_path):
                try:
                    return int(os.path.basename(adapter_path).split("_")[-1])
                except (ValueError, IndexError):
                    return 0
            
            adapter_dirs_sorted = sorted(adapter_dirs, key=get_batch_number)
            adapter_to_load = adapter_dirs_sorted[-1]
            print(f"Loading trained adapter from: {adapter_to_load}")
        
    else:
        print(f"No trained adapters found in {log_dir}, using base model")
    
    # Use unified loader from utils
    if adapter_to_load:
        return load_model_for_evaluation(model_path=adapter_to_load, model_type=model_type)
    else:
        return load_model_for_evaluation(use_base_model=True, model_type=model_type)


def find_best_run_for_task(task_type, role):
    """
    Automatically find the best run and adapter for a given task and role.
    
    Args:
        task_type: The task type (e.g., "gsm8k", "arithmetic")
        role: The role string (e.g., "Markovian", "NonMarkovian")
        
    Returns:
        Tuple of (log_file_path, adapter_index) or (None, None) if not found.
    """
    results_dir = os.path.join("results", task_type)
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return None, None
        
    # Glob for directories matching the pattern
    pattern = os.path.join(results_dir, f"*{role}*")
    candidate_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    
    # Filter to ensure strict role matching (exclude NonMarkovian when looking for Markovian)
    if role == "Markovian":
        candidate_dirs = [d for d in candidate_dirs if "NonMarkovian" not in os.path.basename(d)]
    
    if not candidate_dirs:
        print(f"No run directories found for task '{task_type}' with role '{role}'")
        return None, None
        
    # Sort by name (which includes timestamp) to get the latest
    candidate_dirs.sort()
    latest_dir = candidate_dirs[-1]
    print(f"Auto-detected latest run directory for {role}: {latest_dir}")
    
    # Look for best_adapter.json
    best_adapter_path = os.path.join(latest_dir, "best_adapter.json")
    if not os.path.exists(best_adapter_path):
        print(f"Warning: best_adapter.json not found in {latest_dir}. Using latest available log file and adapter.")
        # Fallback: use log.jsonl and None for adapter_index (defaults to latest)
        log_path = os.path.join(latest_dir, "log.jsonl")
        return log_path, None
        
    try:
        with open(best_adapter_path, "r") as f:
            data = json.load(f)
            batch_index = data.get("batch_index")
            # Check if it's a valid integer (it might be 0)
            if batch_index is not None:
                print(f"Found best adapter for {role} at batch index {batch_index}")
                log_path = os.path.join(latest_dir, "log.jsonl")
                return log_path, batch_index
            else:
                print(f"Warning: batch_index not found in {best_adapter_path}")
    except Exception as e:
        print(f"Error reading best_adapter.json: {e}")
        
    # Fallback
    log_path = os.path.join(latest_dir, "log.jsonl")
    return log_path, None


def perturb_CoT(CoT, config):
    """
    Perturb the chain-of-thought (CoT) according to the perturbation configuration.
    """
    perturbed_CoT = CoT

    # Randomly delete a fraction of characters
    if config.get("delete_fraction", 0) > 0:
        chars = list(perturbed_CoT)
        num_to_delete = int(len(chars) * config["delete_fraction"])
        indices_to_delete = random.sample(range(len(chars)), num_to_delete)
        chars = [char for idx, char in enumerate(chars) if idx not in indices_to_delete]
        perturbed_CoT = "".join(chars)

    # Truncate a fraction from either end
    if config.get("truncate_fraction", 0) > 0:
        truncate_length = int(len(perturbed_CoT) * (1 - config["truncate_fraction"]))
        if config.get("truncate_from_front", False):
            perturbed_CoT = (
                perturbed_CoT[-truncate_length:] if truncate_length > 0 else ""
            )
        else:
            perturbed_CoT = perturbed_CoT[:truncate_length]

    # Replace digits with random probability
    if config.get("digit_replace_prob", 0) > 0:
        chars = list(perturbed_CoT)
        for i, char in enumerate(chars):
            if char.isdigit() and random.random() < config["digit_replace_prob"]:
                chars[i] = str(random.randint(0, 9))
        perturbed_CoT = "".join(chars)

    # Replace alphanumeric characters with random probability
    if config.get("char_replace_prob", 0) > 0:
        chars = list(perturbed_CoT)
        alphanumeric = string.ascii_letters + string.digits
        for i, char in enumerate(chars):
            if char in alphanumeric and random.random() < config["char_replace_prob"]:
                chars[i] = random.choice(alphanumeric)
        perturbed_CoT = "".join(chars)

    return perturbed_CoT


# Define perturbation configurations
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERTURB_S3_BUCKET = os.environ.get("PERTURB_S3_BUCKET")
if PERTURB_S3_BUCKET:
    PERTURB_S3_BUCKET = PERTURB_S3_BUCKET.rstrip("/")
_S3_WARNING_PRINTED = False

RUN_SYNC_PATTERNS = [
    "markovian_comparison_accuracy/*.json",
    "markovian_comparison_accuracy/*.png",
    "checkpoint_scan/*.json",
    "checkpoint_scan/*.png",
    "adapter_*/perturb_metadata.json",
]

ADAPTER_SYNC_PATTERNS = [
    "perturb_metadata.json",
]

PERTURBATION_SETS = {
    "delete": {
        "perturbations": {
            f"Delete{int(frac*100)}%": {"delete_fraction": frac}
            for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "description": "Character deletion perturbations",
    },
    "truncate_back": {
        "perturbations": {
            f"TruncateBack{int(frac*100)}%": {
                "truncate_fraction": frac,
                "truncate_from_front": False,
            }
            for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "description": "Text truncation from end perturbations",
    },
    "truncate_front": {
        "perturbations": {
            f"TruncateFront{int(frac*100)}%": {
                "truncate_fraction": frac,
                "truncate_from_front": True,
            }
            for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "description": "Text truncation from start perturbations",
    },
    "digit_replace": {
        "perturbations": {
            f"DigitReplace{int(prob*100)}%": {"digit_replace_prob": prob}
            for prob in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "description": "Random digit replacement perturbations",
    },
    "char_replace": {
        "perturbations": {
            f"CharReplace{int(prob*100)}%": {"char_replace_prob": prob}
            for prob in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        },
        "description": "Random alphanumeric character replacement perturbations",
    },
}


def _perturb_metadata_path(adapter_dir: str) -> str:
    return os.path.join(adapter_dir, "perturb_metadata.json")


def _ensure_metadata_structure(data: dict, adapter_dir: str) -> dict:
    if not isinstance(data, dict):
        data = {}
    data.setdefault("adapter", os.path.basename(adapter_dir))
    data.setdefault("records", {})
    return data


def load_perturb_metadata(adapter_dir: str) -> dict:
    path = _perturb_metadata_path(adapter_dir)
    if not os.path.exists(path):
        return _ensure_metadata_structure({}, adapter_dir)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return _ensure_metadata_structure(data, adapter_dir)


def save_perturb_metadata(adapter_dir: str, metadata: dict):
    path = _perturb_metadata_path(adapter_dir)
    os.makedirs(adapter_dir, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp_path, path)


def build_perturb_metadata_key(
    task_type: str,
    perturb_type: str,
    metric: str,
    paired_role: str,
    paired_adapter_index: int,
    markovian_run: str,
    non_markovian_run: str,
) -> str:
    payload = {
        "task_type": task_type,
        "perturb": perturb_type,
        "metric": metric,
        "paired_role": paired_role,
        "paired_adapter_index": paired_adapter_index,
        "markovian_run": os.path.basename(markovian_run),
        "non_markovian_run": os.path.basename(non_markovian_run),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _record_satisfies(record: dict, required_stride: Optional[int]) -> bool:
    if required_stride is None:
        return True
    record_stride = record.get("stride", 1)
    return record_stride <= required_stride


def metadata_has_record(metadata: dict, key: str, required_stride: Optional[int] = None) -> bool:
    record = metadata.get("records", {}).get(key)
    if not record:
        return False
    return _record_satisfies(record, required_stride)


# Metadata cache to avoid repeated disk IO
_PERTURB_METADATA_CACHE = {}
_PULLED_ADAPTERS = set()
_PULLED_RUN_DIRS = set()


def safe_relpath(path: str, base_dir: str) -> str:
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


def _s3_uri_for(local_path: str) -> Optional[str]:
    if not PERTURB_S3_BUCKET:
        return None
    rel_path = safe_relpath(local_path, PROJECT_ROOT).replace("\\", "/")
    dest = f"{PERTURB_S3_BUCKET}/{rel_path}"
    if dest.startswith("s3:/") and not dest.startswith("s3://"):
        dest = dest.replace("s3:/", "s3://", 1)
    return dest


def _run_s3_sync(cmd: list[str]):
    global _S3_WARNING_PRINTED
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        if not _S3_WARNING_PRINTED:
            print("Warning: aws CLI not found; skipping S3 sync.")
            _S3_WARNING_PRINTED = True
    except subprocess.CalledProcessError as e:
        print(f"S3 sync error: {e}")


def _sync_to_s3(local_path: str, include_patterns: list[str]):
    if not PERTURB_S3_BUCKET:
        return
    dest = _s3_uri_for(local_path)
    if not dest:
        return
    include_args = ["--exclude", "*"]
    for pattern in include_patterns:
        include_args.extend(["--include", pattern])
    cmd = ["aws", "s3", "sync", local_path, dest, *include_args]
    _run_s3_sync(cmd)


def _sync_from_s3(local_path: str, include_patterns: list[str]):
    if not PERTURB_S3_BUCKET:
        return
    source = _s3_uri_for(local_path)
    if not source:
        return
    include_args = ["--exclude", "*"]
    for pattern in include_patterns:
        include_args.extend(["--include", pattern])
    os.makedirs(local_path, exist_ok=True)
    cmd = ["aws", "s3", "sync", source, local_path, *include_args]
    _run_s3_sync(cmd)


def sync_run_dir_outputs(run_dir: str):
    _sync_to_s3(run_dir, RUN_SYNC_PATTERNS)


def sync_run_dir_from_s3(run_dir: str):
    if run_dir in _PULLED_RUN_DIRS:
        return
    _sync_from_s3(run_dir, RUN_SYNC_PATTERNS)
    _PULLED_RUN_DIRS.add(run_dir)


def pull_adapter_metadata(adapter_dir: str):
    if adapter_dir in _PULLED_ADAPTERS:
        return
    _sync_from_s3(adapter_dir, ADAPTER_SYNC_PATTERNS)
    _PULLED_ADAPTERS.add(adapter_dir)


def push_adapter_metadata(adapter_dir: str):
    _sync_to_s3(adapter_dir, ADAPTER_SYNC_PATTERNS)


def get_cached_metadata(adapter_dir: str) -> dict:
    if adapter_dir and os.path.isdir(adapter_dir):
        pull_adapter_metadata(adapter_dir)
    if adapter_dir not in _PERTURB_METADATA_CACHE:
        _PERTURB_METADATA_CACHE[adapter_dir] = load_perturb_metadata(adapter_dir)
    return _PERTURB_METADATA_CACHE[adapter_dir]


def persist_metadata_cache(adapter_dir: str):
    data = _PERTURB_METADATA_CACHE.get(adapter_dir)
    if data is not None:
        save_perturb_metadata(adapter_dir, data)
        push_adapter_metadata(adapter_dir)


def infer_role_from_log_path(log_file_path: str) -> str:
    run_dir = os.path.dirname(log_file_path)
    basename = os.path.basename(run_dir).lower()
    if "nonmarkovian" in basename:
        return "NonMarkovian"
    if "markovian" in basename:
        return "Markovian"
    return "Unknown"


def get_output_paths(log_file, perturb_type, include_question=False):
    """Get standardized paths for output files."""
    # If log_file points to a file, get its directory
    # If log_file points to a directory, use it directly
    if os.path.isfile(log_file):
        base_dir = os.path.dirname(log_file)
    else:
        base_dir = log_file
        
    base_name = f"perturbation_results_{perturb_type}"
    if include_question:
        base_name += "_with_question"
    return {
        "json": os.path.join(base_dir, f"{base_name}.json"),
        "plot": os.path.join(base_dir, f"{base_name}_plot.png"),
        "debug_plot": os.path.join(base_dir, f"{base_name}_debug.png"),
    }


def save_perturbation_results(results, log_file, perturb_type, include_question=False):
    """Save perturbation results to a JSON file."""
    output_file = get_output_paths(log_file, perturb_type, include_question)["json"]
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")


def load_perturbation_results(log_file, perturb_type, include_question=False):
    """Load perturbation results from a JSON file."""
    input_file = get_output_paths(log_file, perturb_type, include_question)["json"]
    with open(input_file, "r") as f:
        return json.load(f)


def run_perturbations(log_file, perturb_type, include_question=False, stride=1, max_index=None, save_interval=10, evaluator="actor", adapter_index=None):
    """
    Run perturbation analysis on the given log file.
    max_index: if provided, only process entries with batch_index <= max_index
    include_question: whether to include the question in the prompt
    save_interval: save intermediate results every this many entries (set to 0 to disable)
    """
    if perturb_type not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    perturbations = PERTURBATION_SETS[perturb_type]["perturbations"]

    # Process the log file to extract perturbation data
    with open(log_file, "r") as f:
        log_data = [json.loads(line) for line in f]

    # Extract hyperparameters from the first line
    hyperparameters = log_data[0]
    task_type = hyperparameters.get("task_type", "gsm8k")
    actor_model, frozen_model, tokenizer, device = load_model_with_adapters(log_file, hyperparameters["model_type"], hyperparameters, adapter_index=adapter_index)
    eval_model = actor_model if evaluator == "actor" else frozen_model

    # Filter log data by batch index if max_index is provided
    if max_index is not None:
        log_data = [entry for entry in log_data if entry.get("Batch Index", float('inf')) <= max_index]
        print(f"Processing entries up to batch index {max_index}")

    # Path for saving results
    output_path = get_output_paths(log_file, perturb_type, include_question)["json"]
    
    # Check if we have previous partial results to resume from
    perturbation_data = []
    last_processed_idx = -1
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                perturbation_data = json.load(f)
                if perturbation_data:
                    # Get the last processed batch index
                    last_processed_idx = perturbation_data[-1]["Batch Index"]
                    print(f"Resuming from entry with batch index {last_processed_idx}")
        except (json.JSONDecodeError, KeyError):
            print(f"Could not parse previous results in {output_path}, starting fresh")
            perturbation_data = []
            last_processed_idx = -1

    # Extract perturbation-related metrics
    entries_to_process = []
    for entry in log_data[1:]:
        if "Example" not in entry:
            continue
        batch_idx = entry.get("Batch Index", -1)
        if batch_idx > last_processed_idx:
            entries_to_process.append(entry)
    
    print(f"Processing {len(entries_to_process)} entries, saving every {save_interval} entries")
    
    for i, entry in enumerate(tqdm(entries_to_process[::stride], desc="Processing entries")):
        if i % 100 == 0:  # Adjust print frequency based on stride
            example = entry["Example"]
            print(f"\nProcessing entry {i*stride}...")
            print_debug_info(
                task_type=task_type,
                q=example.get("Question", ""),
                reasoning_text_first=example["Actor Reasoning"],
                ans=example["Answer"],
                avg_log_prob=entry.get("Training Metrics", {}).get(
                    "Actor Log Probs", None
                ),
                extracted_generated_answers=None,
            )

        example = entry["Example"]
        actor_CoT = example["Actor Reasoning"]
        critic_CoT = example["Critic Reasoning"]
        answer = example["Answer"]
        question = example.get("Question", "")

        # Prepare entry results
        entry_results = {
            "Batch Index": entry.get("Batch Index", None),
            "Log Probs": {
                "Actor": {
                    "Original": None,
                    "Perturbed": {}
                },
                "Comparison": {  # We'll use this for either critic or actor with question
                    "Original": None,
                    "Perturbed": {}
                }
            }
        }

        # Calculate Original log probs for Actor
        actor_log_prob, _ = calculate_answer_log_probs(
            model=eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=[question],
            reasoning=[actor_CoT],
            answers=[answer],
            hyperparameters=hyperparameters,
            include_question=False,  # Always without question for original actor
        )
        entry_results["Log Probs"]["Actor"]["Original"] = actor_log_prob[0].item()

        # Calculate log probs for either:
        # 1. Critic (if include_question=False)
        # 2. Actor with question (if include_question=True)
        comparison_log_prob, _ = calculate_answer_log_probs(
            model=eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=[question],
            reasoning=[actor_CoT if include_question else critic_CoT],
            answers=[answer],
            hyperparameters=hyperparameters,
            include_question=include_question,
        )
        entry_results["Log Probs"]["Comparison"]["Original"] = comparison_log_prob[0].item()

        # Perform perturbations and calculate log probabilities
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                continue

            # Perturb Actor CoT (always without question)
            perturbed_actor_CoT = perturb_CoT(actor_CoT, pert_config)
            actor_perturbed_log_prob, _ = calculate_answer_log_probs(
                model=eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=[question],
                reasoning=[perturbed_actor_CoT],
                answers=[answer],
                hyperparameters=hyperparameters,
                include_question=False,  # Always without question for actor
            )
            entry_results["Log Probs"]["Actor"]["Perturbed"][pert_name] = actor_perturbed_log_prob[0].item()

            # Perturb comparison CoT (either critic or actor-with-question)
            perturbed_critic_CoT = perturb_CoT(critic_CoT, pert_config) if not include_question else None
            comparison_perturbed_log_prob, _ = calculate_answer_log_probs(
                model=eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=[question],
                reasoning=[perturbed_actor_CoT if include_question else perturbed_critic_CoT],
                answers=[answer],
                hyperparameters=hyperparameters,
                include_question=include_question,
            )
            entry_results["Log Probs"]["Comparison"]["Perturbed"][pert_name] = comparison_perturbed_log_prob[0].item()

        perturbation_data.append(entry_results)
        
        # Periodically save intermediate results
        if save_interval > 0 and (i + 1) % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(perturbation_data, f)
            print(f"\nSaved {len(perturbation_data)} results to {output_path}")

    # Save final results
    with open(output_path, "w") as f:
        json.dump(perturbation_data, f)
        
    print(f"Analysis complete. Processed {len(perturbation_data)} entries.")
    return perturbation_data


def run_perturbations_batched(log_file, perturb_type, include_question=False, stride=1, max_index=None, save_interval=10, batch_size=8, evaluator="actor", adapter_index=None):
    """
    Run perturbation analysis on the given log file using batched processing for improved performance.
    
    Args:
        log_file: Path to the log file to analyze
        perturb_type: Type of perturbation to apply
        include_question: Whether to include the question in the prompt
        stride: Process every nth entry of the log file
        max_index: If provided, only process entries with batch_index <= max_index
        save_interval: Save intermediate results every this many examples (set to 0 to disable)
        batch_size: Number of examples to process in each batch
    
    Returns:
        List of perturbation results
    """
    if perturb_type not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    perturbations = PERTURBATION_SETS[perturb_type]["perturbations"]

    # Process the log file to extract perturbation data
    with open(log_file, "r") as f:
        log_data = [json.loads(line) for line in f]

    # Extract hyperparameters from the first line
    hyperparameters = log_data[0]
    task_type = hyperparameters.get("task_type", "gsm8k")
    actor_model, frozen_model, tokenizer, device = load_model_with_adapters(log_file, hyperparameters["model_type"], hyperparameters, adapter_index=adapter_index)
    eval_model = actor_model if evaluator == "actor" else frozen_model

    # Filter log data by batch index if max_index is provided
    if max_index is not None:
        log_data = [entry for entry in log_data if entry.get("Batch Index", float('inf')) <= max_index]
        print(f"Processing entries up to batch index {max_index}")

    # Path for saving results
    output_path = get_output_paths(log_file, perturb_type, include_question)["json"]
    
    # Check if we have previous partial results to resume from
    perturbation_data = []
    last_processed_idx = -1
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                perturbation_data = json.load(f)
                if perturbation_data:
                    # Get the last processed batch index
                    last_processed_idx = perturbation_data[-1]["Batch Index"]
                    print(f"Resuming from entry with batch index {last_processed_idx}")
        except (json.JSONDecodeError, KeyError):
            print(f"Could not parse previous results in {output_path}, starting fresh")
            perturbation_data = []
            last_processed_idx = -1

    # Extract perturbation-related metrics
    entries_to_process = []
    for entry in log_data[1:]:
        if "Example" not in entry:
            continue
        batch_idx = entry.get("Batch Index", -1)
        if batch_idx > last_processed_idx:
            entries_to_process.append(entry)
    
    # Apply stride
    entries_to_process = entries_to_process[::stride]
    
    print(f"Processing {len(entries_to_process)} entries in batches of {batch_size}, saving every {save_interval} examples")
    
    # Track total number of examples processed for save interval
    total_examples_processed = 0
    next_save_threshold = save_interval
    
    # Process in batches
    for batch_idx in tqdm(range(0, len(entries_to_process), batch_size), desc="Processing batches"):
        batch_entries = entries_to_process[batch_idx:batch_idx + batch_size]
        batch_size_actual = len(batch_entries)
        
        # Print debug info for first entry in batch
        if batch_idx % 5 == 0:
            example = batch_entries[0]["Example"]
            print(f"\nProcessing batch starting at index {batch_idx}...")
            print_debug_info(
                task_type=task_type,
                q=example.get("Question", ""),
                reasoning_text_first=example["Actor Reasoning"],
                ans=example["Answer"],
                avg_log_prob=batch_entries[0].get("Training Metrics", {}).get(
                    "Actor Log Probs", None
                ),
                extracted_generated_answers=None,
            )
        
        # Extract batch data
        batch_questions = [entry["Example"].get("Question", "") for entry in batch_entries]
        batch_actor_CoTs = [entry["Example"]["Actor Reasoning"] for entry in batch_entries]
        batch_critic_CoTs = [entry["Example"]["Critic Reasoning"] for entry in batch_entries]
        batch_answers = [entry["Example"]["Answer"] for entry in batch_entries]
        batch_indices = [entry.get("Batch Index", None) for entry in batch_entries]
        
        # Initialize batch results
        batch_results = [
            {
                "Batch Index": idx,
                "Log Probs": {
                    "Actor": {
                        "Original": None,
                        "Perturbed": {}
                    },
                    "Comparison": {
                        "Original": None,
                        "Perturbed": {}
                    }
                }
            }
            for idx in batch_indices
        ]
        
        # Calculate Original log probs for Actor (all without question)
        actor_log_probs, _ = calculate_answer_log_probs(
            model=eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=batch_questions,
            reasoning=batch_actor_CoTs,
            answers=batch_answers,
            hyperparameters=hyperparameters,
            include_question=False,  # Always without question for original actor
        )
        
        # Store original actor log probs
        for i in range(batch_size_actual):
            batch_results[i]["Log Probs"]["Actor"]["Original"] = actor_log_probs[i].item()
        
        # Calculate log probs for comparison (either critic or actor with question)
        comparison_reasoning = batch_actor_CoTs if include_question else batch_critic_CoTs
        comparison_log_probs, _ = calculate_answer_log_probs(
            model=eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=batch_questions,
            reasoning=comparison_reasoning,
            answers=batch_answers,
            hyperparameters=hyperparameters,
            include_question=include_question,
        )
        
        # Store original comparison log probs
        for i in range(batch_size_actual):
            batch_results[i]["Log Probs"]["Comparison"]["Original"] = comparison_log_probs[i].item()
        
        # Process each perturbation type
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                continue
                
            # Perturb all actor CoTs in batch
            perturbed_actor_CoTs = [perturb_CoT(cot, pert_config) for cot in batch_actor_CoTs]
            
            # Calculate perturbed actor log probs (without question)
            actor_perturbed_log_probs, _ = calculate_answer_log_probs(
                model=frozen_model,
                tokenizer=tokenizer,
                device=device,
                questions=batch_questions,
                reasoning=perturbed_actor_CoTs,
                answers=batch_answers,
                hyperparameters=hyperparameters,
                include_question=False,  # Always without question for actor
            )
            
            # Store perturbed actor log probs
            for i in range(batch_size_actual):
                batch_results[i]["Log Probs"]["Actor"]["Perturbed"][pert_name] = actor_perturbed_log_probs[i].item()
            
            # Handle comparison CoTs (either perturbed critic or perturbed actor with question)
            if include_question:
                # Use perturbed actor CoTs with question
                perturbed_comparison_CoTs = perturbed_actor_CoTs
            else:
                # Perturb critic CoTs
                perturbed_comparison_CoTs = [perturb_CoT(cot, pert_config) for cot in batch_critic_CoTs]
            
            # Calculate perturbed comparison log probs
            comparison_perturbed_log_probs, _ = calculate_answer_log_probs(
                model=frozen_model,
                tokenizer=tokenizer,
                device=device,
                questions=batch_questions,
                reasoning=perturbed_comparison_CoTs,
                answers=batch_answers,
                hyperparameters=hyperparameters,
                include_question=include_question,
            )
            
            # Store perturbed comparison log probs
            for i in range(batch_size_actual):
                batch_results[i]["Log Probs"]["Comparison"]["Perturbed"][pert_name] = comparison_perturbed_log_probs[i].item()
        
        # Add batch results to overall results
        perturbation_data.extend(batch_results)
        
        # Update total examples processed
        total_examples_processed += batch_size_actual
        
        # Periodically save intermediate results based on example count
        if save_interval > 0 and total_examples_processed >= next_save_threshold:
            with open(output_path, "w") as f:
                json.dump(perturbation_data, f)
            print(f"\nSaved {len(perturbation_data)} results to {output_path}")
            # Update next save threshold
            next_save_threshold = ((total_examples_processed // save_interval) + 1) * save_interval
    
    # Save final results
    with open(output_path, "w") as f:
        json.dump(perturbation_data, f)
    
    print(f"Analysis complete. Processed {len(perturbation_data)} entries.")
    return perturbation_data


def plot_perturbation_results(
    results, log_file, perturb_type, window_size=40, debug=False, max_index=None, font_size=12, legend_font_size=10, include_question=False
):
    """
    Plot the perturbation results comparing actor and critic log probabilities.

    Args:
        results: The perturbation results data.
        log_file: Path to the log file or results directory.
        perturb_type: The type of perturbation being analyzed.
        window_size: Smoothing window size.
        debug: Whether to generate debug plots.
        max_index: Maximum index to plot.
        font_size: Base font size for plot text elements.
        legend_font_size: Font size for the legend in plots.
        include_question: Whether the question was included in the prompt.
    """
    if not results:
        print("No results to plot.")
        return
        
    # Get all perturbation degrees from the first entry
    if "Log Probs" not in results[0] or "Actor" not in results[0]["Log Probs"] or "Perturbed" not in results[0]["Log Probs"]["Actor"]:
        print("Invalid result format. Cannot find perturbation data.")
        return
        
    perturbation_degrees = list(results[0]["Log Probs"]["Actor"]["Perturbed"].keys())
    print(f"Found perturbation degrees: {perturbation_degrees}")
    
    # Only filter out the exact baseline case (e.g., Delete0%)
    baseline_name = f"{perturb_type.title().replace('_', '')}0%"
    plot_degrees = [deg for deg in perturbation_degrees if deg != baseline_name]
    print(f"Plotting degrees: {plot_degrees}")
    
    if not plot_degrees:
        print("No non-zero perturbation degrees found to plot.")
        return
        
    # Extract batch indices
    batch_indices = [entry["Batch Index"] for entry in results]
    
    if max_index is not None:
        max_index = min(max_index, len(batch_indices))
        results = results[:max_index]
        batch_indices = batch_indices[:max_index]
        
    # Plotting
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_degrees)))
    
    for i, degree in enumerate(plot_degrees):
        # Extract data for this perturbation degree
        actor_original = []
        actor_perturbed = []
        comparison_original = []
        comparison_perturbed = []
        
        for entry in results:
            actor_original.append(entry["Log Probs"]["Actor"]["Original"])
            comparison_original.append(entry["Log Probs"]["Comparison"]["Original"])
            actor_perturbed.append(entry["Log Probs"]["Actor"]["Perturbed"][degree])
            comparison_perturbed.append(entry["Log Probs"]["Comparison"]["Perturbed"][degree])
            
        # Calculate differences
        actor_diff = np.array(actor_original) - np.array(actor_perturbed)
        comparison_diff = np.array(comparison_original) - np.array(comparison_perturbed)
        diff_difference = actor_diff - comparison_diff
        
        # Smoothing
        if window_size > 1 and len(diff_difference) > window_size:
            try:
                effect_smooth = savgol_filter(diff_difference, window_size, 3)
                padding = window_size // 2
                x_values = range(padding, len(diff_difference) - padding)
                effect_smooth = effect_smooth[padding:-padding]
            except ValueError as e:
                print(f"Smoothing error: {e}. Using raw data.")
                x_values = range(len(diff_difference))
                effect_smooth = diff_difference
        else:
            x_values = range(len(diff_difference))
            effect_smooth = diff_difference
            
        # Plot this perturbation degree
        plt.plot(
            x_values,
            effect_smooth,
            label=f"{degree}",
            color=colors[i],
            linewidth=2,
        )
    
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=legend_font_size, loc="best")
    
    plt.xlabel("Training Batch", fontsize=font_size)
    
    # Update y-label based on what we're comparing
    if include_question:
        plt.ylabel("Difference in Perturbation Effect\n(Actor w/o Question - Actor w/ Question)", fontsize=font_size)
    else:
        plt.ylabel("Difference in Perturbation Effect\n(Actor - Critic)", fontsize=font_size)
        
    title = f"Perturbation Analysis: {perturb_type.replace('_', ' ').title()}"
    if include_question:
        title += " (Comparing with/without Question)"
    if window_size > 1:
        title += f" (Smoothing: {window_size})"
    else:
        title += " (Raw Data)"
        
    plt.title(title, fontsize=font_size)
    plt.tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    
    output_file = get_output_paths(log_file, perturb_type, include_question)["plot"]
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_multiple_perturbation_results(
    log_file, perturb_types, window_size=40, max_index=None, font_size=12, legend_font_size=10, include_question=False
):
    """Plot multiple perturbation results in a grid layout."""
    # Calculate grid dimensions
    n_plots = len(perturb_types)
    n_rows = (n_plots + 1) // 2  # 2 columns, round up
    n_cols = min(2, n_plots)  # Use 2 columns unless only 1 plot
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 6 * n_rows))
    
    # Convert axes to array if single row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    for ax, perturb_type in zip(axes_flat, perturb_types):
        try:
            results = load_perturbation_results(log_file, perturb_type, include_question)
            if max_index is not None:
                results = results[:max_index]
                
            # Plot each perturbation degree
            for i, (pert, _) in enumerate(results[0]["Log Probs"]["Actor"]["Perturbed"].items()):
                # Skip baseline case (0% perturbation)
                if pert == f"{perturb_type.title().replace('_', '')}0%":
                    continue
                
                # Calculate differences for Actor and Comparison model
                actor_orig_values = [-entry["Log Probs"]["Actor"]["Original"] for entry in results]
                actor_pert_values = [-entry["Log Probs"]["Actor"]["Perturbed"][pert] for entry in results]
                actor_diff_values = [p - o for p, o in zip(actor_pert_values, actor_orig_values)]
                
                comparison_orig_values = [-entry["Log Probs"]["Comparison"]["Original"] for entry in results]
                comparison_pert_values = [-entry["Log Probs"]["Comparison"]["Perturbed"][pert] for entry in results]
                comparison_diff_values = [p - o for p, o in zip(comparison_pert_values, comparison_orig_values)]
                
                # Calculate effect difference
                effect_difference = [a - c for a, c in zip(actor_diff_values, comparison_diff_values)]
                
                if window_size > 1 and len(effect_difference) > window_size:
                    effect_smooth = savgol_filter(effect_difference, window_size, 3)
                    padding = window_size // 2
                    x_values = range(padding, len(effect_difference) - padding)
                    effect_smooth = effect_smooth[padding:-padding]
                else:
                    x_values = range(len(effect_difference))
                    effect_smooth = effect_difference
                
                ax.plot(x_values, effect_smooth, label=f"{pert}", color=colors[i % len(colors)], linewidth=2)
            
            ax.grid(True)
            ax.legend(fontsize=legend_font_size, loc='best')
            
            if ax.get_subplotspec().is_first_col():
                # Update y-label based on what we're comparing
                if include_question:
                    ax.set_ylabel("Difference in Perturbation Effect\n(Actor w/o Question - Actor w/ Question)", fontsize=font_size)
                else:
                    ax.set_ylabel("Difference in Perturbation Effect\n(Actor - Critic)", fontsize=font_size)
            
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Training Batch", fontsize=font_size)
            
            ax.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            title = f"{perturb_type.replace('_', ' ').title()}"
            if include_question:
                title += " (Comparing with/without Question)"
            
            if window_size > 1:
                title += f" (Smoothing: {window_size})"
            else:
                title += " (Raw Data)"
                
            ax.set_title(title, fontsize=font_size+2)
                
        except FileNotFoundError:
            print(f"No saved results found for {perturb_type}")
            ax.text(0.5, 0.5, f"No data for {perturb_type}", ha='center', va='center', fontsize=font_size)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    suffix = "_comparison_question" if include_question else ""
    output_file = os.path.join(os.path.dirname(log_file), f"combined_perturbation_plot{suffix}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {output_file}")
    plt.close()


def collate_perturbation_results(perturbation_files, output_dir, perturb_type, include_question=False):
    """
    Average perturbation results across multiple runs and save to a new directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    accumulated_results = []
    
    # Process each perturbation result file
    for perturbation_file in perturbation_files:
        try:
            with open(perturbation_file, 'r') as f:
                results = json.load(f)
                accumulated_results.append(results)
        except FileNotFoundError:
            print(f"Warning: No results found in {perturbation_file}")
            continue
    
    if not accumulated_results:
        print("No results to collate.")
        return
    
    num_runs = len(accumulated_results)
    
    # Find minimum length across all runs
    min_length = min(len(run) for run in accumulated_results)
    question_status = "with question" if include_question else "without question"
    print(f"Using {min_length} entries for {perturb_type} ({question_status}) (shortest common length)")
    
    # Initialize structure for averaged results
    averaged_results = []
    for entry_idx in range(min_length):
        avg_entry = {
            "Batch Index": accumulated_results[0][entry_idx]["Batch Index"],
            "Log Probs": {
                "Actor": {
                    "Original": 0.0,
                    "Perturbed": {}
                },
                "Comparison": {
                    "Original": 0.0,
                    "Perturbed": {}
                }
            }
        }
        
        # Average the Original values for both Actor and Critic
        for run in accumulated_results:
            avg_entry["Log Probs"]["Actor"]["Original"] += run[entry_idx]["Log Probs"]["Actor"]["Original"] / num_runs
            avg_entry["Log Probs"]["Comparison"]["Original"] += run[entry_idx]["Log Probs"]["Comparison"]["Original"] / num_runs
        
        # Get perturbation names from first run
        pert_names = accumulated_results[0][entry_idx]["Log Probs"]["Actor"]["Perturbed"].keys()
        
        # Initialize perturbation dictionaries
        for pert_name in pert_names:
            avg_entry["Log Probs"]["Actor"]["Perturbed"][pert_name] = 0.0
            avg_entry["Log Probs"]["Comparison"]["Perturbed"][pert_name] = 0.0
        
        # Average the perturbed values for both Actor and Critic
        for run in accumulated_results:
            for pert_name in pert_names:
                avg_entry["Log Probs"]["Actor"]["Perturbed"][pert_name] += (
                    run[entry_idx]["Log Probs"]["Actor"]["Perturbed"][pert_name] / num_runs
                )
                avg_entry["Log Probs"]["Comparison"]["Perturbed"][pert_name] += (
                    run[entry_idx]["Log Probs"]["Comparison"]["Perturbed"][pert_name] / num_runs
                )
        
        averaged_results.append(avg_entry)
    
    # Save averaged results
    output_file = get_output_paths(output_dir, perturb_type, include_question)["json"]
    with open(output_file, "w") as f:
        json.dump(averaged_results, f)
    print(f"Averaged results for {perturb_type} saved to {output_file}")


def compute_sensitivity_summary(results, perturb_type):
    """
    Compute summary statistics (mean sensitivity diff) for a set of results.
    Returns a dict mapping degree -> mean difference.
    """
    if not results:
        return {}
        
    perturbation_degrees = list(results[0]["Effect Difference"].keys())
    baseline_name = f"{perturb_type.title().replace('_', '')}0%"
    analysis_degrees = [deg for deg in perturbation_degrees if deg != baseline_name]
    
    summary = {}
    for degree in analysis_degrees:
        effect_differences = [entry["Effect Difference"][degree] for entry in results]
        summary[degree] = np.mean(effect_differences)
        
    return summary


def load_scan_results(output_dir, perturb_type):
    """Load previously saved scan results."""
    output_file = os.path.join(output_dir, f"scan_results_{perturb_type}.json")
    if not os.path.exists(output_file):
        print(f"Scan results file not found: {output_file}")
        return None
    with open(output_file, "r") as f:
        return json.load(f)


def run_checkpoint_scan(
    markovian_log_file,
    non_markovian_log_file,
    perturb_type,
    task_type,
    num_samples=128,
    batch_size=8,
    stride: int = 1,
    question_length=None,
    target_length=None,
):
    """
    Iterate over ALL matching adapters in the log directories and compute fresh sensitivity.
    """
    markovian_dir = os.path.dirname(markovian_log_file)
    non_markovian_dir = os.path.dirname(non_markovian_log_file)
    sync_run_dir_from_s3(markovian_dir)
    sync_run_dir_from_s3(non_markovian_dir)
    markovian_role = infer_role_from_log_path(markovian_log_file)
    non_markovian_role = infer_role_from_log_path(non_markovian_log_file)
    
    # Find common adapter indices
    m_adapters = glob.glob(os.path.join(markovian_dir, "adapter_*"))
    nm_adapters = glob.glob(os.path.join(non_markovian_dir, "adapter_*"))
    
    def get_idx(p):
        try: return int(p.split("_")[-1])
        except: return None
        
    m_idxs = set(get_idx(p) for p in m_adapters if get_idx(p) is not None)
    nm_idxs = set(get_idx(p) for p in nm_adapters if get_idx(p) is not None)
    
    common_idxs = sorted(list(m_idxs.intersection(nm_idxs)))
    
    if not common_idxs:
        print("No common adapter indices found between the two runs.")
        return
        
    print(f"Found {len(common_idxs)} common checkpoints to scan: {common_idxs}")
    
    scan_results = []
    
    for idx in tqdm(common_idxs, desc="Scanning checkpoints"):
        # Run eval for this checkpoint
        # Note: We use 'accuracy' metric logic if task is QA, or log prob if not.
        # Assuming QA task based on context.
        # Using run_qa_perturbation_accuracy logic but simplified or calling it directly.
        
        # To avoid reloading base model constantly, ideally we'd refactor.
        # For now, we'll call run_qa_perturbation_accuracy which loads everything.
        # It is slow but robust.
        
        # Suppress print output from sub-function
        # sys.stdout = open(os.devnull, 'w')
        mark_adapter_dir = os.path.join(markovian_dir, f"adapter_{idx}")
        non_adapter_dir = os.path.join(non_markovian_dir, f"adapter_{idx}")
        if not (os.path.isdir(mark_adapter_dir) and os.path.isdir(non_adapter_dir)):
            continue

        metadata_key = build_perturb_metadata_key(
            task_type=task_type,
            perturb_type=perturb_type,
            metric="accuracy",
            paired_role=non_markovian_role,
            paired_adapter_index=idx,
            markovian_run=markovian_dir,
            non_markovian_run=non_markovian_dir,
        )

        mark_metadata = get_cached_metadata(mark_adapter_dir)
        non_metadata = get_cached_metadata(non_adapter_dir)
        if metadata_has_record(mark_metadata, metadata_key, stride) and metadata_has_record(non_metadata, metadata_key, stride):
            print(f"Skipping adapter_{idx}: perturbation metadata already satisfies coverage (stride={stride}).")
            continue

        try:
            comparison_data, _, _ = run_qa_perturbation_accuracy(
                markovian_log_file=markovian_log_file,
                non_markovian_log_file=non_markovian_log_file,
                perturb_type=perturb_type,
                task_type=task_type,
                num_samples=num_samples,
                batch_size=batch_size,
                evaluator="actor",
                adapter_index=idx,
                question_length=question_length,
                target_length=target_length,
                markovian_adapter_index=idx, # Enforce same index
                non_markovian_adapter_index=idx,
                stride=stride,
            )
        finally:
            # sys.stdout = sys.__stdout__
            pass
            
        summary = compute_sensitivity_summary(comparison_data, perturb_type)
        scan_results.append({
            "Batch Index": idx,  # Training batch index
            "Sensitivity Summary": summary
        })

        timestamp = datetime.datetime.utcnow().isoformat()
        record_common = {
            "task_type": task_type,
            "perturbation": perturb_type,
            "metric": "accuracy",
            "num_samples": num_samples,
            "batch_size": batch_size,
            "stride": stride,
            "adapter_index": idx,
            "paired_adapter_index": idx,
            "summary": summary,
            "timestamp": timestamp,
            "scan_results_file": os.path.join("checkpoint_scan", f"scan_results_{perturb_type}.json"),
            "scan_plot_file": os.path.join("checkpoint_scan", f"scan_plot_{perturb_type}.png"),
        }

        mark_record = {
            **record_common,
            "role": markovian_role,
            "paired_role": non_markovian_role,
            "paired_run": os.path.basename(non_markovian_dir),
            "status": "completed",
        }
        non_record = {
            **record_common,
            "role": non_markovian_role,
            "paired_role": markovian_role,
            "paired_run": os.path.basename(markovian_dir),
            "status": "completed",
        }

        mark_metadata["records"][metadata_key] = mark_record
        non_metadata["records"][metadata_key] = non_record
        persist_metadata_cache(mark_adapter_dir)
        persist_metadata_cache(non_adapter_dir)
        
    # Save scan results
    output_dir = os.path.join(markovian_dir, "checkpoint_scan")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"scan_results_{perturb_type}.json")
    
    with open(output_file, "w") as f:
        json.dump(scan_results, f)

    non_output_dir = os.path.join(non_markovian_dir, "checkpoint_scan")
    os.makedirs(non_output_dir, exist_ok=True)
    non_output_file = os.path.join(non_output_dir, os.path.basename(output_file))
    if non_output_file != output_file:
        shutil.copy2(output_file, non_output_file)
        
    print(f"Scan results saved to {output_file}")
    sync_run_dir_outputs(markovian_dir)
    sync_run_dir_outputs(non_markovian_dir)
    return scan_results


def plot_checkpoint_scan_results(scan_results, output_dir, perturb_type):
    """
    Plot Sensitivity vs Training Batch from scan results.
    """
    if not scan_results:
        return

    batches = [r["Batch Index"] for r in scan_results]
    degrees = list(scan_results[0]["Sensitivity Summary"].keys())
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(degrees)))
    
    for i, degree in enumerate(degrees):
        values = [r["Sensitivity Summary"][degree] for r in scan_results]
        plt.plot(batches, values, marker='o', label=degree, color=colors[i])
        
    plt.xlabel("Training Batch")
    plt.ylabel("Mean Sensitivity Difference\n(Markovian - Non-Markovian)")
    plt.title(f"Sensitivity Differential Evolution: {perturb_type}")
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(output_dir, f"scan_plot_{perturb_type}.png")
    plt.savefig(output_file)
    print(f"Scan plot saved to {output_file}")
    plt.close()


def run_markovian_comparison(markovian_log_file, non_markovian_log_file, perturb_type, stride=1, max_index=None, save_interval=10, batch_size=8, evaluator="actor", adapter_index=None, markovian_adapter_index=None, non_markovian_adapter_index=None):
    """
    Compare perturbation sensitivity between Markovian and Non-Markovian models.
    
    Args:
        markovian_log_file: Path to the Markovian model's log file
        non_markovian_log_file: Path to the Non-Markovian model's log file
        perturb_type: Type of perturbation to apply
        stride: Process every nth entry of the log file
        max_index: If provided, only process entries with batch_index <= max_index
        save_interval: Save intermediate results every this many examples
        batch_size: Number of examples to process in each batch
        
    Returns:
        List of comparison results
    """
    if perturb_type not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    perturbations = PERTURBATION_SETS[perturb_type]["perturbations"]

    # Resolve adapter indices
    m_idx = markovian_adapter_index if markovian_adapter_index is not None else adapter_index
    nm_idx = non_markovian_adapter_index if non_markovian_adapter_index is not None else adapter_index

    # Load both log files
    print("Loading Markovian model log file...")
    with open(markovian_log_file, "r") as f:
        markovian_log_data = [json.loads(line) for line in f]
    
    print("Loading Non-Markovian model log file...")
    with open(non_markovian_log_file, "r") as f:
        non_markovian_log_data = [json.loads(line) for line in f]

    # Extract hyperparameters from both files
    markovian_hyperparams = markovian_log_data[0]
    non_markovian_hyperparams = non_markovian_log_data[0]
    
    # Verify they have the expected markovian settings
    markovian_flag = markovian_hyperparams.get("markovian", True)
    non_markovian_flag = non_markovian_hyperparams.get("markovian", True)
    
    print(f"Markovian log file markovian setting: {markovian_flag}")
    print(f"Non-Markovian log file markovian setting: {non_markovian_flag}")
    
    if markovian_flag == non_markovian_flag:
        print("WARNING: Both log files have the same markovian setting!")
    
    # Load both models with their respective adapters
    print("Loading Markovian model...")
    actor_markovian, frozen_markovian, tokenizer, device = load_model_with_adapters(markovian_log_file, markovian_hyperparams["model_type"], markovian_hyperparams, adapter_index=m_idx)
    
    print("Loading Non-Markovian model...")
    actor_non_markovian, frozen_non_markovian, _, _ = load_model_with_adapters(non_markovian_log_file, non_markovian_hyperparams["model_type"], non_markovian_hyperparams, adapter_index=nm_idx)

    # Select evaluator per flag (default actor): use adapter-loaded actor, or frozen baseline
    markovian_eval_model = actor_markovian if evaluator == "actor" else frozen_markovian
    non_markovian_eval_model = actor_non_markovian if evaluator == "actor" else frozen_non_markovian

    # Filter log data by batch index if max_index is provided
    if max_index is not None:
        markovian_log_data = [entry for entry in markovian_log_data if entry.get("Batch Index", float('inf')) <= max_index]
        non_markovian_log_data = [entry for entry in non_markovian_log_data if entry.get("Batch Index", float('inf')) <= max_index]
        print(f"Processing entries up to batch index {max_index}")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(markovian_log_file), "markovian_comparison")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_results_{perturb_type}.json")
    
    # Extract entries to process from both models
    markovian_entries = []
    non_markovian_entries = []
    
    for entry in markovian_log_data[1:]:
        if "Example" in entry:
            markovian_entries.append(entry)
    
    for entry in non_markovian_log_data[1:]:
        if "Example" in entry:
            non_markovian_entries.append(entry)
    
    # Apply stride and ensure we have matching data
    markovian_entries = markovian_entries[::stride]
    non_markovian_entries = non_markovian_entries[::stride]
    
    # Use the minimum length to ensure we have paired data
    min_length = min(len(markovian_entries), len(non_markovian_entries))
    markovian_entries = markovian_entries[:min_length]
    non_markovian_entries = non_markovian_entries[:min_length]
    
    print(f"Processing {min_length} paired entries from both models")
    
    comparison_data = []
    
    # Process in batches
    for batch_idx in tqdm(range(0, min_length, batch_size), desc="Processing comparison batches"):
        batch_markovian = markovian_entries[batch_idx:batch_idx + batch_size]
        batch_non_markovian = non_markovian_entries[batch_idx:batch_idx + batch_size]
        batch_size_actual = len(batch_markovian)
        
        # Extract data for current batch
        questions = [entry["Example"].get("Question", "") for entry in batch_markovian]
        actor_cots_markovian = [entry["Example"]["Actor Reasoning"] for entry in batch_markovian]
        actor_cots_non_markovian = [entry["Example"]["Actor Reasoning"] for entry in batch_non_markovian]
        answers = [entry["Example"]["Answer"] for entry in batch_markovian]
        
        # Initialize batch results
        batch_results = []
        for i in range(batch_size_actual):
            batch_results.append({
                "Batch Index": batch_markovian[i].get("Batch Index", None),
                "Markovian Effects": {},
                "Non_Markovian Effects": {},
                "Effect Difference": {}  # Will be Markovian - Non_Markovian
            })
        
        # Calculate original log probs for both models
        # Markovian: without question, using trained Markovian model
        markovian_original_logprobs, _ = calculate_answer_log_probs(
            model=markovian_eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=questions,
            reasoning=actor_cots_markovian,
            answers=answers,
            hyperparameters=markovian_hyperparams,
            include_question=False,  # Markovian doesn't use question
        )
        
        # Non-Markovian: with question, using trained Non-Markovian model
        non_markovian_original_logprobs, _ = calculate_answer_log_probs(
            model=non_markovian_eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=questions,
            reasoning=actor_cots_non_markovian,
            answers=answers,
            hyperparameters=non_markovian_hyperparams,
            include_question=True,  # Non-Markovian uses question
        )
        
        # Process each perturbation
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                continue
            
            # Perturb reasoning for both models
            perturbed_markovian_cots = [perturb_CoT(cot, pert_config) for cot in actor_cots_markovian]
            perturbed_non_markovian_cots = [perturb_CoT(cot, pert_config) for cot in actor_cots_non_markovian]
            
            # Calculate perturbed log probs
            # Markovian: without question, using trained Markovian model
            markovian_perturbed_logprobs, _ = calculate_answer_log_probs(
                model=markovian_eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=questions,
                reasoning=perturbed_markovian_cots,
                answers=answers,
                hyperparameters=markovian_hyperparams,
                include_question=False,
            )
            
            # Non-Markovian: with question, using trained Non-Markovian model
            non_markovian_perturbed_logprobs, _ = calculate_answer_log_probs(
                model=non_markovian_eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=questions,
                reasoning=perturbed_non_markovian_cots,
                answers=answers,
                hyperparameters=non_markovian_hyperparams,
                include_question=True,
            )
            
            # Calculate perturbation effects for this batch
            for i in range(batch_size_actual):
                markovian_effect = markovian_original_logprobs[i].item() - markovian_perturbed_logprobs[i].item()
                non_markovian_effect = non_markovian_original_logprobs[i].item() - non_markovian_perturbed_logprobs[i].item()
                effect_difference = markovian_effect - non_markovian_effect
                
                batch_results[i]["Markovian Effects"][pert_name] = markovian_effect
                batch_results[i]["Non_Markovian Effects"][pert_name] = non_markovian_effect
                batch_results[i]["Effect Difference"][pert_name] = effect_difference
        
        # Add batch results to overall results
        comparison_data.extend(batch_results)
        
        # Periodically save intermediate results
        if save_interval > 0 and (batch_idx + batch_size_actual) % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(comparison_data, f)
            print(f"\nSaved {len(comparison_data)} comparison results to {output_path}")
    
    # Save final results
    with open(output_path, "w") as f:
        json.dump(comparison_data, f)
    
    print(f"Markovian comparison analysis complete. Processed {len(comparison_data)} entries.")
    print(f"Results saved to {output_path}")
    
    return comparison_data, markovian_hyperparams, non_markovian_hyperparams


def _generate_actor_cots_for_questions(model, tokenizer, device, questions, hyperparameters):
    """Generate actor chain-of-thought texts for a batch of questions using shared helper."""
    return generate_actor_reasoning(
        actor_model=model,
        tokenizer=tokenizer,
        device=device,
        questions=list(questions),
        hyperparameters=hyperparameters,
    )


def run_markovian_comparison_fresh(
    markovian_log_file,
    non_markovian_log_file,
    perturb_type,
    num_samples=128,
    task_type="wiki_continuation",
    question_length=None,
    target_length=None,
    batch_size=8,
    evaluator="actor",
    adapter_index=None,
    markovian_adapter_index=None,
    non_markovian_adapter_index=None,
):
    """Run comparison using fixed checkpoints on fresh datapoints, not training logs."""
    if perturb_type not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    perturbations = PERTURBATION_SETS[perturb_type]["perturbations"]

    # Resolve adapter indices
    m_idx = markovian_adapter_index if markovian_adapter_index is not None else adapter_index
    nm_idx = non_markovian_adapter_index if non_markovian_adapter_index is not None else adapter_index

    # Load hyperparameters from both logs
    with open(markovian_log_file, "r") as f:
        markovian_hyperparams = json.loads(next(f))
    with open(non_markovian_log_file, "r") as f:
        non_markovian_hyperparams = json.loads(next(f))

    # Force desired task type for fresh comparison
    markovian_hyperparams = {**markovian_hyperparams, "task_type": task_type}
    non_markovian_hyperparams = {**non_markovian_hyperparams, "task_type": task_type}

    # If lengths provided, override
    if question_length is not None:
        markovian_hyperparams["question_length"] = int(question_length)
        non_markovian_hyperparams["question_length"] = int(question_length)
    if target_length is not None:
        markovian_hyperparams["target_length"] = int(target_length)
        non_markovian_hyperparams["target_length"] = int(target_length)

    # Load models with specified adapters
    actor_markovian, frozen_markovian, tokenizer, device = load_model_with_adapters(
        markovian_log_file, markovian_hyperparams["model_type"], markovian_hyperparams, adapter_index=m_idx
    )
    actor_non_markovian, frozen_non_markovian, _, _ = load_model_with_adapters(
        non_markovian_log_file, non_markovian_hyperparams["model_type"], non_markovian_hyperparams, adapter_index=nm_idx
    )

    markovian_eval_model = actor_markovian if evaluator == "actor" else frozen_markovian
    non_markovian_eval_model = actor_non_markovian if evaluator == "actor" else frozen_non_markovian

    # Prepare fresh dataset QA pairs
    q_len = int(markovian_hyperparams.get("question_length", 512))
    t_len = int(markovian_hyperparams.get("target_length", 128))
    qa_pairs = list(load_wiki_pairs(tokenizer, q_len, t_len, num_samples, start_index=10000))
    if not qa_pairs:
        raise RuntimeError("No suitable wiki samples found for fresh comparison.")

    # Process in batches
    comparison_data = []
    output_dir = os.path.join(os.path.dirname(markovian_log_file), "markovian_comparison")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_results_fresh_{perturb_type}.json")

    for batch_start in tqdm(range(0, len(qa_pairs), batch_size), desc="Processing comparison batches (fresh)"):
        batch = qa_pairs[batch_start: batch_start + batch_size]
        questions, answers = zip(*batch)

        # Generate actor CoTs for each model on same questions
        actor_cots_markovian = _generate_actor_cots_for_questions(actor_markovian, tokenizer, device, questions, markovian_hyperparams)
        actor_cots_non_markovian = _generate_actor_cots_for_questions(actor_non_markovian, tokenizer, device, questions, non_markovian_hyperparams)

        # Initialize batch results (use sequential index as batch index)
        batch_results = []
        for i in range(len(questions)):
            batch_results.append({
                "Batch Index": batch_start + i,
                "Markovian Effects": {},
                "Non_Markovian Effects": {},
                "Effect Difference": {}
            })

        # Original log probs
        markovian_original_logprobs, _ = calculate_answer_log_probs(
            model=markovian_eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=list(questions),
            reasoning=actor_cots_markovian,
            answers=list(answers),
            hyperparameters=markovian_hyperparams,
            include_question=False,
        )
        non_markovian_original_logprobs, _ = calculate_answer_log_probs(
            model=non_markovian_eval_model,
            tokenizer=tokenizer,
            device=device,
            questions=list(questions),
            reasoning=actor_cots_non_markovian,
            answers=list(answers),
            hyperparameters=non_markovian_hyperparams,
            include_question=True,
        )

        # Perturbations
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                continue
            perturbed_markovian_cots = [perturb_CoT(cot, pert_config) for cot in actor_cots_markovian]
            perturbed_non_markovian_cots = [perturb_CoT(cot, pert_config) for cot in actor_cots_non_markovian]

            markovian_perturbed_logprobs, _ = calculate_answer_log_probs(
                model=markovian_eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=list(questions),
                reasoning=perturbed_markovian_cots,
                answers=list(answers),
                hyperparameters=markovian_hyperparams,
                include_question=False,
            )
            non_markovian_perturbed_logprobs, _ = calculate_answer_log_probs(
                model=non_markovian_eval_model,
                tokenizer=tokenizer,
                device=device,
                questions=list(questions),
                reasoning=perturbed_non_markovian_cots,
                answers=list(answers),
                hyperparameters=non_markovian_hyperparams,
                include_question=True,
            )

            for i in range(len(questions)):
                markovian_effect = markovian_original_logprobs[i].item() - markovian_perturbed_logprobs[i].item()
                non_markovian_effect = non_markovian_original_logprobs[i].item() - non_markovian_perturbed_logprobs[i].item()
                effect_difference = markovian_effect - non_markovian_effect
                batch_results[i]["Markovian Effects"][pert_name] = markovian_effect
                batch_results[i]["Non_Markovian Effects"][pert_name] = non_markovian_effect
                batch_results[i]["Effect Difference"][pert_name] = effect_difference

        comparison_data.extend(batch_results)

        # Periodic save
        with open(output_path, "w") as f:
            json.dump(comparison_data, f)

    print(f"Markovian fresh comparison analysis complete. Processed {len(comparison_data)} entries.")
    print(f"Results saved to {output_path}")

    return comparison_data, markovian_hyperparams, non_markovian_hyperparams


def combine_all_markovian_comparison_plots(base_directory, font_size=12, include_perturbations=None, exclude_perturbations=None, legend_font_size=None):
    """
    Combine all markovian comparison plots from a directory into a single comprehensive figure.
    
    Args:
        base_directory: Base directory containing markovian_comparison subdirectories
        font_size: Base font size for plot elements (deprecated, use legend_font_size)
        include_perturbations: List of perturbation types to include (if None, include all)
        exclude_perturbations: List of perturbation types to exclude (if None, exclude none)
        legend_font_size: Font size for all text elements (if None, uses font_size for backward compatibility)
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path
    import os
    import numpy as np
    
    # Use legend_font_size if provided, otherwise fall back to font_size for backward compatibility
    if legend_font_size is None:
        legend_font_size = font_size
    
    # Find all markovian comparison plot files
    plot_files = []
    perturbation_types = []
    
    markovian_dir = os.path.join(base_directory, "markovian_comparison")
    if os.path.exists(markovian_dir):
        for filename in os.listdir(markovian_dir):
            if filename.startswith("markovian_comparison_") and filename.endswith("_plot.png"):
                # Extract perturbation type from filename
                perturb_type = filename.replace("markovian_comparison_", "").replace("_plot.png", "")
                
                # Apply include/exclude filters
                if include_perturbations is not None and perturb_type not in include_perturbations:
                    continue
                if exclude_perturbations is not None and perturb_type in exclude_perturbations:
                    continue
                
                plot_files.append(os.path.join(markovian_dir, filename))
                perturbation_types.append(perturb_type)
    
    if not plot_files:
        print(f"No markovian comparison plots found in {markovian_dir}")
        return
    
    # Sort by perturbation type for consistent ordering
    sorted_pairs = sorted(zip(plot_files, perturbation_types), key=lambda x: x[1])
    plot_files, perturbation_types = zip(*sorted_pairs)
    
    n_plots = len(plot_files)
    
    # Create subplot layout - try to make it roughly square
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    fig.suptitle('Comprehensive Markovian vs Non-Markovian Perturbation Analysis', 
                fontsize=legend_font_size + 4, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_plots == 1:
        # For single plot, axes is a single matplotlib axis object
        axes = [axes]
    else:
        # For multiple plots, axes is a numpy array
        axes = axes.flatten()
    
    # Load and display each plot
    for i, (plot_file, perturb_type) in enumerate(zip(plot_files, perturbation_types)):
        try:
            img = mpimg.imread(plot_file)
            axes[i].imshow(img)
            axes[i].set_title(f'{perturb_type.replace("_", " ").title()}', 
                            fontsize=legend_font_size + 2, fontweight='bold')
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading {plot_file}: {e}")
            axes[i].text(0.5, 0.5, f'Error loading\n{perturb_type}', 
                        ha='center', va='center', fontsize=legend_font_size)
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Save combined plot
    output_path = os.path.join(markovian_dir, "combined_markovian_comparison_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_path}")
    print(f"Included {n_plots} perturbation types: {', '.join(sorted(perturbation_types))}")
    plt.close()


def plot_markovian_comparison_results(results, output_dir, perturb_type, window_size=40, font_size=12, legend_font_size=10, markovian_hyperparams=None, non_markovian_hyperparams=None):
    """
    Plot the Markovian vs Non-Markovian comparison results.
    
    Args:
        results: The comparison results data
        output_dir: Directory to save the plot
        perturb_type: The type of perturbation being analyzed
        window_size: Smoothing window size
        font_size: Base font size for plot text elements
        legend_font_size: Font size for the legend
    """
    if not results:
        print("No results to plot.")
        return
    
    # Get all perturbation degrees from the first entry
    perturbation_degrees = list(results[0]["Effect Difference"].keys())
    print(f"Found perturbation degrees: {perturbation_degrees}")
    
    # Only plot non-zero perturbation degrees
    baseline_name = f"{perturb_type.title().replace('_', '')}0%"
    plot_degrees = [deg for deg in perturbation_degrees if deg != baseline_name]
    print(f"Plotting degrees: {plot_degrees}")
    
    if not plot_degrees:
        print("No non-zero perturbation degrees found to plot.")
        return
    
    # Extract batch indices
    batch_indices = [entry["Batch Index"] for entry in results]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_degrees)))
    
    for i, degree in enumerate(plot_degrees):
        # Extract effect differences for this perturbation degree
        effect_differences = [entry["Effect Difference"][degree] for entry in results]
        
        # Smoothing
        if window_size > 1 and len(effect_differences) > window_size:
            try:
                smoothed_effects = savgol_filter(effect_differences, window_size, 3)
                padding = window_size // 2
                x_values = range(padding, len(effect_differences) - padding)
                smoothed_effects = smoothed_effects[padding:-padding]
            except ValueError as e:
                print(f"Smoothing error: {e}. Using raw data.")
                x_values = range(len(effect_differences))
                smoothed_effects = effect_differences
        else:
            x_values = range(len(effect_differences))
            smoothed_effects = effect_differences
        
        # Plot this perturbation degree
        plt.plot(
            x_values,
            smoothed_effects,
            label=f"{degree}",
            color=colors[i],
            linewidth=2,
        )
    
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=legend_font_size, loc="best")
    
    # Create x-axis label with batch size information if available
    xlabel = "Training Batch"
    if "fresh" in perturb_type:
        xlabel = "Sample Index (Fresh Evaluation)"
    elif markovian_hyperparams and non_markovian_hyperparams:
        m_batch_size = markovian_hyperparams.get('batch_size', 'unknown')
        nm_batch_size = non_markovian_hyperparams.get('batch_size', 'unknown')
        if m_batch_size == nm_batch_size:
            xlabel += f" (batch size={m_batch_size})"
        else:
            xlabel += f" (Markovian: {m_batch_size}, Non-Markovian: {nm_batch_size})"
    
    plt.xlabel(xlabel, fontsize=legend_font_size)
    plt.ylabel("Perturbation Effect Difference\n(Markovian Effect - Non-Markovian Effect)", fontsize=legend_font_size)
    
    title = f"Markovian vs Non-Markovian Comparison: {perturb_type.replace('_', ' ').title()}"
    if window_size > 1:
        title += f" (Smoothing: {window_size})"
    else:
        title += " (Raw Data)"
    
    plt.title(title, fontsize=legend_font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=legend_font_size)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f"markovian_comparison_{perturb_type}_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Markovian comparison plot saved to {output_file}")
    plt.close()


def analyze_markovian_comparison_summary(results, perturb_type):
    """
    Print a summary analysis of the Markovian comparison results.
    
    Args:
        results: The comparison results data
        perturb_type: The type of perturbation being analyzed
    """
    if not results:
        print("No results to analyze.")
        return
    
    print(f"\n=== MARKOVIAN COMPARISON SUMMARY: {perturb_type.upper()} ===")
    
    perturbation_degrees = list(results[0]["Effect Difference"].keys())
    baseline_name = f"{perturb_type.title().replace('_', '')}0%"
    analysis_degrees = [deg for deg in perturbation_degrees if deg != baseline_name]
    
    for degree in analysis_degrees:
        markovian_effects = [entry["Markovian Effects"][degree] for entry in results]
        non_markovian_effects = [entry["Non_Markovian Effects"][degree] for entry in results]
        effect_differences = [entry["Effect Difference"][degree] for entry in results]
        
        # Calculate statistics
        mean_markovian = np.mean(markovian_effects)
        mean_non_markovian = np.mean(non_markovian_effects)
        mean_difference = np.mean(effect_differences)
        std_difference = np.std(effect_differences)
        
        # Count how often each model is more sensitive
        markovian_more_sensitive = sum(1 for diff in effect_differences if diff > 0)
        non_markovian_more_sensitive = sum(1 for diff in effect_differences if diff < 0)
        
        print(f"\n{degree}:")
        print(f"  Mean Markovian Effect: {mean_markovian:.4f}")
        print(f"  Mean Non-Markovian Effect: {mean_non_markovian:.4f}")
        print(f"  Mean Difference (M - NM): {mean_difference:.4f} ± {std_difference:.4f}")
        print(f"  Markovian more sensitive: {markovian_more_sensitive}/{len(results)} cases")
        print(f"  Non-Markovian more sensitive: {non_markovian_more_sensitive}/{len(results)} cases")
        
        if mean_difference > 0:
            print(f"  → Overall: Markovian model is MORE sensitive to {degree} perturbations")
        elif mean_difference < 0:
            print(f"  → Overall: Non-Markovian model is MORE sensitive to {degree} perturbations")
        else:
            print(f"  → Overall: Similar sensitivity to {degree} perturbations")
    
    print("\n" + "="*60)


def run_qa_perturbation_accuracy(
    markovian_log_file,
    non_markovian_log_file,
    perturb_type,
    task_type,
    num_samples=None,
    batch_size=8,
    evaluator="actor",
    adapter_index=None,
    question_length=None,
    target_length=None,
    markovian_adapter_index=None,
    non_markovian_adapter_index=None,
    stride: int = 1,
):
    """
    Run perturbation analysis measuring ACCURACY drop on QA tasks.
    """
    if perturb_type not in PERTURBATION_SETS:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    perturbations = PERTURBATION_SETS[perturb_type]["perturbations"]

    markovian_dir = os.path.dirname(markovian_log_file)
    non_markovian_dir = os.path.dirname(non_markovian_log_file)
    sync_run_dir_from_s3(markovian_dir)
    sync_run_dir_from_s3(non_markovian_dir)
    markovian_role = infer_role_from_log_path(markovian_log_file)
    non_markovian_role = infer_role_from_log_path(non_markovian_log_file)

    # Resolve adapter indices
    m_idx = markovian_adapter_index if markovian_adapter_index is not None else adapter_index
    nm_idx = non_markovian_adapter_index if non_markovian_adapter_index is not None else adapter_index
    mark_adapter_dir = os.path.join(markovian_dir, f"adapter_{m_idx}") if m_idx is not None else None
    non_adapter_dir = os.path.join(non_markovian_dir, f"adapter_{nm_idx}") if nm_idx is not None else None

    # Load hyperparameters
    with open(markovian_log_file, "r") as f:
        markovian_hyperparams = json.loads(next(f))
    with open(non_markovian_log_file, "r") as f:
        non_markovian_hyperparams = json.loads(next(f))

    # Force task settings
    markovian_hyperparams = {**markovian_hyperparams, "task_type": task_type}
    non_markovian_hyperparams = {**non_markovian_hyperparams, "task_type": task_type}
    
    if question_length:
        markovian_hyperparams["question_length"] = int(question_length)
        non_markovian_hyperparams["question_length"] = int(question_length)
    if target_length:
        markovian_hyperparams["target_length"] = int(target_length)
        non_markovian_hyperparams["target_length"] = int(target_length)

    # Load models
    actor_markovian, frozen_markovian, tokenizer, device = load_model_with_adapters(
        markovian_log_file, markovian_hyperparams["model_type"], markovian_hyperparams, adapter_index=m_idx
    )
    actor_non_markovian, frozen_non_markovian, _, _ = load_model_with_adapters(
        non_markovian_log_file, non_markovian_hyperparams["model_type"], non_markovian_hyperparams, adapter_index=nm_idx
    )

    markovian_eval_model = actor_markovian if evaluator == "actor" else frozen_markovian
    non_markovian_eval_model = actor_non_markovian if evaluator == "actor" else frozen_non_markovian
    
    # Load Data
    qa_pairs = []
    print(f"Loading fresh {task_type} data for accuracy analysis...")
    
    if task_type == "gsm8k":
        qa_pairs = list(load_gsm8k_dataset(split="test"))
    elif task_type == "mmlu":
        subject = markovian_hyperparams.get("mmlu_subject", None)
        qa_pairs = list(load_mmlu_dataset(split="test", subject=subject))
    elif task_type == "math":
        qa_pairs = list(load_math_dataset(split="test"))
    elif task_type == "svamp":
        qa_pairs = list(load_svamp_dataset(split="test"))
    elif task_type == "aqua":
        qa_pairs = list(load_aqua_dataset(split="test"))
    elif task_type == "mathqa":
        qa_pairs = list(load_mathqa_dataset(split="test"))
    elif task_type == "arc":
        qa_pairs = list(load_arc_dataset(split="validation"))
    elif task_type == "arithmetic":
        qa_pairs = list(load_arithmetic_dataset(chunk_size=num_samples, split="test"))
    else:
        raise ValueError(f"Unsupported task type for QA perturbation: {task_type}")

    if not qa_pairs:
        raise RuntimeError(f"No samples found for {task_type}")
    
    stride = max(1, int(stride or 1))
    qa_pairs = qa_pairs[::stride]
    if num_samples:
        qa_pairs = qa_pairs[:num_samples]
    if not qa_pairs:
        raise RuntimeError("No samples left after applying stride/num_samples filters.")
    effective_num_samples = len(qa_pairs)
    print(f"Loaded {effective_num_samples} examples with stride={stride}.")

    # Helper to select evaluation function
    def get_eval_func(tt):
        if tt == "gsm8k": return evaluate_model_on_gsm8k
        if tt == "mmlu": return evaluate_model_on_mmlu
        if tt == "arc": return evaluate_model_on_arc
        if tt == "aqua": return evaluate_model_on_aqua
        if tt == "mathqa": return evaluate_model_on_mathqa
        if tt in ["svamp", "math", "arithmetic"]: return evaluate_model_on_numeric
        return evaluate_model_on_numeric 

    eval_func = get_eval_func(task_type)

    # 1. Generate Original CoTs and Baselines
    print("Generating Original CoTs and Baseline Accuracy...")
    
    # Markovian (no question in Stage 2)
    markovian_hyperparams["markovian"] = True 
    _, m_results, _ = eval_func(
        actor_markovian, markovian_eval_model, tokenizer, device, qa_pairs, markovian_hyperparams,
        batch_size=batch_size, num_samples=len(qa_pairs)
    )
    m_cots = [r["reasoning"] for r in m_results]
    
    # Non-Markovian (question in Stage 2)
    non_markovian_hyperparams["markovian"] = False
    _, nm_results, _ = eval_func(
        actor_non_markovian, non_markovian_eval_model, tokenizer, device, qa_pairs, non_markovian_hyperparams,
        batch_size=batch_size, num_samples=len(qa_pairs)
    )
    nm_cots = [r["reasoning"] for r in nm_results]
    
    # Prepare results structure
    comparison_data = []
    for i in range(len(qa_pairs)):
        comparison_data.append({
            "Batch Index": i,
            "Markovian Effects": {"Original": int(m_results[i]["correct"])},
            "Non_Markovian Effects": {"Original": int(nm_results[i]["correct"])},
            "Effect Difference": {"Original": int(m_results[i]["correct"]) - int(nm_results[i]["correct"])}
        })

    # 2. Run Perturbations
    for pert_name, pert_config in perturbations.items():
        if pert_name == "Original":
            continue
            
        print(f"Evaluating perturbation: {pert_name}")
        
        # Define perturbation function
        def p_func(text):
            return perturb_CoT(text, pert_config)
            
        # Evaluate Markovian
        _, m_pert_results, _ = eval_func(
            actor_markovian, markovian_eval_model, tokenizer, device, qa_pairs, markovian_hyperparams,
            batch_size=batch_size, num_samples=len(qa_pairs),
            precomputed_cots=m_cots,
            perturbation_fn=p_func
        )
        
        # Evaluate Non-Markovian
        _, nm_pert_results, _ = eval_func(
            actor_non_markovian, non_markovian_eval_model, tokenizer, device, qa_pairs, non_markovian_hyperparams,
            batch_size=batch_size, num_samples=len(qa_pairs),
            precomputed_cots=nm_cots,
            perturbation_fn=p_func
        )
        
        for i in range(len(qa_pairs)):
            m_orig = int(m_results[i]["correct"])
            m_pert = int(m_pert_results[i]["correct"])
            m_sensitivity = m_orig - m_pert
            
            nm_orig = int(nm_results[i]["correct"])
            nm_pert = int(nm_pert_results[i]["correct"])
            nm_sensitivity = nm_orig - nm_pert
            
            diff = m_sensitivity - nm_sensitivity
            
            comparison_data[i]["Markovian Effects"][pert_name] = m_sensitivity
            comparison_data[i]["Non_Markovian Effects"][pert_name] = nm_sensitivity
            comparison_data[i]["Effect Difference"][pert_name] = diff

    # Save results
    output_dir = os.path.join(markovian_dir, "markovian_comparison_accuracy")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"comparison_results_accuracy_{perturb_type}.json")
    
    with open(output_path, "w") as f:
        json.dump(comparison_data, f)
        
    non_output_dir = os.path.join(non_markovian_dir, "markovian_comparison_accuracy")
    os.makedirs(non_output_dir, exist_ok=True)
    non_output_path = os.path.join(non_output_dir, os.path.basename(output_path))
    if non_output_path != output_path:
        shutil.copy2(output_path, non_output_path)

    if mark_adapter_dir and os.path.isdir(mark_adapter_dir) and non_adapter_dir and os.path.isdir(non_adapter_dir):
        metadata_key = build_perturb_metadata_key(
            task_type=task_type,
            perturb_type=perturb_type,
            metric="accuracy",
            paired_role=non_markovian_role,
            paired_adapter_index=nm_idx,
            markovian_run=markovian_dir,
            non_markovian_run=non_markovian_dir,
        )
        mark_metadata = get_cached_metadata(mark_adapter_dir)
        non_metadata = get_cached_metadata(non_adapter_dir)
        timestamp = datetime.datetime.utcnow().isoformat()
        record_common = {
            "task_type": task_type,
            "perturbation": perturb_type,
            "metric": "accuracy",
            "num_samples": effective_num_samples,
            "batch_size": batch_size,
            "stride": stride,
            "summary": compute_sensitivity_summary(comparison_data, perturb_type),
            "timestamp": timestamp,
            "comparison_results_file": os.path.join("markovian_comparison_accuracy", os.path.basename(output_path)),
        }
        mark_record = {
            **record_common,
            "adapter_index": m_idx,
            "paired_adapter_index": nm_idx,
            "role": markovian_role,
            "paired_role": non_markovian_role,
            "paired_run": os.path.basename(non_markovian_dir),
            "status": "completed",
        }
        non_record = {
            **record_common,
            "adapter_index": nm_idx,
            "paired_adapter_index": m_idx,
            "role": non_markovian_role,
            "paired_role": markovian_role,
            "paired_run": os.path.basename(markovian_dir),
            "status": "completed",
        }
        mark_metadata["records"][metadata_key] = mark_record
        non_metadata["records"][metadata_key] = non_record
        persist_metadata_cache(mark_adapter_dir)
        persist_metadata_cache(non_adapter_dir)

    print(f"Accuracy perturbation analysis saved to {output_path}")
    sync_run_dir_outputs(markovian_dir)
    sync_run_dir_outputs(non_markovian_dir)
    return comparison_data, markovian_hyperparams, non_markovian_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Perturbation Analysis Tool")
    parser.add_argument("--log_file", help="Log file to analyze or directory containing perturbation results")
    parser.add_argument("--metric", type=str, default="log_prob", choices=["log_prob", "accuracy"], help="Metric to evaluate: 'log_prob' or 'accuracy'")
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Process every nth entry of the log file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Generate debug plots with raw values"
    )
    parser.add_argument("--max_index", type=int, help="Maximum index to plot")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from saved results",
    )
    parser.add_argument(
        "--process_only", action="store_true", help="Only process data without plotting"
    )
    parser.add_argument(
        "--include_question",
        action="store_true",
        help="Include the question text in the prompt when evaluating",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save intermediate results every N entries (0 to disable)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of examples to process in each batch (0 for non-batched processing)",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["actor", "frozen"],
        default="actor",
        help="Which model to use for evaluation: adapter-loaded actor (actor) or frozen baseline (frozen). Default: actor",
    )
    parser.add_argument(
        "--adapter_index",
        type=int,
        help="Force loading a specific adapter index (e.g., 400 will use adapter_400)",
    )
    # Fresh datapoint comparison flags
    parser.add_argument(
        "--fresh_comparison",
        action="store_true",
        help="Run Markovian vs Non-Markovian comparison on fresh datapoints (not training logs)",
    )
    parser.add_argument(
        "--fresh_task_type",
        type=str,
        default="wiki_continuation",
        help="Task type for fresh comparison (e.g., wiki_continuation, wiki_compression)",
    )
    parser.add_argument(
        "--fresh_num_samples",
        type=int,
        default=128,
        help="Number of fresh samples to evaluate in fresh comparison",
    )
    parser.add_argument(
        "--fresh_question_length",
        type=int,
        help="Question/context length (tokens) for fresh wiki tasks",
    )
    parser.add_argument(
        "--fresh_target_length",
        type=int,
        help="Target/answer length (tokens) for fresh wiki tasks",
    )
    
    # New arguments for Markovian comparison
    parser.add_argument(
        "--markovian_comparison",
        action="store_true",
        help="Run Markovian vs Non-Markovian comparison analysis",
    )
    parser.add_argument(
        "--markovian_log",
        type=str,
        help="Path to Markovian model log file (for comparison mode)",
    )
    parser.add_argument(
        "--non_markovian_log", 
        type=str,
        help="Path to Non-Markovian model log file (for comparison mode)",
    )

    # Adjusted to not require --perturb when using --collate
    perturb_group = parser.add_mutually_exclusive_group(required=False)
    perturb_group.add_argument(
        "--perturb",
        nargs="+",
        choices=list(PERTURBATION_SETS.keys()),
        help="Type(s) of perturbation to analyze",
    )
    perturb_group.add_argument(
        "--all", action="store_true", help="Run all perturbation types"
    )

    # Modify the --collate help message
    parser.add_argument(
        "--collate",
        nargs="+",
        help="List of perturbation result JSON files to average"
    )
    parser.add_argument(
        "--output_dir",
        default="perturbation_results",
        help="Output directory for collated results",
    )

    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="Base font size for plot text elements"
    )
    parser.add_argument(
        "--legend_font_size",
        type=int,
        default=10,
        help="Font size for the legend in plots"
    )
    parser.add_argument(
        "--plot_multiple_perturbations",
        action="store_true",
        help="Generate a combined plot for multiple perturbation types"
    )
    parser.add_argument(
        "--combine_all_plots",
        action="store_true",
        help="Combine all existing markovian comparison plots into a single comprehensive figure"
    )
    parser.add_argument(
        "--include_perturbations",
        nargs="+",
        choices=list(PERTURBATION_SETS.keys()),
        help="Include only specified perturbation types in combined plot (for --combine_all_plots)"
    )
    parser.add_argument(
        "--exclude_perturbations",
        nargs="+",
        choices=list(PERTURBATION_SETS.keys()),
        help="Exclude specified perturbation types from combined plot (for --combine_all_plots)"
    )
    parser.add_argument(
        "--regenerate_before_combine",
        action="store_true",
        help="Regenerate individual markovian comparison plots with new parameters before combining (for --combine_all_plots)"
    )

    parser.add_argument(
        "--sweep_checkpoints",
        action="store_true",
        help="Iterate over all common checkpoints and plot sensitivity evolution (requires --fresh_comparison)"
    )
    parser.add_argument(
        "--plot_scan_only",
        action="store_true",
        help="Only plot from existing checkpoint sweep results (requires --sweep_checkpoints)"
    )

    args = parser.parse_args()

    if args.all:
        args.perturb = list(PERTURBATION_SETS.keys())

    # Auto-detection of logs and adapters if missing
    markovian_adapter_index = args.adapter_index
    non_markovian_adapter_index = args.adapter_index
    
    # Determine target task for auto-detection
    target_task = args.fresh_task_type
        
    if (args.fresh_comparison or args.markovian_comparison) and (not args.markovian_log or not args.non_markovian_log):
        if target_task:
            print(f"Attempting to auto-detect logs for task: {target_task}")
            if not args.markovian_log:
                 log, idx = find_best_run_for_task(target_task, "Markovian")
                 if log:
                     print(f"  Markovian: {log} (adapter {idx})")
                     args.markovian_log = log
                     if idx is not None: markovian_adapter_index = idx
            
            if not args.non_markovian_log:
                 log, idx = find_best_run_for_task(target_task, "NonMarkovian")
                 if log:
                     print(f"  NonMarkovian: {log} (adapter {idx})")
                     args.non_markovian_log = log
                     if idx is not None: non_markovian_adapter_index = idx

    # Handle fresh datapoint comparison mode
    if args.fresh_comparison:
        if not args.markovian_log or not args.non_markovian_log:
            print("Error: --fresh_comparison requires both --markovian_log and --non_markovian_log (auto-detection failed)")
            return
        if not args.perturb:
            print("Error: --fresh_comparison requires --perturb argument")
            return
            
        # Checkpoint Sweep Mode
        if args.sweep_checkpoints:
            for perturb_type in args.perturb:
                output_dir = os.path.join(os.path.dirname(args.markovian_log), "checkpoint_scan")
                
                if args.plot_scan_only:
                    print(f"Plotting existing scan results for {perturb_type}...")
                    scan_results = load_scan_results(output_dir, perturb_type)
                    if scan_results:
                        plot_checkpoint_scan_results(scan_results, output_dir, perturb_type)
                    continue

                print(f"Running Checkpoint Sweep for {perturb_type}...")
                scan_results = run_checkpoint_scan(
                    markovian_log_file=args.markovian_log,
                    non_markovian_log_file=args.non_markovian_log,
                    perturb_type=perturb_type,
                    task_type=args.fresh_task_type,
                    num_samples=args.fresh_num_samples,
                    batch_size=args.batch_size,
                    stride=args.stride,
                    question_length=args.fresh_question_length,
                    target_length=args.fresh_target_length,
                )
                plot_checkpoint_scan_results(scan_results, output_dir, perturb_type)
            return

        for perturb_type in args.perturb:
            print(f"Running Fresh Markovian vs Non-Markovian comparison for {perturb_type} (Metric: {args.metric})...")
            
            if args.metric == "accuracy":
                comparison_data, markovian_hyperparams, non_markovian_hyperparams = run_qa_perturbation_accuracy(
                    markovian_log_file=args.markovian_log,
                    non_markovian_log_file=args.non_markovian_log,
                    perturb_type=perturb_type,
                    task_type=args.fresh_task_type,
                    num_samples=args.fresh_num_samples,
                    batch_size=args.batch_size,
                    evaluator=args.evaluator,
                    adapter_index=args.adapter_index,
                    question_length=args.fresh_question_length,
                    target_length=args.fresh_target_length,
                    markovian_adapter_index=markovian_adapter_index,
                    non_markovian_adapter_index=non_markovian_adapter_index,
                    stride=args.stride,
                )
            else:
                comparison_data, markovian_hyperparams, non_markovian_hyperparams = run_markovian_comparison_fresh(
                    markovian_log_file=args.markovian_log,
                    non_markovian_log_file=args.non_markovian_log,
                    perturb_type=perturb_type,
                    num_samples=args.fresh_num_samples,
                    task_type=args.fresh_task_type,
                    question_length=args.fresh_question_length,
                    target_length=args.fresh_target_length,
                    batch_size=args.batch_size,
                    evaluator=args.evaluator,
                    adapter_index=args.adapter_index,
                    markovian_adapter_index=markovian_adapter_index,
                    non_markovian_adapter_index=non_markovian_adapter_index,
                )
            
            output_dir = os.path.join(os.path.dirname(args.markovian_log), f"markovian_comparison_{args.metric}")
            os.makedirs(output_dir, exist_ok=True)
            
            plot_markovian_comparison_results(
                results=comparison_data,
                output_dir=output_dir,
                perturb_type=f"fresh_{perturb_type}",
                window_size=args.window_size,
                font_size=args.font_size,
                legend_font_size=args.legend_font_size,
                markovian_hyperparams=markovian_hyperparams,
                non_markovian_hyperparams=non_markovian_hyperparams,
            )
            analyze_markovian_comparison_summary(comparison_data, f"fresh_{perturb_type}")
            print(f"Fresh markovian comparison for {perturb_type} completed.")
        return

    # Handle markovian comparison mode
    if args.markovian_comparison:
        if not args.markovian_log or not args.non_markovian_log:
            print("Error: --markovian_comparison requires both --markovian_log and --non_markovian_log arguments (auto-detection failed)")
            return
        if not args.perturb:
            print("Error: --markovian_comparison requires --perturb argument")
            return
        
        for perturb_type in args.perturb:
            print(f"Running Markovian vs Non-Markovian comparison for {perturb_type}...")
            comparison_data, markovian_hyperparams, non_markovian_hyperparams = run_markovian_comparison(
                markovian_log_file=args.markovian_log,
                non_markovian_log_file=args.non_markovian_log,
                perturb_type=perturb_type,
                stride=args.stride,
                max_index=args.max_index,
                save_interval=args.save_interval,
                batch_size=args.batch_size,
                evaluator=args.evaluator,
                adapter_index=args.adapter_index,
                markovian_adapter_index=markovian_adapter_index,
                non_markovian_adapter_index=non_markovian_adapter_index,
            )
            
            # Generate plots and analysis
            output_dir = os.path.join(os.path.dirname(args.markovian_log), "markovian_comparison")
            plot_markovian_comparison_results(
                results=comparison_data,
                output_dir=output_dir,
                perturb_type=perturb_type,
                window_size=args.window_size,
                font_size=args.font_size,
                legend_font_size=args.legend_font_size,
                markovian_hyperparams=markovian_hyperparams,
                non_markovian_hyperparams=non_markovian_hyperparams
            )
            
            # Print summary analysis
            analyze_markovian_comparison_summary(comparison_data, perturb_type)
            
            print(f"Markovian comparison for {perturb_type} completed.")
        
        return  # Exit after comparison analysis

    # Handle combine all plots mode
    if args.combine_all_plots:
        if not args.log_file:
            print("Error: --combine_all_plots requires --log_file argument to specify the base directory")
            return
        
        # Validate include/exclude arguments
        if args.include_perturbations and args.exclude_perturbations:
            print("Error: Cannot specify both --include_perturbations and --exclude_perturbations")
            return
        
        # If log_file points to a file, get its directory; if it's a directory, use it directly
        if os.path.isfile(args.log_file):
            base_dir = os.path.dirname(args.log_file)
        else:
            base_dir = args.log_file
            
        # Regenerate individual plots if requested
        if args.regenerate_before_combine:
            # Determine which perturbations to regenerate
            if args.include_perturbations:
                perturb_types_to_regenerate = args.include_perturbations
            else:
                # Use all available perturbation types, minus excluded ones
                perturb_types_to_regenerate = list(PERTURBATION_SETS.keys())
                if args.exclude_perturbations:
                    perturb_types_to_regenerate = [p for p in perturb_types_to_regenerate if p not in args.exclude_perturbations]
            
            print(f"Regenerating individual plots for: {perturb_types_to_regenerate}")
            markovian_dir = os.path.join(base_dir, "markovian_comparison")
            
            for perturb_type in perturb_types_to_regenerate:
                json_file = os.path.join(markovian_dir, f"comparison_results_{perturb_type}.json")
                if os.path.exists(json_file):
                    print(f"Regenerating plot for {perturb_type}...")
                    with open(json_file, 'r') as f:
                        results = json.load(f)
                    plot_markovian_comparison_results(
                        results=results,
                        output_dir=markovian_dir,
                        perturb_type=perturb_type,
                        window_size=args.window_size,
                        font_size=args.font_size,
                        legend_font_size=args.legend_font_size
                    )
                else:
                    print(f"Warning: {json_file} not found, skipping {perturb_type}")
        
        combine_all_markovian_comparison_plots(
            base_dir, 
            font_size=args.font_size,
            include_perturbations=args.include_perturbations,
            exclude_perturbations=args.exclude_perturbations,
            legend_font_size=args.legend_font_size
        )
        return

    if args.collate:
        if not args.output_dir:
            print("Please specify an output directory using --output_dir when using --collate.")
            return
        # Extract perturb_type from the filenames
        perturb_types = set()
        include_question = False
        for file in args.collate:
            basename = os.path.basename(file)
            # Check if file includes question in the name
            if "_with_question.json" in basename:
                include_question = True
                basename = basename.replace("_with_question.json", ".json")
            if basename.startswith("perturbation_results_") and basename.endswith(".json"):
                perturb_type = basename[len("perturbation_results_"):-len(".json")]
                perturb_types.add(perturb_type)
            else:
                print(f"Invalid perturbation result file: {file}")
                return
        if len(perturb_types) != 1:
            print("All perturbation result files must be for the same perturbation type.")
            return
        perturb_type = perturb_types.pop()
        print(f"Collating results for perturbation type: {perturb_type}" + 
              (" (with question)" if include_question else ""))
        collate_perturbation_results(args.collate, args.output_dir, perturb_type, include_question)
        print(f"Collation complete. Results saved to {args.output_dir}")
        if not args.plot_only:
            return
        # Update log_file to point to collated results for plotting
        args.log_file = args.output_dir
        args.perturb = [perturb_type]
        args.include_question = include_question
    else:
        if args.log_file:
            if not args.perturb and not args.all:
                print("Please specify perturbation types using --perturb or --all.")
                return
        else:
            # Get the latest result directory
            log_dir = find_latest_result()
            if log_dir is None:
                print("No result directories found.")
                return
            args.log_file = log_dir
    
    # Run perturbation analysis if not in plot_only mode
    if not args.plot_only:
        for perturb_type in args.perturb:
            question_status = "with" if args.include_question else "without"
            print(f"Running perturbation analysis for {perturb_type} ({question_status} question)...")
            
            # Choose between batched and non-batched processing
            if args.batch_size > 0:
                print(f"Using batched processing with batch size {args.batch_size}")
                results = run_perturbations_batched(
                    args.log_file, 
                    perturb_type, 
                    include_question=args.include_question,
                    stride=args.stride, 
                    max_index=args.max_index,
                    save_interval=args.save_interval,
                    batch_size=args.batch_size,
                    evaluator=args.evaluator,
                    adapter_index=args.adapter_index,
                )
            else:
                print("Using non-batched processing")
                results = run_perturbations(
                    args.log_file, 
                    perturb_type, 
                    include_question=args.include_question,
                    stride=args.stride, 
                    max_index=args.max_index,
                    save_interval=args.save_interval,
                    evaluator=args.evaluator,
                    adapter_index=args.adapter_index,
                )
            
            save_perturbation_results(
                results, 
                args.log_file, 
                perturb_type, 
                include_question=args.include_question
            )
            print(f"Analysis for {perturb_type} completed and saved.")

    # Plot if needed
    if not args.process_only:
        if args.plot_only and args.plot_multiple_perturbations and len(args.perturb) > 1:
            # Create combined plot for multiple perturbation types
            plot_multiple_perturbation_results(
                args.log_file,
                args.perturb,
                window_size=args.window_size,
                max_index=args.max_index,
                font_size=args.font_size,
                legend_font_size=args.legend_font_size,
                include_question=args.include_question
            )
        else:
            for perturb_type in args.perturb:
                result_file = get_output_paths(args.log_file, perturb_type, args.include_question)["json"]
                try:
                    with open(result_file, "r") as f:
                        results = json.load(f)
                    plot_perturbation_results(
                        results,
                        args.log_file,
                        perturb_type,
                        window_size=args.window_size,
                        debug=args.debug,
                        max_index=args.max_index,
                        font_size=args.font_size,
                        legend_font_size=args.legend_font_size,
                        include_question=args.include_question
                    )
                except FileNotFoundError:
                    print(
                        f"No saved results found for {perturb_type}{' with question' if args.include_question else ''} in {args.log_file}. Run the analysis first or check the file path."
                    )
    else:
        print("Process-only mode is selected, but no processing code is provided.")

if __name__ == "__main__":
    main()
