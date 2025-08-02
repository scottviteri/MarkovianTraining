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
from train import calculate_answer_log_probs
from utils import find_latest_result, print_debug_info
from tqdm import tqdm
import string
from pathlib import Path
from utils import load_model
from peft import PeftModel
import glob


def load_model_with_adapters(log_file_path, model_type, hyperparameters):
    """
    Load a model with its trained adapters if they exist.
    
    Args:
        log_file_path: Path to the log file
        model_type: Type of model to load
        hyperparameters: Hyperparameters for the model
        
    Returns:
        tuple: (actor_model, frozen_model, tokenizer, device)
    """
    # Load base models first
    actor_model, frozen_model, tokenizer, device = load_model(model_type, hyperparameters)
    
    # Look for adapter directories in the same directory as the log file
    log_dir = os.path.dirname(log_file_path)
    adapter_pattern = os.path.join(log_dir, "adapter_*")
    adapter_dirs = glob.glob(adapter_pattern)
    
    if adapter_dirs:
        # Sort by batch number to get the latest adapter
        def get_batch_number(adapter_path):
            try:
                return int(os.path.basename(adapter_path).split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        adapter_dirs_sorted = sorted(adapter_dirs, key=get_batch_number)
        latest_adapter = adapter_dirs_sorted[-1]
        print(f"Loading trained adapter from: {latest_adapter}")
        
        # Load the adapter using PEFT
        actor_model = PeftModel.from_pretrained(
            actor_model,  # Base model with LoRA config
            latest_adapter,  # Adapter path
            is_trainable=False  # Set to inference mode
        )
        print(f"Successfully loaded adapter from batch {os.path.basename(latest_adapter)}")
    else:
        print(f"No trained adapters found in {log_dir}, using base model")
    
    return actor_model, frozen_model, tokenizer, device


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


def run_perturbations(log_file, perturb_type, include_question=False, stride=1, max_index=None, save_interval=10):
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
    _, frozen_model, tokenizer, device = load_model_with_adapters(log_file, hyperparameters["model_type"], hyperparameters)

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
            frozen_model=frozen_model,
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
            frozen_model=frozen_model,
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
                frozen_model=frozen_model,
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
                frozen_model=frozen_model,
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


def run_perturbations_batched(log_file, perturb_type, include_question=False, stride=1, max_index=None, save_interval=10, batch_size=8):
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
    _, frozen_model, tokenizer, device = load_model_with_adapters(log_file, hyperparameters["model_type"], hyperparameters)

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
            frozen_model=frozen_model,
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
            frozen_model=frozen_model,
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
                frozen_model=frozen_model,
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
                frozen_model=frozen_model,
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
    
    plt.xlabel("Example Index", fontsize=font_size)
    
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
                ax.set_xlabel("Example Index", fontsize=font_size)
            
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


def run_markovian_comparison(markovian_log_file, non_markovian_log_file, perturb_type, stride=1, max_index=None, save_interval=10, batch_size=8):
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
    _, markovian_model, tokenizer, device = load_model_with_adapters(markovian_log_file, markovian_hyperparams["model_type"], markovian_hyperparams)
    
    print("Loading Non-Markovian model...")
    _, non_markovian_model, _, _ = load_model_with_adapters(non_markovian_log_file, non_markovian_hyperparams["model_type"], non_markovian_hyperparams)

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
            frozen_model=markovian_model,
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
            frozen_model=non_markovian_model,
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
                frozen_model=markovian_model,
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
                frozen_model=non_markovian_model,
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
    
    return comparison_data


def combine_all_markovian_comparison_plots(base_directory, font_size=12):
    """
    Combine all markovian comparison plots from a directory into a single comprehensive figure.
    
    Args:
        base_directory: Base directory containing markovian_comparison subdirectories
        font_size: Font size for plot elements
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from pathlib import Path
    import os
    
    # Find all markovian comparison plot files
    plot_files = []
    perturbation_types = []
    
    markovian_dir = os.path.join(base_directory, "markovian_comparison")
    if os.path.exists(markovian_dir):
        for filename in os.listdir(markovian_dir):
            if filename.startswith("markovian_comparison_") and filename.endswith("_plot.png"):
                plot_files.append(os.path.join(markovian_dir, filename))
                # Extract perturbation type from filename
                perturb_type = filename.replace("markovian_comparison_", "").replace("_plot.png", "")
                perturbation_types.append(perturb_type)
    
    if not plot_files:
        print(f"No markovian comparison plots found in {markovian_dir}")
        return
    
    # Sort by perturbation type for consistent ordering
    sorted_pairs = sorted(zip(plot_files, perturbation_types), key=lambda x: x[1])
    plot_files, perturbation_types = zip(*sorted_pairs)
    
    n_plots = len(plot_files)
    
    # Create subplot layout - try to make it roughly square
    if n_plots <= 4:
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
                fontsize=font_size + 4, fontweight='bold')
    
    # Flatten axes array for easier indexing
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Load and display each plot
    for i, (plot_file, perturb_type) in enumerate(zip(plot_files, perturbation_types)):
        try:
            img = mpimg.imread(plot_file)
            axes[i].imshow(img)
            axes[i].set_title(f'{perturb_type.replace("_", " ").title()}', 
                            fontsize=font_size + 2, fontweight='bold')
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading {plot_file}: {e}")
            axes[i].text(0.5, 0.5, f'Error loading\n{perturb_type}', 
                        ha='center', va='center', fontsize=font_size)
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
    plt.close()
    
    print(f"Combined plot saved to: {output_path}")
    print(f"Included {n_plots} perturbation types: {', '.join(perturbation_types)}")


def plot_markovian_comparison_results(results, output_dir, perturb_type, window_size=40, font_size=12, legend_font_size=10):
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
    
    plt.xlabel("Example Index", fontsize=font_size)
    plt.ylabel("Perturbation Effect Difference\n(Markovian Effect - Non-Markovian Effect)", fontsize=font_size)
    
    title = f"Markovian vs Non-Markovian Comparison: {perturb_type.replace('_', ' ').title()}"
    if window_size > 1:
        title += f" (Smoothing: {window_size})"
    else:
        title += " (Raw Data)"
    
    plt.title(title, fontsize=font_size + 2)
    plt.tick_params(axis="both", which="major", labelsize=font_size)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add interpretation text
    plt.figtext(0.02, 0.02, 
                "Positive values: Markovian model more sensitive to perturbations\n"
                "Negative values: Non-Markovian model more sensitive to perturbations",
                fontsize=font_size-2, style='italic')
    
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


def main():
    parser = argparse.ArgumentParser(description="Perturbation Analysis Tool")
    parser.add_argument("--log_file", help="Log file to analyze or directory containing perturbation results")
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

    args = parser.parse_args()

    if args.all:
        args.perturb = list(PERTURBATION_SETS.keys())

    # Handle markovian comparison mode
    if args.markovian_comparison:
        if not args.markovian_log or not args.non_markovian_log:
            print("Error: --markovian_comparison requires both --markovian_log and --non_markovian_log arguments")
            return
        if not args.perturb:
            print("Error: --markovian_comparison requires --perturb argument")
            return
        
        for perturb_type in args.perturb:
            print(f"Running Markovian vs Non-Markovian comparison for {perturb_type}...")
            comparison_data = run_markovian_comparison(
                markovian_log_file=args.markovian_log,
                non_markovian_log_file=args.non_markovian_log,
                perturb_type=perturb_type,
                stride=args.stride,
                max_index=args.max_index,
                save_interval=args.save_interval,
                batch_size=args.batch_size
            )
            
            # Generate plots and analysis
            output_dir = os.path.join(os.path.dirname(args.markovian_log), "markovian_comparison")
            plot_markovian_comparison_results(
                results=comparison_data,
                output_dir=output_dir,
                perturb_type=perturb_type,
                window_size=args.window_size,
                font_size=args.font_size,
                legend_font_size=args.legend_font_size
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
        
        # If log_file points to a file, get its directory; if it's a directory, use it directly
        if os.path.isfile(args.log_file):
            base_dir = os.path.dirname(args.log_file)
        else:
            base_dir = args.log_file
            
        combine_all_markovian_comparison_plots(base_dir, font_size=args.font_size)
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
                    batch_size=args.batch_size
                )
            else:
                print("Using non-batched processing")
                results = run_perturbations(
                    args.log_file, 
                    perturb_type, 
                    include_question=args.include_question,
                    stride=args.stride, 
                    max_index=args.max_index,
                    save_interval=args.save_interval
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
