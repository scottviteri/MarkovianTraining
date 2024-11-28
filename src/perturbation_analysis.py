import os
import json
import re
import random
import argparse
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


def get_output_paths(log_file, perturb_type):
    """Get standardized paths for output files."""
    base_dir = os.path.dirname(log_file)
    base_name = f"perturbation_results_{perturb_type}"
    return {
        "json": os.path.join(base_dir, f"{base_name}.json"),
        "plot": os.path.join(base_dir, f"{base_name}_plot.png"),
        "debug_plot": os.path.join(base_dir, f"{base_name}_debug.png"),
    }


def save_perturbation_results(results, log_file, perturb_type):
    """Save perturbation results to a JSON file."""
    output_file = get_output_paths(log_file, perturb_type)["json"]
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")


def load_perturbation_results(log_file, perturb_type):
    """Load perturbation results from a JSON file."""
    input_file = get_output_paths(log_file, perturb_type)["json"]
    with open(input_file, "r") as f:
        return json.load(f)


def run_perturbations(log_file, perturb_type, stride=1, max_index=None):
    """
    Run perturbation analysis on the given log file.
    max_index: if provided, only process entries with batch_index <= max_index
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
    frozen_model, tokenizer, device = load_model(hyperparameters["model_type"])

    # Filter log data by batch index if max_index is provided
    if max_index is not None:
        log_data = [entry for entry in log_data if entry.get("Batch Index", float('inf')) <= max_index]
        print(f"Processing entries up to batch index {max_index}")

    # Extract perturbation-related metrics
    perturbation_data = []
    for i, entry in enumerate(tqdm(log_data[1::stride], desc="Processing entries")):
        if "Example" not in entry:
            continue

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
                "Critic": {
                    "Original": None,
                    "Perturbed": {}
                }
            }
        }

        # Calculate Original log probs for both Actor and Critic
        actor_log_prob, _ = calculate_answer_log_probs(
            frozen_model=frozen_model,
            tokenizer=tokenizer,
            device=device,
            questions=[question],
            reasoning=[actor_CoT],
            answers=[answer],
            hyperparameters=hyperparameters,
        )
        critic_log_prob, _ = calculate_answer_log_probs(
            frozen_model=frozen_model,
            tokenizer=tokenizer,
            device=device,
            questions=[question],
            reasoning=[critic_CoT],
            answers=[answer],
            hyperparameters=hyperparameters,
        )
        entry_results["Log Probs"]["Actor"]["Original"] = actor_log_prob[0].item()
        entry_results["Log Probs"]["Critic"]["Original"] = critic_log_prob[0].item()

        # Perform perturbations and calculate log probabilities for both Actor and Critic
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                continue

            # Perturb Actor CoT
            perturbed_actor_CoT = perturb_CoT(actor_CoT, pert_config)
            actor_perturbed_log_prob, _ = calculate_answer_log_probs(
                frozen_model=frozen_model,
                tokenizer=tokenizer,
                device=device,
                questions=[question],
                reasoning=[perturbed_actor_CoT],
                answers=[answer],
                hyperparameters=hyperparameters,
            )
            entry_results["Log Probs"]["Actor"]["Perturbed"][pert_name] = actor_perturbed_log_prob[0].item()

            # Perturb Critic CoT
            perturbed_critic_CoT = perturb_CoT(critic_CoT, pert_config)
            critic_perturbed_log_prob, _ = calculate_answer_log_probs(
                frozen_model=frozen_model,
                tokenizer=tokenizer,
                device=device,
                questions=[question],
                reasoning=[perturbed_critic_CoT],
                answers=[answer],
                hyperparameters=hyperparameters,
            )
            entry_results["Log Probs"]["Critic"]["Perturbed"][pert_name] = critic_perturbed_log_prob[0].item()

        perturbation_data.append(entry_results)

    return perturbation_data


def plot_perturbation_results(results, log_file, perturb_type, window_size=40, debug=False, max_index=None, font_size=12):
    """
    Plot the results of perturbation analysis.
    No smoothing is applied if window_size=1.
    font_size controls the size of text elements in the plot.
    """
    if not results:
        print("No data found to plot.")
        return

    # Get output paths at the start
    output_paths = get_output_paths(log_file, perturb_type)
    output_file = output_paths["debug_plot"] if debug else output_paths["plot"]

    # Threshold results by array index
    if max_index is not None:
        results = results[:max_index]
        if not results:
            print(f"No data found within max_index {max_index}")
            return

    colors = list(mcolors.TABLEAU_COLORS.values())

    if debug:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

        # Plot perturbation results in first three axes
        for i, (pert, _) in enumerate(results[0]["Log Probs"]["Actor"]["Perturbed"].items()):
            # Skip baseline case
            if pert == f"{perturb_type.title().replace('_', '')}0%":
                continue

            # Get Actor values
            actor_orig_values = [-entry["Log Probs"]["Actor"]["Original"] for entry in results]
            actor_pert_values = [-entry["Log Probs"]["Actor"]["Perturbed"][pert] for entry in results]
            actor_diff_values = [p - o for p, o in zip(actor_pert_values, actor_orig_values)]

            # Get Critic values
            critic_orig_values = [-entry["Log Probs"]["Critic"]["Original"] for entry in results]
            critic_pert_values = [-entry["Log Probs"]["Critic"]["Perturbed"][pert] for entry in results]
            critic_diff_values = [p - o for p, o in zip(critic_pert_values, critic_orig_values)]

            if window_size > 1 and len(actor_diff_values) > window_size:
                actor_smooth = savgol_filter(actor_diff_values, window_size, 3)
                critic_smooth = savgol_filter(critic_diff_values, window_size, 3)
                effect_diff_smooth = savgol_filter([a - c for a, c in zip(actor_diff_values, critic_diff_values)], window_size, 3)

                padding = window_size // 2
                x_values = range(padding, len(actor_diff_values) - padding)
                actor_smooth = actor_smooth[padding:-padding]
                critic_smooth = critic_smooth[padding:-padding]
                effect_diff_smooth = effect_diff_smooth[padding:-padding]
            else:
                x_values = range(len(actor_diff_values))
                actor_smooth = actor_diff_values
                critic_smooth = critic_diff_values
                effect_diff_smooth = [a - c for a, c in zip(actor_diff_values, critic_diff_values)]

            ax1.plot(x_values, actor_smooth, label=f"{pert} (Actor)", color=colors[i])
            ax2.plot(x_values, critic_smooth, label=f"{pert} (Critic)", color=colors[i])
            ax3.plot(x_values, effect_diff_smooth, label=f"{pert} (Actor - Critic)", color=colors[i])

        # Plot Actor vs Critic original difference
        actor_values = [-entry["Log Probs"]["Actor"]["Original"] for entry in results]
        critic_values = [-entry["Log Probs"]["Critic"]["Original"] for entry in results]
        ac_diff_values = [a - c for a, c in zip(actor_values, critic_values)]

        if window_size > 1 and len(ac_diff_values) > window_size:
            ac_diff_smooth = savgol_filter(ac_diff_values, window_size, 3)
            padding = window_size // 2
            x_values = range(padding, len(ac_diff_values) - padding)
            ac_diff_smooth = ac_diff_smooth[padding:-padding]
        else:
            x_values = range(len(ac_diff_values))
            ac_diff_smooth = ac_diff_values

        ax4.plot(x_values, ac_diff_smooth, label="Actor - Critic (Original)", color="purple")

        # Update font sizes in debug plots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylabel("Change in Negative Log Probability", fontsize=font_size+2)
            ax.legend(fontsize=font_size)
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            ax.grid(True)
        
        if window_size > 1:
            ax1.set_title(f"Actor Perturbation Effect (smoothing={window_size})", fontsize=font_size+4)
            ax2.set_title(f"Critic Perturbation Effect (smoothing={window_size})", fontsize=font_size+4)
            ax3.set_title(f"Difference in Effects (Actor - Critic) (smoothing={window_size})", fontsize=font_size+4)
            ax4.set_title(f"Original Actor vs Critic Difference (smoothing={window_size})", fontsize=font_size+4)
        else:
            ax1.set_title("Actor Perturbation Effect (raw)", fontsize=font_size+4)
            ax2.set_title("Critic Perturbation Effect (raw)", fontsize=font_size+4)
            ax3.set_title("Difference in Effects (Actor - Critic) (raw)", fontsize=font_size+4)
            ax4.set_title("Original Actor vs Critic Difference (raw)", fontsize=font_size+4)

        plt.tight_layout()
    else:
        plt.figure(figsize=(12, 6))

        # Plot differences for each perturbation type
        for i, (pert, _) in enumerate(results[0]["Log Probs"]["Actor"]["Perturbed"].items()):
            # Calculate differences for Actor
            actor_orig_values = [-entry["Log Probs"]["Actor"]["Original"] for entry in results]
            actor_pert_values = [-entry["Log Probs"]["Actor"]["Perturbed"][pert] for entry in results]
            actor_diff_values = [p - o for p, o in zip(actor_pert_values, actor_orig_values)]

            # Calculate differences for Critic
            critic_orig_values = [-entry["Log Probs"]["Critic"]["Original"] for entry in results]
            critic_pert_values = [-entry["Log Probs"]["Critic"]["Perturbed"][pert] for entry in results]
            critic_diff_values = [p - o for p, o in zip(critic_pert_values, critic_orig_values)]

            # Calculate the difference between Actor and Critic perturbation effects
            effect_difference = [a - c for a, c in zip(actor_diff_values, critic_diff_values)]

            if window_size > 1 and len(effect_difference) > window_size:
                effect_smooth = savgol_filter(effect_difference, window_size, 3)
                padding = window_size // 2
                x_values = range(padding, len(effect_difference) - padding)
                effect_smooth = effect_smooth[padding:-padding]
            else:
                x_values = range(len(effect_difference))
                effect_smooth = effect_difference

            plt.plot(x_values, effect_smooth, label=f"{pert}", color=colors[i])

        plt.grid(True)
        plt.legend(fontsize=font_size)
        plt.ylabel("Difference in Perturbation Effect\n(Actor - Critic)", fontsize=font_size+2)
        plt.xlabel("Example Index", fontsize=font_size+2)
        plt.tick_params(axis='both', which='major', labelsize=font_size)

        if window_size > 1:
            plt.title(f"Perturbation Analysis: Actor vs Critic Effect (smoothing={window_size})", 
                     fontsize=font_size+4)
        else:
            plt.title("Perturbation Analysis: Actor vs Critic Effect (raw)", 
                     fontsize=font_size+4)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_multiple_perturbation_results(log_file, perturb_types, window_size=40, max_index=None, font_size=12):
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
            results = load_perturbation_results(log_file, perturb_type)
            if max_index is not None:
                results = results[:max_index]
            
            # Plot each perturbation degree
            for i, (pert, _) in enumerate(results[0]["Log Probs"]["Actor"]["Perturbed"].items()):
                # Skip baseline case (0% perturbation)
                if pert == f"{perturb_type.title().replace('_', '')}0%":
                    continue
                
                # Calculate differences for Actor and Critic
                actor_orig_values = [-entry["Log Probs"]["Actor"]["Original"] for entry in results]
                actor_pert_values = [-entry["Log Probs"]["Actor"]["Perturbed"][pert] for entry in results]
                actor_diff_values = [p - o for p, o in zip(actor_pert_values, actor_orig_values)]
                
                critic_orig_values = [-entry["Log Probs"]["Critic"]["Original"] for entry in results]
                critic_pert_values = [-entry["Log Probs"]["Critic"]["Perturbed"][pert] for entry in results]
                critic_diff_values = [p - o for p, o in zip(critic_pert_values, critic_orig_values)]
                
                # Calculate effect difference
                effect_difference = [a - c for a, c in zip(actor_diff_values, critic_diff_values)]
                
                if window_size > 1 and len(effect_difference) > window_size:
                    effect_smooth = savgol_filter(effect_difference, window_size, 3)
                    padding = window_size // 2
                    x_values = range(padding, len(effect_difference) - padding)
                    effect_smooth = effect_smooth[padding:-padding]
                else:
                    x_values = range(len(effect_difference))
                    effect_smooth = effect_difference
                
                ax.plot(x_values, effect_smooth, label=f"{pert}", color=colors[i], linewidth=2)
            
            ax.grid(True)
            ax.legend(fontsize=font_size-2, loc='best')
            
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Diff in Perturbation Effect\n(Actor - Critic)", fontsize=font_size)
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Example Index", fontsize=font_size)
            
            ax.tick_params(axis='both', which='major', labelsize=font_size-2)
            
            if window_size > 1:
                ax.set_title(f"{perturb_type.replace('_', ' ').title()} (smoothing={window_size})", 
                           fontsize=font_size+2)
            else:
                ax.set_title(f"{perturb_type.replace('_', ' ').title()} (raw)", 
                           fontsize=font_size+2)
                
        except FileNotFoundError:
            print(f"No saved results found for {perturb_type}")
            ax.text(0.5, 0.5, f"No data for {perturb_type}", 
                   ha='center', va='center', fontsize=font_size)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(log_file), "combined_perturbation_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {output_file}")


def collate_perturbation_results(log_files, output_dir):
    """
    Average perturbation results across multiple runs and save to a new directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    accumulated_results = {}
    
    # Process each log file directory
    for log_file in log_files:
        base_dir = os.path.dirname(log_file)
        
        for perturb_type in PERTURBATION_SETS.keys():
            try:
                results = load_perturbation_results(log_file, perturb_type)
                
                if perturb_type not in accumulated_results:
                    accumulated_results[perturb_type] = {
                        'results': [],
                        'count': 0
                    }
                
                accumulated_results[perturb_type]['results'].append(results)
                accumulated_results[perturb_type]['count'] += 1
                
            except FileNotFoundError:
                print(f"Warning: No results found for {perturb_type} in {log_file}")
                continue
    
    # Average the results for each perturbation type
    for perturb_type, acc_data in accumulated_results.items():
        if acc_data['count'] == 0:
            continue
            
        results_list = acc_data['results']
        num_runs = len(results_list)
        
        # Find minimum length across all runs
        min_length = min(len(run) for run in results_list)
        print(f"Using {min_length} entries for {perturb_type} (shortest common length)")
        
        # Initialize structure for averaged results
        averaged_results = []
        for entry_idx in range(min_length):
            avg_entry = {
                "Batch Index": results_list[0][entry_idx]["Batch Index"],
                "Log Probs": {
                    "Actor": {
                        "Original": 0.0,
                        "Perturbed": {}
                    },
                    "Critic": {
                        "Original": 0.0,
                        "Perturbed": {}
                    }
                }
            }
            
            # Average the Original values for both Actor and Critic
            for run in results_list:
                avg_entry["Log Probs"]["Actor"]["Original"] += run[entry_idx]["Log Probs"]["Actor"]["Original"] / num_runs
                avg_entry["Log Probs"]["Critic"]["Original"] += run[entry_idx]["Log Probs"]["Critic"]["Original"] / num_runs
            
            # Get perturbation names from first run
            pert_names = results_list[0][entry_idx]["Log Probs"]["Actor"]["Perturbed"].keys()
            
            # Initialize perturbation dictionaries
            for pert_name in pert_names:
                avg_entry["Log Probs"]["Actor"]["Perturbed"][pert_name] = 0.0
                avg_entry["Log Probs"]["Critic"]["Perturbed"][pert_name] = 0.0
            
            # Average the perturbed values for both Actor and Critic
            for run in results_list:
                for pert_name in pert_names:
                    avg_entry["Log Probs"]["Actor"]["Perturbed"][pert_name] += (
                        run[entry_idx]["Log Probs"]["Actor"]["Perturbed"][pert_name] / num_runs
                    )
                    avg_entry["Log Probs"]["Critic"]["Perturbed"][pert_name] += (
                        run[entry_idx]["Log Probs"]["Critic"]["Perturbed"][pert_name] / num_runs
                    )
            
            averaged_results.append(avg_entry)
        
        # Save averaged results
        output_file = os.path.join(output_dir, f"perturbation_results_{perturb_type}.json")
        with open(output_file, "w") as f:
            json.dump(averaged_results, f)
        print(f"Averaged results for {perturb_type} saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Perturbation Analysis Tool")
    parser.add_argument("--log_file", help="Log file to analyze")
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

    # Create a mutually exclusive group for --perturb and --all
    perturb_group = parser.add_mutually_exclusive_group(required=True)
    perturb_group.add_argument(
        "--perturb",
        nargs="+",
        choices=list(PERTURBATION_SETS.keys()),
        help="Type(s) of perturbation to analyze",
    )
    perturb_group.add_argument(
        "--all", action="store_true", help="Run all perturbation types"
    )

    # Add new collate argument
    parser.add_argument(
        "--collate",
        nargs="+",
        help="List of log file locations to average results from",
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

    args = parser.parse_args()

    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_result(return_log=True)

    if not log_file:
        print("No log file found.")
        return

    # If --all is used, set perturb to all available types
    if args.all:
        args.perturb = list(PERTURBATION_SETS.keys())

    print(f"Using log file: {log_file}")
    print(f"Perturbation types: {', '.join(args.perturb)}")

    # Handle collation if requested
    if args.collate:
        print(f"Collating results from {len(args.collate)} runs...")
        collate_perturbation_results(args.collate, args.output_dir)
        print(f"Collation complete. Results saved to {args.output_dir}")
        if not args.plot_only:
            return
        # Update log_file to point to collated results for plotting
        log_file = os.path.join(args.output_dir, "dummy.log")

    # Process data if needed
    if not args.plot_only:
        for perturb_type in args.perturb:
            print(f"\nProcessing {perturb_type}...")
            results = run_perturbations(
                log_file, 
                perturb_type, 
                stride=args.stride,
                max_index=args.max_index
            )
            if results:
                save_perturbation_results(results, log_file, perturb_type)
            else:
                print(f"No data found to save for {perturb_type}.")

    # Plot if needed
    if not args.process_only:
        if args.plot_only and not args.debug and len(args.perturb) > 1:
            # Create combined plot for multiple perturbation types
            plot_multiple_perturbation_results(
                log_file,
                args.perturb,
                window_size=args.window_size,
                max_index=args.max_index,
                font_size=args.font_size
            )
        else:
            # Original single-perturbation plotting
            for perturb_type in args.perturb:
                try:
                    results = load_perturbation_results(log_file, perturb_type)
                    plot_perturbation_results(
                        results,
                        log_file,
                        perturb_type,
                        window_size=args.window_size,
                        debug=args.debug,
                        max_index=args.max_index,
                        font_size=args.font_size
                    )
                except FileNotFoundError:
                    print(
                        f"No saved results found for {perturb_type}. Run without --plot_only first."
                    )


if __name__ == "__main__":
    main()
