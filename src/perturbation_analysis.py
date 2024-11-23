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
from train import calculate_answer_log_probs, find_latest_result, print_debug_info
from tqdm import tqdm
import string
import shutil
from pathlib import Path


def load_model_and_tokenizer(model_type):
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


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


def run_perturbations(log_file, perturb_type, stride=1):
    """
    Run perturbation analysis on the given log file.
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
    frozen_model, tokenizer, device = load_model_and_tokenizer(
        hyperparameters["model_type"]
    )

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


def plot_perturbation_results(
    results, log_file, perturb_type, window_size=40, debug=False, max_index=None
):
    """
    Plot the results of perturbation analysis.
    No smoothing is applied if window_size=1.
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
        for i, (pert, values) in enumerate(results[0]["Avg Log Probs"].items()):
            if (
                f"{perturb_type.title().replace('_', '')}0%" == pert
            ):  # Skip baseline case
                continue

            perturbed_values = [-entry["Avg Log Probs"][pert] for entry in results]
            baseline_key = f"{perturb_type.title().replace('_', '')}0%"
            baseline_values = [
                -entry["Avg Log Probs"][baseline_key] for entry in results
            ]
            diff_values = [p - o for p, o in zip(perturbed_values, baseline_values)]

            if window_size > 1 and len(perturbed_values) > window_size:
                perturbed_smooth = savgol_filter(perturbed_values, window_size, 3)
                baseline_smooth = savgol_filter(baseline_values, window_size, 3)
                diff_smooth = savgol_filter(diff_values, window_size, 3)

                padding = window_size // 2
                x_values = range(padding, len(perturbed_values) - padding)
                perturbed_smooth = perturbed_smooth[padding:-padding]
                baseline_smooth = baseline_smooth[padding:-padding]
                diff_smooth = diff_smooth[padding:-padding]
            else:
                x_values = range(len(perturbed_values))
                perturbed_smooth = perturbed_values
                baseline_smooth = baseline_values
                diff_smooth = diff_values

            ax1.plot(x_values, perturbed_smooth, label=f"{pert}", color=colors[i])
            ax2.plot(x_values, baseline_smooth, label=f"Baseline (0%)", color=colors[i])
            ax3.plot(x_values, diff_smooth, label=f"{pert} diff", color=colors[i])

        # Plot Actor vs Critic difference
        actor_values = [-entry["Original"]["Actor"] for entry in results]
        critic_values = [-entry["Original"]["Critic"] for entry in results]
        ac_diff_values = [a - c for a, c in zip(actor_values, critic_values)]

        if window_size > 1 and len(ac_diff_values) > window_size:
            ac_diff_smooth = savgol_filter(ac_diff_values, window_size, 3)
            padding = window_size // 2
            x_values = range(padding, len(ac_diff_values) - padding)
            ac_diff_smooth = ac_diff_smooth[padding:-padding]
        else:
            x_values = range(len(ac_diff_values))
            ac_diff_smooth = ac_diff_values

        ax4.plot(x_values, ac_diff_smooth, label="Actor - Critic", color="purple")

        # Set titles and labels
        if window_size > 1:
            ax1.set_title(f"Perturbed Values (smoothing={window_size})", fontsize=16)
            ax2.set_title(f"Baseline Values (smoothing={window_size})", fontsize=16)
            ax3.set_title(
                f"Differences (Perturbed - Baseline) (smoothing={window_size})",
                fontsize=16,
            )
            ax4.set_title(
                f"Actor vs Critic Difference (smoothing={window_size})", fontsize=16
            )
        else:
            ax1.set_title("Perturbed Values (raw)", fontsize=16)
            ax2.set_title("Baseline Values (raw)", fontsize=16)
            ax3.set_title("Differences (Perturbed - Baseline) (raw)", fontsize=16)
            ax4.set_title("Actor vs Critic Difference (raw)", fontsize=16)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_ylabel("Negative Log Probability", fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True)

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

            plt.plot(x_values, effect_smooth, 
                    label=f"{pert} (Actor vs Critic diff)", 
                    color=colors[i])

        plt.grid(True)
        plt.legend(fontsize=12)
        plt.ylabel("Difference in Perturbation Effect (Actor - Critic)", fontsize=14)

        if window_size > 1:
            plt.title(f"Perturbation Analysis: Actor vs Critic Effect (smoothing={window_size})")
        else:
            plt.title("Perturbation Analysis: Actor vs Critic Effect (raw)")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_multiple_perturbation_results(log_file, perturb_types, window_size=40, max_index=None):
    """
    Create a figure with subplots in a 2-column grid for multiple perturbation types.
    """
    # Calculate number of rows needed for 2 columns
    n_rows = (len(perturb_types) + 1) // 2  # Ceiling division
    n_cols = 2
    
    # Create figure with shared axes
    fig = plt.figure(figsize=(20, 6 * n_rows))
    
    # Create subplot grid with shared axes
    axes = []
    for i in range(len(perturb_types)):
        row = i // 2
        col = i % 2
        if i == 0:
            ax = plt.subplot2grid((n_rows, n_cols), (row, col))
            first_ax = ax
        else:
            # Share y-axis within row, share x-axis within column
            share_y = axes[i-2] if i >= 2 else None  # Share y with plot 2 rows up
            share_x = axes[i-1] if i % 2 == 1 else None  # Share x with previous plot if in right column
            ax = plt.subplot2grid((n_rows, n_cols), (row, col), 
                                sharex=share_x, sharey=share_y)
        axes.append(ax)

    colors = list(mcolors.TABLEAU_COLORS.values())

    for ax, perturb_type in zip(axes, perturb_types):
        try:
            results = load_perturbation_results(log_file, perturb_type)

            if max_index is not None:
                results = results[:max_index]

            # Plot differences for each perturbation type
            for i, (pert, values) in enumerate(results[0]["Avg Log Probs"].items()):
                if pert == f"{perturb_type.title().replace('_', '')}0%":
                    continue

                perturbed_values = [-entry["Avg Log Probs"][pert] for entry in results]
                baseline_values = [-entry["Avg Log Probs"][f"{perturb_type.title().replace('_', '')}0%"]
                                 for entry in results]
                diff_values = [p - o for p, o in zip(perturbed_values, baseline_values)]

                if window_size > 1 and len(diff_values) > window_size:
                    diff_smooth = savgol_filter(diff_values, window_size, 3)
                    padding = window_size // 2
                    x_values = range(padding, len(diff_values) - padding)
                    diff_smooth = diff_smooth[padding:-padding]
                else:
                    x_values = range(len(diff_values))
                    diff_smooth = diff_values

                ax.plot(x_values, diff_smooth, label=f"{pert}", color=colors[i])

            ax.grid(True)
            ax.legend(fontsize=10)
            
            # Only show y-label for leftmost plots
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("Diff in Neg Log Prob", fontsize=12)
            
            # Only show x-label for bottom plots
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("Example Index", fontsize=12)

            if window_size > 1:
                ax.set_title(f"{perturb_type.replace('_', ' ').title()} (smoothing={window_size})")
            else:
                ax.set_title(f"{perturb_type.replace('_', ' ').title()} (raw)")

        except FileNotFoundError:
            ax.text(0.5, 0.5, f"No saved results found for {perturb_type}",
                   ha="center", va="center")
            ax.set_title(f"{perturb_type.replace('_', ' ').title()}")

    plt.tight_layout()

    # Use a simple, fixed filename for combined plots
    output_file = os.path.join(os.path.dirname(log_file), "perturbation_results_combined.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {output_file}")
    plt.close()


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
                "Avg Log Probs": {},
                "Original": {"Actor": 0.0, "Critic": 0.0}
            }
            
            # Average the Original values
            for run in results_list:
                avg_entry["Original"]["Actor"] += run[entry_idx]["Original"]["Actor"] / num_runs
                avg_entry["Original"]["Critic"] += run[entry_idx]["Original"]["Critic"] / num_runs
            
            # Get perturbation names from first run
            pert_names = results_list[0][entry_idx]["Avg Log Probs"].keys()
            
            # Average the Log Probs for each perturbation level
            for pert_name in pert_names:
                avg_entry["Avg Log Probs"][pert_name] = sum(
                    run[entry_idx]["Avg Log Probs"][pert_name] 
                    for run in results_list
                ) / num_runs
            
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
            results = run_perturbations(log_file, perturb_type, stride=args.stride)
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
                    )
                except FileNotFoundError:
                    print(
                        f"No saved results found for {perturb_type}. Run without --plot_only first."
                    )


if __name__ == "__main__":
    main()
