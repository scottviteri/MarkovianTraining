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
from train import calculate_answer_log_probs, find_latest_result


def load_model_and_tokenizer(model_name):
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
    # Remove leading "Reasoning:" if present
    CoT = re.sub(r"^Reasoning:\s*", "", CoT.strip())
    perturbed_CoT = CoT

    # Randomly delete a fraction of characters
    if config.get("delete_fraction", 0) > 0:
        chars = list(perturbed_CoT)
        num_to_delete = int(len(chars) * config["delete_fraction"])
        indices_to_delete = random.sample(range(len(chars)), num_to_delete)
        chars = [char for idx, char in enumerate(chars) if idx not in indices_to_delete]
        perturbed_CoT = "".join(chars)

    # Randomly replace digits
    if config.get("digit_change_prob", 0) > 0:

        def replace_digit(match):
            if random.random() < config["digit_change_prob"]:
                return str(random.randint(0, 9))
            else:
                return match.group(0)

        perturbed_CoT = re.sub(r"\d", replace_digit, perturbed_CoT)

    # Truncate a fraction from the end
    if config.get("truncate_fraction", 0) > 0:
        truncate_length = int(len(perturbed_CoT) * (1 - config["truncate_fraction"]))
        perturbed_CoT = perturbed_CoT[:truncate_length]

    return perturbed_CoT


def run_perturbations(log_file):
    """
    Run perturbation analysis on the given log file.

    Args:
        log_file (str): Path to the log file to analyze

    Returns:
        list: Perturbation analysis results
    """
    # Load models
    mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_model, mistral_tokenizer, mistral_device = load_model_and_tokenizer(
        mistral_model_name
    )

    # Define perturbation configurations
    perturbations = {
        "Original": {},
        "DigitChange30%": {"digit_change_prob": 0.30},
        "Delete30%": {"delete_fraction": 0.30},
        "Truncate10%": {"truncate_fraction": 0.10},
    }

    perturbation_data = []

    # Process the log file to extract perturbation data
    with open(log_file, "r") as f:
        log_data = [json.loads(line) for line in f]

    # Extract hyperparameters from the first line of the log file
    hyperparameters = log_data[0]

    # Extract perturbation-related metrics
    for entry in log_data:
        # Skip entries without Action or Observation
        if "Action" not in entry or "Observation" not in entry:
            continue

        CoT = entry["Action"]
        observation = re.sub(r"^Answer:\s*", "", entry["Observation"].strip())
        question = entry.get("Question", "")

        # Prepare entry results
        entry_results = {
            "Batch Index": entry.get("Batch Index", None),
            "Avg Log Probs": {},
        }

        # Perform perturbations and calculate log probabilities
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                perturbed_CoT = re.sub(r"^Reasoning:\s*", "", CoT.strip())
            else:
                perturbed_CoT = perturb_CoT(CoT, pert_config)

            # Tokenize the reasoning
            tokenized_input = mistral_tokenizer(
                perturbed_CoT,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).input_ids.to(mistral_device)

            # Extract task_type and model_type from hyperparameters
            task_type = hyperparameters.get("task_type", "gsm8k")
            model_type = hyperparameters.get("model_type", "mistral")

            avg_log_prob, _ = calculate_answer_log_probs(
                model=mistral_model,
                tokenizer=mistral_tokenizer,
                device=mistral_device,
                questions=[question],
                reasoning_tokens=tokenized_input,
                answers=[observation],
                task_type=task_type,
                model_type=model_type,
                hyperparameters=hyperparameters,
            )
            avg_log_prob_value = avg_log_prob[0].item()

            entry_results["Avg Log Probs"][pert_name] = avg_log_prob_value

        perturbation_data.append(entry_results)

    return perturbation_data


def plot_perturbation_results(log_file, window_size=40):
    """
    Plot the results of perturbation analysis.

    Args:
        log_file (str): Path to the log file to plot
        window_size (int): Smoothing window size
    """
    # Always regenerate the analysis results
    results = run_perturbations(log_file)

    # Check if we have any data to plot
    if not results:
        print("No data found to plot.")
        return

    # Initialize the averaged data
    averaged_data = {
        pert: [] for pert in results[0]["Avg Log Probs"].keys() if pert != "Original"
    }

    # Calculate the differences from the original
    for pert in averaged_data.keys():
        values = [
            -entry["Avg Log Probs"][pert] - (-entry["Avg Log Probs"]["Original"])
            for entry in results
        ]
        averaged_data[pert] = values

    plt.figure(figsize=(12, 6))

    # Generate a color for each perturbation type
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_index = 0

    for pert, values in averaged_data.items():
        if len(values) > window_size:
            # Apply Savitzky-Golay filter for smoothing
            smoothed_values = savgol_filter(values, window_size, 3)

            # Only plot the central part not affected by edge effects
            half_window = window_size // 2
            x_values = range(half_window, len(smoothed_values) - half_window)
            y_values = smoothed_values[half_window:-half_window]

            plt.plot(
                x_values,
                y_values,
                label=pert,
                color=colors[color_index % len(colors)],
                linewidth=2,
            )
        else:
            # If we can't smooth, plot the original values
            plt.plot(
                values,
                label=pert,
                color=colors[color_index % len(colors)],
                linewidth=2,
            )

        color_index += 1

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Difference in Negated Log Probability", fontsize=16)
    plt.title(f"Perturbation Results (Smoothing Window: {window_size})", fontsize=16)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Increase font size for tick labels
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Save the plot in the same directory as the input log file
    output_file = os.path.join(
        os.path.dirname(log_file),
        f"perturbation_results_plot_smooth{window_size}.png",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Perturbation Analysis Tool")
    parser.add_argument("--log_file", help="Log file to analyze")
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size"
    )

    args = parser.parse_args()

    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_result(return_log=True)

    if not log_file:
        print("No log file found.")
        return

    print(f"Using log file: {log_file}")

    # Plot the results
    plot_perturbation_results(log_file, window_size=args.window_size)


if __name__ == "__main__":
    main()
