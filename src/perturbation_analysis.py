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


# Define perturbation configurations
perturbations = {
    "Original": {},  # No perturbation
    "DigitChange30%": {"digit_change_prob": 0.30},
    "Delete30%": {"delete_fraction": 0.30},
    "Truncate10%": {"truncate_fraction": 0.10},
}


def run_perturbations(log_file, stride=1):
    """
    Run perturbation analysis on the given log file.
    """
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
        CoT = example["Actor Reasoning"]
        answer = example["Answer"]
        question = example.get("Question", "")

        # Prepare entry results
        entry_results = {
            "Batch Index": entry.get("Batch Index", None),
            "Avg Log Probs": {},
        }

        # Perform perturbations and calculate log probabilities
        for pert_name, pert_config in perturbations.items():
            if pert_name == "Original":
                perturbed_CoT = CoT
            else:
                perturbed_CoT = perturb_CoT(CoT, pert_config)

            avg_log_prob, _ = calculate_answer_log_probs(
                frozen_model=frozen_model,
                tokenizer=tokenizer,
                device=device,
                questions=[question],
                reasoning=[perturbed_CoT],
                answers=[answer],
                hyperparameters=hyperparameters,
            )
            avg_log_prob_value = avg_log_prob[0].item()

            entry_results["Avg Log Probs"][pert_name] = avg_log_prob_value

        perturbation_data.append(entry_results)

    return perturbation_data


def plot_perturbation_results(
    results, log_file, window_size=40, debug=False, max_index=None
):
    """
    Plot the results of perturbation analysis.
    No smoothing is applied if window_size=1.
    """
    if not results:
        print("No data found to plot.")
        return

    # Threshold results by array index
    if max_index is not None:
        results = results[:max_index]
        if not results:
            print(f"No data found within max_index {max_index}")
            return

    colors = list(mcolors.TABLEAU_COLORS.values())

    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

        for i, (pert, values) in enumerate(results[0]["Avg Log Probs"].items()):
            if pert == "Original":
                continue
            perturbed_values = [-entry["Avg Log Probs"][pert] for entry in results]
            original_values = [-entry["Avg Log Probs"]["Original"] for entry in results]
            diff_values = [p - o for p, o in zip(perturbed_values, original_values)]

            if window_size > 1 and len(perturbed_values) > window_size:
                perturbed_smooth = savgol_filter(perturbed_values, window_size, 3)
                original_smooth = savgol_filter(original_values, window_size, 3)
                diff_smooth = savgol_filter(diff_values, window_size, 3)

                # Only plot valid indices (excluding edges affected by smoothing)
                padding = window_size // 2
                x_values = range(padding, len(perturbed_values) - padding)
                perturbed_smooth = perturbed_smooth[padding:-padding]
                original_smooth = original_smooth[padding:-padding]
                diff_smooth = diff_smooth[padding:-padding]
            else:
                x_values = range(len(perturbed_values))
                perturbed_smooth = perturbed_values
                original_smooth = original_values
                diff_smooth = diff_values

            ax1.plot(x_values, perturbed_smooth, label=f"{pert}", color=colors[i])
            ax2.plot(x_values, original_smooth, label=f"Original", color=colors[i])
            ax3.plot(x_values, diff_smooth, label=f"{pert} diff", color=colors[i])

        ax1.set_title(f"Perturbed Values (smoothing={window_size})", fontsize=16)
        ax1.set_ylabel("Negative Log Probability", fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True)

        ax2.set_title(f"Original Values (smoothing={window_size})", fontsize=16)
        ax2.set_ylabel("Negative Log Probability", fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)

        ax3.set_title(
            f"Differences (Perturbed - Original) (smoothing={window_size})", fontsize=16
        )
        ax3.set_ylabel("Difference in Negative Log Probability", fontsize=14)
        ax3.legend(fontsize=12)
        ax3.grid(True)

        plt.tight_layout()
        output_file = os.path.join(
            os.path.dirname(log_file), f"perturbation_results_debug.png"
        )
    else:
        # Original plotting code here
        plt.figure(figsize=(12, 6))
        # ... rest of original plotting code ...
        output_file = os.path.join(
            os.path.dirname(log_file), "perturbation_results_plot.png"
        )

        # Add smoothing info to title if smoothing was applied
        if window_size > 1:
            plt.title(f"Perturbation Analysis Results (smoothing={window_size})")
        else:
            plt.title("Perturbation Analysis Results (raw)")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def save_perturbation_results(results, log_file):
    """Save perturbation results to a JSON file."""
    output_file = os.path.join(os.path.dirname(log_file), "perturbation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")


def load_perturbation_results(log_file):
    """Load perturbation results from a JSON file."""
    input_file = os.path.join(os.path.dirname(log_file), "perturbation_results.json")
    with open(input_file, "r") as f:
        return json.load(f)


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
    parser.add_argument("--max_index", type=int, help="Maximum batch index to plot")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from saved results",
    )
    parser.add_argument(
        "--process_only", action="store_true", help="Only process data without plotting"
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

    # Process data if needed
    if not args.plot_only:
        results = run_perturbations(log_file, stride=args.stride)
        if results:
            save_perturbation_results(results, log_file)
        else:
            print("No data found to save.")
            return

    # Plot if needed
    if not args.process_only:
        try:
            results = load_perturbation_results(log_file)
            plot_perturbation_results(
                results,
                log_file,
                window_size=args.window_size,
                debug=args.debug,
                max_index=args.max_index,
            )
        except FileNotFoundError:
            print("No saved results found. Run without --plot_only first.")


if __name__ == "__main__":
    main()
