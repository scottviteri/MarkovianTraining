import json
import re
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from policy_gradient_normalized import calculate_answer_log_probs
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors
import warnings


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

    return perturbed_CoT


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def process_file(
    log_file,
    perturbations,
    mistral_model,
    mistral_tokenizer,
    mistral_device,
    llama_model,
    llama_tokenizer,
    llama_device,
):
    results = []
    with open(log_file, "r") as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(log_file)}"):
            entry = json.loads(line)
            if "Action" not in entry or "Observation" not in entry:
                continue
            CoT = entry["Action"]
            observation = entry["Observation"]

            observation_clean = re.sub(r"^Answer:\s*", "", observation.strip())

            entry_results = {
                "Batch Index": entry.get("Batch Index", None),
                "Avg Log Probs": {},
            }

            for pert_name, pert_config in perturbations.items():
                if pert_name == "Original":
                    perturbed_CoT = re.sub(r"^Reasoning:\s*", "", CoT.strip())
                else:
                    perturbed_CoT = perturb_CoT(CoT, pert_config)

                tokenized_input = mistral_tokenizer(
                    perturbed_CoT,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=False,
                ).input_ids.to(mistral_device)

                avg_log_prob = calculate_answer_log_probs(
                    mistral_model,
                    mistral_tokenizer,
                    mistral_device,
                    tokenized_input,
                    [observation_clean],
                    use_gsm8k=False,
                )
                avg_log_prob_value = avg_log_prob[0].item()
                entry_results["Avg Log Probs"][pert_name] = avg_log_prob_value

            # Calculate log probabilities for the Llama 7B model
            tokenized_input = llama_tokenizer(
                re.sub(r"^Reasoning:\s*", "", CoT.strip()),
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).input_ids.to(llama_device)

            avg_log_prob = calculate_answer_log_probs(
                llama_model,
                llama_tokenizer,
                llama_device,
                tokenized_input,
                [observation_clean],
                use_gsm8k=False,
            )
            avg_log_prob_value = avg_log_prob[0].item()
            entry_results["Avg Log Probs"]["Llama"] = avg_log_prob_value

            results.append(entry_results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Perturb Chain-of-Thought and calculate losses"
    )
    parser.add_argument(
        "--log_files",
        nargs="+",
        help="Paths to the log files within the results subfolder",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot results from existing JSON files"
    )
    parser.add_argument(
        "--plot_llama",
        action="store_true",
        help="Plot average original run versus average Llama run",
    )
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size for plotting"
    )
    parser.add_argument(
        "--results_subfolder",
        default="9-28-24",
        help="Subfolder within ./results/ to use (default: 9-28-24)",
    )
    args = parser.parse_args()

    results_path = os.path.join("./results", args.results_subfolder)

    if args.plot:
        plot_results(args.log_files, args.window_size, results_path)
    elif args.plot_llama:
        plot_original_vs_llama(args.log_files, args.window_size, results_path)
    else:
        # Define perturbation configurations
        perturbations = {
            "Original": {},
            "DigitChange15%": {"digit_change_prob": 0.15},
            "DigitChange30%": {"digit_change_prob": 0.30},
            "Delete15%": {"delete_fraction": 0.15},
            "Delete30%": {"delete_fraction": 0.30},
        }

        # Load Mistral model and tokenizer
        mistral_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        mistral_model, mistral_tokenizer, mistral_device = load_model_and_tokenizer(
            mistral_model_name
        )

        # Load Llama 7B model and tokenizer
        llama_model_name = "meta-llama/Llama-2-7b-hf"
        llama_model, llama_tokenizer, llama_device = load_model_and_tokenizer(
            llama_model_name
        )

        for log_file in args.log_files:
            input_path = os.path.join(results_path, log_file)
            results = process_file(
                input_path,
                perturbations,
                mistral_model,
                mistral_tokenizer,
                mistral_device,
                llama_model,
                llama_tokenizer,
                llama_device,
            )

            # Save results to a JSON file, overwriting if it exists
            output_file = os.path.join(
                results_path, f"analysis_results_{os.path.basename(log_file)}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(results, f_out, indent=2)
            print(f"Results saved to {output_file}")


def plot_original_vs_llama(log_files, window_size=40, results_path="./results/9-28-24"):
    all_data = []
    for log_file in log_files:
        result_file = os.path.join(
            results_path, f"analysis_results_{os.path.basename(log_file)}.json"
        )
        with open(result_file, "r") as f:
            all_data.append(json.load(f))

    # Find the minimum length among all datasets
    min_length = min(len(data) for data in all_data)

    # Initialize the averaged data
    averaged_data = {"Original": [], "Llama": []}

    # Calculate the average across all datasets
    for i in range(min_length):
        original_values = [data[i]["Avg Log Probs"]["Original"] for data in all_data]
        llama_values = [data[i]["Avg Log Probs"]["Llama"] for data in all_data]
        averaged_data["Original"].append(np.mean(original_values))
        averaged_data["Llama"].append(np.mean(llama_values))

    plt.figure(figsize=(12, 6))
    colors = ["#e41a1c", "#377eb8"]

    for model, values in averaged_data.items():
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
                label=model,
                color=colors[0] if model == "Original" else colors[1],
                linewidth=2,
            )
        else:
            # If we can't smooth, plot the original values
            plt.plot(
                values,
                label=model,
                color=colors[0] if model == "Original" else colors[1],
                linewidth=2,
            )

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Average Log Probability", fontsize=16)
    plt.title(
        f"Average Original vs Llama Results (Smoothing Window: {window_size})",
        fontsize=16,
    )
    plt.legend(fontsize=20, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Increase font size for tick labels
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = os.path.join(
        results_path, f"average_original_vs_llama_plot_smooth{window_size}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def plot_results(log_files, window_size=40, results_path="./results/9-28-24"):
    all_data = []
    for log_file in log_files:
        result_file = os.path.join(
            results_path, f"analysis_results_{os.path.basename(log_file)}.json"
        )
        with open(result_file, "r") as f:
            all_data.append(json.load(f))

    # Find the minimum length among all datasets
    min_length = min(len(data) for data in all_data)

    # Initialize the averaged data
    averaged_data = {
        pert: [] for pert in all_data[0][0]["Avg Log Probs"].keys() if pert != "Llama"
    }

    # Calculate the average across all datasets
    for i in range(min_length):
        original_values = [-data[i]["Avg Log Probs"]["Original"] for data in all_data]
        avg_original = np.mean(original_values)

        for pert in averaged_data.keys():
            values = [
                -data[i]["Avg Log Probs"][pert]
                - (-data[i]["Avg Log Probs"]["Original"])
                for data in all_data
            ]
            averaged_data[pert].append(np.mean(values))

    plt.figure(figsize=(12, 6))

    # Generate a color for each perturbation type
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_index = 0

    for pert, values in averaged_data.items():
        if pert != "Original":  # Skip plotting the Original, as it will always be 0
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
    plt.ylabel("Average Difference in Negated Log Probability", fontsize=16)
    plt.title(
        f"Average Perturbation Results (Smoothing Window: {window_size})", fontsize=16
    )
    plt.legend(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Increase font size for tick labels
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = os.path.join(
        results_path, f"average_perturbation_results_plot_smooth{window_size}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
