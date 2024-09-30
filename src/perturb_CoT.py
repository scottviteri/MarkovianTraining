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


def main():
    parser = argparse.ArgumentParser(
        description="Perturb Chain-of-Thought and calculate losses"
    )
    parser.add_argument(
        "--log_file", help="Path to the log file within ./results/9-28-24/"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot results from existing JSON file"
    )
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size for plotting"
    )
    args = parser.parse_args()

    log_file = os.path.join("./results/9-28-24", args.log_file)

    if args.plot:
        plot_results(log_file, args.window_size)
    else:
        # Load tokenizer and model
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Define perturbation configurations
        perturbations = {
            "Original": {},
            "DigitChange15%": {"digit_change_prob": 0.15},
            "DigitChange30%": {"digit_change_prob": 0.30},
            "Delete15%": {"delete_fraction": 0.15},
            "Delete30%": {"delete_fraction": 0.30},
        }

        results = []
        total_log_probs = {pert_name: 0.0 for pert_name in perturbations}
        count = 0

        with open(log_file, "r") as f:
            for line in tqdm(f):
                entry = json.loads(line)
                if "Action" not in entry or "Observation" not in entry:
                    continue
                CoT = entry["Action"]
                observation = entry["Observation"]

                # Remove leading "Answer:" from observation if present
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

                    tokenized_input = tokenizer(
                        perturbed_CoT,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                        add_special_tokens=False,
                    ).input_ids.to(device)

                    avg_log_prob = calculate_answer_log_probs(
                        model,
                        tokenizer,
                        device,
                        tokenized_input,
                        [observation_clean],  # Pass a list with a single answer string
                        use_gsm8k=False,
                    )
                    avg_log_prob_value = avg_log_prob[
                        0
                    ].item()  # Convert to Python float
                    entry_results["Avg Log Probs"][pert_name] = avg_log_prob_value
                    total_log_probs[pert_name] += avg_log_prob_value

                    # Compare with the original "Avg Log Prob"
                    if pert_name == "Original":
                        original_avg_log_prob = entry["Avg Log Prob"]

                        # Check if it's a list or array
                        if isinstance(original_avg_log_prob, (list, np.ndarray)):
                            original_avg_log_prob = original_avg_log_prob[0]

                        # if not np.isclose(
                        #    avg_log_prob_value, original_avg_log_prob, atol=0.2, rtol=0
                        # ):
                        #    warnings.warn(
                        #        f"Mismatch in log probabilities: Original={avg_log_prob_value}, Avg Log Prob={original_avg_log_prob}"
                        #    )
                        #    print(f"Full entry: {entry}")
                        #    print(f"Calculated avg_log_prob: {avg_log_prob}")

                results.append(entry_results)
                count += 1

        # Save results to a JSON file, overwriting if it exists
        output_file = os.path.join("./results/9-28-24", "perturbation_results.json")
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(results, f_out, indent=2)


def plot_results(log_file, window_size=40):
    with open(log_file, "r") as f:
        results = json.load(f)

    perturbations = list(results[0]["Avg Log Probs"].keys())
    data = {pert: [] for pert in perturbations}

    for entry in results:
        original_value = -entry["Avg Log Probs"]["Original"]
        for pert, value in entry["Avg Log Probs"].items():
            # Negate the log prob and subtract the original value
            adjusted_value = -value - original_value
            data[pert].append(adjusted_value)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(perturbations) - 1))
    color_index = 0

    for pert, values in data.items():
        if pert != "Original":  # Skip plotting the Original, as it will always be 0
            if len(values) > window_size:
                # Apply Savitzky-Golay filter for smoothing
                smoothed_values = savgol_filter(values, window_size, 3)

                # Only plot the central part not affected by edge effects
                half_window = window_size // 2
                x_values = range(half_window, len(smoothed_values) - half_window)
                y_values = smoothed_values[half_window:-half_window]

                plt.plot(x_values, y_values, label=pert, color=colors[color_index])
            else:
                # If we can't smooth, plot the original values
                plt.plot(values, label=pert, color=colors[color_index])

            color_index += 1

    plt.xlabel("Sample")
    plt.ylabel("Difference in Negated Log Probability")
    plt.title(f"Perturbation Results (Smoothing Window: {window_size})")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.splitext(log_file)[0] + f"_plot_smooth{window_size}.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    main()
