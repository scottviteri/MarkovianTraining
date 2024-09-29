import json
import re
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from policy_gradient_normalized import calculate_answer_log_probs


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

    # Randomly replace digits
    if config.get("digit_change_prob", 0) > 0:

        def replace_digit(match):
            if random.random() < config["digit_change_prob"]:
                return str(random.randint(0, 9))
            else:
                return match.group(0)

        perturbed_CoT = re.sub(r"\d", replace_digit, perturbed_CoT)

    # Introduce padding tokens (e.g., extra spaces)
    if config.get("pad_fraction", 0) > 0:
        chars = list(perturbed_CoT)
        num_to_pad = int(len(chars) * config["pad_fraction"])
        indices_to_pad = random.sample(range(len(chars)), num_to_pad)
        for idx in indices_to_pad:
            chars[idx] = chars[idx] + " "  # Add extra space
        perturbed_CoT = "".join(chars)

    return perturbed_CoT


def main():
    parser = argparse.ArgumentParser(
        description="Perturb Chain-of-Thought and calculate losses"
    )
    parser.add_argument(
        "log_file", help="Path to the log file within ./results/9-28-24/"
    )
    args = parser.parse_args()

    log_file = os.path.join("./results/9-28-24", args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    # Define perturbation configurations
    perturbations = {
        "Original": {},
        "Delete30%Chars": {"delete_fraction": 0.3},
        "DigitChange10%": {"digit_change_prob": 0.1},
        "Pad20%Chars": {"pad_fraction": 0.2},
    }

    results = []

    with open(log_file, "r") as f:
        for line in tqdm(f):
            entry = json.loads(line)
            if "Action" not in entry or "Observation" not in entry:
                continue
            CoT = entry["Action"]
            observation = entry["Observation"]
            question = entry.get("Prev Observation", "")
            combined_input = question + "\n" + CoT

            entry_results = {
                "Batch Index": entry.get("Batch Index", None),
                "Avg Log Probs": {},
            }

            for pert_name, pert_config in perturbations.items():
                if pert_name == "Original":
                    perturbed_CoT = CoT
                else:
                    perturbed_CoT = perturb_CoT(CoT, pert_config)
                perturbed_input = question + "\n" + perturbed_CoT
                avg_log_prob = calculate_answer_log_probs(
                    model,
                    tokenizer,
                    device,
                    perturbed_input,
                    observation,
                    use_gsm8k=False,
                )
                entry_results["Avg Log Probs"][pert_name] = avg_log_prob

            results.append(entry_results)

    # Save results to a JSON file
    output_file = os.path.join("./results/9-28-24", "perturbation_results.json")
    with open(output_file, "w") as f_out:
        json.dump(results, f_out, indent=2)


if __name__ == "__main__":
    main()
