import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from train import (
    find_latest_result,
    calculate_answer_log_probs,
    print_debug_info,
    construct_prompts,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_evaluation_model(model_type="mistral"):
    """Load a frozen model for evaluation."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError("model_type must be either 'mistral' or 'llama'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def run_cross_model_evaluation(log_files, stride=1, debug_freq=100, max_index=None):
    """
    Evaluate log files using the opposite model as evaluator.
    
    Args:
        log_files (list): List of log file paths to evaluate
        stride (int): Process every nth entry
        debug_freq (int): How often to print debug info
        max_index (int): If provided, only process entries with batch_index <= max_index
    """
    results = {
        "files": log_files,
        "evaluations": [],
        "generator_model": None,
        "evaluator_model": None,
    }

    for file in log_files:
        file_results = []
        with open(file, "r") as f:
            lines = [json.loads(line) for line in f]
            hyperparameters = lines[0].copy()

            # Store model types
            results["generator_model"] = hyperparameters["model_type"]
            results["evaluator_model"] = (
                "mistral" if hyperparameters["model_type"] == "llama" else "llama"
            )

            # Update hyperparameters for the evaluator model
            hyperparameters["model_type"] = results["evaluator_model"]
            if results["evaluator_model"] == "mistral":
                hyperparameters["model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
            else:
                hyperparameters["model_name"] = "meta-llama/Llama-3.1-8B-Instruct"

            # Load the evaluation model
            frozen_model, tokenizer, device = load_evaluation_model(
                model_type=results["evaluator_model"]
            )

            # Filter entries by batch index if max_index is provided
            entries = [entry for entry in lines[1:] if "Example" in entry]
            if max_index is not None:
                entries = [entry for entry in entries if entry.get("Batch Index", float('inf')) <= max_index]
                print(f"Processing entries up to batch index {max_index}")
            
            # Apply stride
            entries = entries[::stride]
            pbar = tqdm(entries, desc="Processing examples")

            # Process each example
            for i, entry in enumerate(pbar):
                example = entry["Example"]
                question = example["Question"]
                actor_reasoning = example["Actor Reasoning"]
                critic_reasoning = example["Critic Reasoning"]
                answer = example["Answer"]

                # Calculate log probabilities for both reasonings using eval model's hyperparameters
                actor_log_probs, _ = calculate_answer_log_probs(
                    frozen_model,
                    tokenizer,
                    device,
                    [question],
                    [actor_reasoning],
                    [answer],
                    hyperparameters,
                )

                critic_log_probs, _ = calculate_answer_log_probs(
                    frozen_model,
                    tokenizer,
                    device,
                    [question],
                    [critic_reasoning],
                    [answer],
                    hyperparameters,
                )

                # Print debug info periodically
                if i % debug_freq == 0:
                    print("\nDebug Info for Actor Reasoning:")
                    print_debug_info(
                        hyperparameters["task_type"],
                        question,
                        actor_reasoning,
                        answer,
                        actor_log_probs.mean().item(),
                    )
                    print("\nDebug Info for Critic Reasoning:")
                    print_debug_info(
                        hyperparameters["task_type"],
                        question,
                        critic_reasoning,
                        answer,
                        critic_log_probs.mean().item(),
                    )

                # Store results including original normalized reward
                result = {
                    "Batch Index": entry["Batch Index"],
                    "Avg Log Probs": {
                        "Actor": actor_log_probs.mean().item(),
                        "Critic": critic_log_probs.mean().item(),
                    },
                    "Original Reward": entry["Training Metrics"]["Normalized Reward"],
                    "Example": example,
                    "Metrics": entry.get("Training Metrics", {}),
                }
                file_results.append(result)

                # Update progress bar description with current scores
                pbar.set_description(
                    f"Actor: {actor_log_probs.mean().item():.3f}, "
                    f"Critic: {critic_log_probs.mean().item():.3f}"
                )

        results["evaluations"].append(file_results)

    return results


def plot_cross_model_comparison(results, log_file, window_size=40, max_index=None):
    """
    Plot Cross-Model Actor-Critic Difference alongside Original Normalized Reward.
    """
    all_data = results["evaluations"]
    
    # Apply max_index to limit plotting range
    if max_index is not None:
        all_data = [data[:max_index] for data in all_data]
        print(f"Plotting up to index {max_index}")
    
    min_length = min(len(data) for data in all_data)

    # Calculate Actor-Critic differences (cross-model normalized reward)
    cross_model_rewards = []
    original_rewards = []
    for i in range(min_length):
        # Cross-model rewards
        diffs = [
            data[i]["Avg Log Probs"]["Actor"] - data[i]["Avg Log Probs"]["Critic"]
            for data in all_data
        ]
        cross_model_rewards.append(np.mean(diffs))
        
        # Original rewards
        rewards = [data[i]["Original Reward"] for data in all_data]
        original_rewards.append(np.mean(rewards))

    plt.figure(figsize=(12, 6))
    
    # Plot both metrics with smoothing if enough data points
    if len(cross_model_rewards) > window_size:
        # Cross-model rewards
        smoothed_cross = savgol_filter(cross_model_rewards, window_size, 3)
        half_window = window_size // 2
        x_values = range(half_window, len(smoothed_cross) - half_window)
        y_values = smoothed_cross[half_window:-half_window]
        plt.plot(x_values, y_values, 
                label=f"{results['evaluator_model'].title()} Normalized Reward", 
                color="#e41a1c", 
                linewidth=2)
        
        # Original rewards
        smoothed_orig = savgol_filter(original_rewards, window_size, 3)
        y_values_orig = smoothed_orig[half_window:-half_window]
        plt.plot(x_values, y_values_orig, 
                label=f"{results['generator_model'].title()} Normalized Reward", 
                color="#377eb8", 
                linewidth=2)
    else:
        plt.plot(cross_model_rewards, 
                label=f"{results['evaluator_model'].title()} Normalized Reward", 
                color="#e41a1c", 
                linewidth=2)
        plt.plot(original_rewards, 
                label=f"{results['generator_model'].title()} Normalized Reward", 
                color="#377eb8", 
                linewidth=2)

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Normalized Reward", fontsize=16)
    plt.title(
        f"{results['generator_model'].title()} Generated, {results['evaluator_model'].title()} Evaluated\n"
        f"Normalized Rewards Comparison (Smoothing: {window_size})",
        fontsize=16,
    )
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()

    output_file = os.path.join(os.path.dirname(log_file), "cross_model_evaluation.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def save_evaluation_results(results, log_file):
    """Save evaluation results to a new jsonl file."""
    output_file = os.path.join(
        os.path.dirname(log_file), "cross_model_evaluation_results.jsonl"
    )

    with open(output_file, "w") as f:
        # Write metadata as first line
        json.dump(
            {
                "files": results["files"],
                "generator_model": results["generator_model"],
                "evaluator_model": results["evaluator_model"],
            },
            f,
        )
        f.write("\n")

        # Write evaluation results
        for eval_set in results["evaluations"]:
            for entry in eval_set:
                json.dump(entry, f)
                f.write("\n")

    print(f"Evaluation results saved to {output_file}")
    return output_file


def load_evaluation_results(results_file):
    """Load evaluation results from jsonl file."""
    results = {"evaluations": [[]]}  # Initialize with single evaluation set

    with open(results_file, "r") as f:
        # First line contains metadata
        metadata = json.loads(f.readline())
        results["files"] = metadata["files"]
        results["generator_model"] = metadata["generator_model"]
        results["evaluator_model"] = metadata["evaluator_model"]

        # Read evaluation results
        for line in f:
            results["evaluations"][0].append(json.loads(line))

    return results


def collate_cross_model_results(log_files, output_dir):
    """
    Average cross-model evaluation results across multiple runs and save to a new directory.
    
    Args:
        log_files (list): List of paths to cross-model evaluation result files
        output_dir (str): Directory to save collated results
    """
    os.makedirs(output_dir, exist_ok=True)
    accumulated_results = {
        'results': [],
        'count': 0,
        'metadata': None
    }
    
    # Process each evaluation results file
    for log_file in log_files:
        try:
            results = load_evaluation_results(log_file)
            
            # Store metadata from first file
            if accumulated_results['metadata'] is None:
                accumulated_results['metadata'] = {
                    'files': results['files'],
                    'generator_model': results['generator_model'],
                    'evaluator_model': results['evaluator_model']
                }
            else:
                # Verify consistency across files
                if (results['generator_model'] != accumulated_results['metadata']['generator_model'] or
                    results['evaluator_model'] != accumulated_results['metadata']['evaluator_model']):
                    print(f"Warning: Inconsistent models in {log_file}, skipping")
                    continue
            
            accumulated_results['results'].append(results['evaluations'][0])  # Assuming single evaluation set
            accumulated_results['count'] += 1
            
        except FileNotFoundError:
            print(f"Warning: No results found in {log_file}")
            continue
    
    if accumulated_results['count'] == 0:
        print("No valid results found to collate")
        return
    
    # Find minimum length across all runs
    min_length = min(len(run) for run in accumulated_results['results'])
    print(f"Using {min_length} entries (shortest common length)")
    
    # Initialize structure for averaged results
    averaged_results = []
    for entry_idx in range(min_length):
        avg_entry = {
            "Batch Index": accumulated_results['results'][0][entry_idx]["Batch Index"],
            "Avg Log Probs": {
                "Actor": 0.0,
                "Critic": 0.0
            },
            "Original Reward": 0.0,  # Initialize Original Reward
            "Example": accumulated_results['results'][0][entry_idx]["Example"],
            "Metrics": accumulated_results['results'][0][entry_idx].get("Metrics", {})
        }
        
        # Average the log probabilities and original reward
        num_runs = accumulated_results['count']
        for run in accumulated_results['results']:
            avg_entry["Avg Log Probs"]["Actor"] += run[entry_idx]["Avg Log Probs"]["Actor"] / num_runs
            avg_entry["Avg Log Probs"]["Critic"] += run[entry_idx]["Avg Log Probs"]["Critic"] / num_runs
            if "Original Reward" in run[entry_idx]:
                avg_entry["Original Reward"] += run[entry_idx]["Original Reward"] / num_runs
        
        averaged_results.append(avg_entry)
    
    # Save averaged results
    output_file = os.path.join(output_dir, "cross_model_evaluation_results.jsonl")
    with open(output_file, "w") as f:
        # Write metadata as first line
        json.dump(accumulated_results['metadata'], f)
        f.write("\n")
        
        # Write averaged results
        for entry in averaged_results:
            json.dump(entry, f)
            f.write("\n")
    
    print(f"Averaged results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Evaluation Tool")
    parser.add_argument("--log_file", help="Log file to evaluate")
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size"
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every nth entry")
    parser.add_argument(
        "--debug_freq",
        type=int,
        default=100,
        help="Print debug info every n processed entries",
    )
    parser.add_argument(
        "--process_only", action="store_true", help="Only process data without plotting"
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from saved results",
    )
    parser.add_argument(
        "--collate",
        nargs="+",
        help="List of evaluation result files to average"
    )
    parser.add_argument(
        "--output_dir",
        default="cross_model_results",
        help="Output directory for collated results"
    )
    parser.add_argument(
        "--max_index",
        type=int,
        help="Maximum index to process (batch index for --process_only, array index for --plot_only)"
    )
    
    args = parser.parse_args()

    if args.collate:
        print(f"Collating results from {len(args.collate)} runs...")
        collate_cross_model_results(args.collate, args.output_dir)
        print(f"Collation complete. Results saved to {args.output_dir}")
        if not args.plot_only:
            return
        # Update log_file to point to collated results for plotting
        log_file = os.path.join(args.output_dir, "cross_model_evaluation_results.jsonl")
    else:
        if args.log_file:
            log_file = args.log_file
        else:
            log_file = find_latest_result(return_log=True)

    if not log_file:
        print("No log file found.")
        return

    print(f"Using log file: {log_file}")

    # Determine evaluation results file path
    eval_results_file = os.path.join(
        os.path.dirname(log_file), "cross_model_evaluation_results.jsonl"
    )

    # Process data if needed
    if not args.plot_only:
        print(f"Processing every {args.stride}th entry")
        results = run_cross_model_evaluation(
            [log_file], 
            stride=args.stride, 
            debug_freq=args.debug_freq,
            max_index=args.max_index  # Pass max_index for batch index filtering
        )
        save_evaluation_results(results, log_file)

    # Plot if needed
    if not args.process_only:
        try:
            results = load_evaluation_results(eval_results_file)
            plot_cross_model_comparison(
                results, 
                log_file, 
                window_size=args.window_size,
                max_index=args.max_index  # Pass max_index for plot range limiting
            )
        except FileNotFoundError:
            print(f"No saved results found at {eval_results_file}. Run without --plot_only first.")


if __name__ == "__main__":
    main()
