import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from train import calculate_answer_log_probs
from utils import print_debug_info, find_latest_result
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_evaluation_model(model_type="mistral"):
    """Load a frozen model for evaluation."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_type == "gpt2":
        model_name = "openai-community/gpt2"
    elif model_type == "tinystories":
        model_name = "roneneldan/TinyStories"
    else:
        raise ValueError("model_type must be either 'mistral', 'llama', 'gpt2', or 'tinystories'")

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


def run_cross_model_evaluation(log_files, stride=1, debug_freq=100, max_index=None, critic_model_type=None):
    """
    Evaluate log files using the specified critic model.
    
    Args:
        log_files (list): List of log file paths to evaluate
        stride (int): Process every nth entry
        debug_freq (int): How often to print debug info
        max_index (int): If provided, only process entries with batch_index <= max_index
        critic_model_type (str): Model type for evaluation
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
            
            # Use the specified critic model type
            if critic_model_type:
                results["evaluator_model"] = critic_model_type
            else:
                raise ValueError("A critic model type must be specified.")

            # Update hyperparameters for the evaluator model
            hyperparameters["model_type"] = results["evaluator_model"]
            if results["evaluator_model"] == "mistral":
                hyperparameters["model_name"] = "mistralai/Mistral-7B-Instruct-v0.2"
            elif results["evaluator_model"] == "llama":
                hyperparameters["model_name"] = "meta-llama/Llama-3.1-8B-Instruct"
            elif results["evaluator_model"] == "gpt2":
                hyperparameters["model_name"] = "openai-community/gpt2"
            elif results["evaluator_model"] == "tinystories":
                hyperparameters["model_name"] = "roneneldan/TinyStories"
            else:
                raise ValueError("Unsupported model type")

            # Load the evaluation model
            frozen_model, tokenizer, device = load_evaluation_model(
                model_type=results["evaluator_model"]
            )

            # Filter entries and validate required fields
            entries = []
            for entry in lines[1:]:
                if ("Example" in entry and 
                    "Training Metrics" in entry and 
                    "Normalized Reward" in entry["Training Metrics"] and
                    "Batch Index" in entry):
                    entries.append(entry)

            print(f"Found {len(entries)} valid entries")

            if max_index is not None:
                entries = [entry for entry in entries if entry["Batch Index"] <= max_index]
                print(f"Processing entries up to batch index {max_index}")

            # Apply stride
            entries = entries[::stride]
            print(f"Processing {len(entries)} entries after applying stride {stride}")

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


def plot_cross_model_comparison(results, log_file, window_size=40, max_index=None, show_log_probs=False):
    """
    Plot evaluation results.
    
    Args:
        results: Evaluation results dictionary
        log_file: Path to log file
        window_size: Window size for smoothing
        max_index: Maximum index to plot
        show_log_probs: Whether to show actor/critic log probabilities
    """
    all_data = results["evaluations"]
    
    if max_index is not None:
        all_data = [data[:max_index] for data in all_data]
        print(f"Plotting up to index {max_index}")
    
    min_length = min(len(data) for data in all_data)

    # Initialize arrays for plotting
    actor_values = []
    critic_values = []
    computed_rewards = []  # Difference between actor and critic log probs
    original_rewards = []  # Original normalized rewards
    
    for i in range(min_length):
        # Store absolute log probs
        actor_probs = [data[i]["Avg Log Probs"]["Actor"] for data in all_data]
        critic_probs = [data[i]["Avg Log Probs"]["Critic"] for data in all_data]
        actor_values.append(np.mean(actor_probs))
        critic_values.append(np.mean(critic_probs))
        
        # Compute reward as difference between actor and critic
        diffs = [
            data[i]["Avg Log Probs"]["Actor"] - data[i]["Avg Log Probs"]["Critic"]
            for data in all_data
        ]
        computed_rewards.append(np.mean(diffs))
        
        # Store original rewards
        rewards = [data[i]["Original Reward"] for data in all_data]
        original_rewards.append(np.mean(rewards))

    plt.figure(figsize=(12, 6))
    
    if len(actor_values) > window_size:
        half_window = window_size // 2
        x_values = range(half_window, len(actor_values) - half_window)
        
        if show_log_probs:
            # Plot absolute log probs
            smoothed_actor = savgol_filter(actor_values, window_size, 3)
            smoothed_critic = savgol_filter(critic_values, window_size, 3)
            plt.plot(x_values, smoothed_actor[half_window:-half_window], 
                    label="Actor Log Probs", color="#e41a1c", linewidth=2)
            plt.plot(x_values, smoothed_critic[half_window:-half_window], 
                    label="Critic Log Probs", color="#377eb8", linewidth=2)
        
        # Always plot both rewards
        smoothed_computed = savgol_filter(computed_rewards, window_size, 3)
        smoothed_orig = savgol_filter(original_rewards, window_size, 3)
        
        plt.plot(x_values, smoothed_computed[half_window:-half_window], 
                label=f"{results['evaluator_model'].title()} Computed Reward", 
                color="#4daf4a", linewidth=2)
        plt.plot(x_values, smoothed_orig[half_window:-half_window], 
                label=f"{results['generator_model'].title()} Original Reward", 
                color="#984ea3", linewidth=2)
    else:
        if show_log_probs:
            plt.plot(actor_values, label="Actor Log Probs", color="#e41a1c", linewidth=2)
            plt.plot(critic_values, label="Critic Log Probs", color="#377eb8", linewidth=2)
        
        plt.plot(computed_rewards, 
                label=f"{results['evaluator_model'].title()} Computed Reward", 
                color="#4daf4a", linewidth=2)
        plt.plot(original_rewards, 
                label=f"{results['generator_model'].title()} Original Reward", 
                color="#984ea3", linewidth=2)

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Log Probability / Reward", fontsize=16)
    plt.title(
        f"{results['generator_model'].title()} Generated, {results['evaluator_model'].title()} Evaluated\n"
        f"{'Log Probabilities and ' if show_log_probs else ''}Rewards Comparison (Smoothing: {window_size})",
        fontsize=16,
    )
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()

    # Save plots with evaluator model type in filename
    plot_name = f"evaluation_results_{results['evaluator_model']}.png"
    output_file = os.path.join(os.path.dirname(log_file), plot_name)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def save_evaluation_results(results, log_file):
    """Save evaluation results to a new jsonl file."""
    # Include critic model type in filename
    filename = f"evaluation_results_{results['evaluator_model']}.jsonl"
    output_file = os.path.join(os.path.dirname(log_file), filename)

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


def collate_cross_model_results(paths, output_dir):
    """
    Collate results from multiple evaluation runs.
    
    Args:
        paths: List of paths (can be files or directories)
        output_dir: Directory to save collated results
    """
    accumulated_results = {
        'metadata': None,
        'results': [],
        'count': 0
    }
    
    for path in paths:
        if os.path.isdir(path):
            # If directory, look for evaluation results file
            eval_file = None
            for model_type in ["mistral", "llama", "gpt2", "tinystories"]:
                potential_file = os.path.join(path, f"evaluation_results_{model_type}.jsonl")
                if os.path.exists(potential_file):
                    eval_file = potential_file
                    break
            if eval_file is None:
                print(f"No evaluation results found in directory: {path}")
                continue
        else:
            # If file path, use directly
            eval_file = path
            
        if not os.path.exists(eval_file):
            print(f"File not found: {eval_file}")
            continue
            
        print(f"Processing: {eval_file}")
        
        try:
            with open(eval_file, 'r') as f:
                # Read metadata (first line)
                metadata = json.loads(f.readline())
                
                # Initialize metadata if not done yet
                if accumulated_results['metadata'] is None:
                    accumulated_results['metadata'] = metadata
                elif metadata != accumulated_results['metadata']:
                    print(f"Warning: Metadata mismatch in {eval_file}")
                    continue
                
                # Read results
                results = []
                for line in f:
                    results.append(json.loads(line))
                
                accumulated_results['results'].append(results)
                accumulated_results['count'] += 1
                
        except Exception as e:
            print(f"Error processing {eval_file}: {e}")
            continue
    
    if accumulated_results['count'] == 0:
        print("No valid results files found to collate")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find shortest common length
    min_length = min(len(results) for results in accumulated_results['results'])
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
            "Original Reward": 0.0,
            "Example": accumulated_results['results'][0][entry_idx]["Example"],
            "Metrics": accumulated_results['results'][0][entry_idx].get("Metrics", {}),
            "Std Dev": 0.0  # Add standard deviation field
        }
        
        # Collect values for computing mean and std
        actor_values = []
        critic_values = []
        reward_values = []
        
        for run in accumulated_results['results']:
            actor_val = run[entry_idx]["Avg Log Probs"]["Actor"]
            critic_val = run[entry_idx]["Avg Log Probs"]["Critic"]
            actor_values.append(actor_val)
            critic_values.append(critic_val)
            reward_values.append(actor_val - critic_val)
            
            avg_entry["Avg Log Probs"]["Actor"] += actor_val / num_runs
            avg_entry["Avg Log Probs"]["Critic"] += critic_val / num_runs
            if "Original Reward" in run[entry_idx]:
                avg_entry["Original Reward"] += run[entry_idx]["Original Reward"] / num_runs
        
        # Compute standard deviation of the computed rewards
        if len(reward_values) > 1:
            avg_entry["Std Dev"] = np.std(reward_values, ddof=1)
        
        averaged_results.append(avg_entry)
    
    # Extract critic model type from metadata
    critic_model = accumulated_results['metadata'].get('evaluator_model', 'unknown')
    
    # Save averaged results with model-specific name
    output_file = os.path.join(output_dir, f"evaluation_results_{critic_model}.jsonl")
    with open(output_file, "w") as f:
        # Write metadata as first line
        json.dump(accumulated_results['metadata'], f)
        f.write("\n")
        
        # Write averaged results
        for entry in averaged_results:
            json.dump(entry, f)
            f.write("\n")
    
    print(f"Averaged results saved to {output_file}")


def plot_multiple_critics_comparison(log_dir, window_size=40, max_index=None, show_log_probs=False, show_error_bars=False):
    """
    Create a combined plot comparing different critic models' evaluations.
    
    Args:
        log_dir: Directory containing evaluation results for different critics
        window_size: Window size for smoothing
        max_index: Maximum index to plot
        show_log_probs: Whether to show actor/critic log probabilities
        show_error_bars: Whether to show error bars (standard deviation) across runs
    """
    plt.figure(figsize=(12, 6))
    
    # Color scheme for different models
    colors = {
        "llama": "#e41a1c",
        "mistral": "#377eb8",
        "gpt2": "#4daf4a",
        "tinystories": "#984ea3"
    }
    
    generator_model = None
    
    for model_type in ["mistral", "llama", "gpt2", "tinystories"]:
        results_file = os.path.join(log_dir, f"evaluation_results_{model_type}.jsonl")
        if not os.path.exists(results_file):
            continue
            
        try:
            results = load_evaluation_results(results_file)
            if generator_model is None:
                generator_model = results["generator_model"]
                
            all_data = results["evaluations"][0]
            
            if max_index is not None:
                all_data = all_data[:max_index]
            
            # Extract computed rewards and their standard deviations if available
            computed_rewards = []
            reward_stds = []
            
            for entry in all_data:
                reward = entry["Avg Log Probs"]["Actor"] - entry["Avg Log Probs"]["Critic"]
                computed_rewards.append(reward)
                if "Std Dev" in entry:
                    reward_stds.append(entry["Std Dev"])
            
            if len(computed_rewards) > window_size:
                half_window = window_size // 2
                x_values = range(half_window, len(computed_rewards) - half_window)
                smoothed = savgol_filter(computed_rewards, window_size, 3)
                
                plt.plot(x_values, smoothed[half_window:-half_window],
                        label=f"{model_type.title()} Critic",
                        color=colors[model_type],
                        linewidth=2)
                
                # Add error bars if available and requested
                if show_error_bars and reward_stds:
                    smoothed_std = savgol_filter(reward_stds, window_size, 3)
                    plt.fill_between(x_values,
                                   smoothed[half_window:-half_window] - smoothed_std[half_window:-half_window],
                                   smoothed[half_window:-half_window] + smoothed_std[half_window:-half_window],
                                   color=colors[model_type],
                                   alpha=0.2)
            else:
                plt.plot(computed_rewards,
                        label=f"{model_type.title()} Critic",
                        color=colors[model_type],
                        linewidth=2)
                
                if show_error_bars and reward_stds:
                    plt.fill_between(range(len(computed_rewards)),
                                   np.array(computed_rewards) - np.array(reward_stds),
                                   np.array(computed_rewards) + np.array(reward_stds),
                                   color=colors[model_type],
                                   alpha=0.2)
                        
        except FileNotFoundError:
            print(f"No results found for {model_type} critic")
            continue
    
    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Computed Reward (Actor - Critic Log Prob)", fontsize=16)
    title = f"Comparison of Different Critics Evaluating {generator_model.title()} Generator\n"
    if show_error_bars:
        title += f"(Smoothing: {window_size}, with Standard Deviation)"
    else:
        title += f"(Smoothing: {window_size})"
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(log_dir, "multiple_critics_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Model Evaluation Tool")
    parser.add_argument("--log_file", help="Log file to evaluate")
    parser.add_argument(
        "--window_size", type=int, default=40, help="Smoothing window size for plots"
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
        "--plot_multiple_critics",
        action="store_true",
        help="Create a combined plot comparing different critic models' evaluations"
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
        help="Maximum index to process/plot"
    )
    parser.add_argument(
        "--critic_model",
        type=str,
        choices=["mistral", "llama", "gpt2", "tinystories"],
        help="Specify which model to use as the critic"
    )
    parser.add_argument(
        "--show_log_probs",
        action="store_true",
        help="Show actor and critic log probabilities in the plot"
    )
    parser.add_argument(
        "--show_error_bars",
        action="store_true",
        help="Show error bars (standard deviation) in the plot when multiple runs are available"
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

    # If we're only plotting multiple critics, skip the evaluation step
    if args.plot_multiple_critics:
        log_dir = os.path.dirname(log_file)
        plot_multiple_critics_comparison(
            log_dir,
            window_size=args.window_size,
            max_index=args.max_index,
            show_log_probs=args.show_log_probs,
            show_error_bars=args.show_error_bars
        )
        return

    # Process data if needed
    if not args.plot_only:
        if not args.critic_model:
            raise ValueError("--critic_model must be specified when running evaluation")
        print(f"Processing every {args.stride}th entry")
        results = run_cross_model_evaluation(
            [log_file], 
            stride=args.stride, 
            debug_freq=args.debug_freq,
            max_index=args.max_index,
            critic_model_type=args.critic_model
        )
        save_evaluation_results(results, log_file)

    # Handle single critic plotting
    if not args.process_only:
        eval_results_file = os.path.join(
            os.path.dirname(log_file), 
            f"evaluation_results_{args.critic_model}.jsonl"
        )
        try:
            results = load_evaluation_results(eval_results_file)
            plot_cross_model_comparison(
                results, 
                log_file, 
                window_size=args.window_size,
                max_index=args.max_index,
                show_log_probs=args.show_log_probs
            )
        except FileNotFoundError:
            print(f"No saved results found at {eval_results_file}. Run without --plot_only first.")


if __name__ == "__main__":
    main()
