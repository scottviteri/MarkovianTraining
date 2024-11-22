import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import argparse
import math
from train import get_latest_log_file
from constants import EI_SKIP_INITIAL
import sys


def get_nested_value(entry, path):
    """Helper function to get nested dictionary values using dot notation"""
    value = entry
    for key in path.split("."):
        if value is None or key not in value:
            return None
        value = value[key]
    return value


def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def plot_metrics(
    file_path, window_size=10, output_file=None, normalized_reward_only=False
):
    # Read the entire file contents first
    with open(file_path, "r") as f:
        file_contents = f.readlines()

    # Check if the first line is a JSON object with hyperparameters
    hyperparameters = json.loads(file_contents[0].strip())
    print("Hyperparameters:")
    print(hyperparameters)

    # Setup output file path
    if output_file is None:
        log_dir = os.path.dirname(file_path)
        log_filename = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(log_dir, exist_ok=True)
        output_file = os.path.join(log_dir, f"{log_filename}_metrics.png")

    # Parse all entries
    entries = [json.loads(line) for line in file_contents[1:]]

    if normalized_reward_only:
        # For arithmetic, include both contains answer and actor log probs
        if task_type == 'arithmetic':
            metrics_to_plot = [
                ("Example.Contains Answer", "Contains Answer", "Batch", "Fraction", {"ylim": (-0.01, 1.01)}),
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Batch", "Value")
            ]
            num_rows, num_cols = 1, 2  # Horizontal layout
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6))
            axs = np.array(axs).reshape(-1)
        else:
            # Keep original normalized reward for non-arithmetic tasks
            metrics_to_plot = [
                ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value", 
                 {"ylim": (-0.5, 0.5)} if task_type == 'wiki_prediction' else {})
            ]
            num_rows, num_cols = 1, 1
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
            axs = np.array([axs])
    else:
        # Define the metrics to plot with their paths in the JSON structure
        plot_info = [
            ("Training Metrics.Loss", "Total Loss", "Batch", "Loss"),
            (
                "Training Metrics.Policy Gradient Loss",
                "Policy Gradient Loss",
                "Batch",
                "Loss",
            ),
            (
                "Training Metrics.Actor Log Probs",
                "Actor Log Probs",
                "Batch",
                "Log Prob",
            ),
            ("Training Metrics.KL", "KL Divergence", "Batch", "KL"),
            ("Training Metrics.Gradient Norm", "Gradient Norm", "Batch", "Norm"),
            ("Training Metrics.Advantage", "Advantage", "Batch", "Value"),
            (
                "Training Metrics.Normalized Reward",
                "Normalized Reward",
                "Batch",
                "Value",
            ),
            (
                "Training Metrics.Active Samples.Fraction",
                "Fraction of Active Samples",
                "Batch",
                "Fraction",
                {"ylim": (0, 1)},
            ),
        ]

        # Add PPO metrics if present
        if (
            entries
            and "Training Metrics" in entries[0]
            and "PPO Ratio" in entries[0]["Training Metrics"]
        ):
            plot_info.extend(
                [
                    ("Training Metrics.PPO Ratio", "PPO Ratio", "Batch", "Ratio"),
                    (
                        "Training Metrics.PPO Clipped Ratio",
                        "PPO Clipped Ratio",
                        "Batch",
                        "Ratio",
                    ),
                ]
            )

        # Add EI metrics if present
        if (
            entries
            and "EI Metrics" in entries[0]
            and entries[0]["EI Metrics"]["Use EI"]
        ):
            plot_info.extend(
                [
                    (
                        "EI Metrics.Mean Previous Advantage",
                        "Mean Previous Advantage",
                        "Batch",
                        "Value",
                    ),
                    (
                        "EI Metrics.Std Previous Advantage",
                        "Std Previous Advantage",
                        "Batch",
                        "Value",
                    ),
                    ("EI Metrics.Threshold", "EI Threshold", "Batch", "Value"),
                ]
            )

        # Create the plots
        num_plots = len(plot_info)
        num_cols = 2
        num_rows = math.ceil(num_plots / num_cols)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        fig.suptitle(f"Training Metrics (Window Size: {window_size})")

        # Flatten axs if it's a 2D array
        axs = axs.flatten() if num_plots > 2 else [axs] if num_plots == 1 else axs

    # Add title suffix based on KL type
    def get_plot_title(entry, title):
        if (
            title == "KL Divergence"
            and "Training Metrics" in entry
            and "KL Type" in entry["Training Metrics"]
        ):
            return f"{title} ({entry['Training Metrics']['KL Type']})"
        return title

    # Modify plotting loop to use dynamic titles
    for i, (metric_path, title, xlabel, ylabel, *extra) in enumerate(plot_info):
        data = [
            get_nested_value(entry, metric_path)
            for entry in entries[EI_SKIP_INITIAL:]
            if get_nested_value(entry, metric_path) is not None
        ]

        if data:
            smoothed_data = moving_average(data, window_size)
            offset = window_size // 2

            # Get dynamic title for KL plot
            plot_title = get_plot_title(entries[0], title)

            # Only show scatter plot if not in normalized_reward_only mode
            if not normalized_reward_only:
                axs[i].scatter(range(len(data)), data, alpha=0.3)

            axs[i].plot(
                range(offset, offset + len(smoothed_data)),
                smoothed_data,
                color="red",
                linewidth=2,
            )
            axs[i].set_title(plot_title)
            axs[i].set_xlabel(xlabel)
            axs[i].set_ylabel(ylabel)
            if extra:
                for key, value in extra[0].items():
                    getattr(axs[i], f"set_{key}")(value)

    # Remove any unused subplots
    for i in range(num_plots, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


def plot_combined_metrics(file_paths, host_names, window_size=10, output_file=None, plot_summary=False):
    """Plot metrics from multiple files on the same plot."""
    # Read first file to get task type and check available metrics
    with open(file_paths[0], "r") as f:
        file_contents = f.readlines()
        hyperparameters = json.loads(file_contents[0].strip())
        first_entry = json.loads(file_contents[1].strip())  # Read first log entry
    
    task_type = hyperparameters.get('task_type', 'unknown')
    has_answer_logprobs = "Actor Answer Log Probs" in first_entry.get("Training Metrics", {})
    
    if output_file is None:
        output_file = f"combined_metrics_{task_type}.png"
    
    if plot_summary:
        # For arithmetic tasks
        if task_type == 'arithmetic':
            metrics_to_plot = [
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Batch", "Value"),
                ("Example.Contains Answer", "Contains Answer", "Batch", "Fraction", {"ylim": (-0.01, 1.01)})
            ]
        # For wiki tasks
        elif task_type.startswith('wiki_'):
            if has_answer_logprobs:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value", {"ylim": (-0.5, 0.5)}),
                    ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Batch", "Value")
                ]
            else:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value", {"ylim": (-0.5, 0.5)})
                ]
        else:
            # For other tasks, just show normalized reward
            metrics_to_plot = [
                ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value")
            ]
            
        # Use horizontal layout for multiple plots, vertical for single plot
        if len(metrics_to_plot) > 1:
            num_rows, num_cols = 1, 2
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 6))
            axs = np.array(axs).reshape(-1)
        else:
            num_rows, num_cols = 1, 1
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
            axs = np.array([axs])
    else:
        # Add Contains Answer to full metrics plot for arithmetic
        base_metrics = [
            ("Training Metrics.Loss", "Total Loss", "Batch", "Loss"),
            ("Training Metrics.Policy Gradient Loss", "Policy Gradient Loss", "Batch", "Loss"),
            ("Training Metrics.Actor Log Probs", "Actor Log Probs", "Batch", "Log Prob"),
            ("Training Metrics.KL", "KL Divergence", "Batch", "KL"),
            ("Training Metrics.Gradient Norm", "Gradient Norm", "Batch", "Norm"),
            ("Training Metrics.Advantage", "Advantage", "Batch", "Value"),
            ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value", 
             {"ylim": (-0.5, 0.5)} if task_type == 'wiki_prediction' else {}),
            ("Training Metrics.Active Samples.Fraction", "Fraction of Active Samples", "Batch", "Fraction", {"ylim": (0, 1)})
        ]
        
        if task_type == 'arithmetic':
            base_metrics.append(
                ("Example.Contains Answer", "Contains Answer", "Batch", "Fraction", {"ylim": (-0.01, 1.01)})  # Updated ylim
            )
        
        metrics_to_plot = base_metrics
        num_rows = (len(metrics_to_plot) + 1) // 2
        num_cols = 2
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

    axs = axs.flatten()

    for metric_idx, (metric_path, title, xlabel, ylabel, *extra) in enumerate(
        metrics_to_plot
    ):
        for file_path, host_name in zip(file_paths, host_names):
            with open(file_path, "r") as f:
                file_contents = f.readlines()
            
            # Get hyperparameters from first line
            hyperparameters = json.loads(file_contents[0].strip())
            
            # Create label from hyperparameters, keeping use_ei as is
            label = f"t{hyperparameters['temperature']}ei{hyperparameters['use_ei']}CoT{hyperparameters['cot_length']}"
            
            entries = [json.loads(line) for line in file_contents[1:]]
            data = [
                get_nested_value(entry, metric_path)
                for entry in entries[EI_SKIP_INITIAL:]
                if get_nested_value(entry, metric_path) is not None
            ]
            
            if data:
                # Convert "Contains Answer" to 0/1 if that's the metric
                if metric_path == "Example.Contains Answer":
                    data = [1 if x else 0 for x in data]
                
                smoothed_data = moving_average(data, window_size)
                offset = window_size // 2
                
                line = axs[metric_idx].plot(
                    range(offset, offset + len(smoothed_data)),
                    smoothed_data,
                    linewidth=2,
                    label=label
                )
        
        # Only add legend if there are labeled lines in the plot
        if len(axs[metric_idx].get_lines()) > 0:
            axs[metric_idx].legend()
            
        axs[metric_idx].set_title(title)
        axs[metric_idx].set_xlabel(xlabel)
        axs[metric_idx].set_ylabel(ylabel)
        if extra:
            for key, value in extra[0].items():
                getattr(axs[metric_idx], f"set_{key}")(value)

    # Remove any unused subplots
    for i in range(len(metrics_to_plot), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for moving average (default: 10)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output plot file path (optional)",
    )
    parser.add_argument(
        "--plot_summary",
        action="store_true",
        help="Plot summary metrics (varies by task type)",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        help="Indices of machines to include (1-based indexing)",
    )
    args = parser.parse_args()

    # Define the list of hosts (keep in sync with download.sh)
    hosts = [
        "left",
        "mid",
        "right",
        "riight",
        "left2",
        "mid2",
        "right2",
        "riight2",
        "left3",
        "mid3",
        "right3",
        "riight3"
    ]

    if len(args.indices) > 0:
        # Collect files and corresponding host names to plot
        files_to_plot = []
        host_names_to_plot = []
        valid_files = False

        for i in args.indices:
            if i < 1 or i > len(hosts):
                print(f"Invalid index: {i} (must be between 1 and {len(hosts)})")
                continue
            hostname = hosts[i - 1].split(":")[0]  # Remove port if present
            log_path = f"./results_{i}_{hostname}/log.jsonl"
            if os.path.exists(log_path):
                files_to_plot.append(log_path)
                host_names_to_plot.append(hosts[i - 1])
                valid_files = True
            else:
                print(f"Warning: Could not find log file for index {i} ({log_path})")

        if valid_files:
            plot_combined_metrics(
                files_to_plot,
                host_names_to_plot,
                window_size=args.window_size,
                output_file=args.output_file,
                plot_summary=args.plot_summary,
            )
        else:
            print("No valid files found to plot")
            sys.exit(1)
    else:
        # Single file plotting (use latest log file)
        log_file = get_latest_log_file()
        plot_metrics(
            log_file,
            window_size=args.window_size,
            output_file=args.output_file,
            normalized_reward_only=args.plot_summary,
        )
