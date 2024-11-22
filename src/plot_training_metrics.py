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
                ("Example.Contains Answer", "Contains Answer", "Batch", "Fraction")
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
        # Check if Actor Answer Log Probs is available
        has_answer_logprobs = "Actor Answer Log Probs" in first_entry.get("Training Metrics", {})
        
        # Base metrics that are always included
        base_metrics = [
            ("Training Metrics.Loss", "Total Loss", "Batch", "Loss"),
            ("Training Metrics.Policy Gradient Loss", "Policy Gradient Loss", "Batch", "Loss"),
            ("Training Metrics.Actor Log Probs", "Actor Log Probs", "Batch", "Log Prob"),
            ("Training Metrics.KL", "KL Divergence", "Batch", "KL"),
            ("Training Metrics.Gradient Norm", "Gradient Norm", "Batch", "Norm"),
            ("Training Metrics.Advantage", "Advantage", "Batch", "Value"),
            ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value", 
             {"ylim": (-0.5, 0.5)} if task_type == 'wiki_prediction' else {}),
            ("Training Metrics.Active Samples.Fraction", "Fraction of Active Samples", "Batch", "Fraction")
        ]
        
        # Add Actor Answer Log Probs if available
        if has_answer_logprobs:
            base_metrics.append(
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Batch", "Value")
            )
        
        # Add Contains Answer for arithmetic tasks
        if task_type == 'arithmetic':
            base_metrics.append(
                ("Example.Contains Answer", "Contains Answer", "Batch", "Fraction", {"ylim": (-0.01, 1.01)})
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
            label = f"c{hyperparameters['cot_length']}t{hyperparameters['temperature']}ei{hyperparameters['use_ei']}kl{hyperparameters.get('kl_penalty', 'NA')}"
            
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
        "riight3",
        "left4"
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
    else:
        # Single file plotting (use latest log file)
        log_file = get_latest_log_file()
        files_to_plot = [log_file]
        host_names_to_plot = ["local"]
        valid_files = True

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
