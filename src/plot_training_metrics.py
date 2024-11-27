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


def get_nested_value(entry, path, metrics_dict=None):
    """Helper function to get nested dictionary values using dot notation, with optional derived metrics"""
    if path == "Training Metrics.Critic Answer Log Probs" and metrics_dict is not None:
        # Calculate Critic Answer Log Probs if not directly available
        actor_probs = get_nested_value(entry, "Training Metrics.Actor Answer Log Probs")
        norm_reward = get_nested_value(entry, "Training Metrics.Normalized Reward")
        if actor_probs is not None and norm_reward is not None:
            return actor_probs - norm_reward
        return None

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


def plot_combined_metrics(file_paths, host_names, window_size=10, output_file=None, plot_summary=False, max_index=None, average=False, show_std=False, show_legend=True, label_size=10, show_title=True):
    """Plot metrics from multiple files on the same plot."""
    # Read first file to get task type and check available metrics
    with open(file_paths[0], "r") as f:
        file_contents = f.readlines()
        hyperparameters = json.loads(file_contents[0].strip())
        first_entry = json.loads(file_contents[1].strip())  # Read first log entry
    
    task_type = hyperparameters.get('task_type', 'unknown')
    has_answer_logprobs = "Actor Answer Log Probs" in first_entry.get("Training Metrics", {})
    has_critic_probs = "Critic Answer Log Probs" in first_entry.get("Training Metrics", {})
    
    # Initialize metrics_dict for deriving critic probs if needed
    metrics_dict = None if has_critic_probs else {"derive_critic": True}
    
    if output_file is None:
        output_file = f"combined_metrics_{task_type}.png"
    
    if plot_summary:
        # For arithmetic tasks or gsm8k
        if task_type in ['arithmetic', 'gsm8k']:
            metrics_to_plot = [
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)"),
                ("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)"),
                ("Example.Contains Answer", "Contains Answer", "Training Batch No. []", "Fraction")
            ]
        # For wiki tasks
        elif task_type.startswith('wiki_'):
            if has_answer_logprobs:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')"),
                    ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)"),
                    ("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)")
                ]
            else:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')")
                ]
        else:
            metrics_to_plot = [
                ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')")
            ]
    else:
        base_metrics = [
            ("Training Metrics.Loss", "Total Loss", "Training Batch No. []", "Loss"),
            ("Training Metrics.Policy Gradient Loss", "Policy Gradient Loss", "Training Batch No. []", "Loss"),
            ("Training Metrics.KL", "KL Divergence", "Training Batch No. []", "KL"),
            ("Training Metrics.Gradient Norm", "Gradient Norm", "Training Batch No. []", "Norm"),
            ("Training Metrics.Advantage", "Advantage", "Training Batch No. []", "Value"),
            ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')"),
            ("Training Metrics.Active Samples.Fraction", "Fraction of Active Samples", "Training Batch No. []", "Fraction")
        ]
        
        if has_answer_logprobs:
            base_metrics.extend([
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)"),
                ("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)")
            ])
        metrics_to_plot = base_metrics

    # Calculate required number of rows and columns for subplots
    if len(metrics_to_plot) > 1:
        num_cols = 2
        num_rows = math.ceil(len(metrics_to_plot) / 2)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axs = np.array(axs).reshape(-1)
    else:
        num_rows, num_cols = 1, 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
        axs = np.array([axs])

    axs = axs.flatten()

    for metric_idx, (metric_path, title, xlabel, ylabel, *extra) in enumerate(
        metrics_to_plot
    ):
        all_data = []  # Store data from all files for averaging
        
        for file_path, host_name in zip(file_paths, host_names):
            with open(file_path, "r") as f:
                file_contents = f.readlines()
            
            hyperparameters = json.loads(file_contents[0].strip())
            label = f"c{hyperparameters['cot_length']}t{hyperparameters['temperature']}ei{hyperparameters['use_ei']}kl{hyperparameters.get('kl_penalty', 'NA')}"
            
            entries = [json.loads(line) for line in file_contents[1:]]
            if max_index is not None:
                entries = entries[:max_index]
            
            data = [
                get_nested_value(entry, metric_path, metrics_dict)
                for entry in entries[EI_SKIP_INITIAL:]
                if get_nested_value(entry, metric_path, metrics_dict) is not None
            ]
            
            if data:
                if metric_path == "Example.Contains Answer":
                    data = [1 if x else 0 for x in data]
                
                if average:
                    all_data.append(data)
                else:
                    smoothed_data = moving_average(data, window_size)
                    offset = window_size // 2
                    axs[metric_idx].plot(
                        range(offset, offset + len(smoothed_data)),
                        smoothed_data,
                        linewidth=2,
                        label=label
                    )
        
        if average and all_data:
            # Convert all data to float arrays first
            all_data = [np.array(d, dtype=float) for d in all_data]
            
            # Pad shorter sequences with NaN
            max_len = max(len(d) for d in all_data)
            padded_data = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in all_data]
            padded_array = np.array(padded_data)
            
            # Calculate mean and std, ignoring NaN values
            mean_data = np.nanmean(padded_array, axis=0)
            std_data = np.nanstd(padded_array, axis=0) if show_std else None
            
            # Remove trailing NaN values
            valid_mask = ~np.isnan(mean_data)
            mean_data = mean_data[valid_mask]
            if show_std:
                std_data = std_data[valid_mask]
            
            # Smooth mean and std data
            smoothed_mean = moving_average(mean_data, window_size)
            if show_std:
                smoothed_std = moving_average(std_data, window_size)
            
            offset = window_size // 2
            x_range = range(offset, offset + len(smoothed_mean))
            
            # Plot mean line
            line = axs[metric_idx].plot(
                x_range,
                smoothed_mean,
                linewidth=2,
                label="Average",
                color='blue'
            )[0]
            
            # Add std bands if requested
            if show_std:
                axs[metric_idx].fill_between(
                    x_range,
                    smoothed_mean - smoothed_std,
                    smoothed_mean + smoothed_std,
                    alpha=0.2,
                    color=line.get_color(),
                    label='±1 SD'
                )

        # Only add legend if there are labeled lines in the plot and show_legend is True
        if len(axs[metric_idx].get_lines()) > 0 and show_legend:
            axs[metric_idx].legend(fontsize=label_size)
            
        # Set labels and their sizes
        axs[metric_idx].set_xlabel(xlabel, fontsize=label_size)
        axs[metric_idx].set_ylabel(ylabel, fontsize=label_size)
        # Set tick label sizes
        axs[metric_idx].tick_params(axis='both', which='major', labelsize=label_size)
        # Add grid with light gray lines
        axs[metric_idx].grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add smoothing window info in bottom right corner
        axs[metric_idx].text(
            0.98, 0.02,  # Changed y position to 0.02 for bottom right
            f"Smoothing window = {window_size}",
            transform=axs[metric_idx].transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=label_size * 0.8,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='black',
                pad=3,
                boxstyle='round,pad=0.5'
            )
        )
        
        if extra:
            for key, value in extra[0].items():
                getattr(axs[metric_idx], f"set_{key}")(value)

        # Set title if enabled
        if show_title:
            axs[metric_idx].set_title(title, fontsize=label_size)

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
        "--files", "-f",
        nargs="+",
        type=str,
        help="Direct paths to log files to plot",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        help="Indices of machines to include (1-based indexing)",
    )
    parser.add_argument(
        "--max_index",
        type=int,
        help="Maximum index to plot (truncates data after this point)",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average the values across all input files",
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="Show standard deviation bands when averaging",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Don't show the legend in the plots",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=10,
        help="Font size for all labels (axis labels, tick labels, and legend)",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Don't show titles on the plots",
    )
    args = parser.parse_args()

    if args.files:
        # Use directly specified files
        files_to_plot = []
        host_names_to_plot = []
        valid_files = False

        for file_path in args.files:
            if os.path.exists(file_path):
                files_to_plot.append(file_path)
                # Use filename as host name for the legend
                host_names_to_plot.append(os.path.basename(os.path.dirname(file_path)))
                valid_files = True
            else:
                print(f"Warning: Could not find log file: {file_path}")
    elif len(args.indices) > 0:
        # Define the list of hosts (keep in sync with download.sh)
        hosts = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
        ]
        # Collect files and corresponding host names to plot
        files_to_plot = []
        host_names_to_plot = []
        valid_files = False

        for i in args.indices:
            if i < 1 or i > len(hosts):
                print(f"Invalid index: {i} (must be between 1 and {len(hosts)})")
                continue
            hostname = hosts[i - 1].split(":")[0]  # Remove port if present
            log_path = f"./results_{hostname}/log.jsonl"
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
            max_index=args.max_index,
            average=args.average,
            show_std=args.show_std,
            show_legend=not args.no_legend,
            label_size=args.label_size,
            show_title=not args.no_title,
        )
    else:
        print("No valid files found to plot")
        sys.exit(1)
