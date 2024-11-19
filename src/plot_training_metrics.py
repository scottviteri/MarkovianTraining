import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import argparse
import math
from train import get_latest_log_file
from constants import EI_SKIP_INITIAL


def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def plot_metrics(file_path, window_size=10, output_file=None):
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

    # Define the metrics to plot with their paths in the JSON structure
    plot_info = [
        ("Training Metrics.Loss", "Total Loss", "Batch", "Loss"),
        ("Training Metrics.Policy Gradient Loss", "Policy Gradient Loss", "Batch", "Loss"),
        ("Training Metrics.KL Penalty", "KL Penalty", "Batch", "Loss"),
        ("Training Metrics.Gradient Norm", "Gradient Norm", "Batch", "Norm"),
        ("Training Metrics.Advantage", "Advantage", "Batch", "Value"),
        ("Training Metrics.Normalized Reward", "Normalized Reward", "Batch", "Value"),
        ("Training Metrics.Active Samples.Fraction", "Fraction of Active Samples", "Batch", "Fraction", {"ylim": (0, 1)}),
    ]

    # Add PPO metrics if present
    if entries and "Training Metrics" in entries[0] and "PPO Ratio" in entries[0]["Training Metrics"]:
        plot_info.extend([
            ("Training Metrics.PPO Ratio", "PPO Ratio", "Batch", "Ratio"),
            ("Training Metrics.PPO Clipped Ratio", "PPO Clipped Ratio", "Batch", "Ratio"),
        ])

    # Add EI metrics if present
    if entries and "EI Metrics" in entries[0] and entries[0]["EI Metrics"]["Use EI"]:
        plot_info.extend([
            ("EI Metrics.Mean Previous Advantage", "Mean Previous Advantage", "Batch", "Value"),
            ("EI Metrics.Std Previous Advantage", "Std Previous Advantage", "Batch", "Value"),
            ("EI Metrics.Threshold", "EI Threshold", "Batch", "Value"),
        ])

    # Create the plots
    num_plots = len(plot_info)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(f"Training Metrics (Window Size: {window_size})")

    # Flatten axs if it's a 2D array
    axs = axs.flatten() if num_plots > 2 else [axs] if num_plots == 1 else axs

    def get_nested_value(entry, path):
        """Helper function to get nested dictionary values using dot notation"""
        value = entry
        for key in path.split('.'):
            if value is None or key not in value:
                return None
            value = value[key]
        return value

    # Plot the metrics
    for i, (metric_path, title, xlabel, ylabel, *extra) in enumerate(plot_info):
        # Extract data using the metric path
        data = [
            get_nested_value(entry, metric_path) 
            for entry in entries[EI_SKIP_INITIAL:]
            if get_nested_value(entry, metric_path) is not None
        ]
        
        if data:  # Only plot if we have data
            smoothed_data = moving_average(data, window_size)
            offset = window_size // 2
            
            axs[i].scatter(range(len(data)), data, alpha=0.3)
            axs[i].plot(
                range(offset, offset + len(smoothed_data)),
                smoothed_data,
                color="red",
                linewidth=2
            )
            axs[i].set_title(title)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for moving average (default: 10)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file to analyze (optional)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output plot file path (optional)",
    )
    args = parser.parse_args()

    # Use provided log file or find the latest one
    log_file = args.log_file or get_latest_log_file()

    plot_metrics(log_file, window_size=args.window_size, output_file=args.output_file)
