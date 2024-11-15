import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import argparse
import math
from src.train import get_latest_log_file


def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def plot_metrics(file_path, window_size=10, output_file=None):
    # Read the entire file contents first
    with open(file_path, "r") as f:
        file_contents = f.readlines()

    # Check if the first line is a JSON object with hyperparameters
    first_line = file_contents[0].strip()
    try:
        hyperparameters = json.loads(first_line)
        if "hyperparameters" in hyperparameters:
            hyperparameters = hyperparameters["hyperparameters"]
    except json.JSONDecodeError:
        # If first line is not JSON, use default hyperparameters
        hyperparameters = {}

    print("Hyperparameters:")
    print(hyperparameters)

    # If no output file is specified, create one in the same directory as the log file
    if output_file is None:
        # Create a plot filename based on the log file
        log_dir = os.path.dirname(file_path)
        log_filename = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(log_dir, exist_ok=True)
        output_file = os.path.join(log_dir, f"{log_filename}_metrics.png")

    normalize_loss = hyperparameters.get("normalize_loss", True)

    # Parse the log entries
    metrics = defaultdict(list)
    for line in file_contents[1:]:  # Skip the first line with hyperparameters
        try:
            entry = json.loads(line)
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
                elif key == "Is Correct" and isinstance(value, bool):
                    metrics["Fraction Correct"].append(int(value))
                elif key == "Reasoning Contains Answer":
                    metrics["Reasoning Contains Answer"].append(int(value))
        except json.JSONDecodeError:
            continue

    # Determine if this is an EI run
    use_ei = hyperparameters.get("use_ei", False)

    # Define the metrics to plot
    plot_info = [
        ("Aggregate loss", "Aggregate Loss", "Batch", "Loss"),
        ("Grad Norm", "Gradient Norm", "Batch", "Norm"),
        (
            "Avg Log Prob",
            "Average Log Probability",
            "Batch",
            "Log Probability",
            {"ylim": (None, 0)},
        ),
        (
            "Reasoning Contains Answer",
            "Reasoning Contains Answer",
            "Batch",
            "Proportion",
            {"ylim": (0, 1)},
        ),
    ]

    if normalize_loss:
        plot_info.extend(
            [
                ("Normalized Reward", "Normalized Reward", "Batch", "Reward"),
                ("Advantage", "Advantage", "Batch", "Advantage"),
            ]
        )

    if "PPO Ratio" in metrics:
        plot_info.extend(
            [
                ("PPO Ratio", "PPO Ratio", "Batch", "Ratio"),
                ("PPO Clipped Ratio", "PPO Clipped Ratio", "Batch", "Ratio"),
            ]
        )

    if use_ei:
        plot_info.extend(
            [
                (
                    "Mean Previous Advantage",
                    "Mean Previous Advantage",
                    "Batch",
                    "Advantage",
                ),
                ("EI Threshold", "EI Threshold", "Batch", "Threshold"),
                (
                    "Fraction Active Samples",
                    "Fraction of Active Samples",
                    "Batch",
                    "Fraction",
                    {"ylim": (0, 1)},
                ),
            ]
        )

    if "Fraction Correct" in metrics:
        plot_info.append(
            (
                "Fraction Correct",
                "Fraction Correct",
                "Batch",
                "Fraction",
                {"ylim": (0, 1)},
            )
        )

    # Create the figure and axes
    num_plots = len(plot_info)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(f"Training Metrics (Window Size: {window_size})")

    # Flatten axs if it's a 2D array
    axs = axs.flatten() if num_plots > 2 else [axs] if num_plots == 1 else axs

    # Plot the metrics
    for i, (metric, title, xlabel, ylabel, *extra) in enumerate(plot_info):
        if metric in metrics and len(metrics[metric]) > window_size:
            smoothed_data = moving_average(metrics[metric], window_size)
            axs[i].plot(smoothed_data)
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
