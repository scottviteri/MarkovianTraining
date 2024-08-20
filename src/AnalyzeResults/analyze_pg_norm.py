import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import glob
import argparse
import math


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def get_latest_log_file():
    log_files = glob.glob("src/AnalyzeResults/PolicyGradientNormalized_*.log")
    if not log_files:
        raise FileNotFoundError("No PolicyGradientNormalized log files found.")
    return max(log_files, key=os.path.getctime)


def plot_metrics(
    file_path, window_size=16, output_file="src/AnalyzeResults/pg_norm_plot.png"
):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse the hyperparameters from the first line
    hyperparameters = json.loads(lines[0])
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    normalize_loss = hyperparameters.get("normalize_loss", True)

    # Parse the log entries
    metrics = defaultdict(list)
    for line in lines[1:]:
        entry = json.loads(line)
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                metrics[key].append(value)
            elif key == "Is Correct" and isinstance(value, bool):
                metrics["Fraction Correct"].append(int(value))

    # Determine if this is an EI run
    use_ei = any("Use EI" in entry and entry["Use EI"] for entry in map(json.loads, lines[1:]))

    # Define the metrics to plot
    plot_info = [
        ("Aggregate loss", "Aggregate Loss", "Batch", "Loss"),
        ("Grad Norm", "Gradient Norm", "Batch", "Norm"),
        ("Avg Log Prob", "Average Log Probability", "Batch", "Log Probability", {"ylim": (None, 0)}),
        ("Reasoning Contains Answer", "Reasoning Contains Answer", "Batch", "Proportion", {"ylim": (0, 1)}),
    ]

    if normalize_loss:
        plot_info.extend([
            ("Normalized Reward", "Normalized Reward", "Batch", "Reward"),
            ("Advantage", "Advantage", "Batch", "Advantage"),
        ])

    if "PPO Ratio" in metrics:
        plot_info.extend([
            ("PPO Ratio", "PPO Ratio", "Batch", "Ratio"),
            ("PPO Clipped Ratio", "PPO Clipped Ratio", "Batch", "Ratio"),
        ])

    if use_ei:
        plot_info.extend([
            ("Mean Previous Advantage", "Mean Previous Advantage", "Batch", "Advantage"),
            ("EI Threshold", "EI Threshold", "Batch", "Threshold"),
        ])

    if "Fraction Correct" in metrics:
        plot_info.append(("Fraction Correct", "Fraction Correct", "Batch", "Fraction", {"ylim": (0, 1)}))

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
        if metric in metrics:
            axs[i].plot(moving_average(metrics[metric], window_size))
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
    parser = argparse.ArgumentParser(
        description="Analyze Policy Gradient Normalized results."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Window size for moving average (default: 16)",
    )
    args = parser.parse_args()

    latest_log_file = get_latest_log_file()
    plot_metrics(latest_log_file, window_size=args.window_size)
    print(f"Plot saved as pg_norm_plot.png with window size {args.window_size}")