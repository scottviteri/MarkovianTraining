import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import glob
import argparse


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

    # Determine the number of rows needed for the plot
    num_rows = 3 if "PPO Ratio" not in metrics else 4
    if not normalize_loss:
        num_rows -= 1  # Remove one row if not using normalization

    # Create the figure and axes
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    fig.suptitle(f"Training Metrics (Window Size: {window_size})")

    # Plot Aggregate Loss
    axs[0, 0].plot(moving_average(metrics["Aggregate loss"], window_size))
    axs[0, 0].set_title("Aggregate Loss")
    axs[0, 0].set_xlabel("Batch")
    axs[0, 0].set_ylabel("Loss")

    # Plot Gradient Norm
    axs[0, 1].plot(moving_average(metrics["Grad Norm"], window_size))
    axs[0, 1].set_title("Gradient Norm")
    axs[0, 1].set_xlabel("Batch")
    axs[0, 1].set_ylabel("Norm")

    if normalize_loss:
        # Plot Normalized Reward
        axs[1, 0].plot(moving_average(metrics["Normalized Reward"], window_size))
        axs[1, 0].set_title("Normalized Reward")
        axs[1, 0].set_xlabel("Batch")
        axs[1, 0].set_ylabel("Reward")

        # Plot Advantage
        axs[1, 1].plot(moving_average(metrics["Advantage"], window_size))
        axs[1, 1].set_title("Advantage")
        axs[1, 1].set_xlabel("Batch")
        axs[1, 1].set_ylabel("Advantage")

        # Plot Average Log Prob
        axs[2, 0].plot(moving_average(metrics["Avg Log Prob"], window_size))
        axs[2, 0].set_title("Average Log Probability")
        axs[2, 0].set_xlabel("Batch")
        axs[2, 0].set_ylabel("Log Probability")
        axs[2, 0].set_ylim(top=0)  # Set maximum to 0

        # Plot Reasoning Contains Answer
        contains_answer = [int(x) for x in metrics["Reasoning Contains Answer"]]
        axs[2, 1].plot(moving_average(contains_answer, window_size))
        axs[2, 1].set_title("Reasoning Contains Answer")
        axs[2, 1].set_xlabel("Batch")
        axs[2, 1].set_ylabel("Proportion")
        axs[2, 1].set_ylim(0, 1)  # Set range from 0 to 1
    else:
        # Plot Average Log Prob
        axs[1, 0].plot(moving_average(metrics["Avg Log Prob"], window_size))
        axs[1, 0].set_title("Average Log Probability")
        axs[1, 0].set_xlabel("Batch")
        axs[1, 0].set_ylabel("Log Probability")
        axs[1, 0].set_ylim(top=0)  # Set maximum to 0

        # Plot Reasoning Contains Answer
        contains_answer = [int(x) for x in metrics["Reasoning Contains Answer"]]
        axs[1, 1].plot(moving_average(contains_answer, window_size))
        axs[1, 1].set_title("Reasoning Contains Answer")
        axs[1, 1].set_xlabel("Batch")
        axs[1, 1].set_ylabel("Proportion")
        axs[1, 1].set_ylim(0, 1)  # Set range from 0 to 1

    # If PPO is used, plot PPO-specific metrics
    if "PPO Ratio" in metrics:
        row = 3 if normalize_loss else 2
        axs[row, 0].plot(moving_average(metrics["PPO Ratio"], window_size))
        axs[row, 0].set_title("PPO Ratio")
        axs[row, 0].set_xlabel("Batch")
        axs[row, 0].set_ylabel("Ratio")

        axs[row, 1].plot(moving_average(metrics["PPO Clipped Ratio"], window_size))
        axs[row, 1].set_title("PPO Clipped Ratio")
        axs[row, 1].set_xlabel("Batch")
        axs[row, 1].set_ylabel("Ratio")

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
