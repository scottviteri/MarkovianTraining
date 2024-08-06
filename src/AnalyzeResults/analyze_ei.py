import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def smooth_data(data, window_size):
    return moving_average(data, window_size)


def get_latest_log_file():
    log_files = glob.glob("src/AnalyzeResults/ExpertIterationDictionary_*.log")
    if not log_files:
        raise FileNotFoundError("No ExpertIterationDictionary log files found.")
    return max(log_files, key=os.path.getctime)


def plot_metrics(
    file_path, window_size=16, output_file="src/AnalyzeResults/ei_plot.png"
):
    # Load the log file into a list of dictionaries
    with open(file_path, "r") as file:
        # Read the first line as hyperparameters
        hyperparameters = json.loads(file.readline().strip())
        # Read the rest of the lines as expert iteration data
        expert_iteration_data = [json.loads(line) for line in file if line.strip()]

    print(f"Loaded hyperparameters: {hyperparameters}")
    print(f"Loaded {len(expert_iteration_data)} entries from the log file.")
    print("First data entry:", expert_iteration_data[0])

    # Extract batch indices and reasoning contains answer values
    batch_indices = [entry["Batch Index"] for entry in expert_iteration_data]
    reasoning_contains_answer = [
        1 if entry["Reasoning Contains Answer"] else 0
        for entry in expert_iteration_data
    ]

    # Extract and smooth the average log prob data (negated aggregate loss)
    average_log_prob = [-entry["Aggregate loss"] for entry in expert_iteration_data]

    # Extract gradient norm values
    gradient_norms = [entry.get("Gradient Norm", 0) for entry in expert_iteration_data]

    # Print debugging information
    print(f"Length of batch_indices: {len(batch_indices)}")
    print(f"Length of reasoning_contains_answer: {len(reasoning_contains_answer)}")
    print(f"Length of average_log_prob: {len(average_log_prob)}")

    # Only smooth if we have enough data points
    if len(batch_indices) >= window_size:
        smoothed_data_reasoning = smooth_data(reasoning_contains_answer, window_size)
        smoothed_data_log_prob = smooth_data(average_log_prob, window_size)
        smoothed_data_grad_norm = smooth_data(gradient_norms, window_size)
    else:
        print(f"Warning: Not enough data points for smoothing. Using raw data.")
        smoothed_data_reasoning = reasoning_contains_answer
        smoothed_data_log_prob = average_log_prob
        smoothed_data_grad_norm = gradient_norms

    # Create a new plot with raw data, smoothed data, training example indicators, average log prob, and gradient norm
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Calculate the number of padded points to exclude
    exclude_points = window_size // 2 if len(batch_indices) >= window_size else 0

    # Ensure all data arrays have the same length
    plot_length = min(
        len(batch_indices),
        len(reasoning_contains_answer),
        len(smoothed_data_reasoning),
        len(smoothed_data_log_prob),
        len(smoothed_data_grad_norm),
    )

    # Plot raw data
    ax1.plot(
        batch_indices[:plot_length],
        reasoning_contains_answer[:plot_length],
        marker="o",
        linestyle="",
        markersize=2,
        alpha=0.3,
        label="Raw Data",
    )

    # Plot smoothed data
    ax1.plot(
        batch_indices[exclude_points:plot_length],
        smoothed_data_reasoning[: plot_length - exclude_points],
        color="red",
        linewidth=2,
        label="Smoothed Data",
    )

    # Extract training example data if available
    training_example = [
        1 if entry.get("Training Example") == "True" else 0
        for entry in expert_iteration_data
    ]

    # Plot training example indicators if available
    if any(training_example):
        ax1.scatter(
            [
                batch_indices[i]
                for i in range(len(batch_indices) - exclude_points)
                if training_example[i] == 1
            ],
            [1.05] * sum(training_example[:-exclude_points]),
            marker="^",
            color="green",
            s=20,
            label="Training Example",
        )

    ax1.set_xlabel("Batch Index")
    ax1.set_ylabel("Reasoning Contains Answer (1: True, 0: False)")
    ax1.set_ylim(-0.1, 1.1)

    # Create a second y-axis for average log prob
    ax2.plot(
        batch_indices[exclude_points:plot_length],
        smoothed_data_log_prob[: plot_length - exclude_points],
        color="purple",
        linewidth=2,
        label="Smoothed Average Log Prob",
    )
    ax1.set_xlabel("Batch Index")
    ax2.set_ylabel("Average Log Prob")
    ax2.set_ylim(top=0)

    # Plot gradient norm on ax3
    ax3.plot(
        batch_indices[exclude_points:plot_length],
        smoothed_data_grad_norm[: plot_length - exclude_points],
        color="orange",
        linewidth=2,
        label="Smoothed Gradient Norm",
    )
    ax3.set_xlabel("Batch Index")
    ax3.set_ylabel("Gradient Norm")
    ax3.legend()

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.15, 0.9),
    )

    plt.title(
        f"Reasoning Contains Answer, Average Log Prob, and Gradient Norm vs Batch Index\n"
        f"(Window Size: {window_size})"
    )
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Save the new plot
    plt.savefig(output_file)
    plt.close()

    print(
        f"Smoothed plot with training example indicators, average log prob, and gradient norm "
        f"saved as {output_file}"
    )

    print(f"Plot length: {plot_length}")
    print(f"Exclude points: {exclude_points}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Expert Iteration results.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Window size for moving average (default: 16)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="src/AnalyzeResults/ei_plot.png",
        help="Output file path for the plot (default: src/AnalyzeResults/ei_plot.png)",
    )
    args = parser.parse_args()

    latest_log_file = get_latest_log_file()
    plot_metrics(
        latest_log_file, window_size=args.window_size, output_file=args.output_file
    )
    print(f"Plot saved as {args.output_file} with window size {args.window_size}")
