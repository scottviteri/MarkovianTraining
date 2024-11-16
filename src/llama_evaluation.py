import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from train import find_latest_result


def run_llama_evaluation(log_files):
    """
    Evaluate log files using Llama as the evaluator.

    Args:
        log_files (list): List of log file paths to evaluate

    Returns:
        dict: Llama evaluation results
    """
    llama_results = {"files": log_files, "evaluations": []}

    for file in log_files:
        with open(file, "r") as f:
            # Assuming log file contains JSON lines with evaluation data
            file_results = [json.loads(line) for line in f]
            llama_results["evaluations"].append(file_results)

    return llama_results


def plot_original_vs_llama(llama_results, log_file, window_size=40):
    all_data = llama_results["evaluations"]

    # Find the minimum length among all datasets
    min_length = min(len(data) for data in all_data)

    # Initialize the averaged data
    averaged_data = {"Original": [], "Llama": []}

    # Calculate the average across all datasets
    for i in range(min_length):
        original_values = [data[i]["Avg Log Probs"]["Original"] for data in all_data]
        llama_values = [data[i]["Avg Log Probs"]["Llama"] for data in all_data]
        averaged_data["Original"].append(np.mean(original_values))
        averaged_data["Llama"].append(np.mean(llama_values))

    plt.figure(figsize=(12, 6))
    colors = ["#e41a1c", "#377eb8"]

    for model, values in averaged_data.items():
        if len(values) > window_size:
            # Apply Savitzky-Golay filter for smoothing
            smoothed_values = savgol_filter(values, window_size, 3)

            # Only plot the central part not affected by edge effects
            half_window = window_size // 2
            x_values = range(half_window, len(smoothed_values) - half_window)
            y_values = smoothed_values[half_window:-half_window]

            plt.plot(
                x_values,
                y_values,
                label=model,
                color=colors[0] if model == "Original" else colors[1],
                linewidth=2,
            )
        else:
            # If we can't smooth, plot the original values
            plt.plot(
                values,
                label=model,
                color=colors[0] if model == "Original" else colors[1],
                linewidth=2,
            )

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Average Log Probability", fontsize=16)
    plt.title(
        f"Average Original vs Llama Results (Smoothing Window: {window_size})",
        fontsize=16,
    )
    plt.legend(fontsize=20, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Increase font size for tick labels
    plt.tick_params(axis="both", which="major", labelsize=14)

    output_file = os.path.join(
        os.path.dirname(log_file),
        f"average_original_vs_llama_plot_smooth{window_size}.png",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Llama Evaluation Tool")
    parser.add_argument("--log_file", help="Log file to evaluate")

    args = parser.parse_args()

    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_result(return_log=True)

    if not log_file:
        print("No log file found.")
        return

    print(f"Using log file: {log_file}")

    # Run evaluation
    results = run_llama_evaluation([log_file])

    # Plot original vs llama results
    plot_original_vs_llama(results, log_file)


if __name__ == "__main__":
    main()
