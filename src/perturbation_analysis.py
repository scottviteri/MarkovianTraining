import os
import json
import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter


def find_latest_log_file():
    """
    Find the most recent log.jsonl file in the checkpoints directory.

    Returns:
        str: Path to the most recent log file, or None if no log file found
    """
    checkpoints_dir = "checkpoints"
    log_files = []

    # Walk through all subdirectories in checkpoints
    for root, dirs, files in os.walk(checkpoints_dir):
        for file in files:
            if file == "log.jsonl":
                full_path = os.path.join(root, file)
                log_files.append((os.path.getmtime(full_path), full_path))

    # Sort by modification time and return the most recent
    if log_files:
        return sorted(log_files, key=lambda x: x[0], reverse=True)[0][1]

    return None


def run_perturbations(log_files):
    """
    Run perturbation analysis on the given log files.

    Args:
        log_files (list): List of log file paths to analyze

    Returns:
        dict: Perturbation analysis results
    """
    perturbation_results = {"files": log_files, "perturbations": []}

    for file in log_files:
        with open(file, "r") as f:
            # Assuming log file contains JSON lines with perturbation data
            file_results = [json.loads(line) for line in f]
            perturbation_results["perturbations"].append(file_results)

    return perturbation_results


def save_perturbation_results(results, output_file=None):
    """
    Save perturbation analysis results to a JSON file.

    Args:
        results (dict): Perturbation analysis results
        output_file (str, optional): Path to save the results
    """
    if output_file is None:
        # Create a timestamped output file in results/perturbations/
        os.makedirs("results/perturbations", exist_ok=True)
        output_file = f"results/perturbations/perturbation_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Perturbation analysis results saved to {output_file}")


def plot_perturbation_results(log_files, window_size=40):
    """
    Plot the results of perturbation analysis.

    Args:
        log_files (list): List of log file paths to plot
        window_size (int): Smoothing window size
    """
    all_data = []
    for log_file in log_files:
        # Check if a pre-existing analysis result file exists
        result_file = os.path.join(
            "results/perturbations",
            f"analysis_results_{os.path.basename(log_file)}.json",
        )

        if os.path.exists(result_file):
            # If pre-existing result file exists, load it
            with open(result_file, "r") as f:
                all_data.append(json.load(f))
        else:
            # If no pre-existing file, process the log file directly
            print(
                f"No pre-existing analysis found for {log_file}. Processing log file directly."
            )

            # Process the log file to extract perturbation data
            with open(log_file, "r") as f:
                log_data = [json.loads(line) for line in f]

            # Extract perturbation-related metrics
            # This is a placeholder - you'll need to adapt this to your specific log file structure
            perturbation_data = []
            for entry in log_data:
                # Example extraction - modify based on your actual log file structure
                pert_entry = {
                    "Avg Log Probs": {
                        "Original": entry.get("Avg Log Prob", 0),
                        # Add other perturbation types as needed
                    }
                }
                perturbation_data.append(pert_entry)

            all_data.append(perturbation_data)

    # Check if we have any data to plot
    if not all_data:
        print("No data found to plot.")
        return

    # Find the minimum length among all datasets
    min_length = min(len(data) for data in all_data)

    # Initialize the averaged data
    # Dynamically extract perturbation types from the first dataset
    averaged_data = {
        pert: [] for pert in all_data[0][0]["Avg Log Probs"].keys() if pert != "Llama"
    }

    # Calculate the average across all datasets
    for i in range(min_length):
        for pert in averaged_data.keys():
            values = [
                -data[i]["Avg Log Probs"][pert]
                - (-data[i]["Avg Log Probs"]["Original"])
                for data in all_data
            ]
            averaged_data[pert].append(np.mean(values))

    plt.figure(figsize=(12, 6))

    # Generate a color for each perturbation type
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_index = 0

    for pert, values in averaged_data.items():
        if pert != "Original":  # Skip plotting the Original, as it will always be 0
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
                    label=pert,
                    color=colors[color_index % len(colors)],
                    linewidth=2,
                )
            else:
                # If we can't smooth, plot the original values
                plt.plot(
                    values,
                    label=pert,
                    color=colors[color_index % len(colors)],
                    linewidth=2,
                )

            color_index += 1

    plt.xlabel("Sample", fontsize=16)
    plt.ylabel("Average Difference in Negated Log Probability", fontsize=16)
    plt.title(
        f"Average Perturbation Results (Smoothing Window: {window_size})", fontsize=16
    )
    plt.legend(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Increase font size for tick labels
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Ensure results directory exists
    os.makedirs("results/plots", exist_ok=True)
    output_file = os.path.join(
        "results/plots", f"average_perturbation_results_plot_smooth{window_size}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Perturbation Analysis Tool")
    parser.add_argument("log_files", nargs="*", help="Log files to analyze (optional)")
    parser.add_argument("--plot", action="store_true", help="Plot perturbation results")
    parser.add_argument("--output", type=str, help="Output file for analysis results")

    args = parser.parse_args()

    # If no log files provided, find the most recent one
    if not args.log_files:
        latest_log_file = find_latest_log_file()
        if latest_log_file:
            args.log_files = [latest_log_file]
            print(f"Using most recent log file: {latest_log_file}")
        else:
            print("No log files found in checkpoints directory.")
            return

    # Resolve log file paths
    resolved_log_files = []
    for file in args.log_files:
        if not os.path.exists(file):
            # Try to find in checkpoints directory
            checkpoint_path = os.path.join("checkpoints", file)
            if os.path.exists(checkpoint_path):
                file = checkpoint_path
        resolved_log_files.append(file)

    # Run analysis
    results = run_perturbations(resolved_log_files)

    if args.plot:
        plot_perturbation_results(resolved_log_files)
    else:
        save_perturbation_results(results, args.output)


if __name__ == "__main__":
    main()
