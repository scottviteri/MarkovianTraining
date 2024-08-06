import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def smooth_data(data, window_size):
    return moving_average(data, window_size)


def get_latest_log_file():
    log_files = glob.glob("src/AnalyzeResults/ExpertIterationDictionary_*.log")
    if not log_files:
        raise FileNotFoundError("No ExpertIterationDictionary log files found.")
    return max(log_files, key=os.path.getctime)


# Set the window size for smoothing
max_window_size = 16

# Load the log file into a list of dictionaries
log_file_path = get_latest_log_file()
with open(log_file_path, "r") as file:
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
    1 if entry["Reasoning Contains Answer"] else 0 for entry in expert_iteration_data
]

# Extract and smooth the average log prob data (negated aggregate loss)
average_log_prob = [-entry["Aggregate loss"] for entry in expert_iteration_data]
smoothed_data_log_prob = smooth_data(average_log_prob, max_window_size)

# Extract training example data if available
training_example = [
    1 if entry.get("Training Example") == "True" else 0
    for entry in expert_iteration_data
]

# Smooth the reasoning contains answer data
smoothed_data_reasoning = smooth_data(reasoning_contains_answer, max_window_size)

# Create a new plot with raw data, smoothed data, training example indicators (if available), and average log prob
fig, ax1 = plt.subplots(figsize=(12, 6))

# Calculate the number of padded points to exclude
exclude_points = max_window_size // 2

# Plot raw data excluding padded points
ax1.plot(
    batch_indices[exclude_points:-exclude_points],
    reasoning_contains_answer[exclude_points:-exclude_points],
    marker="o",
    linestyle="",
    markersize=2,
    alpha=0.3,
    label="Raw Data",
)

# Plot smoothed data excluding padded points
ax1.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_reasoning[:-exclude_points],
    color="red",
    linewidth=2,
    label="Smoothed Data",
)

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
ax1.set_ylim(-0.1, 1.15)

# Create a second y-axis for average log prob
ax2 = ax1.twinx()
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_log_prob[:-exclude_points],
    color="purple",
    linewidth=2,
    label="Smoothed Average Log Prob",
)
ax2.set_ylabel("Average Log Prob")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.15, 0.9)
)

plt.title(
    f"Reasoning Contains Answer and Average Log Prob vs Batch Index\n(Window Size: {max_window_size})"
)
plt.grid(True, linestyle="--", alpha=0.7)

# Adjust the layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
# Save the new plot
plt.savefig("src/AnalyzeResults/smoothed_reasoning_and_log_prob_plot.png")
plt.close()

print(
    "Smoothed plot with training example indicators (if available) and average log prob saved as smoothed_reasoning_and_log_prob_plot.png"
)
