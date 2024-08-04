import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

max_window_size = 4

# Load the log file into a list of dictionaries
with open(
    max(
        glob.glob("src/AnalyzeResults/PolicyGradientDictionary_*.log"),
        key=os.path.getctime,
    ),
    "r",
) as file:
    # Read the first line as hyperparameters
    hyperparameters = json.loads(file.readline().strip())
    # Read the rest of the lines as expert iteration data
    expert_iteration_data = [json.loads(line) for line in file if line.strip()]

print(f"Loaded hyperparameters: {hyperparameters}")
print(f"Using PPO: {hyperparameters.get('use_ppo', False)}")
print(f"PPO Epsilon: {hyperparameters.get('ppo_epsilon', 'N/A')}")
print(f"Loaded {len(expert_iteration_data)} entries from the log file.")
print("First data entry:", expert_iteration_data[0])

# Extract batch indices and reasoning contains answer values
batch_indices = [entry["Batch Index"] for entry in expert_iteration_data]
reasoning_contains_answer = [
    1 if entry["Reasoning Contains Answer"] else 0 for entry in expert_iteration_data
]


def smooth_data(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


# Smooth the reasoning contains answer data
padded_data_reasoning = np.pad(
    reasoning_contains_answer,
    (max_window_size // 2, max_window_size // 2),
    mode="edge",
)
smoothed_data_reasoning = smooth_data(padded_data_reasoning, max_window_size)[:-1]

# Extract and smooth the average log prob data, value predictions, and advantage data
avg_log_probs = [entry["Avg Log Prob"] for entry in expert_iteration_data]
value_predictions = [entry["Value Prediction"] for entry in expert_iteration_data]
advantages = [entry["Advantage"] for entry in expert_iteration_data]
value_losses = [entry["Value Loss"] for entry in expert_iteration_data]
initial_advantages = [entry["Initial Advantage"] for entry in expert_iteration_data]

# Add these lines to extract PPO-specific data if available
if hyperparameters.get("use_ppo", False):
    ratios = [entry.get("PPO Ratio", 1.0) for entry in expert_iteration_data]
    clipped_ratios = [
        entry.get("PPO Clipped Ratio", 1.0) for entry in expert_iteration_data
    ]

# Smooth all the data
smoothed_data_log_prob = smooth_data(
    np.pad(avg_log_probs, (max_window_size // 2, max_window_size // 2), mode="edge"),
    max_window_size,
)[:-1]
smoothed_data_value_pred = smooth_data(
    np.pad(
        value_predictions, (max_window_size // 2, max_window_size // 2), mode="edge"
    ),
    max_window_size,
)[:-1]
smoothed_data_advantage = smooth_data(
    np.pad(advantages, (max_window_size // 2, max_window_size // 2), mode="edge"),
    max_window_size,
)[:-1]
smoothed_data_value_loss = smooth_data(
    np.pad(value_losses, (max_window_size // 2, max_window_size // 2), mode="edge"),
    max_window_size,
)[:-1]
smoothed_data_initial_advantage = smooth_data(
    np.pad(
        initial_advantages, (max_window_size // 2, max_window_size // 2), mode="edge"
    ),
    max_window_size,
)[:-1]

# Add these lines for PPO data smoothing
if hyperparameters.get("use_ppo", False):
    smoothed_data_ratio = smooth_data(
        np.pad(ratios, (max_window_size // 2, max_window_size // 2), mode="edge"),
        max_window_size,
    )[:-1]
    smoothed_data_clipped_ratio = smooth_data(
        np.pad(
            clipped_ratios, (max_window_size // 2, max_window_size // 2), mode="edge"
        ),
        max_window_size,
    )[:-1]

# Create a new plot with raw data, smoothed data, log prob series, and advantage
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

# Calculate the number of padded points to exclude
exclude_points = max_window_size // 2

# Plot on the first subplot (ax1)
ax1.plot(
    batch_indices[exclude_points:-exclude_points],
    reasoning_contains_answer[exclude_points:-exclude_points],
    marker="o",
    linestyle="",
    markersize=2,
    alpha=0.3,
    label="Raw Data",
)

ax1.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_reasoning[exclude_points:-exclude_points],
    color="red",
    linewidth=2,
    label="Smoothed Data",
)

ax1.set_ylabel("Reasoning Contains Answer (1: True, 0: False)")
ax1.set_ylim(-0.1, 1.15)
ax1.legend(loc="upper left")

# Plot on the second subplot (ax2)
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_log_prob[exclude_points:-exclude_points],
    color="purple",
    linewidth=2,
    label="Smoothed Avg Log Prob",
)
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_value_pred[exclude_points:-exclude_points],
    color="green",
    linewidth=2,
    label="Smoothed Value Prediction",
)
ax2.set_ylabel("Log Prob / Value Prediction")
ax2.legend(loc="upper left")

# Plot on the third subplot (ax3)
ax3.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_initial_advantage[exclude_points:-exclude_points],
    color="blue",
    linewidth=2,
    label="Smoothed Initial Advantage",
)
ax3.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_advantage[exclude_points:-exclude_points],
    color="orange",
    linewidth=2,
    label="Smoothed Advantage",
)
ax3.set_ylabel("Initial Advantage / Advantage")
ax3.legend(loc="upper left")

# Plot on the fourth subplot (ax4)
ax4.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_value_loss[exclude_points:-exclude_points],
    color="red",
    linewidth=2,
    label="Smoothed Value Loss",
)

if hyperparameters.get("use_ppo", False):
    ax4.plot(
        batch_indices[exclude_points:-exclude_points],
        smoothed_data_ratio[exclude_points:-exclude_points],
        color="cyan",
        linewidth=2,
        label="Smoothed PPO Ratio",
    )
    ax4.plot(
        batch_indices[exclude_points:-exclude_points],
        smoothed_data_clipped_ratio[exclude_points:-exclude_points],
        color="magenta",
        linewidth=2,
        label="Smoothed PPO Clipped Ratio",
    )

ax4.set_xlabel("Batch Index")
ax4.set_ylabel("Value Loss / PPO Ratios")
ax4.legend(loc="upper left")

# Update the plot title
algorithm_name = "PPO" if hyperparameters.get("use_ppo", False) else "Policy Gradient"
plt.suptitle(
    f"{algorithm_name}: Training Metrics vs Batch Index\n"
    f"(Window Size: {max_window_size})"
)

plt.grid(True, linestyle="--", alpha=0.7)

# Adjust the layout to make room for the legend
plt.tight_layout()
# Save the new plot
plt.savefig("src/AnalyzeResults/pg_plot.png")
plt.close()

print(f"Smoothed plot for {algorithm_name} with training metrics saved as pg_plot.png")
