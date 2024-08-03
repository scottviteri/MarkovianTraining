import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

"""
{"Aggregate loss": 5.644328594207764, "Batch Index": 0, "Prev Observation": "Question: 64 + 77 + 53 + 93 + 46 + 38 + 79 + 28 + 96 + 5 + 73 + 72 + 60 + 8 + 75", "Action": "Reasoning: \n\n1. Add the first two numbers: 64 + 77 = 141\n2. Add the next two numbers: 141 + 53 = 194\n3. Add the next two numbers: 194 + 93 = 287\n4. Add the next two numbers: 287 + 46 = 333\n5. Add the next two numbers: 333 + 38 = 371\n6. Add the next two numbers: 371 + 79 = 449\n7. Add the next two numbers: 449 + 28 = 477\n8. Add the next two numbers: 477 + 96 = 573\n9. Add the final two numbers: 573 + 5 = 578\n\nAnswer: The sum of all the given numbers is 578.\n\nIn total, the problem was broken down into 8 subproblems, each requiring a single addition operation. After solving each subproblem, we could add the result to the overall sum. This method simplified the original problem and made it easier to understand and solve. By decomposing problems into smaller parts and addressing each subproblem individually, complex problems can be effectively tackled.\n\nIn a nutshell, solve the problem as follows: \n\nStep 1: Add the first two numbers (64 and 77)\nStep 2: Find the sum of the result and the next two numbers\nStep 3: Repeat the process until we find the sum of all the given numbers.\n\nOverall, this process is known as the repeated subtraction method for addition, which is an efficient and streamlined approach towards solving complex addition problems. It enables", "Observation": "Answer: 867", "Reasoning Contains Answer": false, "Training Example": "False"}
{"Aggregate loss": 2.581582546234131, "Batch Index": 1, "Prev Observation": "Question: 22 + 98 + 43 + 93 + 36 + 84 + 33 + 76 + 48 + 45 + 33 + 10 + 24 + 39 + 4", "Action": "Reasoning: \n\n1. Addition of two numbers: 22 + 98\n2. Carry the sum to the next column if necessary and add the next numbers: 11 + 43\n3. Carry the sum to the next column if necessary and add the next numbers: 111 + 93\n4. Carry the sum to the next column if necessary and add the next numbers: 127 + 36\n5. Carry the sum to the next column if necessary and add the next numbers: 163 + 84\n6. Carry the sum to the next column if necessary and add the next numbers: 247 + 33\n7. Carry the sum to the next column if necessary and add the next numbers: 280 + 76\n8. Carry the sum to the next column if necessary and add the next numbers: 356 + 48\n9. Carry the sum to the next column if necessary and add the next numbers: 404 + 45\n10. Carry the sum to the next column if necessary and add the next numbers: 449 + 33\n11. Carry the sum to the next column if necessary and add the next numbers: 482 + 10\n12. Carry the sum to the next column if necessary and add the next numbers: 492 + 24\n13. Carry the sum to the next column if necessary and add the last numbers: 516 + 39\n\nSo, 22 + 98 + 43 + 93 + 36 + 84 + 33 + 76 + 48 + 45 + 33 + 10 + 24 +", "Observation": "Answer: 688", "Reasoning Contains Answer": false, "Training Example": "True"}
...
"""
# Load the log file into a list of dictionaries
with open(
    max(
        glob.glob("src/AnalyzeResults/ExpertIterationDictionary_*.log"),
        key=os.path.getctime,
    ),
    "r",
) as file:
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

# Create the plot
# plt.figure(figsize=(12, 6))
# plt.plot(
#    batch_indices, reasoning_contains_answer, marker="o", linestyle="", markersize=2
# )
# plt.title("Reasoning Contains Answer vs Batch Index")
# plt.xlabel("Batch Index")
# plt.ylabel("Reasoning Contains Answer (1: True, 0: False)")
# plt.ylim(-0.1, 1.1)  # Set y-axis limits to show clear separation between 0 and 1
# plt.grid(True, linestyle="--", alpha=0.7)
#
## Save the plot
# plt.savefig("src/AnalyzeResults/reasoning_contains_answer_plot.png")
# plt.close()
#
# print("Plot saved as reasoning_contains_answer_plot.png")


def smooth_data(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


# Determine the maximum window size
max_window_size = 4  # Using the larger of the two original window sizes

# Smooth the reasoning contains answer data
padded_data_reasoning = np.pad(
    reasoning_contains_answer,
    (max_window_size // 2, max_window_size // 2),
    mode="edge",
)
smoothed_data_reasoning = smooth_data(padded_data_reasoning, max_window_size)[:-1]

# Extract and smooth the average log prob data (negated aggregate loss)
average_log_prob = [-entry["Aggregate loss"] for entry in expert_iteration_data]
padded_data_log_prob = np.pad(
    average_log_prob, (max_window_size // 2, max_window_size // 2), mode="edge"
)
smoothed_data_log_prob = smooth_data(padded_data_log_prob, max_window_size)[:-1]

# Extract training example data
training_example = [
    1 if entry["Training Example"] == "True" else 0 for entry in expert_iteration_data
]

# Create a new plot with raw data, smoothed data, training example indicators, and average log prob
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot reasoning contains answer data
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
    smoothed_data_reasoning[exclude_points:-exclude_points],
    color="red",
    linewidth=2,
    label="Smoothed Data",
)

# Plot training example indicators excluding padded points
ax1.scatter(
    [
        batch_indices[i]
        for i in range(len(batch_indices) - exclude_points)
        if training_example[i] == 1
    ],
    [1.05]
    * sum(
        training_example[:-exclude_points]
    ),  # Slightly above the top of the plot
    marker="^",
    color="green",
    s=20,
    label="Training Example",
)

ax1.set_xlabel("Batch Index")
ax1.set_ylabel("Reasoning Contains Answer (1: True, 0: False)")
ax1.set_ylim(
    -0.1, 1.15
)  # Increased upper limit to accommodate training example indicators

# Create a second y-axis for average log prob
ax2 = ax1.twinx()
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_log_prob[exclude_points:-exclude_points],
    color="purple",
    linewidth=2,
    label="Smoothed Average Log Prob",
)
ax2.set_ylabel("Average Log Prob")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.15, 0.9))

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
    "Smoothed plot with training example indicators and average log prob saved as smoothed_reasoning_and_log_prob_plot.png"
)
