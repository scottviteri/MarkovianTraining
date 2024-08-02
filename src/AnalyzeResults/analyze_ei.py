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
    expert_iteration_data = [json.loads(line) for line in file if line.strip()]

print(f"Loaded {len(expert_iteration_data)} entries from the log file.")
print("First entry:", expert_iteration_data[0])


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


# Smooth the data
window_size = 4  # Adjust this value to change the smoothing level
half_window = window_size // 2
padded_data = np.pad(reasoning_contains_answer, (half_window, half_window), mode="edge")
smoothed_data = smooth_data(padded_data, window_size)[:-1]

# Create a new plot with both raw and smoothed data
plt.figure(figsize=(12, 6))
plt.plot(
    batch_indices,
    reasoning_contains_answer,
    marker="o",
    linestyle="",
    markersize=2,
    alpha=0.3,
    label="Raw Data",
)
plt.plot(batch_indices, smoothed_data, color="red", linewidth=2, label="Smoothed Data")
plt.title(f"Reasoning Contains Answer vs Batch Index (Window Size: {window_size})")
plt.xlabel("Batch Index")
plt.ylabel("Reasoning Contains Answer (1: True, 0: False)")
plt.ylim(-0.1, 1.1)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Save the new plot
plt.savefig("src/AnalyzeResults/smoothed_reasoning_contains_answer_plot.png")
plt.close()

print("Smoothed plot saved as smoothed_reasoning_contains_answer_plot.png")
