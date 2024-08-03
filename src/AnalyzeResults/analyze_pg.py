import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

"""
{"hyperparameters": {"learning_rate": 0.0001, "batch_size": 6, "gradient_accumulation_steps": 8, "effective_batch_size": 48, "num_batches": 10000}}
{"Aggregate loss": -0.8916961550712585, "Batch Index": 0, "Prev Observation": "Question: 71 + 76 + 77 + 31 + 33 + 91 + 76 + 7 + 76 + 13 + 55 + 87 + 16 + 81 + 65", "Action": "StepByStep: \n1. Add the first three numbers: 71 + 76 + 77 = 224\n2. Add the next three numbers: 31 + 33 + 91 = 155\n3. Add the next two numbers: 76 + 7 = 83\n4. Add the next number: 76 = 76\n5. Add the next number: 13 = 13\n6. Add the next number: 55 = 55\n7. Add the next number: 87 = 87\n8. Add the next number: 16 = 16\n9. Add the next number: 81 = 81\n10. Add the next number: 65 = 65\n11. Add the numbers calculated in steps 1-10: 224 + 155 + 83 + 76 + 13 + 55 + 87 + 16 + 81 + 65 = 1316\n\nTherefore, the sum of all the given numbers is 1316.\n\nConfidence: 95%. It's possible that I've made a mistake due to typing errors, but I've followed each step precisely and performed them in the correct order. The final answer has been double-checked for accuracy.\n\nAnswer: The sum of all the given numbers is 1316.\n\nLet me know if you need further assistance or explanation. (I'm here to help!) Have a great day! :) - The friendly AI assistant! :)\n\np.s. Remember, keep breaking down complex problems into smaller, manageable tasks, and you'll tackle even the toughest challenges with ease! :) - The wise", "Observation": "Answer: 855", "Reasoning Contains Answer": false, "Avg Log Prob": -4.864299297332764, "Baseline Avg Log Prob": -4.202348709106445, "Advantage": -0.6619505882263184}
{"Aggregate loss": -0.28632110357284546, "Batch Index": 1, "Prev Observation": "Question: 13 + 33 + 99 + 42 + 97 + 71 + 21 + 59 + 77 + 26 + 3 + 58 + 52 + 73 + 45", "Action": "StepByStep: \n1. Add the first two numbers: 13 + 33 = 46\n2. Add the next number to the result: 46 + 99 = 145\n3. Repeat this process for the remaining numbers: 145 + 42 = 187, 187 + 97 = 284, 284 + 71 = 355, 355 + 21 = 376, 376 + 59 = 435, 435 + 77 = 512, 512 + 26 = 538, 538 + 3 = 541, 541 + 58 = 609, 609 + 52 = 661, 661 + 73 = 734, 734 + 45 = 779\n\nSo, the sum of all the given numbers is 779.\nAnswer: \u27a2 The sum of all given numbers is 779. \ud83c\udf89\nLet me know if you need help with anything else! \ud83d\ude0a#MathProblems #StepByStepSolutions #Addition #SumOfNumbers #MathHelp #Education #Learning #StudentLife #StudyTips #MathTutor #Numbers #Sum #Decomposition #CrispCommunication #EffectiveStudy #LearnMathEasy #MentalMath #Skills #MathIsFun #TeachingMath #BrainPower #CriticalThinking #ProblemSolving #QuickMath #OnlineMathHelp #MathFacts #LearnMathFast #MathMaster #N", "Observation": "Answer: 769", "Reasoning Contains Answer": false, "Avg Log Prob": -6.650873184204102, "Baseline Avg Log Prob": -3.574218273162842, "Advantage": -3.0766549110412598}
...
"""
max_window_size = 8

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

# Extract and smooth the average log prob data, baseline log prob data, and advantage data
avg_log_probs = [entry["Avg Log Prob"] for entry in expert_iteration_data]
baseline_avg_log_probs = [entry["Baseline Avg Log Prob"] for entry in expert_iteration_data]
initial_advantages = [entry["Initial Advantage"] for entry in expert_iteration_data]
final_advantages = [entry["Final Advantage"] for entry in expert_iteration_data]

padded_data_log_prob = np.pad(
    avg_log_probs, (max_window_size // 2, max_window_size // 2), mode="edge"
)
smoothed_data_log_prob = smooth_data(padded_data_log_prob, max_window_size)[:-1]

padded_data_baseline_log_prob = np.pad(
    baseline_avg_log_probs, (max_window_size // 2, max_window_size // 2), mode="edge"
)
smoothed_data_baseline_log_prob = smooth_data(padded_data_baseline_log_prob, max_window_size)[:-1]

padded_data_initial_advantage = np.pad(
    initial_advantages, (max_window_size // 2, max_window_size // 2), mode="edge"
)
smoothed_data_initial_advantage = smooth_data(padded_data_initial_advantage, max_window_size)[:-1]

padded_data_final_advantage = np.pad(
    final_advantages, (max_window_size // 2, max_window_size // 2), mode="edge"
)
smoothed_data_final_advantage = smooth_data(padded_data_final_advantage, max_window_size)[:-1]

# Create a new plot with raw data, smoothed data, log prob series, and advantage
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

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
    smoothed_data_baseline_log_prob[exclude_points:-exclude_points],
    color="green",
    linewidth=2,
    label="Smoothed Baseline Avg Log Prob",
)
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_initial_advantage[exclude_points:-exclude_points],
    color="orange",
    linewidth=2,
    label="Smoothed Initial Advantage",
)
ax2.plot(
    batch_indices[exclude_points:-exclude_points],
    smoothed_data_final_advantage[exclude_points:-exclude_points],
    color="red",
    linewidth=2,
    label="Smoothed Final Advantage",
)

ax2.set_xlabel("Batch Index")
ax2.set_ylabel("Log Prob / Advantage")
ax2.legend(loc="upper left")

plt.suptitle(
    f"Reasoning Contains Answer, Log Probs, and Advantage vs Batch Index\n(Window Size: {max_window_size})"
)
plt.grid(True, linestyle="--", alpha=0.7)

# Adjust the layout to make room for the legend
plt.tight_layout()
# Save the new plot
plt.savefig("src/AnalyzeResults/pg_plot.png")
plt.close()

print(
    "Smoothed plot with average log prob, baseline log prob, and advantage saved as pg_plot.png"
)