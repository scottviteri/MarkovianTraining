import re
from typing import List, Dict
import matplotlib.pyplot as plt
from typing import List
from torchtyping import TensorType
import numpy as np


def parse_markovian_run_log(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        content = file.read()
    entries = content.split("\n\n")
    parsed_data = []

    for entry in entries:
        entry_dict = {}
        lines = entry.strip().split("\n")

        # Parse the lines
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                entry_dict[key.strip()] = value.strip()

        # Check if "Action" and "Observation" are in the entry
        if "Action" in entry_dict and "Observation" in entry_dict:
            answer_in_action = check_answer_in_action(entry_dict)
            entry_dict["answer_in_action"] = answer_in_action

        # Check if "Default Action" and "Observation" are in the entry
        if "Default Action" in entry_dict and "Observation" in entry_dict:
            default_answer_in_action = check_answer_in_action(
                entry_dict, default_action=True
            )
            entry_dict["default_answer_in_action"] = default_answer_in_action

        parsed_data.append(entry_dict)

    return parsed_data


def check_answer_in_action(entry, default_action=False) -> bool:
    observation = entry["Observation"]
    action = entry["Default Action"] if default_action else entry["Action"]

    # Extract the answer from the Observation
    answer_match = re.search(r"Answer: (\d+)", observation)
    if answer_match:
        answer = answer_match.group(1)

        # Check if the answer is in the Action
        if answer in action:
            return True

    return False


def calculate_answer_in_action_fractions(parsed_data, n):
    # Sort the parsed data by batch index
    sorted_data = sorted(parsed_data, key=lambda x: int(x.get("Batch Index", 0)))

    # Initialize the lists to store fractions and counts
    fractions = []
    counts = []
    default_action_fractions = []

    # Get the maximum batch index
    max_batch_index = int(sorted_data[-1].get("Batch Index", 0))

    # Iterate through the data in chunks of size n
    for i in range(0, max_batch_index + 1, n):
        chunk = [
            entry
            for entry in sorted_data
            if i <= int(entry.get("Batch Index", 0)) < i + n
        ]

        # Count entries where answer_in_action is True
        true_count = sum(1 for entry in chunk if entry.get("answer_in_action", False))

        # Calculate the fraction
        fraction = true_count / len(chunk) if chunk else 0
        fractions.append(fraction)

        # Count entries that have an answer_in_action entry
        count = sum(1 for entry in chunk if "answer_in_action" in entry)
        counts.append(float(count) / n)

        # Calculate fraction of default actions containing the answer
        default_true_count = sum(
            1 for entry in chunk if entry.get("default_answer_in_action", False)
        )
        default_fraction = default_true_count / len(chunk) if chunk else 0
        default_action_fractions.append(default_fraction)

    return fractions, counts, default_action_fractions


def calculate_smoothed_fractions(parsed_data, n):
    # Sort the parsed data by batch index
    sorted_data = sorted(parsed_data, key=lambda x: int(x.get("Batch Index", 0)))

    # Filter data points with all required attributes
    filtered_data = [
        x
        for x in sorted_data
        if "Batch Index" in x
        and "answer_in_action" in x
        and "default_answer_in_action" in x
    ]

    answer_in_actions = [float(x["answer_in_action"]) for x in filtered_data]
    default_answer_in_actions = [
        float(x["default_answer_in_action"]) for x in filtered_data
    ]
    batch_indices = [int(x["Batch Index"]) for x in filtered_data]

    def smooth_data(data: TensorType["N"], window_size: int) -> TensorType["N"]:
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    half_window = n // 2
    padded_answer = np.pad(answer_in_actions, (half_window, half_window), mode="edge")
    padded_default = np.pad(
        default_answer_in_actions, (half_window, half_window), mode="edge"
    )

    smoothed_answer: TensorType["N"] = smooth_data(padded_answer, n)[:-1]
    smoothed_default: TensorType["N"] = smooth_data(padded_default, n)[:-1]

    return smoothed_answer.tolist(), smoothed_default.tolist(), batch_indices


# Example usage
log_data = parse_markovian_run_log("MarkovianCleaned.log")
print(f"Parsed {len(log_data)} entries from the log file.")
print("First entry:", log_data[0])

# Example usage
n = 20  # You can adjust this parameter as needed
smoothed_fractions, smoothed_default_fractions, batch_indices = (
    calculate_smoothed_fractions(log_data, n)
)
print(f"Smoothed fractions (n={n}):")
for i, (fraction, default_fraction, batch_index) in enumerate(
    zip(smoothed_fractions, smoothed_default_fractions, batch_indices)
):
    print(
        f"Batch {batch_index}: Fraction = {fraction:.2f}, Default Action Fraction = {default_fraction:.2f}"
    )


# Update the plotting function
def plot_smoothed_fractions(
    smoothed_fractions,
    smoothed_default_fractions,
    batch_indices,
    n,
    output_file="smoothed_answer_in_action_fractions.png",
):
    plt.figure(figsize=(12, 6))
    plt.plot(batch_indices, smoothed_fractions, marker="o", label="Smoothed Fractions")
    plt.plot(
        batch_indices,
        smoothed_default_fractions,
        marker="^",
        label="Smoothed Default Action Fractions",
    )
    plt.title(f"Smoothed Fraction of Entries with Answer in Action (n={n})")
    plt.xlabel("Batch Index")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"Smoothed plot saved to {output_file}")


# Plot the smoothed fractions
plot_smoothed_fractions(
    smoothed_fractions, smoothed_default_fractions, batch_indices, n
)
print()


## Example usage
# n = 20  # You can adjust this parameter as needed
# fractions, counts, default_action_fractions = calculate_answer_in_action_fractions(
#    log_data, n
# )
# print(f"Fractions and counts of entries with answer_in_action=True (n={n}):")
# for i, (fraction, count, default_fraction) in enumerate(
#    zip(fractions, counts, default_action_fractions)
# ):
#    print(
#        f"Chunk {i}: Fraction = {fraction:.2f}, Count = {count:.2f}, Default Action Fraction = {default_fraction:.2f}"
#    )
#
#
# def plot_fractions_and_counts(
#    fractions,
#    counts,
#    default_action_fractions,
#    n,
#    output_file="answer_in_action_fractions_and_counts.png",
# ):
#    plt.figure(figsize=(12, 6))
#    x = range(len(fractions))
#    plt.plot(x, fractions, marker="o", label="Fractions")
#    plt.plot(x, counts, marker="s", label="Counts")
#    plt.plot(x, default_action_fractions, marker="^", label="Default Action Fractions")
#    plt.title(f"Fraction and Count of Entries with Answer in Action (n={n})")
#    plt.xlabel("Chunk Index")
#    plt.ylabel("Value")
#    plt.ylim(0, 1)
#    plt.legend()
#    plt.grid(True)
#    plt.savefig(output_file)
#    plt.close()
#    print(f"Plot saved to {output_file}")
#
#
## Plot the fractions and counts
# plot_fractions_and_counts(fractions, counts, default_action_fractions, n)
# print()
#
#
# def clean_markovian_log(input_file: str, output_file: str) -> None:
#    """
#    Read MarkovianRun.log and write a cleaned version to MarkovianCleaned.log.
#
#    This function filters out empty lines, lines with only underscores,
#    progress bars, lines starting with "Updated: " or "Default: ",
#    and lines immediately following "Batch Index: " lines.
#
#    Args:
#        input_file (str): Path to the input log file (MarkovianRun.log)
#        output_file (str): Path to the output cleaned log file (MarkovianCleaned.log)
#    """
#    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
