import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # For color maps

# Increase font size for all text elements
# plt.rcParams.update({"font.size": 12})

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate CoT answer accuracy.")
parser.add_argument(
    "--use-max",
    action="store_true",
    help="Use element-wise maximum instead of average.",
)
parser.add_argument(
    "--use-best-run",
    action="store_true",
    help="Plot the run with the highest average in each category.",
)
parser.add_argument(
    "--view-all-runs",
    type=str,
    help="View all runs of a specified kind (e.g., EI, PG, PPO).",
)
args = parser.parse_args()


def import_log_file(file_path):
    full_dict = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            in_dict = json.loads(line.strip())
            if isinstance(in_dict, dict):
                full_dict.append(
                    [
                        in_dict["Batch Index"],
                        in_dict["Reasoning Contains Answer"],
                        in_dict["Avg Log Prob"],
                    ]
                )
    return full_dict


file_dicts = {
    "PG": [
        "./results/9-28-24/EPG1.log",
        "./results/9-28-24/EPG2.log",
        "./results/9-28-24/EPG3.log",
        "./results/9-28-24/EPG4.log",
    ],
    "PPO": [
        "./results/9-28-24/EPPO1_R1.log",
        "./results/9-28-24/EPPO2_R1.log",
        "./results/9-28-24/EPPO3_R1.log",
        "./results/9-28-24/EPPO4_R1.log",
    ],
    "EI": [
        "./results/9-28-24/EI1_bak2.log",
        "./results/9-28-24/EI2_bak2.log",
        "./results/9-28-24/EI3_bak2.log",
        "./results/9-28-24/EI4_bak2.log",
    ],
}

full_labs = {
    "EI": "Expert Iteration",
    "PG": "Policy Gradient",
    "PPO": "Proximal Policy Optimization",
}

cols = {
    "EI": "#0072B2",
    "PG": "#E69F00",
    "PPO": "#CC79A7",
}

# Modify the subplot configuration to have two columns
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=False)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def calculate_aggregate(*res_y_lists, x_min=None, use_max=False):
    if x_min is None:
        x_min = min(len(lst) for lst in res_y_lists)
    stacked_arrays = np.vstack([lst[:x_min] for lst in res_y_lists])
    if use_max:
        y_agg = np.max(stacked_arrays, axis=0)
    else:
        y_agg = np.mean(stacked_arrays, axis=0)
    return y_agg


# === New Block for --view-all-runs Option ===
if args.view_all_runs:
    alg_type = args.view_all_runs
    if alg_type not in file_dicts:
        print(f"Error: '{alg_type}' is not a valid algorithm type.")
        exit(1)

    window_size = 40
    num_runs = len(file_dicts[alg_type])
    color_map = cm.get_cmap("tab10", num_runs)  # Choose a suitable colormap

    # Initialize lists to store smoothed data and their lengths
    acc_smoothed_list = []
    logprob_smoothed_list = []
    x_smoothed_list = []

    # Collect data for each run
    for i_file, file in enumerate(file_dicts[alg_type]):
        imported_dicts = import_log_file(file)
        data = np.array(imported_dicts).T
        logprob_smoothed = moving_average(data[2], window_size)
        acc_smoothed = moving_average(data[1], window_size)
        x_smoothed = data[0][window_size - 1 :]

        # Ensure x and y_smoothed have the same length
        min_len = min(len(acc_smoothed), len(x_smoothed), len(logprob_smoothed))
        acc_smoothed = acc_smoothed[:min_len]
        logprob_smoothed = logprob_smoothed[:min_len]
        x_smoothed = x_smoothed[:min_len]

        # Store smoothed data
        acc_smoothed_list.append(acc_smoothed)
        logprob_smoothed_list.append(logprob_smoothed)
        x_smoothed_list.append(x_smoothed)

    # Set x_max to 1000
    x_max = 1000

    # Plot each run up to x_max
    for i_file in range(num_runs):
        acc_smoothed = acc_smoothed_list[i_file]
        logprob_smoothed = logprob_smoothed_list[i_file]
        x_smoothed = x_smoothed_list[i_file]

        # Truncate data to x_max if necessary
        x_indices = np.where(x_smoothed <= x_max)[0]
        acc_smoothed = acc_smoothed[x_indices]
        logprob_smoothed = logprob_smoothed[x_indices]
        x_smoothed = x_smoothed[x_indices]

        # Plot each run with a different color
        axs[0].plot(
            x_smoothed,
            acc_smoothed,
            label=f"{full_labs[alg_type]} Run {i_file + 1}",
            c=color_map(i_file),
            lw=2,
        )
        axs[1].plot(
            x_smoothed,
            logprob_smoothed,
            c=color_map(i_file),
            lw=2,
        )

    # Adjust legends and labels
    axs[0].legend(loc="lower right", fontsize=12)

    # Enlarge text on each axis and labels
    axs[0].tick_params(axis="both", labelsize=14)
    axs[1].tick_params(axis="both", labelsize=14)
    axs[0].set_xlabel("Training Batch No.", fontsize=14)
    axs[1].set_xlabel("Training Batch No.", fontsize=14)
    axs[0].set_ylabel("P(Answer contained in CoT)", fontsize=14)
    axs[1].set_ylabel("ln P(Answer | CoT)", fontsize=14)

    # Set y-limits for consistency
    axs[0].set_ylim([0.0, 1.0])
    axs[1].set_ylim([None, 0.0])  # ymin will auto-adjust

    # Set x-limits to [0, 1000]
    axs[0].set_xlim([0, x_max])
    axs[1].set_xlim([0, x_max])

    # Move the 'Smoothing window = 40' text to the right plot and enlarge text
    axs[1].text(
        0.05,
        0.95,
        f"Smoothing window = {window_size}",
        transform=axs[1].transAxes,
        verticalalignment="top",
        fontsize=16,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
    )

    # Add grid lines to both plots
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("rebuttal_plot.pdf", dpi=300)
    plt.show()
    exit(0)  # Exit after plotting

# === End of --view-all-runs Block ===

# Proceed if --view-all-runs is not specified
for alg_type in file_dicts:
    print(alg_type)
    window_size = 40  # Increased from 20 to 40
    res_x_list = []
    res_acc_list = []
    res_logprob_list = []
    avg_acc_list = []  # To store average accuracies for each run
    avg_logprob_list = []  # To store average log probabilities for each run

    for i_file, file in enumerate(file_dicts[alg_type]):
        imported_dicts = import_log_file(file)
        data = np.array(imported_dicts).T
        logprob_smoothed = moving_average(data[2], window_size)
        acc_smoothed = moving_average(data[1], window_size)
        x_smoothed = data[0][window_size - 1 :]

        # Ensure x and y_smoothed have the same length
        x_smoothed = x_smoothed[: len(acc_smoothed)]
        acc_smoothed = acc_smoothed[: len(x_smoothed)]
        logprob_smoothed = logprob_smoothed[: len(x_smoothed)]

        res_x_list.append(x_smoothed)
        res_acc_list.append(acc_smoothed)
        res_logprob_list.append(logprob_smoothed)

        # Compute the average accuracy and log probability for this run
        avg_acc = np.mean(acc_smoothed)
        avg_logprob = np.mean(logprob_smoothed)
        avg_acc_list.append(avg_acc)
        avg_logprob_list.append(avg_logprob)

    if args.use_best_run:
        # Select the run with the highest average accuracy
        best_index = np.argmax(avg_acc_list)
        print(
            f"Best run for {alg_type}: Run {best_index+1} with average accuracy {avg_acc_list[best_index]:.4f}"
        )
        x_plot = res_x_list[best_index]
        acc_plot = res_acc_list[best_index]
        logprob_plot = res_logprob_list[best_index]
        label_suffix = " (Best Of 4)"
    else:
        # Proceed with aggregation (average or max)
        x_min = np.min([len(xs) for xs in res_x_list])
        # x_min = min(x_min, 750)

        acc_plot = calculate_aggregate(*res_acc_list, x_min=x_min, use_max=args.use_max)
        logprob_plot = calculate_aggregate(
            *res_logprob_list, x_min=x_min, use_max=args.use_max
        )
        x_plot = res_x_list[0][:x_min]
        label_suffix = ""

    # Plotting
    axs[0].plot(
        x_plot,
        acc_plot,
        label=full_labs[alg_type] + label_suffix,
        c=cols[alg_type],
        lw=2,  # Increase line width
    )
    axs[1].plot(
        x_plot,
        logprob_plot,
        c=cols[alg_type],
        lw=2,  # Increase line width
    )

# Adjust legends and labels
axs[0].legend(loc="lower right", fontsize=14)
axs[0].set_xlabel("Training Batch No. [ ]", fontsize=14)
axs[1].set_xlabel("Training Batch No. [ ]", fontsize=14)
axs[0].set_ylabel("P(Answer contained in CoT) [ ]", fontsize=15)
axs[1].set_ylabel("ln P(Answer | CoT) [ ]", fontsize=15)
axs[0].set_ylim([0.0, 1.0])

# Set ymin for axs[1] to the minimum value of logprob_plot
axs[1].set_ylim([np.min(logprob_plot), 0.0])

axs[0].set_xlim([0.0, 1000])
axs[1].set_xlim([0.0, 1000])

# Enlarge text on each axis
axs[0].tick_params(axis="both", labelsize=14)
axs[1].tick_params(axis="both", labelsize=14)

# Move the 'Smoothing window = 40' text to the right plot and enlarge text
axs[1].text(
    0.05,
    0.95,
    f"Smoothing window = {window_size}",
    transform=axs[1].transAxes,
    verticalalignment="top",
    fontsize=16,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
)

# Add grid lines to both plots
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("rebuttal_plot.pdf", dpi=300)
plt.show()
