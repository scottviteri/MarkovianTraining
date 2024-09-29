import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

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
        "./results/9-28-24/EPPO1.log",
        "./results/9-28-24/EPPO2.log",
        "./results/9-28-24/EPPO3.log",
        "./results/9-28-24/EPPO4.log",
    ],
    "EI": [
        "./results/9-28-24/EI1_bak.log",
        "./results/9-28-24/EI2_bak.log",
        "./results/9-28-24/EI3_bak.log",
        "./results/9-28-24/EI4_bak.log",
    ],
}
# max_cuts = {
#    "EI": 285,
#    "PG": 342,
#    "PPO": 448,
# }
full_labs = {
    "EI": "Expert Iteration",
    "PG": "Policy Gradient",
    "PPO": "Proximal Policy Optimization",
}
cols = {
    "EI": "#0072B2",
    "PG": "#E69F00",
    "PPO": "#CC79A7",  # "#D55E00",
}

fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)


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


for alg_type in file_dicts:
    print(alg_type)
    window_size = 40
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
        x_plot = res_x_list[best_index]
        acc_plot = res_acc_list[best_index]
        logprob_plot = res_logprob_list[best_index]
        label_suffix = " (Best Of 4)"
    else:
        # Proceed with aggregation (average or max)
        x_min = np.min([len(xs) for xs in res_x_list])
        x_min = min(x_min, 750)

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
        lw=2,
    )
    axs[1].plot(
        x_plot,
        logprob_plot,
        label=full_labs[alg_type] + label_suffix,
        c=cols[alg_type],
        lw=2,
    )

    # Optional dashed lines (if needed)
    axs[0].plot(
        x_plot,
        acc_plot,
        label=None,
        c=cols[alg_type],
        alpha=0.5,
        ls="--",
    )
    axs[1].plot(
        x_plot,
        logprob_plot,
        label=None,
        c=cols[alg_type],
        alpha=0.5,
        ls="--",
    )

axs[0].legend()
axs[1].set_xlabel("Training Batch No. [ ]")
axs[0].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
axs[0].set_ylabel("P(Answer contained in CoT) [ ]")
axs[0].set_ylim([0.0, 1.0])
axs[0].set_xlim([0.0, 800])
axs[0].text(
    0.05,
    0.65,
    f"Smoothing window = {window_size}",
    transform=axs[0].transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"),
)
# axs[1].set_ylabel("LogProb [ ]")
axs[1].set_ylabel("ln P(Answer | CoT) [ ]")
plt.subplots_adjust(hspace=0.05)
# caption = "Figure 1: This is a caption for the sine wave plot. This is a caption for the sine wave plot. This is a caption for the sine wave plot. This is a caption for the sine wave plot."
# fig.text(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10)
plt.savefig("rebuttal_plot.png", dpi=300)
plt.show()


# Print the imported dictionaries
# for i, d in enumerate(imported_dicts):
#     print(json.dumps(d, indent=2))
