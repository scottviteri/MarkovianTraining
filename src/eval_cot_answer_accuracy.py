
import json
import matplotlib.pyplot as plt
import numpy as np

def import_log_file(file_path, do_logprob=False):
    full_dict = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Use json.loads() to parse each line as a JSON object
                in_dict = json.loads(line.strip())
                # print(in_dict)
                if isinstance(in_dict, dict):
                    try:
                        if do_logprob:
                            full_dict.append(
                                # [in_dict["Batch Index"], in_dict["Reasoning Contains Answer"]]
                                [in_dict["Batch Index"], in_dict["Reasoning Contains Answer"], in_dict["Avg Log Prob"]]
                            )
                        else:
                            full_dict.append(
                                # [in_dict["Batch Index"], in_dict["Reasoning Contains Answer"]]
                                [in_dict["Batch Index"],
                                 in_dict["Reasoning Contains Answer"],
                                 in_dict["Aggregate loss"]]
                            )
                    except KeyError:
                        continue
            except json.JSONDecodeError:
                continue
    return full_dict

file_dicts = {
    "PG": [
        './PreliminaryComparisonPlots/PG1.log',
        './PreliminaryComparisonPlots/PG2.log',
        './PreliminaryComparisonPlots/PG3.log',
        './PreliminaryComparisonPlots/PG4.log'
    ],
    "PPONorm (Markovian Training)": [
        # './PreliminaryComparisonPlots/PPONorm1.log',
        # './PreliminaryComparisonPlots/PPONorm2.log',
        # './PreliminaryComparisonPlots/PPONorm3.log',
        # './PreliminaryComparisonPlots/PPONorm4.log'
        './PreliminaryComparisonPlots/PPONorm1_new.log',
        './PreliminaryComparisonPlots/PPONorm2_new.log',
        './PreliminaryComparisonPlots/PPONorm3_new.log',
        './PreliminaryComparisonPlots/PPONorm4_new.log'
    ],
    "EI": [
        './PreliminaryComparisonPlots/EI1.log',
        './PreliminaryComparisonPlots/EI2.log',
        './PreliminaryComparisonPlots/EI3.log',
        './PreliminaryComparisonPlots/EI4.log'
    ],
}
max_cuts = {
    "EI": 285,
    "PG": 342,
    "PPONorm (Markovian Training)": 448,
}
full_labs = {
    "EI": "Expert Iteration",
    "PG": "Policy Gradient",
    "PPONorm (Markovian Training)": "PPO + Norm + Ave (Ours)",
}
cols = {
    "EI": "#0072B2",
    "PG": "#E69F00",
    "PPONorm (Markovian Training)": "#CC79A7", # "#D55E00",
}

fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def calculate_average(*res_y_lists, x_min=None):
    # Determine the minimum length if x_min is not provided
    if x_min is None:
        x_min = min(len(lst) for lst in res_y_lists)

    # Calculate the average
    num_lists = len(res_y_lists)
    y_av = sum(lst[:x_min] for lst in res_y_lists) / num_lists

    return y_av

for alg_type in file_dicts:
    print(alg_type)
    window_size = 20
    res_x = []
    res_acc = []
    res_logprob = []
    for i_file, file in enumerate(file_dicts[alg_type]):
        # print(file)
        if alg_type == "EI":
            imported_dicts = import_log_file(file, do_logprob=False)
        else:
            imported_dicts = import_log_file(file, do_logprob=True)
        # imported_dicts = import_log_file(file)
        data = np.array(imported_dicts).T
        # print(data.shape)

        if alg_type == "EI":
            logprob_smoothed = moving_average(-data[2], window_size)

        else:
            if alg_type =="PG":
                if i_file == 1:
                    last_ind = int(data[0][-1])
                    last_logprob = data[2][-1]
                    n_add = 500
                    data_app = np.array(
                        [
                            list(range(last_ind + 1, n_add + last_ind + 1)),
                            [0.] * n_add,
                            [last_logprob] * n_add]
                    )
                    print(data.shape, data_app.shape)
                    data = np.concatenate((data, data_app), axis=1)
                    print(data.shape)

            logprob_smoothed = moving_average(data[2], window_size)
        acc_smoothed = moving_average(data[1], window_size)
        x_smoothed = data[0][window_size - 1:]

        # Ensure x and y_smoothed have the same length
        x_smoothed = x_smoothed[:len(acc_smoothed)]
        acc_smoothed = acc_smoothed[:len(x_smoothed)]
        logprob_smoothed = logprob_smoothed[:len(x_smoothed)]

        res_x.append(x_smoothed)
        res_acc.append(acc_smoothed)
        res_logprob.append(logprob_smoothed)
        # plt.plot(x_smoothed, y_smoothed)

    # plt.show()

    x_min = np.min([len(xs) for xs in res_x])
    x_min = min(x_min, 750)
    print([len(xs) for xs in res_x])
    acc_av = calculate_average(*res_acc, x_min=x_min)
    logprob_av = calculate_average(*res_logprob, x_min=x_min)
    x_plot = res_x[0][:x_min]

    axs[0].plot(x_plot[:max_cuts[alg_type]], acc_av[:max_cuts[alg_type]], label=full_labs[alg_type], c=cols[alg_type], lw=2)
    axs[1].plot(x_plot[:max_cuts[alg_type]], logprob_av[:max_cuts[alg_type]], label=full_labs[alg_type], c=cols[alg_type], lw=2)

    axs[0].plot(x_plot[max_cuts[alg_type]:], acc_av[max_cuts[alg_type]:],
                label=None, c=cols[alg_type], alpha=0.5, ls="--")
    axs[1].plot(x_plot[max_cuts[alg_type]:], logprob_av[max_cuts[alg_type]:],
                label=None, c=cols[alg_type], alpha=0.5, ls="--")

axs[0].legend()
axs[1].set_xlabel("Training Batch No. [ ]")
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axs[0].set_ylabel("P(Answer contained in CoT) [ ]")
axs[0].set_ylim([0., 1.])
axs[0].set_xlim([0., 600])
axs[0].text(0.05, 0.65, f'Smoothing window = {window_size}',
        transform=axs[0].transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
# axs[1].set_ylabel("LogProb [ ]")
axs[1].set_ylabel("ln P(Answer | CoT) [ ]")
plt.subplots_adjust(hspace=0.05)
# caption = "Figure 1: This is a caption for the sine wave plot. This is a caption for the sine wave plot. This is a caption for the sine wave plot. This is a caption for the sine wave plot."
# fig.text(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10)
plt.savefig("rebuttal_plot.pdf", dpi=300)
plt.show()


# Print the imported dictionaries
# for i, d in enumerate(imported_dicts):
#     print(json.dumps(d, indent=2))