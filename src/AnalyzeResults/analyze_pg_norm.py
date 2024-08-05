import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import glob

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def get_latest_log_file():
    log_files = glob.glob("src/AnalyzeResults/PolicyGradientNormalized_*.log")
    if not log_files:
        raise FileNotFoundError("No PolicyGradientNormalized log files found.")
    return max(log_files, key=os.path.getctime)

def plot_metrics(file_path, window_size=1, output_file='src/AnalyzeResults/pg_norm_plot.png'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse the hyperparameters from the first line
    hyperparameters = json.loads(lines[0])
    print("Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")

    # Parse the log entries
    metrics = defaultdict(list)
    for line in lines[1:]:
        entry = json.loads(line)
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                metrics[key].append(value)

    # Plot the metrics
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training Metrics')

    # Plot Aggregate Loss
    axs[0, 0].plot(moving_average(metrics['Aggregate loss'], window_size))
    axs[0, 0].set_title('Aggregate Loss (Moving Average)')
    axs[0, 0].set_xlabel('Batch')
    axs[0, 0].set_ylabel('Loss')

    # Plot Policy Loss
    axs[0, 1].plot(moving_average(metrics['Policy Loss'], window_size))
    axs[0, 1].set_title('Policy Loss (Moving Average)')
    axs[0, 1].set_xlabel('Batch')
    axs[0, 1].set_ylabel('Loss')

    # Plot Normalized Reward
    axs[1, 0].plot(moving_average(metrics['Normalized Reward'], window_size))
    axs[1, 0].set_title('Normalized Reward (Moving Average)')
    axs[1, 0].set_xlabel('Batch')
    axs[1, 0].set_ylabel('Reward')

    # Plot Advantage
    axs[1, 1].plot(moving_average(metrics['Advantage'], window_size))
    axs[1, 1].set_title('Advantage (Moving Average)')
    axs[1, 1].set_xlabel('Batch')
    axs[1, 1].set_ylabel('Advantage')

    # Plot Average Log Prob
    axs[2, 0].plot(moving_average(metrics['Avg Log Prob'], window_size))
    axs[2, 0].set_title('Average Log Probability (Moving Average)')
    axs[2, 0].set_xlabel('Batch')
    axs[2, 0].set_ylabel('Log Probability')

    # Plot Reasoning Contains Answer
    contains_answer = [int(x) for x in metrics['Reasoning Contains Answer']]
    axs[2, 1].plot(moving_average(contains_answer, window_size))
    axs[2, 1].set_title('Reasoning Contains Answer (Moving Average)')
    axs[2, 1].set_xlabel('Batch')
    axs[2, 1].set_ylabel('Proportion')

    # If PPO is used, plot PPO-specific metrics
    if 'PPO Ratio' in metrics:
        fig_ppo, axs_ppo = plt.subplots(1, 2, figsize=(15, 5))
        fig_ppo.suptitle('PPO Metrics')

        axs_ppo[0].plot(moving_average(metrics['PPO Ratio'], window_size))
        axs_ppo[0].set_title('PPO Ratio (Moving Average)')
        axs_ppo[0].set_xlabel('Batch')
        axs_ppo[0].set_ylabel('Ratio')

        axs_ppo[1].plot(moving_average(metrics['PPO Clipped Ratio'], window_size))
        axs_ppo[1].set_title('PPO Clipped Ratio (Moving Average)')
        axs_ppo[1].set_xlabel('Batch')
        axs_ppo[1].set_ylabel('Ratio')

        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '_ppo.png'))
        print(f"PPO plot saved to {output_file.replace('.png', '_ppo.png')}")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    else:
        log_file_path = get_latest_log_file()
    
    plot_metrics(log_file_path)