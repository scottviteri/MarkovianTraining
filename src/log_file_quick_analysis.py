import json
import numpy as np
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import argparse
import os

def load_and_analyze_metrics(file_paths, window=50, target_batch_index=None):
    # Initialize empty lists
    actor_answer_log_probs = []
    critic_answer_log_probs = []
    normalized_rewards = []
    answer_log_probs = []
    contains_answer = []
    all_entries = []
    batch_indices = []
    
    # Process each file
    for file_path in file_paths:
        try:
            with Path(file_path).open() as f:
                next(f)  # Skip first line
                
                for line_num, line in enumerate(f, 2):
                    try:
                        data = json.loads(line)
                        metrics = data["Training Metrics"]
                        batch_indices.append(data["Batch Index"])
                        
                        # Get metrics directly from the JSON
                        if "Actor Answer Log Probs" in metrics:
                            actor_answer_log_probs.append(metrics["Actor Answer Log Probs"])
                        if "Critic Answer Log Probs" in metrics:
                            critic_answer_log_probs.append(metrics["Critic Answer Log Probs"])
                        if "Normalized Reward" in metrics:
                            normalized_rewards.append(metrics["Normalized Reward"])
                        if "Answer Log Probs" in metrics:
                            answer_log_probs.append(metrics["Answer Log Probs"])
                            
                        contains_answer.append(data["Example"].get("Contains Answer", False))
                        all_entries.append(data)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Error processing line {line_num} in {file_path}: {e}")
                        continue
            print(f"Successfully processed {file_path}")
            
        except FileNotFoundError:
            print(f"Error: Could not find file {file_path}")
            continue
        except Exception as e:
            print(f"Error: An unexpected error occurred with {file_path}: {e}")
            continue

    if not normalized_rewards:
        print("No valid data found in any of the input files")
        return

    # Convert to numpy arrays
    normalized_rewards = np.array(normalized_rewards)
    if actor_answer_log_probs:
        actor_answer_log_probs = np.array(actor_answer_log_probs)
    if critic_answer_log_probs:
        critic_answer_log_probs = np.array(critic_answer_log_probs)
    
    # Create plots directory if it doesn't exist
    log_dir = str(Path(file_paths[0]).parent)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Analysis', fontsize=16)
    
    # Plot 1: Normalized Rewards
    ax1.plot(normalized_rewards, alpha=0.3, label='Raw')
    ax1.plot(np.convolve(normalized_rewards, np.ones(window)/window, mode='valid'),
             label=f'Moving Avg (w={window})')
    ax1.set_title('Normalized Rewards')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Value')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Actor Answer Log Probs
    if actor_answer_log_probs.size > 0:
        ax2.plot(actor_answer_log_probs, alpha=0.3, label='Raw')
        ax2.plot(np.convolve(actor_answer_log_probs, np.ones(window)/window, mode='valid'),
                label=f'Moving Avg (w={window})')
        ax2.set_title('Actor Answer Log Probs')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Value')
        ax2.grid(True)
        ax2.legend()

    # Plot 3: Critic Answer Log Probs
    if critic_answer_log_probs.size > 0:
        ax3.plot(critic_answer_log_probs, alpha=0.3, label='Raw')
        ax3.plot(np.convolve(critic_answer_log_probs, np.ones(window)/window, mode='valid'),
                label=f'Moving Avg (w={window})')
        ax3.set_title('Critic Answer Log Probs')
        ax3.set_xlabel('Batch')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        ax3.legend()

    # Plot 4: Contains Answer
    contains_answer = np.array(contains_answer)
    window_avg = np.convolve(contains_answer, np.ones(window)/window, mode='valid')
    ax4.plot(contains_answer, 'o', alpha=0.1, label='Raw')
    ax4.plot(np.arange(len(window_avg)), window_avg, label=f'Moving Avg (w={window})')
    ax4.set_title('Contains Answer')
    ax4.set_xlabel('Batch')
    ax4.set_ylabel('Rate')
    ax4.grid(True)
    ax4.legend()

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # Print statistics
    print(f"\nProcessed {len(normalized_rewards)} total entries from {len(file_paths)} files")
    print("\nRaw Statistics:")
    
    if actor_answer_log_probs.size > 0:
        print("Actor Answer Log Probs:")
        print(f"  Mean: {actor_answer_log_probs.mean():.4f}")
        print(f"  Std:  {actor_answer_log_probs.std():.4f}")
        print(f"  Max:  {actor_answer_log_probs.max():.4f}")
    
    if critic_answer_log_probs.size > 0:
        print("\nCritic Answer Log Probs:")
        print(f"  Mean: {critic_answer_log_probs.mean():.4f}")
        print(f"  Std:  {critic_answer_log_probs.std():.4f}")
        print(f"  Max:  {critic_answer_log_probs.max():.4f}")
    
    print("\nNormalized Rewards:")
    print(f"  Mean: {normalized_rewards.mean():.4f}")
    print(f"  Std:  {normalized_rewards.std():.4f}")
    print(f"  Max:  {normalized_rewards.max():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='Log files to analyze')
    parser.add_argument('--window_size', type=int, default=50, help='Smoothing window size')
    parser.add_argument('--batch_index', type=int, help='Print specific batch index entry without plotting')
    args = parser.parse_args()
    
    load_and_analyze_metrics(args.files, args.window_size, args.batch_index)