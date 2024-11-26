import json
import numpy as np
from pathlib import Path
import sys
from pprint import pprint
import matplotlib.pyplot as plt
import argparse

def load_and_analyze_metrics(file_paths, window=50, target_batch_index=None):
    # Initialize empty lists
    actor_log_probs = []
    normalized_rewards = []
    answer_log_probs = []
    all_entries = []
    batch_indices = []  # New: track batch indices
    
    # Process each file
    for file_path in file_paths:
        try:
            with Path(file_path).open() as f:
                # Skip first line
                next(f)
                
                for line_num, line in enumerate(f, 2):
                    try:
                        data = json.loads(line)
                        metrics = data["Training Metrics"]
                        batch_indices.append(data["Batch Index"])  # New: store batch index
                        if "Actor Answer Log Probs" in metrics:
                            actor_log_probs.append(metrics["Actor Answer Log Probs"])
                            normalized_rewards.append(metrics["Normalized Reward"])
                        else:
                            normalized_rewards.append(metrics["Normalized Reward"])
                        if "Answer Log Probs" in metrics:
                            answer_log_probs.append(metrics["Answer Log Probs"])
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
    
    # Print raw statistics
    print(f"\nProcessed {len(normalized_rewards)} total entries from {len(file_paths)} files")
    print("\nRaw Statistics:")
    
    if actor_log_probs:  # If we have actor log probs, show both statistics
        actor_log_probs = np.array(actor_log_probs)
        print("Actor Answer Log Probs:")
        print(f"  Mean: {actor_log_probs.mean():.4f}")
        print(f"  Std:  {actor_log_probs.std():.4f}")
        print(f"  Max:  {actor_log_probs.max():.4f}")
        print("\nNormalized Rewards:")
        print(f"  Mean: {normalized_rewards.mean():.4f}")
        print(f"  Std:  {normalized_rewards.std():.4f}")
        print(f"  Max:  {normalized_rewards.max():.4f}")
        
        # Calculate critic log probs
        critic_log_probs = actor_log_probs - normalized_rewards
        print("\nCritic Log Probs (Actor - Normalized Reward):")
        print(f"  Mean: {critic_log_probs.mean():.4f}")
        print(f"  Std:  {critic_log_probs.std():.4f}")
        print(f"  Max:  {critic_log_probs.max():.4f}")
    else:  # If we only have normalized rewards, use those directly
        print("Normalized Reward:")
        print(f"  Mean: {normalized_rewards.mean():.4f}")
        print(f"  Std:  {normalized_rewards.std():.4f}")
        print(f"  Max:  {normalized_rewards.max():.4f}")

    # If a specific batch index was requested
    if target_batch_index is not None:
        try:
            idx = batch_indices.index(target_batch_index)
            entry = all_entries[idx]
            
            # ANSI color codes
            BLUE = "\033[94m"
            RESET = "\033[0m"
            
            print(f"\nEntry for Batch Index {target_batch_index}:")
            if actor_log_probs:
                print(f"Actor Answer Log Prob: {actor_log_probs[idx]:.4f}")
                print(f"Normalized Reward: {normalized_rewards[idx]:.4f}")
                print(f"Critic Log Prob: {critic_log_probs[idx]:.4f}")
            else:
                print(f"Normalized Reward: {normalized_rewards[idx]:.4f}")
            
            print("\nEntry Details:")
            print(f"{BLUE}Question:{RESET}")
            print(entry["Example"]["Question"])
            print(f"\n{BLUE}Actor Reasoning:{RESET}")
            print(entry["Example"]["Actor Reasoning"])
            print(f"\n{BLUE}Critic Reasoning:{RESET}")
            print(entry["Example"]["Critic Reasoning"])
            print(f"\n{BLUE}Answer:{RESET}")
            print(entry["Example"]["Answer"])
            return  # Skip plotting
        except ValueError:
            print(f"\nError: Batch index {target_batch_index} not found in the data")
            return  # Skip plotting
        except Exception as e:
            print(f"\nError accessing batch index {target_batch_index}: {e}")
            return  # Skip plotting

    # Only create plot if no specific batch index was requested
    # ... existing plotting code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='Log files to analyze')
    parser.add_argument('--window_size', type=int, default=50, help='Smoothing window size')
    parser.add_argument('--batch_index', type=int, help='Print specific batch index entry without plotting')
    args = parser.parse_args()
    
    load_and_analyze_metrics(args.files, args.window_size, args.batch_index)