import json
import numpy as np
from pathlib import Path
import sys
from pprint import pprint

def load_and_analyze_metrics(file_paths):
    # Initialize empty lists
    actor_log_probs = []
    normalized_rewards = []
    answer_log_probs = []
    all_entries = []
    
    # Process each file
    for file_path in file_paths:
        try:
            with Path(file_path).open() as f:
                # Skip first line
                next(f)
                
                for line_num, line in enumerate(f, 2):  # Start counting from 2 since we skipped line 1
                    try:
                        data = json.loads(line)
                        metrics = data["Training Metrics"]
                        # Try to get Actor Answer Log Probs, if not available use Normalized Reward
                        if "Actor Answer Log Probs" in metrics:
                            actor_log_probs.append(metrics["Actor Answer Log Probs"])
                            normalized_rewards.append(metrics["Normalized Reward"])
                        else:
                            # If using old format, just track Normalized Reward
                            normalized_rewards.append(metrics["Normalized Reward"])
                        # Only append answer_log_probs if it exists
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
        
        # Normalize scores for finding best example
        actor_norm = (actor_log_probs - actor_log_probs.mean()) / actor_log_probs.std()
        reward_norm = (normalized_rewards - normalized_rewards.mean()) / normalized_rewards.std()
        combined_scores = actor_norm + reward_norm
    else:  # If we only have normalized rewards, use those directly
        print("Normalized Reward:")
        print(f"  Mean: {normalized_rewards.mean():.4f}")
        print(f"  Std:  {normalized_rewards.std():.4f}")
        print(f"  Max:  {normalized_rewards.max():.4f}")
        combined_scores = normalized_rewards  # Just use normalized rewards for finding best example
    
    # Find index of best combined score
    best_idx = np.argmax(combined_scores)
    best_entry = all_entries[best_idx]
    
    # ANSI color codes
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    print("\nBest Entry:")
    if actor_log_probs:
        print(f"Actor Answer Log Prob: {actor_log_probs[best_idx]:.4f}")
        print(f"Normalized Reward: {normalized_rewards[best_idx]:.4f}")
        print(f"Critic Log Prob: {critic_log_probs[best_idx]:.4f}")
    else:
        print(f"Normalized Reward: {normalized_rewards[best_idx]:.4f}")
    
    print("\nEntry Details:")
    print(f"{BLUE}Question:{RESET}")
    print(best_entry["Example"]["Question"])
    print(f"\n{BLUE}Actor Reasoning:{RESET}")
    print(best_entry["Example"]["Actor Reasoning"])
    print(f"\n{BLUE}Critic Reasoning:{RESET}")
    print(best_entry["Example"]["Critic Reasoning"])
    print(f"\n{BLUE}Answer:{RESET}")
    print(best_entry["Example"]["Answer"])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file1> [log_file2 ...]")
        sys.exit(1)
    
    load_and_analyze_metrics(sys.argv[1:])