#!/usr/bin/env python3
"""
Plot all wiki_continuation runs in a 3x4 grid and extract maximum smoothed rewards.
Usage: python plot_wiki_grid_with_max.py --window_size 100 --output wiki_grid.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import os
from pathlib import Path

def moving_average(data, window_size):
    """Calculate moving average, properly handling NaN values"""
    if len(data) < window_size:
        return data
        
    data_array = np.array(data, dtype=float)
    result = np.zeros(len(data_array) - window_size + 1)
    
    for i in range(len(result)):
        window = data_array[i:i+window_size]
        valid_values = window[~np.isnan(window)]
        if len(valid_values) > 0:
            result[i] = np.mean(valid_values)
        else:
            result[i] = np.nan
    
    return result

def load_log_data(log_file, window_size=100):
    """Load normalized reward data from log file and compute smoothed version"""
    with open(log_file, 'r') as f:
        lines = f.readlines()
        hyperparameters = json.loads(lines[0])
        entries = [json.loads(line) for line in lines[1:]]
    
    # Extract normalized rewards
    rewards = []
    for entry in entries:
        if "Training Metrics" in entry and "Normalized Reward" in entry["Training Metrics"]:
            reward = entry["Training Metrics"]["Normalized Reward"]
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
            else:
                rewards.append(np.nan)
        else:
            rewards.append(np.nan)
    
    if len(rewards) == 0:
        return hyperparameters, None, None, None
    
    rewards = np.array(rewards)
    
    # Smooth the data
    smoothed = moving_average(rewards, window_size)
    
    # Find max of smoothed (ignoring NaN)
    valid_smoothed = smoothed[~np.isnan(smoothed)]
    max_reward = np.max(valid_smoothed) if len(valid_smoothed) > 0 else np.nan
    
    # Create x coordinates
    offset = (window_size - 1) // 2
    x_coords = np.arange(offset, offset + len(smoothed))
    
    return hyperparameters, rewards, smoothed, x_coords, max_reward

def plot_wiki_continuation_grid(results_dir, window_size=100, output_file="wiki_grid.png", figsize=(20, 15)):
    """Create 3x4 grid of all wiki_continuation runs"""
    
    # Define the 12 configurations in order matching the table
    configs = [
        ("20250802_031333_left3", "llama", 1.2, "N", "Y"),
        ("20250802_031211_left2", "llama", 1.2, "Y", "N"),
        ("20250802_014430_left", "llama", 1.3, "Y", "Y"),
        ("20250802_021039_riight3", "mistral", 1.3, "N", "Y"),
        ("20250802_004155_riight2", "mistral", 1.4, "Y", "Y"),
        ("20250802_031442_riight", "mistral", 1.4, "Y", "N"),
        ("20250802_021022_mid3", "phi", 1.3, "N", "Y"),
        ("20250802_001111_mid2", "phi", 1.4, "Y", "Y"),
        ("20250802_031443_mid", "phi", 1.4, "Y", "N"),
        ("20250802_021028_right3", "qwen3", 1.3, "N", "Y"),
        ("20250802_001130_right2", "qwen3", 1.4, "Y", "Y"),
        ("20250802_031445_right", "qwen3", 1.4, "Y", "N"),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    results = []
    
    for idx, (dirname, model, temp, parallel, markov) in enumerate(configs):
        log_file = os.path.join(results_dir, dirname, "log.jsonl")
        
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found")
            axes[idx].text(0.5, 0.5, "Log file not found", ha='center', va='center')
            axes[idx].set_title(f"{model.title()} T={temp} P={parallel} M={markov}")
            continue
        
        hyperparams, raw_rewards, smoothed, x_coords, max_reward = load_log_data(log_file, window_size)
        
        if smoothed is None:
            axes[idx].text(0.5, 0.5, "No data", ha='center', va='center')
            axes[idx].set_title(f"{model.title()} T={temp} P={parallel} M={markov}")
            continue
        
        # Store results
        batch_size = hyperparams.get('batch_size', '?')
        results.append({
            'model': model,
            'temp': temp,
            'batch': batch_size,
            'parallel': parallel,
            'markov': markov,
            'max_reward': max_reward,
            'batches': len(raw_rewards)
        })
        
        # Plot raw data (faint)
        axes[idx].plot(raw_rewards, alpha=0.2, color='gray', linewidth=0.5)
        
        # Plot smoothed data
        mask = ~np.isnan(smoothed)
        if np.any(mask):
            axes[idx].plot(x_coords[mask], smoothed[mask], linewidth=2, color='blue')
        
        # Mark maximum
        if not np.isnan(max_reward):
            max_idx = np.nanargmax(smoothed)
            axes[idx].axhline(y=max_reward, color='red', linestyle='--', alpha=0.5, linewidth=1)
            axes[idx].scatter([x_coords[max_idx]], [max_reward], color='red', s=50, zorder=5)
        
        # Styling
        title = f"{model.title()} T={temp} P={parallel} M={markov}"
        if markov == "Y" and parallel == "Y":
            title += " ⭐"  # Mark the 4 used in Figure 1b
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel("Batch", fontsize=8)
        axes[idx].set_ylabel("Normalized Reward", fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(labelsize=8)
        
        # Add max value text
        if not np.isnan(max_reward):
            axes[idx].text(0.95, 0.95, f"Max: {max_reward:.3f}", 
                          transform=axes[idx].transAxes,
                          fontsize=9, ha='right', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f"Wikipedia Continuation: All 12 Runs (Smoothing Window={window_size})", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    return results

def print_table(results):
    """Print LaTeX table with only varying parameters and max reward"""
    print("\n" + "="*80)
    print("LATEX TABLE (Only Varying Parameters + Max Smoothed Reward)")
    print("="*80)
    
    print(r"""
\begin{table}[ht]
    \centering
    \caption{Wikipedia continuation experiments showing maximum smoothed normalized reward. Non-Markovian models achieve higher peaks but may be less interpretable. Runs marked with * were used in Figure~\ref{fig:loss}.}
    \label{tab:wiki_hyperparams}
    \begin{tabular}{lllllll}
        \toprule
        \textbf{Model} & \textbf{Temp} & \textbf{Batch} & \textbf{Par.} & \textbf{Mar.} & \textbf{Max Reward} & \textbf{Fig} \\
        \midrule""")
    
    for r in results:
        fig_marker = "*" if r['markov'] == "Y" and r['parallel'] == "Y" else ""
        print(f"        {r['model']:8} & {r['temp']:.1f} & {r['batch']:2} & {r['parallel']:1} & {r['markov']:1} & ${r['max_reward']:+.3f}$ & {fig_marker} \\\\")
    
    print(r"""        \bottomrule
    \end{tabular}
\end{table}
""")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Group by Markovian vs Non-Markovian
    markov_yes = [r for r in results if r['markov'] == 'Y']
    markov_no = [r for r in results if r['markov'] == 'N']
    
    print(f"\nMarkovian (M=Y): {len(markov_yes)} runs")
    print(f"  Max rewards: {[f'{r['max_reward']:.3f}' for r in markov_yes]}")
    print(f"  Range: {min(r['max_reward'] for r in markov_yes):.3f} to {max(r['max_reward'] for r in markov_yes):.3f}")
    print(f"  Mean: {np.mean([r['max_reward'] for r in markov_yes]):.3f}")
    
    print(f"\nNon-Markovian (M=N): {len(markov_no)} runs")
    print(f"  Max rewards: {[f'{r['max_reward']:.3f}' for r in markov_no]}")
    print(f"  Range: {min(r['max_reward'] for r in markov_no):.3f} to {max(r['max_reward'] for r in markov_no):.3f}")
    print(f"  Mean: {np.mean([r['max_reward'] for r in markov_no]):.3f}")
    
    # Group by Parallel
    parallel_yes = [r for r in results if r['parallel'] == 'Y']
    parallel_no = [r for r in results if r['parallel'] == 'N']
    
    print(f"\nParallel/GRPO (P=Y): {len(parallel_yes)} runs")
    print(f"  Range: {min(r['max_reward'] for r in parallel_yes):.3f} to {max(r['max_reward'] for r in parallel_yes):.3f}")
    print(f"  Mean: {np.mean([r['max_reward'] for r in parallel_yes]):.3f}")
    
    print(f"\nNon-Parallel/EI (P=N): {len(parallel_no)} runs")
    print(f"  Range: {min(r['max_reward'] for r in parallel_no):.3f} to {max(r['max_reward'] for r in parallel_no):.3f}")
    print(f"  Mean: {np.mean([r['max_reward'] for r in parallel_no]):.3f}")
    
    # The four star runs
    star_runs = [r for r in results if r['markov'] == 'Y' and r['parallel'] == 'Y']
    print(f"\n⭐ The 4 runs used in Figure 1b (M=Y, P=Y):")
    for r in star_runs:
        print(f"  {r['model']:8} T={r['temp']:.1f}: {r['max_reward']:+.3f}")

def main():
    parser = argparse.ArgumentParser(description="Plot wiki continuation runs and extract max rewards")
    parser.add_argument("--results_dir", type=str, 
                       default="MarkovianTraining/results/wiki_continuation",
                       help="Directory containing wiki_continuation results")
    parser.add_argument("--window_size", type=int, default=100,
                       help="Smoothing window size (default: 100)")
    parser.add_argument("--output", type=str, default="wiki_continuation_grid.png",
                       help="Output filename for plot")
    parser.add_argument("--figsize", type=str, default="20,15",
                       help="Figure size as width,height (default: 20,15)")
    
    args = parser.parse_args()
    
    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Create plot and extract results
    results = plot_wiki_continuation_grid(
        args.results_dir, 
        window_size=args.window_size,
        output_file=args.output,
        figsize=figsize
    )
    
    # Print table
    if results:
        print_table(results)
    else:
        print("No results to display")

if __name__ == "__main__":
    main()

