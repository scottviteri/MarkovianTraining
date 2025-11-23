import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def load_eval_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        # Skip metadata line
        f.readline()
        for line in f:
            try:
                entry = json.loads(line)
                # "Original Reward" is Llama's improvement
                # "Avg Log Probs" Actor - Critic is Evaluator's improvement
                
                # Check for required keys
                if "Original Reward" in entry and "Avg Log Probs" in entry:
                    actor_prob = entry["Avg Log Probs"]["Actor"]
                    critic_prob = entry["Avg Log Probs"]["Critic"]
                    evaluator_improvement = actor_prob - critic_prob
                    original_reward = entry["Original Reward"]
                    
                    data.append({
                        "original_reward": original_reward,
                        "evaluator_improvement": evaluator_improvement,
                        "batch_index": entry.get("Batch Index", 0)
                    })
            except json.JSONDecodeError:
                continue
    return data

def plot_alignment_scatter(directory, output_file="alignment_scatter.png"):
    eval_files = [f for f in os.listdir(directory) if f.startswith("evaluation_results_") and f.endswith(".jsonl")]
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        "mistral": "#377eb8",
        "phi": "#ff7f00",
        "gpt2": "#4daf4a",
        "qwen3": "#984ea3",
        "llama3.2-1b": "#e41a1c",
        "tinystories": "#a65628"
    }
    
    for filename in eval_files:
        model_name = filename.replace("evaluation_results_", "").replace(".jsonl", "")
        file_path = os.path.join(directory, filename)
        
        data = load_eval_data(file_path)
        if not data:
            continue
            
        x = np.array([d["original_reward"] for d in data])
        y = np.array([d["evaluator_improvement"] for d in data])
        
        # Create bins for X
        # We want to see the trend: As Llama improves (X increases), does Evaluator improve (Y increases)?
        # X range is roughly -0.5 to 1.0
        bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 15)
        bin_indices = np.digitize(x, bins)
        
        bin_x = []
        bin_y = []
        bin_err = []
        
        for i in range(1, len(bins)):
            indices = (bin_indices == i)
            if np.sum(indices) > 10: # Only plot bins with enough data
                bin_x.append(np.mean(x[indices]))
                bin_y.append(np.mean(y[indices]))
                bin_err.append(np.std(y[indices]) / np.sqrt(np.sum(indices)))
        
        color = colors.get(model_name, "gray")
        
        # Plot connected line with error bars
        plt.errorbar(bin_x, bin_y, yerr=bin_err, label=model_name.title(), 
                    fmt='-o', capsize=3, linewidth=2, markersize=6, color=color)
        
        # Calculate correlation on binned data
        if len(bin_x) > 2:
            slope, intercept, r_value, p_value, std_err = linregress(bin_x, bin_y)
            print(f"{model_name} (Binned): R^2 = {r_value**2:.3f}, Slope = {slope:.3f}")

    # Add y=x line
    plt.plot([-0.2, 0.8], [-0.2, 0.8], 'k--', alpha=0.3, label="y=x Reference")
    
    plt.xlabel("Llama (Actor) Improvement (nats)", fontsize=14)
    plt.ylabel("Evaluator Model Improvement (nats)", fontsize=14)
    plt.title("Cross-Model Value Alignment (Binned)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved binned scatter plot to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing evaluation_results_*.jsonl files")
    parser.add_argument("--output", default="alignment_scatter.png", help="Output filename")
    args = parser.parse_args()
    
    plot_alignment_scatter(args.directory, args.output)

