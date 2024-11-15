import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def run_llama_evaluation(log_files):
    """
    Evaluate log files using Llama as the evaluator.
    
    Args:
        log_files (list): List of log file paths to evaluate
    
    Returns:
        dict: Llama evaluation results
    """
    llama_results = {
        "files": log_files,
        "evaluations": []
    }
    
    for file in log_files:
        with open(file, 'r') as f:
            # Assuming log file contains JSON lines with evaluation data
            file_results = [json.loads(line) for line in f]
            llama_results["evaluations"].append(file_results)
    
    return llama_results

def plot_original_vs_llama(log_files):
    """
    Plot comparison between original and Llama evaluations.
    
    Args:
        log_files (list): List of log file paths to plot
    """
    plt.figure(figsize=(12, 6))
    
    for file in log_files:
        # Load and process data
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # TODO: Implement specific plotting logic for Llama evaluation
        
    plt.title("Original vs Llama Evaluation")
    plt.xlabel("Metric")
    plt.ylabel("Performance")
    
    # Save the plot
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/llama_evaluation.png")
    plt.close()

def save_llama_evaluation(results, output_file=None):
    """
    Save Llama evaluation results to a JSON file.
    
    Args:
        results (dict): Llama evaluation results
        output_file (str, optional): Path to save the results
    """
    if output_file is None:
        # Create a timestamped output file in results/llama_evals/
        os.makedirs("results/llama_evals", exist_ok=True)
        output_file = f"results/llama_evals/llama_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Llama evaluation results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Llama Evaluation Tool")
    parser.add_argument(
        "log_files", 
        nargs="+", 
        help="Log files to evaluate"
    )
    parser.add_argument(
        "--plot_llama", 
        action="store_true", 
        help="Plot Llama vs Original evaluation"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file for evaluation results"
    )
    
    args = parser.parse_args()
    
    # Resolve log file paths (assuming they might be in checkpoints)
    resolved_log_files = []
    for file in args.log_files:
        if not os.path.exists(file):
            # Try to find in checkpoints directory
            checkpoint_path = os.path.join("checkpoints", file)
            if os.path.exists(checkpoint_path):
                file = checkpoint_path
        resolved_log_files.append(file)
    
    # Run evaluation
    results = run_llama_evaluation(resolved_log_files)
    
    if args.plot_llama:
        plot_original_vs_llama(resolved_log_files)
    else:
        save_llama_evaluation(results, args.output)

if __name__ == "__main__":
    main() 