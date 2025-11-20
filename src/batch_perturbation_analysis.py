import subprocess
import sys
import os
import argparse

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run batch perturbation analysis for all datasets")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    datasets = ["gsm8k", "mmlu", "arc", "svamp", "aqua", "mathqa", "arithmetic"]
    perturbations = ["delete", "truncate_back", "truncate_front", "digit_replace", "char_replace"]
    perturb_str = " ".join(perturbations)
    
    # First, run delete and truncate_back for quick feedback
    # perturbations = ["delete", "truncate_back"]
    
    for dataset in datasets:
        print(f"\n{'='*50}\nProcessing dataset: {dataset}\n{'='*50}")
        
        # Construct command
        # We rely on auto-detection of logs inside perturbation_analysis.py
        cmd = (
            f"python src/perturbation_analysis.py "
            f"--fresh_comparison "
            f"--sweep_checkpoints "
            f"--metric accuracy "
            f"--fresh_task_type {dataset} "
            f"--perturb {perturb_str} "
            f"--batch_size {args.batch_size}"
        )
        
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {dataset}: {e}")
            print("Continuing to next dataset...")

if __name__ == "__main__":
    main()

