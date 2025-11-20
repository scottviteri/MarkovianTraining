import subprocess
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
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
            f"--batch_size 8"
        )
        
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {dataset}: {e}")
            print("Continuing to next dataset...")

if __name__ == "__main__":
    main()

