#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --output=output.txt
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Load the PyTorch module (replace with the appropriate version)
module load python/3.9.0
module load py-pytorch/1.11.0_py39

pip install torchvision

# Execute the training script
python3 my_training_script.py
