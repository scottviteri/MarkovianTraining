# MarkovianTraining 

This project implements and evaluates various reinforcement learning algorithms for training language models on question-answering tasks, with a focus on chain-of-thought (CoT) reasoning.

## Installation
```
pip install scipy transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes scipy scikit-learn 
```

## Scripts

### 1. Policy Gradient Training (`policy_gradient_normalized.py`)

This script implements the main training loop for policy gradient methods, including:
- Policy Gradient (PG)
- Proximal Policy Optimization (PPO)
- Expert Iteration (EI)

Usage:
```
python src/policy_gradient_normalized.py --use_gsm8k --resume --use_ei/use_ppo/use_pg
```

Key features:
- Supports GSM8K dataset and custom arithmetic questions
- Implements normalized rewards and advantages
- Allows resuming training from checkpoints
- Logs detailed training information

### 2. CoT Answer Accuracy Evaluation (`eval_cot_answer_accuracy.py`)

This script evaluates and visualizes the performance of trained models.

Usage:
```
python src/eval_cot_answer_accuracy.py --use-max
```

Key features:
- Plots accuracy and log probability metrics
- Supports aggregation across multiple runs
- Customizable plot appearance and smoothing

### 3. Chain-of-Thought Perturbation Analysis (`perturb_CoT.py`)

This script analyzes the robustness of trained models by applying perturbations to the chain-of-thought reasoning.

Usage:
```
# Generate perturbation and Llama comparison data:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official

# Plot perturbation results:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot

# Plot Llama comparison:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot_llama
```

Key features:
- Applies various perturbations to CoT reasoning
- Compares performance with Llama 2 model
- Generates visualizations of perturbation effects

### 4. GSM8K Evaluation (`eval_gsm8k.py`)

This script evaluates the trained model on the GSM8K dataset.

Usage:
```
python src/AnalyzeResults/eval_gsm8k.py --model_path <path_to_model> --num_samples <number_of_samples> --batch_size <batch_size>
```

Key features:
- Evaluates model accuracy on GSM8K test set
- Supports batch processing for efficiency
- Generates detailed evaluation results

## Results

All results, including plots and log files, are stored in the `results/Official` directory.

## Reproduction Steps

1. Train the model using `policy_gradient_normalized.py` with desired settings.
2. Evaluate CoT answer accuracy:
   ```
   python src/eval_cot_answer_accuracy.py --use-max
   ```
3. Perform perturbation analysis:
   ```
   python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official
   python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot
   python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot_llama
   ```
4. Evaluate on GSM8K dataset:
   ```
   python src/AnalyzeResults/eval_gsm8k.py --model_path <path_to_trained_model>
   ```

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- NumPy
- Matplotlib
- tqdm
- datasets (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)

Please ensure all dependencies are installed before running the scripts.

## Notes

- The `--use-max` flag in `eval_cot_answer_accuracy.py` is required to reproduce the results as presented.
- Adjust batch sizes and other parameters as needed based on your hardware capabilities.
- Some scripts may require significant computational resources, especially when working with large language models.