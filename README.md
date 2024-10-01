# MarkovianTraining 

This project implements and evaluates various reinforcement learning algorithms for training language models on question-answering tasks, with a focus on chain-of-thought (CoT) reasoning.

## Installation
```
pip install scipy transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes scipy scikit-learn 
```

## Scripts

### 1. Policy Gradient Training (`src/policy_gradient_normalized.py`)

This script implements the main training loop for policy gradient methods.

Usage:
```
# Policy Gradient:
python src/policy_gradient_normalized.py --use_ei --use_pg

# Proximal Policy Optimization:
python src/policy_gradient_normalized.py --use_ei --use_ppo

# Expert Iteration:
python src/policy_gradient_normalized.py --use_ei
```

### 5. GSM8K Training and Evaluation

To reproduce the GSM8K results using Proximal Policy Optimization (PPO) with Expert Iteration (EI), run the following command:

```
python src/policy_gradient_normalized.py --use_gsm8k --use_ei --use_ppo
```

This command will train a model using PPO with EI on the GSM8K dataset and evaluate its performance.

### 6. GSM8K Training and Evaluation

Note: The `--use_gsm8k` flag trains the model on the GSM8K training set. The trained model weights are saved for later evaluation in SavedModels/.

### 2. CoT Answer Accuracy Evaluation (`src/eval_cot_answer_accuracy.py`)

Evaluates and visualizes the performance of trained models.

Usage:
```
python src/eval_cot_answer_accuracy.py --use-max
```

### 3. Chain-of-Thought Perturbation Analysis (`src/perturb_CoT.py`)

Analyzes the robustness of trained models by applying perturbations to the chain-of-thought reasoning.

Usage:
```
# Generate perturbation and Llama comparison data:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official

# Plot perturbation results:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot

# Plot Llama comparison:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot_llama
```

### 4. GSM8K Evaluation (`src/AnalyzeResults/eval_gsm8k.py`)

Evaluates the trained model on the GSM8K test set.

Usage:
```
python src/AnalyzeResults/eval_gsm8k.py --model_path <path_to_model> --num_samples <number_of_samples> --batch_size <batch_size>
```

This script is designed to evaluate a model that has been trained using the `--use_gsm8k` flag in `policy_gradient_normalized.py`. It uses the saved model weights to perform evaluation on the GSM8K test set.

## Results

All results, including plots and log files, are stored in the `results/Official` directory.

## AnalyzeResults Directory

The `src/AnalyzeResults` directory gets populated with detailed log files when running `src/policy_gradient_normalized.py`. Here's how it works:

1. Log File Creation: At the start of training, a new log file is created in `src/AnalyzeResults` with the format:
   ```
   PolicyGradientNormalized_{dataset_type}_{timestamp}.log
   ```
   Where `{dataset_type}` is either "GSM8K" or "Arithmetic".

2. Logging During Training: For each batch, the script logs detailed information including:
   - Batch index
   - Question and generated reasoning
   - Answer and its log probability
   - Rewards and advantages
   - Various loss metrics
   - Training statistics (e.g., gradient norm)

3. Continuous Logging: The script appends to this log file throughout the entire training process.

4. Resuming Training: If training is resumed, the script continues appending to the most recent log file.

These log files provide a comprehensive record of the training process, allowing for detailed analysis, plotting of learning curves, and debugging.

### Analyzing Results

To plot the loss curve from an existing logged run:

```
python src/AnalyzeResults/analyze_pg_norm.py
```

This script reads the log files in `src/AnalyzeResults`, processes the data, and generates a loss curve plot. The resulting plot is saved as `src/AnalyzeResults/pg_norm_plot.png`.

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