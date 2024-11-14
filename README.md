# MarkovianTraining 

This project implements and evaluates various reinforcement learning algorithms for training language models on question-answering tasks, with a focus on chain-of-thought (CoT) reasoning. The code implements the figures for "Markovian Transformers for Informative Language Modeling" (https://arxiv.org/abs/2404.18988).

## Installation

Install the package dependencies:

```bash
pip install transformers wandb scipy datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes scipy scikit-learn
```

## Testing

To run tests, first set the PYTHONPATH to include the current directory:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest
```

## Scripts

### 1. Policy Gradient Main (`src/policy_gradient_normalized.py`)
#### Model Selection
The script supports both Llama (default) and Mistral models. You can specify the model using the `--model_type` flag:

```bash
# Using Llama (default):
python src/policy_gradient_normalized.py --task_type arithmetic --use_pg

# Using Mistral:
python src/policy_gradient_normalized.py --task_type arithmetic --use_pg --model_type mistral
```

#### Task Types
The script supports several different tasks:

```bash
# Basic Arithmetic:
python src/policy_gradient_normalized.py --task_type arithmetic --use_pg

# Arithmetic with Negative Numbers:
python src/policy_gradient_normalized.py --task_type arithmetic-negative --use_pg

# GSM8K Math Problems:
python src/policy_gradient_normalized.py --task_type gsm8k --use_pg

# Wikipedia Text Compression:
python src/policy_gradient_normalized.py --task_type wiki_compression --use_pg

# Wikipedia Text Continuation:
python src/policy_gradient_normalized.py --task_type wiki_continuation --use_pg
```

#### Training Methods
For any task type, you can use different training methods:

```bash
# Policy Gradient:
python src/policy_gradient_normalized.py --task_type <task> --use_pg

# Proximal Policy Optimization:
python src/policy_gradient_normalized.py --task_type <task> --use_ppo

# Expert Iteration:
python src/policy_gradient_normalized.py --task_type <task> --use_ei
```

#### GSM8K Evaluation (`src/AnalyzeResults/eval_gsm8k.py`)

Evaluates the trained model on the GSM8K test set.

Usage:
```bash
# Using Llama (default):
python src/AnalyzeResults/eval_gsm8k.py --model_path <path_to_model> --num_samples <number_of_samples> --batch_size <batch_size>

# Using Mistral:
python src/AnalyzeResults/eval_gsm8k.py --model_path <path_to_model> --num_samples <number_of_samples> --batch_size <batch_size> --model_type mistral
```

### 2. CoT Answer Accuracy Evaluation (`src/eval_cot_answer_accuracy.py`)

Evaluates and visualizes the performance of trained models (Figure 2).

Usage:
```
python src/eval_cot_answer_accuracy.py --use-max
```

### 3. Chain-of-Thought Perturbation Analysis (`src/perturb_CoT.py`)

Analyzes the robustness of trained models by applying perturbations to the chain-of-thought reasoning (Figure 3).

Usage:
```
# Generate perturbation and Llama comparison data:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official

# Plot perturbation results:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot

# Plot Llama comparison:
python src/perturb_CoT.py --log_file PPO1.log PPO2.log PPO3.log PPO4.log --results_subfolder Official --plot_llama
```

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

This project requires the following main dependencies:
- Python 3.10+
- PyTorch 2.1.0+
- Transformers 4.35.2+
- Datasets 2.14.6+
- Accelerate 0.24.1+
- Weights & Biases 0.15.12+
- Pytest 7.4.3+
- Pytest-cov 4.1.0+
- Multiprocess 0.70.17

Please ensure all dependencies are installed before running the scripts.

## Notes

- The `--use-max` flag in `eval_cot_answer_accuracy.py` is required to reproduce the results as presented.
- When using `--model_type llama`, ensure you have appropriate access to the Llama model weights.
- Adjust batch sizes and other parameters as needed based on your hardware capabilities.
- Some scripts may require significant computational resources, especially when working with large language models.

## GSM8K Results

To generate the GSM8K Performance plot from the appendix, use the following command:

```
python src/AnalyzeResults/analyze_pg_norm.py --gsm8k_plot --window_size=100 --log_file results/Official/GSM8K_training.log
```

This command will create a plot showing two metrics over the course of training:
1. "Fraction Correct (per batch)": The fraction of correct answers in each batch.
2. "Reasoning Contains Answer": The proportion of reasoning steps that contain the correct answer.

The plot uses a moving average with a window size of 100 batches to smooth out short-term fluctuations and highlight longer-term trends.
