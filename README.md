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
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
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

### 2. Model Evaluation Scripts

#### GSM8K Evaluation (`src/evaluate_gsm8k.py`)
Evaluates model performance on the GSM8K test set with detailed metrics and analysis.

```bash
# Basic evaluation:
python src/evaluate_gsm8k.py --model_path <path> --num_samples <n> --batch_size <b>

# Additional options:
--use_base_model        # Evaluate base model without loading weights
--model_type mistral    # Use Mistral instead of default Llama
--stride <n>           # Evaluate every nth example
--training_index <n>   # Evaluate specific training checkpoint
--all_checkpoints      # Evaluate all checkpoints in directory
```

#### Cross-Model Evaluation (`src/evaluate_cross_model.py`)
Evaluates and compares performance across different model configurations.

```bash
# Basic evaluation:
python src/evaluate_cross_model.py --log_file <path>

# Additional options:
--window_size <n>      # Smoothing window size (default: 40)
--stride <n>           # Process every nth entry
--debug_freq <n>       # Print debug info frequency
--process_only         # Only process data without plotting
--plot_only           # Only generate plots from saved results
--max_index <n>       # Maximum index to process
--use_same_model      # Use same model type as generator
```

#### Training Metrics Visualization (`src/plot_training_metrics.py`)
Generates detailed plots of training metrics and performance.

```bash
# Basic usage:
python src/plot_training_metrics.py [indices]

# Additional options:
--window_size <n>      # Moving average window size
--output_file <path>   # Save plot to file
--plot_summary        # Plot summary metrics
--files <paths>       # Direct paths to log files
--max_index <n>       # Maximum index to plot
--average            # Average values across files
--show_std           # Show standard deviation bands
--no_legend          # Hide plot legends
--label-size <n>     # Font size for labels
--no_title          # Hide plot titles
```

### 3. Results Analysis

All results are stored in the following structure:
- `results/<task_type>/<timestamp>/` - Contains training outputs
- `results/evaluations/` - Contains evaluation results
- `results/Official/` - Contains final results and figures

#### Log File Format
Training logs (`log.jsonl`) contain:
1. First line: Hyperparameters configuration
2. Subsequent lines: Per-batch metrics including:
   - Loss values (total, policy gradient)
   - Log probabilities (actor, critic)
   - KL divergence
   - Gradient norms
   - Advantages
   - Normalized rewards
   - Active samples fraction
   - Answer log probabilities (when available)

#### Evaluation Results
Evaluation results are saved as JSON files containing:
- Accuracy metrics
- Detailed per-example results
- Model configuration
- Test parameters

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

- The evaluation scripts support both Llama and Mistral models
- Cross-model evaluation allows comparison of different model configurations
- Training metrics can be visualized with various smoothing and averaging options
- Results are automatically versioned by timestamp
- Checkpoints can be evaluated individually or in batch

## GSM8K Results

To generate the GSM8K Performance plot from the appendix, use the following command:

```
python src/AnalyzeResults/analyze_pg_norm.py --gsm8k_plot --window_size=100 --log_file results/Official/GSM8K_training.log
```

This command will create a plot showing two metrics over the course of training:
1. "Fraction Correct (per batch)": The fraction of correct answers in each batch.
2. "Reasoning Contains Answer": The proportion of reasoning steps that contain the correct answer.

The plot uses a moving average with a window size of 100 batches to smooth out short-term fluctuations and highlight longer-term trends.
