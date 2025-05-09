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

### 1. Main Training Script (`src/train.py`)
The main training script supports various tasks and models with extensive configuration options:

```bash
python src/train.py [options]
```

#### Task Types
```bash
--task_type <type>     # Choose from:
                       # - arithmetic
                       # - arithmetic_negative
                       # - gsm8k
                       # - wiki_compression
                       # - wiki_continuation
```

#### Model Selection
```bash
--model_type <type>    # Choose from:
                       # - llama (default, 8B version)
                       # - mistral (7B-Instruct-v0.2)
```

#### Training Methods
```bash
--use_ppo             # Use Proximal Policy Optimization
--use_ei <float>      # Use Expert Iteration with specified std deviations
```

#### Key Configuration Options
```bash
--cot_length <int>          # Length of chain-of-thought (default: 50)
--temperature <float>       # Sampling temperature (default: 1.0)
--question_length <int>     # Max question length (default: 50)
--target_length <int>       # Max target length (default: 50)
--batch_size <int>         # Batch size (default: 8)
--lr <float>              # Learning rate (default: 1e-4)
--num_batches <int>       # Number of training batches (default: 100000)
--kl_penalty <float>      # KL divergence penalty (default: 0.1)
--ppo_epsilon <float>     # PPO clipping parameter (default: 0.2)
```

#### Additional Options
```bash
--resume                   # Resume training from latest checkpoint
--gradient_accumulation_steps <int>  # Steps before update (default: 1)
--normalize_loss          # Enable loss normalization (default: True)
```

### 2. Model Evaluation Scripts

#### GSM8K Training and Evaluation

### Training
To train a model on the GSM8K dataset:

```bash
python src/train.py --task_type gsm8k --model_type mistral [options]
```

Key training options:
```bash
--use_ppo             # Enable PPO training
--use_ei <float>      # Use Expert Iteration with specified std deviations
--cot_length <int>    # Length of chain-of-thought reasoning
--batch_size <int>    # Training batch size
--num_batches <int>   # Total number of training batches
```

For GSM8K specifically:
- Checkpoints are saved every 500 batches with unique timestamps
- Training automatically tracks dataset epochs
- Progress can be monitored through the training logs

### Evaluation
To evaluate a trained model:

```bash
python src/evaluate_gsm8k.py --model_path <path> --num_samples <n>
```

The evaluation script will:
1. Load the specified model checkpoint
2. Run inference on the GSM8K test set
3. Calculate accuracy metrics
4. Save detailed results to `results/evaluations/`

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

The training script automatically saves:
- Model checkpoints (`.pt` files)
  - For GSM8K: Includes batch index and timestamp
  - For other tasks: Overwrites previous checkpoint
- Training logs (`log.jsonl`)
  - First line: Complete hyperparameter configuration
  - Subsequent lines: Per-batch metrics including:
    - Loss values (total, policy gradient)
    - Log probabilities (actor, critic)
    - KL divergence
    - Gradient norms
    - Advantages
    - Normalized rewards
    - Active samples fraction
    - Answer log probabilities

Results are stored in:
```
results/
├── <task_type>/
│   └── <timestamp>/
│       ├── model.pt
│       └── log.jsonl
├── evaluations/
│   └── <evaluation_results>.json
└── Official/
    └── <final_results_and_figures>
```

### 4. Log Analysis Tool (`src/log_file_quick_analysis.py`)

For quick analysis of training logs, use the log file analysis tool:

```bash
# Plot metrics with default window size (50)
python src/log_file_quick_analysis.py results/task_type/timestamp/log.jsonl

# Plot with custom smoothing window
python src/log_file_quick_analysis.py results/task_type/timestamp/log.jsonl --window_size 100

# Print statistics and examine specific batch without plotting
python src/log_file_quick_analysis.py results/task_type/timestamp/log.jsonl --batch_index 1000
```

The tool provides:
- Overall statistics for actor/critic log probabilities and normalized rewards
- Moving average plots of training metrics
- Detailed examination of specific training batches
- Support for analyzing multiple log files simultaneously

For GSM8K training specifically, the tool can help track:
- Actor and critic reasoning log probabilities
- Actor and critic answer log probabilities
- Normalized rewards
- Training progression across checkpoints

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

To generate evaluate an existing "model.pt" file on the GSM8K test set, run:

```
python src/evaluate_gsm8k.py --model_path <path> --num_samples <n>
```

This command will create a plot showing two metrics over the course of training:
1. "Fraction Correct (per batch)": The fraction of correct answers in each batch.
2. "Reasoning Contains Answer": The proportion of reasoning steps that contain the correct answer.

The plot uses a moving average with a window size of 100 batches to smooth out short-term fluctuations and highlight longer-term trends.

## Model Architecture

The training script uses LoRA (Low-Rank Adaptation) for efficient fine-tuning with the following configuration:
- Task type: Causal Language Modeling
- Rank (r): 8
- Alpha: 16
- Dropout: 0.1
- Target: All linear layers

Both Llama (8B) and Mistral (7B-Instruct-v0.2) models are supported with automatic mixed precision (bfloat16) and automated device placement.
