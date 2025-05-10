# MarkovianTraining 

This project implements and evaluates various reinforcement learning algorithms for training language models on question-answering tasks, with a focus on chain-of-thought (CoT) reasoning. The code implements the figures for "Markovian Transformers for Informative Language Modeling" (https://arxiv.org/abs/2404.18988).

## Recent Additions: Vector Quantization

The repository now includes experimental Vector Quantization (VQ) techniques for language model training. This approach:

- Implements efficient parallel encoding/decoding by finding the nearest token embedding for each hidden state in a single forward pass
- Supports backpropagation through critic models without weight updates
- Creates clean separation between encoder and decoder to prevent information leakage
- Enables efficient memorization with minimal token count

## Installation

Install the package dependencies:

```bash
pip install transformers wandb scipy datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes scipy scikit-learn
```

Alternatively, install the package in development mode:

```bash
pip install -e .
```

## Testing

To run tests, first set the PYTHONPATH to include the current directory:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest
```

## Project Structure

```
markovian_training/
├── src/                     # Main source code
│   ├── vq/                  # Vector Quantization module
│   │   ├── vq_parallel.py   # Core VQ implementation
│   │   └── train_vq_parallel.py # Training with VQ
│   ├── utils.py             # Utility functions
│   └── train.py             # Main training code
├── tests/                   # Test suite
│   └── vq/                  # VQ-specific tests
├── setup.py                 # Package installation configuration
└── requirements.txt         # Dependencies
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

### 2. Vector Quantization Training (`src/vq/train_vq_parallel.py`)

The VQ training script demonstrates how to use parallel vector quantization:

```bash
python -m src.vq.train_vq_parallel [options]
```

#### VQ-specific Options:
```bash
--vq_codebook_weight <float>   # Weight for the codebook loss (default: 1.0)
--vq_token_temp <float>        # Temperature for token generation (default: 1.0)
--vq_reason_length <int>       # Length of reasoning to generate with VQ (default: 50)
```

### 3. VQ Testing

Run tests for the Vector Quantization implementation:

```bash
# Test memorization and token reduction
python -m tests.vq.test_vq_simple --model google/gemma-3-1b-it

# Run example training 
python -m tests.vq.example_vq_training --train
```

## Important Implementation Details

- The critic model has `requires_grad=False` for all parameters, but does not use `torch.no_grad()` during forward passes. This allows gradient flow through the critic without updating its weights.
- The implementation uses a "fresh context" approach to eliminate information leakage between encoder and decoder.

For more details on the original Markovian Training approach, please see the paper: [Markovian Transformers for Informative Language Modeling](https://arxiv.org/abs/2404.18988).
