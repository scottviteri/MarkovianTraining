# MarkovianTraining

This project implements and evaluates Markovian Transformers for informative language modeling, focusing on chain-of-thought (CoT) reasoning through reinforcement learning. The codebase supports the research presented in "Markovian Transformers for Informative Language Modeling" and provides comprehensive tools for training, evaluation, and analysis of language models using Group Relative Policy Optimization (GRPO) and related techniques.

## 🎯 Key Features

- **Markovian vs Non-Markovian Training**: Compare P(answer|CoT) vs P(answer|question,CoT) reward formulations
- **Group Relative Policy Optimization (GRPO)**: Parallel sampling with standardized batch baselines
- **Comprehensive Model Support**: 11 different language models from 124M to 12B parameters
- **Perturbation Analysis**: Systematic robustness evaluation framework
- **Advanced Evaluation Tools**: Cross-model evaluation, visualization, and analysis capabilities
- **Actor Reward Gradients**: Novel training approach using actor model for rewards

## 🚀 Installation

### Standard Installation
```bash
pip install transformers wandb scipy datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib && pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && pip install openai bitsandbytes==0.41.3 scipy scikit-learn
```

If you want to use the this repo using to train the phi language model, the dependencies are slightly different:
```bash
pip install "transformers==4.46.3" wandb scipy datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib && pip install "numpy<2" --force-reinstall && pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && pip install openai bitsandbytes==0.41.3 scipy scikit-learn
```

### For Phi Models
```bash
pip install "transformers==4.46.3" wandb scipy datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib && pip install "numpy<2" --force-reinstall && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes==0.41.3 scipy scikit-learn
```

## 🧪 Testing

The automated suite has been rebuilt around the current `utils/`, `train/`, and `evaluation/` stacks:

- Unit tests live in `tests/unit/` and cover arithmetic loaders, answer extraction, result saving, and helper math utilities.
- GPT-2 integration tests live in `tests/integration/` and exercise a single training step plus the numeric evaluation pipeline end-to-end.
- `tests/conftest.py` provides session-scoped GPT-2 fixtures so weights are loaded once per run.

Run everything (requires ~13s once GPT-2 is cached):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest
```

To skip the heavier GPT-2 integration tests:

```bash
pytest -m "not slow"
```

## 📊 Evaluation System

The project includes a comprehensive evaluation framework with multiple answer extraction methods and validation metrics. See **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** for detailed documentation on:

- Markovian framework and two-stage generation
- Answer extraction methods (word boundary, anchor, LLM)
- Model selection (Actor vs Critic)
- Evaluation metrics and validation (including Haiku gold-standard)
- Cost tracking and best practices

## 📖 Training

### Main Training Script (`src/train.py`)

```bash
python src/train.py [options]
```

#### Task Types
```bash
--task_type <type>     # Choose from:
                       # - arithmetic: Basic math problems
                       # - arithmetic_negative: Math with negative numbers  
                       # - gsm8k: Grade school math dataset
                       # - mmlu: Massive multitask language understanding
                       # - arc: AI2 ARC (Challenge/Easy) science MCQ
                       # - wiki_compression: Wikipedia compression tasks
                       # - wiki_continuation: Wikipedia continuation tasks
```

#### Model Selection (11 Supported Models)
```bash
--model_type <type>    # Choose from:
                       # 🚀 High Performance (8-12B):
                       # - llama (meta-llama/Llama-3.1-8B-Instruct) [default]
                       # - mistral (mistralai/Mistral-7B-Instruct-v0.2)
                       # - gemma-3 (google/gemma-3-12b-it)
                       #
                       # ⚡ Medium Performance (1-4B):
                       # - qwen3 (Qwen/Qwen3-4B)
                       # - qwen3-14b (Qwen/Qwen3-14B)
                       # - phi (microsoft/Phi-3.5-mini-instruct)
                       # - phi-4 (microsoft/phi-4)
                       # - llama3.2-1b (meta-llama/Llama-3.2-1B-Instruct)
                       # - gemma-3-small (google/gemma-3-1b-it)
                       #
                       # 💡 Testing/Development (<1B):
                       # - gpt2 (openai-community/gpt2)
                       # - tinystories (roneneldan/TinyStories)
```

#### Training Methods
```bash
--use_ppo                    # Use Proximal Policy Optimization
--use_ei <float>            # Expert Iteration with std deviations threshold
--parallel                  # GRPO: parallel sampling with batch standardization
--no-markovian             # Non-Markovian: P(answer|question,CoT) [default: Markovian]
--actor_reward_weight <float>  # Weight for actor reward gradients (default: 1.0)
```

#### Core Configuration
```bash
--cot_length <int>          # Chain-of-thought length (default: 50)
--temperature <float>       # Sampling temperature (default: 1.0)
--batch_size <int>         # Batch size (default: 8)
--lr <float>               # Learning rate (default: 1e-4)
--num_batches <int>        # Number of training batches (default: 100000)
--kl_penalty <float>       # KL divergence penalty (default: 0.1)
--ppo_epsilon <float>      # PPO clipping parameter (default: 0.2)
```

#### Advanced Options
```bash
--gradient_accumulation_steps <int>  # Gradient accumulation (default: 1)
--lora_rank <int>                   # LoRA rank (default: 8)
--lora_alpha <float>               # LoRA alpha scaling (default: 16)
--checkpoint_frequency <int>       # Checkpoint frequency (default: 500 for GSM8K)
--normalize_loss                   # Loss normalization (default: True)
--resume                          # Resume from latest checkpoint
```

### Example Training Commands

```bash
# Markovian training with GRPO on GSM8K
python src/train.py --task_type gsm8k --model_type llama --parallel --use_ppo

# Non-Markovian training with actor rewards
python src/train.py --task_type gsm8k --no-markovian  # defaults to actor_reward_weight=1.0

# Expert Iteration with Mistral
python src/train.py --task_type arithmetic --model_type mistral --use_ei 1.0

# Quick test with small model
python src/train.py --task_type arithmetic --model_type gpt2 --num_batches 100

# MMLU training with Qwen3
python src/train.py --task_type mmlu --model_type qwen3 --cot_length 150
```

## 📊 Evaluation

### Unified Evaluation CLI
All evaluation tasks are now consolidated in a single `evaluation.py` script:

```bash
# Evaluate GSM8K
python src/evaluation.py --task_type gsm8k --model_path results/gsm8k/20241201_143022/adapter_500

# Evaluate MMLU
python src/evaluation.py --task_type mmlu --model_path <path> --num_samples 500

# Evaluate ARC-Challenge
python src/evaluation.py --task_type arc --model_path <path>

# Evaluate other tasks: svamp, aqua, mathqa, arithmetic
python src/evaluation.py --task_type svamp --model_path <path>
```

### Common Evaluation Options
```bash
# Evaluate all adapter directories
python src/evaluation.py --task_type gsm8k --run_dir results/gsm8k/20241201_143022 --all_adapters

# Evaluate the latest adapter from a run directory
python src/evaluation.py --task_type gsm8k --model_path results/gsm8k/20241201_143022

# Quick evaluation with stride
python src/evaluation.py --task_type gsm8k --stride 10 --num_samples 100

# Use base model
python src/evaluation.py --task_type gsm8k --use_base_model --model_type llama

# Include Haiku extraction metric (requires ANTHROPIC_API_KEY)
python src/evaluation.py --task_type gsm8k --haiku_metric
```

> `--all_adapters` scans for all `adapter_*` directories within the specified run directory and evaluates each sequentially.

### Task-Specific Options
```bash
# MMLU with specific subject
python src/evaluation.py --task_type mmlu --mmlu_subject mathematics

# ARC with subset selection
python src/evaluation.py --task_type arc --arc_subset ARC-Easy
ARC_SUBSET=ARC-Easy python src/evaluation.py --task_type arc  # via env var

# Answer extraction methods (for numeric tasks)
python src/evaluation.py --task_type gsm8k --answer_extraction_method anchor
```

### Cross-Model Evaluation
Compare different model configurations and training approaches:

```bash
# Basic cross-model evaluation
python src/evaluate_cross_model.py --log_file results/gsm8k/20241201_143022/log.jsonl

# Compare with different critic model
python src/evaluate_cross_model.py --log_file results/gsm8k/20241201_143022/log.jsonl --critic_model mistral

# Plot multiple critics comparison
python src/evaluate_cross_model.py --plot_multiple_critics --log_file results/gsm8k/
```

### Answer Extraction Methods

The evaluation system supports three methods for extracting answers from model-generated text:

| Method | Speed | Cost | Description | Use Case |
|--------|-------|------|-------------|----------|
| **simple** | Fast | Free | Basic regex-based extraction | Bulk evaluation |
| **anchor** | Fast | Free | Context-aware extraction with anchors | Better accuracy, still free |
| **llm** | Slow | ~$0.00001/sample | Claude Haiku gold-standard | Calibration & validation |

#### Setup for LLM Extraction

```bash
# Install Anthropic SDK
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### Comparing Extraction Methods

Use the demo script to compare all three methods on your checkpoints:

```bash
# Compare on base model
python compare_extraction_demo.py --use_base_model --model_type llama --num_samples 50

# Compare on trained checkpoint
python compare_extraction_demo.py --model_path results/gsm8k/20241201_143022/adapter_500

# Specify methods to compare
python compare_extraction_demo.py --use_base_model --methods simple anchor llm
```

#### Batch Calibration Across Checkpoints

Run systematic calibration across multiple training runs:

```bash
# Calibrate GSM8K, SVAMP, and ARC checkpoints
python src/batch_calibrate_extractions.py --datasets gsm8k svamp arc --num_samples 50

# Control number of checkpoints per run
python src/batch_calibrate_extractions.py --datasets gsm8k --max_checkpoints_per_run 3

# All options
python src/batch_calibrate_extractions.py \
  --datasets gsm8k svamp arc aqua \
  --num_samples 100 \
  --model_type llama \
  --max_checkpoints_per_run 5 \
  --output_file my_calibration.json
```

#### Using in Your Code

```python
from evaluation import compare_extraction_methods, extract_answer
from utils import load_model_for_evaluation, load_gsm8k_dataset

# Load model and data
actor, critic, tokenizer, device = load_model_for_evaluation(
    model_path="results/gsm8k/.../adapter_500", 
    model_type="llama"
)
test_data = list(load_gsm8k_dataset(split="test"))[:50]

# Compare extraction methods
results = compare_extraction_methods(
    actor, critic, tokenizer, device, test_data,
    hyperparameters={"task_type": "gsm8k", "model_type": "llama", 
                     "cot_length": 100, "temperature": 1.0},
    methods=["simple", "anchor", "llm"],
    answer_format="numeric"  # or "A-D", "A-E" for MCQ
)

# Or extract answer with specific method
answer_text = "Let me calculate: 2 + 2 = 4. The answer is 4."
extracted = extract_answer(answer_text, method="llm", answer_format="numeric")
```

#### Recommended Workflow

1. **Development**: Use `simple` or `anchor` methods for fast iteration
2. **Calibration**: Periodically compare heuristics against `llm` gold-standard
3. **Validation**: Use LLM extraction on critical evaluations or when debugging failures
4. **Cost Management**: LLM extraction is ~$0.00001 per sample, very affordable for validation sets

## 🔬 Perturbation Analysis

Comprehensive robustness evaluation framework for analyzing model sensitivity to CoT perturbations:

### Available Perturbation Types
- **`delete`**: Random character deletion (0%, 20%, 40%, 60%, 80%, 100%)
- **`truncate_back`**: Truncation from end
- **`truncate_front`**: Truncation from beginning  
- **`digit_replace`**: Random digit replacement
- **`char_replace`**: Random alphanumeric character replacement

### Running Perturbation Analysis
```bash
# Single perturbation type
python src/perturbation_analysis.py --log_file results/gsm8k/20241201_143022/log.jsonl --perturb delete

# Multiple perturbation types with plotting
python src/perturbation_analysis.py --log_file results/gsm8k/20241201_143022/log.jsonl --perturb delete truncate_back digit_replace --plot_multiple_perturbations

# Markovian vs Non-Markovian comparison
python src/perturbation_analysis.py --markovian_log results/markovian/log.jsonl --non_markovian_log results/non_markovian/log.jsonl --perturb delete

# Batched processing for efficiency
python src/perturbation_analysis.py --log_file results/gsm8k/20241201_143022/log.jsonl --perturb delete --batch_size 16
```

### Analysis Options
```bash
--window_size <int>         # Smoothing window (default: 40)
--max_index <int>          # Maximum batch index to analyze
--stride <int>             # Process every nth entry
--include_question         # Include question in perturbation analysis
--save_interval <int>      # Save intermediate results frequency
```

## 📈 Visualization and Analysis

### Training Metrics Visualization
```bash
# Plot training metrics with smoothing
python src/plot_training_metrics.py --files results/*/log.jsonl --window_size 50

# Compare multiple runs with error bars
python src/plot_training_metrics.py --files results/run1/log.jsonl results/run2/log.jsonl --average --show_std

# Summary plots with custom styling
python src/plot_training_metrics.py --plot_summary --label_size 14 --no_title
```

### Quick Log Analysis
```bash
# Fast overview of training progress
python src/log_file_quick_analysis.py results/gsm8k/20241201_143022/log.jsonl

# Examine specific batch
python src/log_file_quick_analysis.py results/gsm8k/20241201_143022/log.jsonl --batch_index 1000

# Custom smoothing window
python src/log_file_quick_analysis.py results/gsm8k/20241201_143022/log.jsonl --window_size 100
```

### Gaussian Process Smoothed Plotting
```bash
# Create GP-smoothed plots for publication
python src/create_gp_smoothed_plot.py --input results/gsm8k/20241201_143022/log.jsonl --output publication_plot.png
```

### Base Log Probability Analysis
```bash
# Analyze base model log probabilities vs context length
python src/analyze_base_logprobs.py --model_type llama --context_lengths 50,100,200,400 --output base_analysis.png
```

## 📁 Results Structure

```
results/
├── <task_type>/
│   └── <timestamp>/
│       ├── adapter_0/          # LoRA adapter checkpoints
│       ├── adapter_500/
│       ├── adapter_1000/
│       ├── log.jsonl          # Training metrics
│       └── evaluations/       # Evaluation results
├── samples/                   # Example log files for reference
│   ├── sample_llama_gsm8k_log.jsonl
│   └── sample_llama_wiki_log.jsonl
├── perturbation_analysis/     # Perturbation analysis results
│   ├── delete_results.json
│   ├── truncate_results.json
│   └── comparison_plots/
└── cross_model_evaluation/    # Cross-model comparison results

LatexFolder/                   # Paper and official figures
├── Figures/                   # All publication-ready figures
│   ├── combined_markovian_comparison_plots.png
│   ├── average_perturbation_results_plot_smooth40.png
│   ├── gsm8k_multiple_critics_comparison.png
│   ├── cross_model_evaluation.png
│   ├── legacy_cross_model/   # Legacy analysis figures
│   └── ...                    # Additional paper figures
├── anonymous-submission-latex-2026.tex  # Main paper
├── appendix.tex              # Supplementary material
├── ReproducibilityChecklist/ # AAAI reproducibility requirements
└── *.pdf                     # Compiled papers
```

### Training Log Structure
Each `log.jsonl` contains:
- **First line**: Complete hyperparameter configuration
- **Subsequent lines**: Per-batch metrics including:
  - Loss values (total, policy gradient, reward gradient)
  - Log probabilities (actor/critic reasoning and answers)
  - KL divergence and weighted KL
  - PPO ratios and clipping
  - Advantages and normalized rewards
  - Expert Iteration thresholds and active sample counts
  - Gradient norms and weight verification

## 🔧 Model Architecture Details

### LoRA Configuration
- **Rank (r)**: 8 (configurable via `--lora_rank`)
- **Alpha**: 16 (configurable via `--lora_alpha`)
- **Dropout**: 0.1
- **Target**: All linear layers
- **Task**: Causal Language Modeling

### Checkpoint System
- **Format**: LoRA adapters (efficient storage)
- **Frequency**: Configurable (default: 500 batches for GSM8K)
- **Content**: Adapter weights + optimizer state + metadata
- **Resumption**: Automatic detection of latest checkpoint

### Memory Optimization
- **Mixed Precision**: bfloat16 for all models
- **Device Mapping**: Automatic (`device_map="auto"`)
- **Gradient Accumulation**: Configurable steps before update
- **Batch Processing**: Optimized for parallel sampling

## 🏛️ Supported Models Authentication

### 🔐 Gated Models (Require HuggingFace Login)
1. **Llama 3.1 8B**: Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. **Llama 3.2 1B**: Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct  
3. **Mistral 7B**: Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
4. **Gemma 3 12B**: Visit https://google/gemma-3-12b-it
5. **Gemma 3 1B**: Visit https://google/gemma-3-1b-it

### ✅ Open Access Models
- GPT-2, TinyStories, Qwen3 variants, Phi variants

## 📚 Advanced Usage

### Markovian vs Non-Markovian Training
```bash
# Markovian: P(answer | CoT) - default
python src/train.py --task_type gsm8k --model_type llama

# Non-Markovian: P(answer | question, CoT)  
python src/train.py --task_type gsm8k --model_type llama --no-markovian
```

### Group Relative Policy Optimization (GRPO)
```bash
# Enable parallel sampling with standardized baselines
python src/train.py --task_type gsm8k --parallel --batch_size 16
```

### Actor Reward Gradients
```bash
# Use actor model for rewards with specified weight
python src/train.py --task_type gsm8k --actor_reward_weight 1.0
```

### Expert Iteration with Dynamic Thresholding
```bash
# Use 1.5 standard deviations above mean as threshold
python src/train.py --task_type gsm8k --use_ei 1.5
```

## 🔍 Research Applications

This codebase supports research into:
- **Markovian language modeling**: Information bottleneck through CoT
- **Robustness analysis**: Perturbation sensitivity across model architectures
- **Training methodology comparison**: PPO vs EI vs GRPO
- **Cross-model generalization**: Transfer of reasoning patterns
- **Reward formulation**: Markovian vs Non-Markovian objectives

## 📄 Dependencies

**Core Requirements:**
- Python 3.10+
- PyTorch 2.1.0+
- Transformers 4.35.2+ (4.46.3+ for Phi models)
- Datasets 2.14.6+
- PEFT 0.4.0+
- bitsandbytes 0.41.3+

**Analysis & Visualization:**
- matplotlib, scipy, scikit-learn
- wandb (for experiment tracking)
- tqdm (for progress bars)

**Optional:**
- flash-attention (for memory efficiency)
- openai (for GPT evaluation baselines)

## 🎯 Quick Start Examples

```bash
# 1. Quick test with small model
python src/train.py --model_type gpt2 --task_type arithmetic --num_batches 10

# 2. Full GSM8K training with GRPO
python src/train.py --task_type gsm8k --model_type llama --parallel --use_ppo --num_batches 5000

# 3. Perturbation analysis
python src/perturbation_analysis.py --log_file results/gsm8k/latest/log.jsonl --perturb delete

# 4. Cross-model evaluation
python src/evaluate_cross_model.py --log_file results/gsm8k/latest/log.jsonl --critic_model mistral
```
