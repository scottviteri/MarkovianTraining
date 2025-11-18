# Evaluation Guide

This guide provides comprehensive documentation on the evaluation system used in MarkovianTraining, including the Markovian framework, answer extraction methods, and evaluation metrics.

## Table of Contents

1. [Overview](#overview)
2. [Markovian Framework](#markovian-framework)
3. [Answer Extraction Methods](#answer-extraction-methods)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Model Selection: Actor vs Critic](#model-selection-actor-vs-critic)
6. [Usage Examples](#usage-examples)
7. [Cost Tracking](#cost-tracking)

## Overview

The evaluation system in MarkovianTraining follows a structured pipeline:

1. **Generation Stage**: Models generate chain-of-thought (CoT) reasoning and final answers
2. **Extraction Stage**: Task-specific extractors parse answers from generated text
3. **Comparison Stage**: Extracted answers are compared with gold answers
4. **Metrics Stage**: Accuracy and optional validation metrics (Haiku, word boundary) are computed

All evaluation code is centralized in `src/evaluation.py` for consistency and maintainability.

## Markovian Framework

The evaluation follows the **Markovian property** for chain-of-thought reasoning:

### Two-Stage Generation

1. **Stage 1: CoT Generation**
   - Actor model generates reasoning at training temperature (stochastic)
   - Prompt: `Question: {question}\nReasoning:`
   - Output: Chain-of-thought reasoning text

2. **Stage 2: Answer Generation**
   - A model generates the final answer deterministically (temperature=0)
   - Prompt: `Reasoning: {cot}\nAnswer:`
   - Output: Final answer text
   - Which model is used depends on the training mode (see below)

### Markovian Property

The key principle is that **the answer should depend only on the CoT, not the original question**:

```
P(answer | CoT)  ✓ Markovian (standard)
P(answer | question, CoT)  ✗ Non-Markovian (alternative mode)
```

This is controlled by the `markovian` hyperparameter:
- `markovian=True` (default): Question is NOT included in answer generation prompt
- `markovian=False`: Question IS included in answer generation prompt

## Model Selection: Actor vs Critic

The choice of which model generates the final answer depends on the training mode:

### Standard Markovian Mode (`actor_reward_weight = 0`)

**Actor Model**:
- Generates CoT reasoning
- Receives reward based on whether critic produces correct answer
- Training objective: Learn to write CoT that leads critic to correct answers

**Critic Model**:
- Generates final answer (frozen, provides stable grading)
- Not trained, serves as answer oracle
- Ensures consistent evaluation across training

**Why this works**: The frozen critic provides a stable "grading rubric" - the actor learns to write reasoning that the critic can reliably convert into correct answers.

### Actor-Only Mode (`actor_reward_weight > 0`)

**Actor Model**:
- Generates both CoT reasoning AND final answer
- Receives reward directly for answer correctness
- Training objective: Learn complete reasoning + answering pipeline

**Critic Model**:
- Not used during evaluation
- Actor must learn to be self-sufficient

**Why this works**: Actor receives direct feedback on answer quality, learning end-to-end generation without depending on critic.

### Evaluation Consistency

The evaluation system automatically selects the appropriate model:

```python
actor_reward_weight = hyperparameters.get("actor_reward_weight", 0.0)
answer_model = actor_model if actor_reward_weight > 0 else critic_model
```

This ensures evaluation matches the training setup, providing accurate performance measurement.

## Answer Extraction Methods

Answer extraction is task-specific and uses different methods depending on the task type.

### Multiple Choice Questions (MCQ)

MCQ tasks (MMLU, ARC, AQuA, MathQA) use letter extraction with two methods:

#### Primary: Word Boundary Extraction (Recommended)

Uses regex word boundaries to avoid false matches:

```python
def extract_letter_word_boundary(text: str) -> str:
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else "X"
```

**Advantages**:
- Avoids false matches in words like "The" (contains E), "Select" (contains E)
- More accurate representation of model's intended answer
- Default method as of the latest version

**Examples**:
- "The answer is B" → B ✓
- "Select option C" → C ✓
- "The best choice" → X (no isolated letter)

#### Legacy: First Letter Match

Finds first occurrence of any valid letter:

```python
def extract_letter_legacy(text: str) -> str:
    matches = re.findall(r"[A-E]", text.upper())
    return matches[0] if matches else "X"
```

**Issues**:
- "The answer is D" → E (matches 'E' in "The")
- "Select B" → E (matches 'E' in "Select")
- Still reported as `accuracy_legacy` for backward compatibility

### Numeric Tasks (GSM8K, SVAMP, MATH)

Numeric tasks use three extraction methods:

#### 1. Anchor Method (Default, Recommended)

Prioritizes the "Answer:" label that models are trained to use:

```python
def extract_answer_with_anchor(answer: str):
    # 1) Look for "Answer:" label (case-insensitive)
    if "answer" in answer.lower():
        # Extract first integer after the label
        ...
    # 2) Else look for "=" sign
    elif "=" in answer:
        # Extract first integer after "="
        ...
    # 3) Else extract first integer anywhere
    ...
```

**Priority**:
1. Integer after "Answer:" or "answer" label
2. Integer after "=" sign
3. First integer in text

**Examples**:
- "Answer: 42" → 42
- "The result is 10. Answer: 42" → 42 (prioritizes label)
- "x = 10, so 10 + 5 = 15" → 10 (after "Answer:", finds first integer)

#### 2. Simple Method (Legacy)

Original method without label prioritization:

```python
def extract_answer_simple(answer: str):
    if "=" in answer:
        answer = answer.split("=")[-1]
    # Extract first integer
    matches = re.findall(r"-?\d+", answer)
    return int(matches[0]) if matches else "[invalid]"
```

**Issues**:
- Doesn't respect "Answer:" label
- Can grab wrong number if multiple are present

#### 3. LLM Method (Gold Standard)

Uses Claude Haiku as a gold-standard extractor:

```python
def extract_answer_with_llm(answer: str, answer_format: str):
    # Call Claude Haiku to extract answer
    # Requires ANTHROPIC_API_KEY environment variable
    ...
```

**Advantages**:
- Most accurate extraction
- Handles complex formats and edge cases
- Useful for validation and comparison

**Cost**: ~$0.0001 per call (tracked automatically)

### Normalization and Comparison

All numeric answers are normalized before comparison:

```python
def normalize_numeric(text: str) -> str:
    s = text.strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)  # Remove LaTeX
    s = s.replace("$", "").replace("\\", "")      # Remove math symbols
    s = re.sub(r"\s+", "", s)                     # Remove whitespace
    return s
```

This ensures "42", " 42 ", "$42$", and "\\boxed{42}" are all treated as equivalent.

## Evaluation Metrics

### Primary Metrics

#### 1. Accuracy

The main metric for all tasks:

```python
accuracy = correct_predictions / total_examples
```

- **MCQ tasks**: Uses word boundary extraction (correct method)
- **Numeric tasks**: Uses anchor extraction (recommended method)

#### 2. Legacy/Alternative Accuracy

For backward compatibility and comparison:

- **MCQ tasks** (`accuracy_legacy`): First letter match without word boundaries
- **Numeric tasks**: Currently only one accuracy reported, but extraction method is configurable

### Validation Metrics

#### 3. Haiku Accuracy (Optional)

When `ANTHROPIC_API_KEY` is set, Haiku extraction runs automatically:

```python
haiku_metrics = {
    "accuracy": 0.85,        # Accuracy using Claude Haiku extraction
    "cost_usd": 0.0100,      # Estimated cost in USD
    "num_calls": 100         # Number of API calls made
}
```

**Purpose**:
- Validate that heuristic extraction methods are working correctly
- Identify cases where extraction logic fails
- Provides gold-standard comparison

**Usage**: Compare primary accuracy with Haiku accuracy. Large differences indicate extraction issues.

## Usage Examples

### Running Evaluation

#### Standalone Evaluation

```bash
# Evaluate GSM8K with latest checkpoint
python -m evaluation --task_type gsm8k

# Evaluate MMLU with specific checkpoint
python -m evaluation --task_type mmlu --model_path results/mmlu/20250101_120000/adapter_500

# Evaluate with specific extraction method
python -m evaluation --task_type gsm8k --answer_extraction_method anchor

# Enable Haiku validation (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your_key_here
python -m evaluation --task_type mmlu --haiku_metric
```

#### During Training

Evaluation runs automatically during training based on `eval_frequency`:

```python
# In your training script
hyperparameters = {
    "task_type": "gsm8k",
    "eval_frequency": 100,  # Evaluate every 100 batches
    "batch_size": 12,
    # ...
}
```

### Programmatic Usage

```python
from evaluation import evaluate_model_on_gsm8k

# Load your models
actor_model, critic_model, tokenizer, device = load_models(...)

# Prepare test data
test_data = [(question1, answer1), (question2, answer2), ...]

# Run evaluation
accuracy, results, haiku_metrics = evaluate_model_on_gsm8k(
    actor_model=actor_model,
    critic_model=critic_model,
    tokenizer=tokenizer,
    device=device,
    test_data=test_data,
    hyperparameters=hyperparameters,
    batch_size=16,
    answer_extraction_method="anchor",  # Use recommended method
    enable_haiku_metric=True,            # Enable Haiku validation
)

print(f"Accuracy: {accuracy:.2%}")
if haiku_metrics:
    print(f"Haiku Accuracy: {haiku_metrics['accuracy']:.2%}")
    print(f"Haiku Cost: ${haiku_metrics['cost_usd']:.4f}")
```

### Results Format

Results are saved as JSONL with detailed per-example information:

```json
{
  "timestamp": "20250118_143022",
  "task_type": "gsm8k",
  "batch_index": 500,
  "accuracy": 0.73,
  "haiku_accuracy": 0.75,
  "haiku_cost_usd": 0.0131,
  "haiku_num_calls": 131,
  "num_examples": 131,
  "detailed_results": [
    {
      "question": "Janet's ducks lay 16 eggs per day...",
      "reasoning": "Let me work through this step by step...",
      "generated_answer": "Answer: 18",
      "predicted": 18,
      "answer": "18",
      "correct": true
    }
  ]
}
```

## Cost Tracking

### Haiku API Costs

The system automatically tracks costs for Haiku extraction:

- **Rate**: ~$0.0001 per call (Claude 3.5 Haiku pricing)
- **Rate Limiting**: Max 50 requests/second (built-in)
- **Reporting**: Cost displayed in console and saved in results

**Example Output**:
```
Haiku Metric: Running Claude Haiku extraction on 500 samples...
Haiku Metric: Accuracy: 76.00% | Cost: $0.0500 (500 calls)
```

### Cost Estimation

For a typical evaluation:
- **GSM8K test set** (1,319 examples): ~$0.13
- **MMLU** (500 examples per subject): ~$0.05
- **Full training run** (eval every 100 batches × 10 evals): ~$1.30

### Disabling Haiku

To skip Haiku extraction:

```python
# Option 1: Don't set ANTHROPIC_API_KEY
# Option 2: Disable explicitly
accuracy, results, _, haiku_metrics = evaluate_model_on_mmlu(
    ...,
    enable_haiku_metric=False
)
```

## Best Practices

### 1. Use Recommended Extraction Methods

- **MCQ**: Word boundary extraction (default)
- **Numeric**: Anchor extraction (default)

These are now the defaults as of the latest version.

### 2. Monitor Haiku Consistency

If Haiku accuracy differs significantly from primary accuracy:
- Check extraction logic for edge cases
- Review failed examples in detailed results
- Consider updating extraction method

### 3. Track Costs

Enable Haiku only when needed:
- During development/debugging: Enable to validate extraction
- During production training: Consider disabling to save costs
- For paper results: Enable for final validation

### 4. Understand Actor/Critic Modes

Match evaluation expectations to training mode:
- Standard Markovian: Actor writes CoT, critic answers
- Actor-only: Actor does everything, critic unused

### 5. Reproducibility

The evaluation system is now deterministic:
- Sampling is sequential (not random)
- Temperature=0 for answer generation
- Same checkpoint + same data = same results

## Troubleshooting

### Low Accuracy Issues

1. **Check extraction method**: Verify you're using the recommended method
2. **Inspect failed examples**: Look at `detailed_results` in output
3. **Compare with Haiku**: Large difference suggests extraction problems
4. **Review prompts**: Ensure models are generating in expected format

### Haiku Extraction Fails

1. **API Key**: Verify `ANTHROPIC_API_KEY` is set correctly
2. **Rate Limits**: System handles rate limiting automatically
3. **Network Issues**: Check internet connection and Anthropic API status
4. **Fallback**: System falls back to heuristic methods on failure

### Memory Issues

For large test sets:
- Reduce `batch_size` parameter
- Use `num_samples` to limit evaluation size
- Monitor GPU memory usage

## Summary

The evaluation system provides:

✓ **Consistent**: Single source of truth in `evaluation.py`  
✓ **Accurate**: Word boundary and anchor extraction methods  
✓ **Validated**: Optional Haiku gold-standard comparison  
✓ **Transparent**: Detailed per-example results and cost tracking  
✓ **Reproducible**: Deterministic sampling and fixed temperature  

For more implementation details, see the docstrings in `src/evaluation.py`.

