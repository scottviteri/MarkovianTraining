# Qwen2.5-7B-Instruct Integration Summary

This document summarizes all changes made to add support for Qwen/Qwen2.5-7B-Instruct to the MarkovianTraining codebase.

## Changes Made

### 1. Constants Definition (`src/constants.py`)
- Added Qwen2.5 special tokens:
  ```python
  QWEN25_IM_START = "<|im_start|>"
  QWEN25_IM_END = "<|im_end|>"
  ```

### 2. Model Loading (`src/utils.py`)
- Added `qwen25` to the model type options in `load_model()` function
- Added model name mapping: `"Qwen/Qwen2.5-7B-Instruct"`
- Imported Qwen2.5 constants

### 3. Prompt Formatting (`src/utils.py`)
- Added Qwen2.5 support to `get_model_specific_tokens()` function
- Added Qwen2.5 ChatML format handling in `construct_prompts()` function
- Format uses system/user/assistant roles with `<|im_start|>` and `<|im_end|>` tokens

### 4. Answer Position Finding (`src/train.py`)
- Added Qwen2.5 to the models that use token ID 16141 for "Answer"
- Updated `find_answer_start_position()` to handle Qwen2.5's tokenization

### 5. Command Line Arguments
- **`src/train.py`**: Added "qwen25" to model type choices
- **`src/evaluate_gsm8k.py`**: Added "qwen25" to model type choices
- **`src/evaluate_cross_model.py`**: Added "qwen25" to critic model choices
- **`src/test_tokenizers.py`**: Added "qwen25" to tokenizer test choices

### 6. Evaluation Scripts
- Updated `evaluate_gsm8k.py` to support all model types including qwen25
- Updated `evaluate_cross_model.py` to handle qwen25 model name mapping

## Key Technical Details

### Tokenization
- "Answer:" is tokenized as `[16141, 25]` in Qwen2.5
- EOS token: `<|im_end|>` (ID: 151645)
- Pad token: `<|endoftext|>` (ID: 151643)

### Prompt Format Example
```
<|im_start|>system
You are a helpful AI assistant that solves problems step by step.<|im_end|>
<|im_start|>user
[Question or task]<|im_end|>
<|im_start|>assistant
[Response]
```

### Usage
To use Qwen2.5 in training:
```bash
python src/train.py --model_type qwen25 --task_type arithmetic
```

To evaluate with Qwen2.5:
```bash
python src/evaluate_gsm8k.py --model_type qwen25 --model_path [path]
```

## Notes
- Qwen2.5 uses ChatML format similar to Phi-4 but without the `<|im_sep|>` token
- The model does not require `trust_remote_code=True`
- Supports context length up to 131,072 tokens 