# Complete Guide: Dataset Answer Formats and Handling Functions

## Overview

This guide walks through every dataset in the repository, showing:
1. **Raw answer format** from the dataset source
2. **Processed format** after loading
3. **Model output format** (what the model generates)
4. **Extraction function** used
5. **Comparison logic** for correctness

---

## Table of Contents

1. [GSM8K (Grade School Math)](#1-gsm8k)
2. [SVAMP (Math Word Problems)](#2-svamp)
3. [MMLU (Multiple Choice)](#3-mmlu)
4. [AQuA (Multiple Choice Math)](#4-aqua)
5. [MathQA (Multiple Choice Math)](#5-mathqa)
6. [ARC (Science Multiple Choice)](#6-arc)
7. [MATH (Competition Math) - Legacy](#7-math-legacy)
8. [Arithmetic (Synthetic) - Training Only](#8-arithmetic)

---

## 1. GSM8K

### Dataset: Grade School Math (8K problems)

#### Raw Answer Format
```python
# From HuggingFace dataset
'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n
She makes 9 * 2 = $<<9*2=18>>18 every day.\n
#### 18'
```

#### After load_gsm8k_dataset() Processing
```python
# File: src/utils.py, line 578
answers = list(map(lambda x: x[x.index("####") + 5 :], ds[split]["answer"]))

# Result: '18' (clean string)
```

**Sample values**: `'18'`, `'3'`, `'70000'`

#### Model Output Format
```
"The answer is 18" 
"18\nExplanation: ..."
"42"
"= 18"
```

#### Extraction Function
```python
# File: src/utils.py, lines 636-656
def extract_answer(answer):
    """Extract numerical answer from various text formats."""
    # Handle GSM8K format with ####
    if "####" in answer:
        answer = answer[answer.index("####") + 5:].strip()
    
    # Handle answers with = sign
    if "=" in answer:
        answer = answer.split("=")[-1].strip()
    
    # Remove commas
    answer = answer.replace(",", "")
    
    # Find first number (including negative)
    matches = re.findall(r"-?\d+", answer.strip())
    if matches:
        return int(matches[0])  # ← Returns INTEGER
    else:
        return "[invalid]"
```

#### Evaluation Pipeline
```python
# File: src/evaluate_gsm8k.py, lines 243-248
extracted_answers = [extract_answer(ans) for ans in generated_answers]

for q, a, cot, gen_a, ext_a in zip(...):
    correct_answer = extract_answer(a)  # ← Gold also extracted
    is_correct = (ext_a == correct_answer)  # ← Integer comparison
```

**Key Points**:
- ✅ Both predicted AND gold go through `extract_answer()`
- ✅ Returns `int` type for numerical values
- ✅ Type-safe comparison: `int == int`
- ✅ Most robust evaluation in the codebase

---

## 2. SVAMP

### Dataset: Simple Variations on Arithmetic Math Word Problems

#### Raw Answer Format
```python
# From HuggingFace dataset (ChilleD/SVAMP or local JSON)
{
    "Body": "Winter is almost here and most animals are migrating...",
    "Answer": 27  # Can be int or string
}
```

#### After load_svamp_dataset() Processing
```python
# File: src/utils.py, lines 875-877
if "Body" in item and "Answer" in item:
    q = item["Body"]
    a = str(item["Answer"]).strip()  # ← Convert to string

# Result: '27' (string)
```

**Sample values**: `'27'`, `'4'`, `'16'`

#### Model Output Format
```
"13\nThe problem requires..."
"The answer is 42"
"= 27"
```

#### Extraction Function
```python
# Same as GSM8K: extract_answer() from src/utils.py
# Returns int or "[invalid]"
```

#### Evaluation Pipeline
```python
# File: src/train.py, lines 1636-1675
def evaluate_model_on_numeric(...):
    def normalize_numeric(text: str) -> str:
        """Normalize by removing LaTeX, whitespace, etc."""
        s = text.strip()
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        s = s.replace("$", "").replace("\\", "")
        s = re.sub(r"\s+", "", s)
        return s
    
    def compare_normalized(pred: Union[int, str], gold: str) -> bool:
        return normalize_numeric(str(pred)) == normalize_numeric(str(gold))
    
    return evaluate_model_generic(
        answer_extractor_fn=extract_answer,  # ← Only on predicted
        answer_comparator_fn=compare_normalized,
        ...
    )
```

**Key Points**:
- ⚠️ Only predicted goes through `extract_answer()`, gold only normalized
- ⚠️ Less robust than GSM8K (no extraction on gold)
- ✅ Fixed Nov 16, 2025 - was using `lambda x: x` before (bug!)
- ⚠️ Different pipeline than GSM8K despite similar task

**Historical Bug** (before commit `1916804`):
```python
# WRONG (before Nov 16, 2025):
answer_extractor_fn=lambda x: x,  # Returns raw text!

# Model generates: "13\nThe problem..."
# Comparator tries: "13\nThe problem..." == "13"
# Result: FALSE even when answer is correct!
```

---

## 3. MMLU

### Dataset: Massive Multitask Language Understanding (57 subjects)

#### Raw Answer Format
```python
# From HuggingFace dataset
{
    "question": "Find the degree for the given field extension...",
    "choices": ["0", "4", "2", "6"],
    "answer": 1  # Index into choices array
}
```

#### After load_mmlu_dataset() Processing
```python
# File: src/utils.py, lines 814-819
for item in data:
    q = format_mmlu_question(item["question"], item["choices"])
    # Convert index to letter
    answer_idx = int(item["answer"])
    answer_letter = chr(ord("A") + answer_idx)
    qa_pairs.append((q, answer_letter))

# Result: 'B' (letter)
```

**Sample values**: `'B'`, `'C'`, `'D'`

#### Question Format
```
Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.

Choices:
A. 0
B. 4
C. 2
D. 6
```

#### Model Output Format
```
"B"
"The answer is B"
"I think B is correct"
"b"  # Lowercase handled
```

#### Extraction Function
```python
# File: src/train.py, lines 1565-1568
def extract_letter(text: str) -> str:
    """Extract first letter A-E from text. Returns 'X' if none found."""
    matches = re.findall(r"[A-E]", text.upper())
    return matches[0] if matches else "X"
```

#### Evaluation Pipeline
```python
# File: src/train.py, lines 1570-1575
return evaluate_model_generic(
    actor_model, critic_model, tokenizer, device, test_data,
    hyperparameters, 
    extract_letter,  # ← Extraction function
    answer_comparator_fn=None,  # ← Uses default equality
    ...
)
```

**Key Points**:
- ✅ Simple letter extraction
- ✅ Case-insensitive (converts to uppercase)
- ✅ Default equality comparison: `'B' == 'B'`
- ✅ Returns 'X' for invalid/no match (never matches A-E)

---

## 4. AQuA

### Dataset: Algebraic Question Answering (Multiple Choice)

#### Raw Answer Format
```python
# From HuggingFace dataset
{
    "question": "A car is being driven...",
    "options": ["A)30", "B)35", "C)40", "D)45", "E)50"],
    "correct": "A"  # Direct letter
}
```

#### After load_aqua_dataset() Processing
```python
# File: src/utils.py, lines 967-973
for item in data:
    if "question" in item and "options" in item and "correct" in item:
        q = format_aqua_question(item["question"], item["options"])
        a = item["correct"].strip()
        qa_pairs.append((q, a))

# Result: 'A' (letter)
```

**Sample values**: `'A'`, `'E'`, `'C'`

#### Question Format
```
A car is being driven towards a tower...

Choices:
A. 30
B. 35
C. 40
D. 45
E. 50
```

#### Extraction & Evaluation
```python
# File: src/train.py, lines 1623-1633
def extract_letter(text: str) -> str:
    """Same as MMLU"""
    matches = re.findall(r"[A-E]", text.upper())
    return matches[0] if matches else "X"

return evaluate_model_generic(
    ..., extract_letter, ...
)
```

**Key Points**:
- ✅ Identical to MMLU extraction
- ✅ 5-choice format (A-E)

---

## 5. MathQA

### Dataset: Math Word Problems (Multiple Choice)

#### Raw Answer Format
```python
# From HuggingFace dataset
{
    "Problem": "a shopkeeper sold an article...",
    "options": "a ) 38 % , b ) 41 % , c ) 42 % , d ) 43 % , e ) 44 %",
    "correct": "a"  # Lowercase letter
}
```

#### After load_mathqa_dataset() Processing
```python
# File: src/utils.py, lines 1050-1068
for item in data:
    if "Problem" in item and "options" in item and "correct" in item:
        # Parse options string into list
        opts = parse_mathqa_options(item["options"])
        q = format_mathqa_question(item["Problem"], opts)
        a = item["correct"].strip().upper()  # ← Uppercase
        qa_pairs.append((q, a))

# Result: 'A' (uppercase letter)
```

**Sample values**: `'A'`, `'C'`, `'E'`

#### Extraction & Evaluation
```python
# Uses same evaluate_model() as GSM8K (can handle both numeric and MCQ)
# Extracts with extract_answer() which looks for letters too
```

**Key Points**:
- ✅ Converted to uppercase during loading
- ✅ 5-choice format (A-E)
- ⚠️ Uses GSM8K evaluation function (more complex than needed)

---

## 6. ARC

### Dataset: AI2 Reasoning Challenge (Science Questions)

#### Raw Answer Format
```python
# From HuggingFace dataset (ARC-Challenge or ARC-Easy)
{
    "question": "An astronomer observes that a planet...",
    "choices": {
        "text": ["The planet's density decreases.", "The planet's gravity decreases.", ...],
        "label": ["A", "B", "C", "D"]
    },
    "answerKey": "C"  # Direct letter
}
```

#### After load_arc_dataset() Processing
```python
# File: src/utils.py, lines 1189-1206
for item in data:
    question_text = item["question"]
    choices_text = item["choices"]["text"]
    choices_label = item["choices"]["label"]
    
    # Format as multiple choice
    q = format_arc_question(question_text, choices_text, choices_label)
    a = item["answerKey"].strip().upper()
    qa_pairs.append((q, a))

# Result: 'C' (uppercase letter)
```

**Sample values**: `'C'`, `'B'`, `'A'`

#### Extraction & Evaluation
```python
# File: src/train.py
# Uses evaluate_model_on_aqua (same letter extraction as MMLU/AQuA)
```

**Key Points**:
- ✅ 4-choice format (A-D) typically
- ✅ Same extraction as other MCQ datasets

---

## 7. MATH (Legacy)

### Dataset: Hendrycks Competition Math

**Note**: This dataset is supported but may not be actively used in current experiments.

#### Raw Answer Format
```python
# From HuggingFace dataset
{
    "problem": "Compute $\\arcsin 1$...",
    "solution": "We have that $\\sin \\frac{\\pi}{2} = 1...\\boxed{\\frac{\\pi}{2}}$"
}
```

#### After load_math_dataset() Processing
```python
# File: src/utils.py, lines 588-611
def extract_math_answer(solution_text: str) -> str:
    """Heuristics:
    - Prefer last \boxed{...} content
    - Fallback to last fraction or integer
    - Else last non-empty line
    """
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    
    frac_or_int = re.findall(r"-?\d+\/\d+|-?\d+", text)
    if frac_or_int:
        return frac_or_int[-1].strip()
    
    # Last non-empty line
    return last_line

# Result: '\\frac{\\pi}{2}' or '42' (varies by problem)
```

#### Evaluation
```python
# Uses evaluate_model_on_numeric (same as SVAMP)
# Normalizes LaTeX formatting
```

**Key Points**:
- ⚠️ Complex answer formats (LaTeX, fractions)
- ✅ Special extraction logic for boxed answers
- ⚠️ May need more robust comparison

---

## 8. Arithmetic (Training Only)

### Synthetic dataset generated on-the-fly

#### Generation
```python
# File: src/train.py
# Generates simple addition/subtraction problems
# Example: "15 + 23 + 45"
# Answer: "83"
```

#### Answer Format
```python
# Clean integer strings: '83', '42', '-15'
```

#### Evaluation
```python
# Uses same extract_answer() and comparison as GSM8K
```

---

## Summary Table

| Dataset | Answer Type | Gold Processing | Predicted Processing | Comparison |
|---------|-------------|-----------------|---------------------|------------|
| **GSM8K** | Numeric (int) | `extract_answer()` | `extract_answer()` | `int == int` ✅ |
| **SVAMP** | Numeric (int) | `normalize_numeric()` | `extract_answer()` | String comparison ⚠️ |
| **MMLU** | Letter (str) | Direct | `extract_letter()` | `str == str` ✅ |
| **AQuA** | Letter (str) | Direct | `extract_letter()` | `str == str` ✅ |
| **MathQA** | Letter (str) | `.upper()` | `extract_answer()` | Complex ⚠️ |
| **ARC** | Letter (str) | `.upper()` | `extract_letter()` | `str == str` ✅ |
| **MATH** | Various | `extract_math_answer()` | `extract_answer()` | Normalized ⚠️ |
| **Arithmetic** | Numeric (int) | Direct | `extract_answer()` | Training only |

---

## Key Functions Reference

### 1. extract_answer() - Primary Numeric Extraction
**Location**: `src/utils.py:636-656`

**Input**: Any string
**Output**: `int` or `"[invalid]"`

**Logic**:
1. Strip `####` prefix if present
2. Take text after `=` if present
3. Remove commas
4. Extract first integer with regex `r"-?\d+"`

**Used by**: GSM8K, SVAMP, Arithmetic, MathQA

### 2. extract_letter() - Multiple Choice Extraction
**Location**: `src/train.py:1565-1568` (MMLU), `1623-1626` (AQuA)

**Input**: Any string
**Output**: First letter A-E (uppercase) or 'X'

**Logic**:
1. Convert to uppercase
2. Find first match of `[A-E]`
3. Return 'X' if no match

**Used by**: MMLU, AQuA, ARC

### 3. normalize_numeric() - LaTeX/Formatting Cleanup
**Location**: `src/train.py:1651-1657`

**Input**: String
**Output**: Normalized string

**Logic**:
1. Strip whitespace
2. Remove `\boxed{...}` LaTeX
3. Remove `$` and `\`
4. Remove all whitespace

**Used by**: SVAMP, MATH (via `compare_normalized`)

### 4. extract_math_answer() - Competition Math Extraction
**Location**: `src/utils.py:588-611`

**Input**: Solution text
**Output**: Extracted answer string

**Logic**:
1. Look for last `\boxed{...}`
2. Fallback to last fraction/integer
3. Fallback to last non-empty line

**Used by**: MATH dataset loader

---

## Recommendations for Consistency

### Issue 1: SVAMP vs GSM8K Inconsistency

**Problem**: SVAMP doesn't extract gold answers, GSM8K does.

**Solution**: Make SVAMP match GSM8K:

```python
def evaluate_model_on_numeric(...):
    return evaluate_model_generic(
        answer_extractor_fn=extract_answer,
        # NEW: Extract both predicted and gold
        answer_comparator_fn=lambda pred, gold: pred == extract_answer(gold),
        ...
    )
```

### Issue 2: Type Safety

**Current**: Mix of `int` and `str` comparisons

**Better**: Always convert to same type before comparison:

```python
# Either both int:
int(pred) == int(gold)

# Or both str:
str(pred) == str(gold)
```

### Issue 3: Error Handling

**Add validation**:
```python
if extract_answer(text) == "[invalid]":
    # Log this case for debugging
    # Count as incorrect
    is_correct = False
```

---

## Testing Your Understanding

Run this to test each dataset's extraction:

```python
from src.utils import extract_answer
from src.train import evaluate_model_on_mmlu

# Test numeric extraction
print(extract_answer("The answer is 42"))  # → 42
print(extract_answer("42\nExplanation"))   # → 42
print(extract_answer("#### 42"))            # → 42

# Test letter extraction (from MMLU code)
def extract_letter(text):
    import re
    matches = re.findall(r"[A-E]", text.upper())
    return matches[0] if matches else "X"

print(extract_letter("I think B"))  # → 'B'
print(extract_letter("b is correct"))  # → 'B'
print(extract_letter("None"))  # → 'X'
```

---

## Quick Reference: Which Function for Which Dataset?

```python
# Import the right functions:
from src.utils import extract_answer  # For: GSM8K, SVAMP, Arithmetic
from src.train import evaluate_model  # For: GSM8K, MathQA
from src.train import evaluate_model_on_mmlu  # For: MMLU
from src.train import evaluate_model_on_aqua  # For: AQuA, ARC
from src.train import evaluate_model_on_numeric  # For: SVAMP, MATH

# Evaluation mapping:
evaluators = {
    "gsm8k": evaluate_model,
    "svamp": evaluate_model_on_numeric,
    "mmlu": evaluate_model_on_mmlu,
    "aqua": evaluate_model_on_aqua,
    "mathqa": evaluate_model,
    "arc": evaluate_model_on_aqua,  # Uses same letter extraction
    "math": evaluate_model_on_numeric,
}
```

