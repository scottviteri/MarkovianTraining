# Comprehensive Training Analysis

## Overview

Analysis of 10 training runs across 7 datasets using Llama 3.1 8B with Markovian training (actor_reward_weight=1.0).

---

## Hyperparameter Summary

### Common Configuration (All Runs)
- **Model**: Llama 3.1 8B (`meta-llama/Llama-3.1-8B-Instruct`)
- **Training Mode**: Markovian with Actor Reward (actor_reward_weight=1.0)
- **Parallel Sampling**: True (GRPO-style)
- **LoRA**: rank=8, alpha=16
- **Learning Rate**: 0.0001
- **KL Penalty**: 0.1
- **Temperature**: 1.0 (1.5 for wiki_continuation)
- **Normalize Loss**: True

### Task-Specific Variations

| Dataset | CoT Length | Batch Size | Use PPO | Num Batches Trained | Final Batch |
|---------|-----------|------------|---------|-------------------|-------------|
| GSM8K | 100 | 12 | No | 8,056 | 8000 |
| MMLU (run 1) | 150 | 4 | No | 5,908 | 5000 |
| MMLU (run 2) | 150 | 12 | No | 5,675 | ~5600 |
| AQuA (run 1) | 50 | 12 | No | 8,672 | 8000 |
| AQuA (run 2) | 50 | 12 | No | 13,195 | ~13000 |
| SVAMP (run 1) | 50 | 12 | No | 10,566 | 10000 |
| SVAMP (run 2) | 50 | 12 | No | 16,355 | ~16000 |
| ARC | 50 | 12 | **Yes** | 6,828 | 6000 |
| MathQA | 150 | 12 | No | 4,393 | ~4000 |
| Wiki Cont. | 50 | 16 | No | 5,195 | ~5000 |

**Key Observation**: Only ARC uses PPO (use_ppo=True), all others use policy gradient.

---

## Performance Results

### Final Accuracies

| Dataset | Task Type | Final Accuracy | Word Boundary Accuracy | Status |
|---------|-----------|---------------|----------------------|--------|
| **GSM8K** | Numeric | 5.23% | N/A | ⚠️ Very Poor |
| **MMLU** | MCQ (4-choice) | 18.00% | 24.00% | ⚠️ Poor |
| **AQuA** | MCQ (5-choice) | 3.54% | 11.42% | ⚠️ Very Poor |
| **SVAMP** | Numeric | 23.33% | N/A | ⚠️ Poor |
| **ARC** | MCQ (4-choice) | 0.00% | 0.00% | ❌ **Mode Collapse** |
| **MathQA** | MCQ (5-choice) | *(not evaluated)* | N/A | Unknown |
| **Wiki Cont.** | Text Gen | *(not evaluated)* | N/A | Unknown |

**Critical Finding**: These are surprisingly low accuracies, especially given the paper claims 54.5% on GSM8K. These appear to be early/failed runs.

---

## Detailed Findings by Dataset

### 1. GSM8K (Grade School Math)

**Hyperparameters**:
- CoT Length: 100 tokens
- Batch Size: 12
- Final Batch: 8000

**Training Dynamics**:
```
Batch 0:    Loss=0.0457, Norm Reward=-2.8125, Actor Log Probs=-3.8125
Batch 999:  Loss=0.0099, Norm Reward=-0.3145, Actor Log Probs=-3.3594
Batch 3999: Loss=-0.098, Norm Reward=0.2314,  Actor Log Probs=-1.5312
Batch 7999: Loss=-0.153, Norm Reward=0.3945,  Actor Log Probs=-0.0894
```

**Analysis**:
- ✅ Stable training progression
- ✅ Actor log probs improve from -3.8 to -0.089 (higher confidence)
- ✅ Normalized reward becomes positive
- ❌ Final accuracy only 5.23% (expected ~50%+)

**Example Correct Reasoning**:
```
Question: "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs..."
Reasoning: "So the steps you will take to solve the problem are 125, 125 * 4, 
           4 *125 = 500, 2 less than 500, 500 - 2 = 498..."
Answer: 623 ✓
```

**Example Incorrect Reasoning**:
```
Question: "Janet's ducks lay 16 eggs per day. She eats three for breakfast..."
Reasoning: "afternoon. If she takes the eggs that are not eaten by the ducks, 
           she can make 2 muffin batter cups per egg..."
Predicted: 13, Actual: 18 ✗
```

**Issues**:
1. Model often misunderstands the problem
2. Arithmetic errors in multi-step reasoning
3. Sometimes extracts wrong number from generated text

---

### 2. ARC (AI2 Reasoning Challenge) - **MODE COLLAPSE**

**Hyperparameters**:
- CoT Length: 50 tokens
- Batch Size: 12
- **Use PPO: True** (unique among all runs)
- Final Batch: 6000

**Training Dynamics**:
```
Batch 0:    Loss=-0.0003, Norm Reward=0.2129,   Actor Log Probs=-1.6328,  PPO Ratio=1.0000
Batch 99:   Loss=-0.0247, Norm Reward=0.2002,   Actor Log Probs=-1.8750,  PPO Ratio=1.1719
Batch 999:  Loss=6.6562,  Norm Reward=-39.2500, Actor Log Probs=-41.0000, PPO Ratio=21.7500 ⚠️
Batch 2999: Loss=9.7500,  Norm Reward=-33.5000, Actor Log Probs=-35.2500, PPO Ratio=26.2500
Batch 5999: Loss=-0.9688, Norm Reward=-33.7500, Actor Log Probs=-35.5000, PPO Ratio=8.1250
```

**Analysis**:
- ❌ **Catastrophic mode collapse at batch ~999**
- ❌ PPO ratio explodes to 21.75 (far beyond epsilon=0.2 clipping range)
- ❌ Actor log probs crash from -1.8 to -41.0
- ❌ Normalized reward becomes highly negative (-39.25)
- ❌ Final accuracy: 0.00%

**Collapsed Output**:
```
Reasoning: "44444444444444444444444444444444444444444444444444..."
Predicted: X (no valid letter extracted)
```

**Root Cause**: PPO instability with actor-reward training. The policy update ratio exceeded clipping bounds, causing the model to converge to a degenerate policy that repeats characters.

**Recommendation**: 
- Use policy gradient (no PPO) for actor-reward mode
- Or reduce learning rate significantly if using PPO
- Or increase PPO clipping epsilon (but this defeats the purpose)

---

### 3. SVAMP (Simple Variations on Arithmetic)

**Hyperparameters**:
- CoT Length: 50 tokens
- Batch Size: 12
- Final Batch: 10000

**Results**:
- Final Accuracy: 23.33%
- **Best performing dataset** in this batch of runs

**Example Correct**:
```
Question: "Paco had 22 sweet cookies and 75 salty cookies. He ate 15 sweet cookies and 54 salty cookies."
Reasoning: "...leaving 22 - 15 = 7 sweet cookies..."
Predicted: 7, Actual: 7 ✓
```

**Example Incorrect**:
```
Question: "Winter is almost here and most animals are migrating to warmer countries. 
          There are 41 bird families..."
Reasoning: "15 birds had moved to a warmer country and 21 birds have died, that's 36 birds 
           that have left their home or died. There is a total of 41 - 36 = 5 bird..."
Predicted: 5, Actual: 27 ✗
```

**Analysis**:
- ✅ Better performance than GSM8K
- ⚠️ Still struggles with multi-entity problems
- ⚠️ Gets confused between "families" and "birds"

---

### 4. MMLU (Massive Multitask Language Understanding)

**Hyperparameters**:
- CoT Length: 150 tokens (longest)
- Run 1: Batch Size 4
- Run 2: Batch Size 12

**Results**:
- Legacy Accuracy: 18.00%
- Word Boundary Accuracy: 24.00%

**Example Incorrect**:
```
Question: "The Coriolis effect is observed on planets because"
Choices: A-D
Reasoning: "C. they are not rotating"
Predicted: C, Actual: A ✗
```

**Analysis**:
- ⚠️ Low accuracy for multiple choice
- ⚠️ Model generates incorrect reasoning
- ⚠️ May need longer CoT for complex reasoning

---

### 5. AQuA (Algebraic Question Answering)

**Hyperparameters**:
- CoT Length: 50 tokens
- Two runs: 8000 and 13000 batches

**Results**:
- Legacy Accuracy: 3.54%
- Word Boundary Accuracy: 11.42%

**Example Incorrect**:
```
Question: "A car is being driven, in a straight line and at a uniform speed, 
          towards the base of a vertical tow..."
Reasoning: "a cliff. One could simply call a driver who leaves the base of the cliff 
           and crashes on the ground with no more pieces than one is given a drop. Howe..."
Predicted: X, Actual: A ✗
```

**Analysis**:
- ❌ Model generates incoherent reasoning
- ❌ Very poor extraction (many "X" predictions)
- ❌ Possible training instability

---

## Key Patterns and Insights

### What's Working

1. **Stable Training (Most Runs)**:
   - Policy gradient without PPO shows stable loss curves
   - Actor log probabilities consistently improve
   - No catastrophic forgetting in most runs

2. **Some Reasoning Capability**:
   - Model can perform basic arithmetic steps
   - Can structure multi-step solutions
   - Shows understanding of problem decomposition

3. **Markovian Property**:
   - Model generates self-contained reasoning
   - CoT doesn't rely on seeing the original question

### What's Failing

1. **PPO + Actor Reward = Mode Collapse**:
   - ARC run collapsed completely with PPO
   - PPO ratio exploded beyond clipping bounds
   - Suggests incompatibility between PPO and actor-reward gradients

2. **Low Absolute Performance**:
   - GSM8K: 5.23% vs. paper claim of 54.5%
   - SVAMP: 23.33% (best result, but still poor)
   - ARC: 0% (collapsed)
   - Suggests these are early/failed experiments

3. **Reasoning Quality Issues**:
   - Misunderstanding problem statements
   - Arithmetic errors
   - Incoherent or repetitive text generation
   - Wrong number extraction

4. **No Evaluation During Training**:
   - All runs have `eval_frequency: null`
   - Only evaluated at the very end
   - Can't see if there was ever good performance during training

---

## Hypotheses for Poor Performance

### 1. Training Not Complete
- These appear to be early/intermediate checkpoints
- Paper results likely from much longer training
- Need 50K+ batches for convergence?

### 2. Actor-Reward Weight Too High
- All runs use `actor_reward_weight=1.0`
- May need to balance with critic gradients
- Paper may use lower values (e.g., 0.1-0.5)

### 3. Batch Size Too Small
- Most runs use batch_size=12
- GRPO benefits from larger batches for baseline estimation
- Paper may use 32-64 batch size

### 4. Learning Rate Too High
- lr=0.0001 may be too aggressive
- Could cause instability in RL fine-tuning
- Try 1e-5 or 5e-5

### 5. Temperature Settings
- Temperature=1.0 during CoT generation
- May need lower temp (0.7-0.8) for more focused reasoning
- Or curriculum from high to low temp

### 6. Evaluation Timing
- No intermediate evaluations
- Can't tell if model peaked earlier
- Should set `eval_frequency=500-1000`

---

## Training Metric Patterns

### Healthy Training (GSM8K):
```
Early:  Loss positive, rewards negative, high entropy
Middle: Loss decreases, rewards increase
Late:   Loss negative, rewards positive, log probs improve
```

### Collapsed Training (ARC):
```
Early:  Normal behavior
Batch 999: PPO ratio explodes (>20x)
Middle: Extreme negative rewards (-40)
Late:   Model stuck in degenerate policy
```

---

## Recommendations

### For Future Training Runs:

1. **Disable PPO with Actor Rewards**:
   ```python
   use_ppo=False  # when actor_reward_weight > 0
   ```

2. **Enable Regular Evaluation**:
   ```python
   eval_frequency=1000  # Check every 1000 batches
   checkpoint_frequency=1000
   ```

3. **Increase Batch Size**:
   ```python
   batch_size=32  # Better for GRPO baseline estimation
   ```

4. **Lower Learning Rate**:
   ```python
   lr=5e-5  # More stable for RL
   ```

5. **Add Gradient Clipping**:
   ```python
   max_grad_norm=1.0  # Prevent exploding gradients
   ```

6. **Start with Lower Actor Weight**:
   ```python
   actor_reward_weight=0.5  # Gradually increase
   ```

7. **Monitor for Collapse**:
   - Watch PPO ratio (should stay near 1.0)
   - Track reward distribution
   - Check for repetitive outputs

### For Analysis:

1. **Check Earlier Checkpoints**:
   - Adapter_1000, adapter_2000, etc.
   - May show better performance before overfitting

2. **Compare with Base Model**:
   - Evaluate Llama 3.1 8B without training
   - Establish baseline performance

3. **Visualize Training Curves**:
   - Plot loss, reward, log probs over time
   - Identify when training went wrong

---

## Evaluation Methodology Notes

### Answer Extraction Methods:

1. **Numeric Tasks (GSM8K, SVAMP)**:
   - Anchor method: Prioritizes "Answer:" label
   - Simple method: First number in text
   - Both have issues with complex outputs

2. **MCQ Tasks (MMLU, ARC, AQuA, MathQA)**:
   - Legacy: First letter A-E found
   - Word Boundary: Isolated letter only (better)
   - ARC showing 0% with both methods (complete failure)

### Data Quality:

- GSM8K: 1,319 test examples
- MMLU: 154 examples (sampled)
- ARC: 294 examples
- SVAMP: 30 examples (very small!)
- AQuA: 254 examples

Small evaluation sets (especially SVAMP) make accuracy estimates noisy.

---

## Summary

**Current Status**: These runs show **early-stage training** with **poor final performance**. The Llama 3.1 8B model is learning some reasoning capability but hasn't converged to high accuracy yet.

**Critical Issue**: ARC run demonstrates that **PPO + Actor Rewards is unstable**, leading to catastrophic mode collapse.

**Next Steps**:
1. Evaluate earlier checkpoints (adapter_1000-4000)
2. Compare against base model performance
3. Run longer training with better hyperparameters
4. Implement recommendations above

**Paper Claims vs. Current Results**:
- Paper: GSM8K 54.5%, ARC 76.9%
- These runs: GSM8K 5.23%, ARC 0%
- **Gap suggests these are preliminary experiments, not final results**

---

*Analysis Date: November 18, 2025*
*Total Training Batches Analyzed: 84,852*
*Datasets: 7 (GSM8K, MMLU, ARC, AQuA, MathQA, SVAMP, Wiki Continuation)*

