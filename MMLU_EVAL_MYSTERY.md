# MMLU Evaluation Mystery

## Problem Statement

There's a significant discrepancy in MMLU evaluation results between two different evaluation methods on the **same checkpoint** (batch 1000):

- **test_mmlu_prompts.py**: 48.23% accuracy (word boundary regex)
- **retroactive_eval.py**: 18.44% accuracy (word boundary regex)

This is a **~30 percentage point difference** that needs investigation.

## What We Know

### Both Methods Appear Identical

Both evaluation methods seem to use:
1. ✅ Same checkpoint: `results/mmlu/20251116_191617/adapter_1000`
2. ✅ Same model: Llama-3.1-8B with LoRA adapter
3. ✅ Same adapter loading: `PeftModel.from_pretrained(base_model, checkpoint_path)`
4. ✅ Same prompts: Markovian mode with `<Redacted>` for questions during answer generation
5. ✅ Same generation parameters: `do_sample=False`, `max_new_tokens=10`
6. ✅ Same test data: MMLU test set with stride=100 (140 examples)

### Output Differences

**retroactive_eval.py output** (18.44% accuracy):
```
Question: Find the degree for the given field extension...
CoT: 6)). The degree of a field extension...
Generated answer: '2 and 3, which are already included in'
Predicted (old): A (extracted from "alre**A**dy")
Gold: B
```

The model is generating garbage text instead of clean letter answers.

**test_mmlu_prompts.py output** (48.23% accuracy):
The model appears to generate better, more coherent outputs that correctly contain answer letters.

## Debugging Steps Taken

1. ✅ Verified actor_reward_weight=1.0 in both cases
2. ✅ Confirmed adapter is loaded (`hasattr(answer_model, 'peft_config')` returns True)
3. ✅ Verified both use actor model for answer generation (after fix)
4. ✅ Checked prompts are identical
5. ✅ Compared generation parameters

## Hypotheses to Investigate

### 1. Model State Difference
- Could there be a difference in model.eval() vs model.train() mode?
- Is there a batch normalization or dropout layer behaving differently?
- Are there any other model flags that differ?

### 2. Adapter Merging
- Does one method merge the adapter while the other doesn't?
- Could there be a merge_and_unload() call somewhere?

### 3. Tokenizer State
- Are padding tokens handled differently?
- Is there a difference in attention masks?
- Could truncation be cutting off important context?

### 4. CUDA/Memory State
- Could CUDA cache state affect inference?
- Are there memory pressure differences?

### 5. Random Seed
- Even with do_sample=False, could there be randomness somewhere?
- Are there any dropout layers still active?

### 6. Generation Context
- Is the full prompt being passed to the model?
- Could there be a difference in how the CoT is concatenated?

### 7. Checkpoint Loading
- Is the checkpoint actually the same between both methods?
- Could there be a checkpoint version mismatch?

## Next Steps

1. Add extensive logging to both evaluation methods to capture:
   - Exact prompts being used
   - Model state flags
   - Adapter configuration
   - First 3 examples with full outputs

2. Run both methods side-by-side on the same checkpoint with identical logging

3. Compare outputs line-by-line to identify the divergence point

4. Once found, implement fix and verify scores align

## Temporary Workaround

Until this is resolved, be cautious about trusting MMLU evaluation scores from retroactive_eval.py. The test_mmlu_prompts.py script appears to give more reasonable results.

## Related Files

- `test_mmlu_prompts.py` (lines 84-204)
- `src/retroactive_eval.py` (evaluation logic)
- `src/train.py` (evaluate_model_generic, lines 1409-1579)

## Status

🔴 **UNRESOLVED** - Needs investigation after evaluation refactor is complete.

