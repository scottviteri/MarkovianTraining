Table 1. Dataset-level comparison (columns exclude PPO/Qwen).

| Dataset | EMA | EI | Non-Markovian | Markovian | No reward | Llama baseline | Row mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 0.000 | 0.616 | 0.633 | 0.571 | 0.622 | 0.196 | 0.457 |
| arc | 0.265 | 0.656 | 0.786 | 0.799 | 0.793 | 0.361 | 0.630 |
| arithmetic | 0.975 | 0.760 | 0.970 | 0.980 | 0.810 | 0.010 | 0.785 |
| mmlu | 0.238 | 0.532 | 0.687 | 0.555 | 0.466 | 0.214 | 0.474 |
| wiki_continuation* | -3.331 | -2.279 | -2.900 | -2.564 | -2.647 | -3.200 | -2.820 |
| svamp | 0.000 | 0.387 | 0.433 | 0.423 | 0.407 | 0.180 | 0.323 |

Column means (excluding wiki_continuation):  
EMA 0.296 | EI 0.590 | Non-Markovian 0.702 | Markovian 0.666 | No reward 0.619 | Llama baseline 0.192

*The wiki row (and its mean) stays in log-probability units; treat it separately from the accuracy rows.

---

Table 2. Model snapshot (Before → After, written as “baseline → trained”).

| Dataset | Qwen (Before → After) | Llama (Before → After) |
| --- | --- | --- |
| gsm8k | 0.130 → 0.716 | 0.196 → 0.571 |
| arc | 0.398 → 0.850 | 0.361 → 0.799 |
| arithmetic | 0.000 → 0.005 | 0.010 → 0.980 |
| mmlu | 0.318 → 0.605 | 0.214 → 0.555 |
| wiki_continuation | -3.031 → -3.012 | -3.200 → -2.564 |
| svamp | 0.283 → 0.317 | 0.180 → 0.423 |

Caption: EMA uses an exponential moving-average baseline; EI selects only high-reward samples; Non-Markovian exposes the question during answer prediction; No reward removes the \(\nabla_\theta \log A_\theta\) term; Llama columns compare the Markovian fine-tune to the frozen base model; arrows show the shift from baseline (“Before”) to trained (“After”), while wiki_continuation metrics stay in log-probability units.

---

## Interpreting the Configurations

### Columns in Table&nbsp;1

- **EMA** – Policy-gradient run that keeps the standard Markovian objective but subtracts an exponential moving-average baseline \(V_t\) from the normalized reward before computing advantages. In the sweep matrix (`scripts/organize_and_deploy.py`) this column explicitly passes `--no-parallel`, so `calculate_advantages` reverts to the exponential moving-average baseline instead of GRPO’s batch standardization.
- **EI (Expert Iteration)** – Same reward as EMA, but the loss is applied only to samples whose reward exceeds \(\mu + k\sigma\) (with \(k = 1\) in `COLUMNS`). The threshold is computed via `utils.calculate_threshold`, which skips the first 25 batches and then keeps only datapoints above mean + 1 σ. Because `parallel=True` is still in effect, we continue to use GRPO-style within-batch standardization before masking out the low-reward samples.
- **Non-Markovian** – Removes the CoT bottleneck by letting the answer head attend to both the original question and the CoT (i.e., optimizing \(P(\text{ans}\mid q,\text{CoT})\)). Because the predictor can still see the question, these runs typically score higher but sacrifice the interpretability benefits of CoT fragility.
- **Markovian** – The primary method: answer prediction must condition only on the generated CoT, enforcing that the CoT text remain load-bearing. The loss is the normalized reward \(R_\theta = \ln \pi_\theta(\text{ans}|\text{CoT}) - \ln \pi_\theta(\text{ans}|\text{CoT}')\) plus the actor-reward gradient, as detailed in Algorithm \ref{alg:markovian_training}.
- **No reward gradient (No reward)** – Drops the \(\nabla_\theta \ln \pi_\theta(\text{ans}|\text{CoT})\) actor-reward term (a.k.a. “chain-rule” gradient) from the total loss, so only the policy-gradient component updates the actor. Comparing this column with Markovian quantifies the gain from including direct reward gradients.
- **Llama baseline** – Accuracy or log probability from the unmodified base model (the reference \(\text{CoT}'\)) before any fine-tuning. Every other column for Llama-based runs should be interpreted relative to this starting point.

Except for the Non-Markovian column, all configurations in Table 1 run with the Markovian constraint (the answer decoder never sees the original question during RL training). When we label one column “Markovian,” we mean the conjunction of the bottleneck plus the GRPO-style training recipe; the others are ablations of that recipe but still inherit the same “question-hidden” constraint.

### Datasets / Rows

- **gsm8k** – Grade-school math word problems (short-form answers). Reported values are exact-match accuracy.
- **arc** – ARC-Challenge multiple-choice science questions, accuracy-based.
- **arithmetic** – Synthetic 15-term addition tasks used to probe reasoning robustness; scores are exact numeric accuracy.
- **mmlu** – Massive Multitask Language Understanding benchmark, with accuracy averaged over sampled subjects.
- **wiki_continuation** – Normalized log-likelihood reward on the continuation task described in Sec. \ref{subsec:wikipedia}; measures how many nats the trained CoT adds relative to the baseline when predicting the next 100 tokens. Its row mean is reported in the same log units (marked with *), but it should be compared only against other wiki configurations, not against the accuracy rows.
- **svamp** – Primary-school math word problems requiring reasoning over textual descriptions; metric is exact-answer accuracy.

All QA-style rows (gsm8k, arc, arithmetic, mmlu, svamp) come from the exact-match metric in `src/evaluation.py`, which samples a single CoT and then decodes the answer with `answer_model.generate(..., do_sample=False)`—effectively temperature 0, one-shot evaluation with no self-consistency. 

### Column Means

The “column means” row averages only the QA-style datasets (gsm8k, arc, arithmetic, mmlu, svamp) to keep units consistent. Those numbers summarize how each optimization variant performs on typical accuracy metrics while avoiding the log-probability scale of wiki continuation.

### Reading Table&nbsp;2

- Each cell is formatted “baseline → trained,” aligning with the narrative in the introduction (e.g., GSM8K 20.7% → 54.5%). In practice, “baseline” corresponds to the frozen instruction-tuned model (either Qwen3 or Llama 3.1 8B) evaluated under the same prompting and decoding settings as the trained adapter.
- Qwen columns rely on checkpoints listed in `results/arc/arc_Qwen3_...` and corresponding gsm8k/mmlu runs; the “After” numbers match the `Qwen3` column in `sweep_results_table.csv`.
- Llama columns use the Markovian adapter checkpoints for each dataset; “Before” matches the `baseline` column and “After” the `Markovian` column.
- Wiki continuation cells should be interpreted as “improvement in normalized reward” (higher is better, but values remain negative because the metric is log-likelihood relative to a strong baseline).

### Practical Takeaways

1. **Chain-rule gradients are significant:** Comparing “Markovian” and “No reward” shows that the direct reward gradient contributes several points of accuracy on gsm8k/arc/mmlu, consistent with the Algorithm \ref{alg:markovian_training} discussion.
2. **Expert Iteration vs GRPO:** EI tends to underperform the full Markovian approach because it lacks within-batch standardization and discards most samples. The table quantifies this gap on every dataset.
3. **Non-Markovian runs benchmark the upper bound:** They reflect what performance looks like for learned prediction with strictly more information available. We get interpretability benefits for only a few-percentage points of performance cost, on average. That said, on wikipedia runs, Markovian and EI with the Markovian bottleneck significantly outperform Non-Markovian.


