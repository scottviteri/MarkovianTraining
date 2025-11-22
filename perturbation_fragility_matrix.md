# Perturbation Fragility Matrix

_Generated 2025-11-22T10:12:00+00:00_

The table reports mean accuracy deltas (Markovian minus Non-Markovian) ± population std for each dataset/perturbation family.

| Dataset | Original | CharReplace | Delete | DigitReplace | TruncateBack | TruncateFront | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc | +0.015 ± 0.467 | +0.320 ± 0.628 | +0.424 ± 0.637 | -0.004 ± 0.062 | +0.069 ± 0.522 | +0.439 ± 0.599 | +0.228 ± 0.572 |
| arithmetic | +0.004 ± 0.242 | -0.016 ± 0.388 | - | -0.043 ± 0.470 | - | - | -0.025 ± 0.410 |
| gsm8k | -0.038 ± 0.533 | +0.059 ± 0.593 | +0.069 ± 0.556 | -0.013 ± 0.554 | +0.105 ± 0.555 | +0.044 ± 0.479 | +0.003 ± 0.551 |
| mmlu | -0.167 ± 0.470 | +0.056 ± 0.416 | +0.124 ± 0.455 | +0.004 ± 0.201 | +0.038 ± 0.525 | -0.001 ± 0.410 | +0.014 ± 0.432 |
| svamp | -0.009 ± 0.471 | +0.154 ± 0.500 | +0.204 ± 0.605 | +0.081 ± 0.546 | +0.076 ± 0.464 | +0.046 ± 0.432 | +0.095 ± 0.512 |
| wiki_continuation | +0.294 ± 0.253 | +0.742 ± 0.477 | +0.750 ± 0.492 | +0.028 ± 0.070 | +0.569 ± 0.499 | +0.381 ± 0.497 | +0.466 ± 0.493 |
| **Overall Avg** | +0.004 ± 0.480 | +0.237 ± 0.581 | +0.314 ± 0.607 | -0.004 ± 0.494 | +0.171 ± 0.552 | +0.182 ± 0.523 | +0.115 ± 0.545 |

## Highlights

- **Most Markovian-fragile dataset:** wiki_continuation (avg Δ=+0.466).
- **Least Markovian-fragile dataset:** arithmetic (avg Δ=-0.025).
- **Most destabilizing perturbation family:** Delete with Δ=+0.314 ± 0.607.
- Positive Δ means the Markovian model loses more accuracy than the Non-Markovian counterpart; negative Δ implies the opposite.
