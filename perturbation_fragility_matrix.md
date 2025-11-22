# Perturbation Fragility Matrix

_Generated 2025-11-22T11:11:28Z_

To keep apples with apples, the QA table below shows **accuracy deltas (Markovian − Non-Markovian)** while the wiki continuation task—scored on log probabilities—is broken out underneath.

## QA Tasks (Accuracy Δ, higher = Markovian loses more accuracy)

| Dataset | Original | CharReplace | Delete | DigitReplace | TruncateBack | TruncateFront | Average |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arc | +0.015 ± 0.467 | +0.320 ± 0.628 | +0.424 ± 0.637 | −0.004 ± 0.062 | +0.069 ± 0.522 | +0.439 ± 0.599 | +0.228 ± 0.572 |
| arithmetic | −0.003 ± 0.278 | −0.016 ± 0.388 | −0.003 ± 0.405 | −0.043 ± 0.470 | +0.001 ± 0.263 | −0.016 ± 0.321 | −0.009 ± 0.341 |
| gsm8k | −0.038 ± 0.533 | +0.059 ± 0.593 | +0.069 ± 0.556 | −0.013 ± 0.554 | +0.105 ± 0.555 | +0.044 ± 0.479 | +0.003 ± 0.551 |
| mmlu | −0.167 ± 0.470 | +0.056 ± 0.416 | +0.124 ± 0.455 | +0.004 ± 0.201 | +0.038 ± 0.525 | −0.001 ± 0.410 | +0.014 ± 0.432 |
| svamp | −0.009 ± 0.471 | +0.154 ± 0.500 | +0.204 ± 0.605 | +0.081 ± 0.546 | +0.076 ± 0.464 | +0.046 ± 0.432 | +0.095 ± 0.512 |
| **QA Overall Avg** | −0.031 ± 0.446 | +0.157 ± 0.555 | +0.102 ± 0.514 | −0.007 ± 0.511 | +0.037 ± 0.414 | +0.059 ± 0.438 | +0.043 ± 0.488 |

## Wiki Continuation (Δ log P, larger = Markovian probability drops faster)

|  | Original | CharReplace | Delete | DigitReplace | TruncateBack | TruncateFront |
| --- | --- | --- | --- | --- | --- | --- |
| Δ log P ± σ | +0.294 ± 0.253 | +0.742 ± 0.477 | +0.750 ± 0.492 | +0.028 ± 0.070 | +0.569 ± 0.499 | +0.381 ± 0.497 |
| Ratio (Markov fragility ÷ Non) | ×1.000 ± 0.000 | ×2.100 ± 1.035 | ×2.118 ± 1.087 | ×1.029 ± 0.016 | ×1.766 ± 0.934 | ×1.464 ± 0.824 |

Overall wiki mean (log space): Δ log P ≈ **+0.466 ± 0.493**, ratio ≈ **×1.53 ± 0.90**.

## Highlights & Takeaways

- **CharReplace remains the most damaging QA perturbation** (+0.157 absolute accuracy, i.e. +15.7 percentage points on average), just edging out Delete (+0.102). Truncation hurts less overall but still biases against Markovian models by roughly 4–6 percentage points.
- **ARC shows the clearest Markovian fragility in accuracy space** (+0.228 absolute accuracy ≈ +22.8 pp), followed by SVAMP (+0.095 ≈ +9.5 pp). Arithmetic is the only task where Markovian accuracy is slightly higher (−0.009 ≈ −0.9 pp).
- **Wiki continuation behaves differently:** Delete and character noise lower Markovian log-probs by ~0.74–0.75 nats (≈2.1× probability loss) relative to Non-Markovian. Reporting it separately avoids polluting the QA averages with unbounded log-prob scales.
- **Variance tells a story:** the QA σ’s stay in the ±0.4–0.6 absolute-accuracy band (±40–60 pp), while wiki’s ratios show tighter spreads—log-prob effects are consistent even though absolute magnitudes dwarf QA deltas.
- **Interpretation:** For QA benchmarks, Markovian vs Non-Markovian gaps are modest but systematic, concentrated in perturbations that disrupt semantic content. For wiki, the Markovian objective is ~1.5× more brittle under severe text corruption, reinforcing the idea that information bottlenecks amplify sensitivity to broad perturbations.
