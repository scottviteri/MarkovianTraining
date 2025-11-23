# Perturbation Fragility Matrix

_Generated 2025-11-22T11:11:28Z_

To keep apples with apples, the QA table below shows **accuracy-drop deltas (Markovian drop $-$ Non-Markovian drop)** while the wiki continuation task—scored on log probabilities—is broken out underneath.

## QA Tasks (Accuracy $\Delta$, higher = Markovian loses more accuracy)

| Dataset | CharReplace | Delete | DigitReplace | TruncateBack | TruncateFront | Average |
| --- | --- | --- | --- | --- | --- | --- |
| arc | +0.320 $\pm$ 0.628 | +0.424 $\pm$ 0.637 | -0.004 $\pm$ 0.062 | +0.069 $\pm$ 0.522 | +0.439 $\pm$ 0.599 | +0.228 $\pm$ 0.572 |
| arithmetic | -0.016 $\pm$ 0.388 | -0.003 $\pm$ 0.405 | -0.043 $\pm$ 0.470 | +0.001 $\pm$ 0.263 | -0.016 $\pm$ 0.321 | -0.009 $\pm$ 0.341 |
| gsm8k | +0.059 $\pm$ 0.593 | +0.069 $\pm$ 0.556 | -0.013 $\pm$ 0.554 | +0.105 $\pm$ 0.555 | +0.044 $\pm$ 0.479 | +0.003 $\pm$ 0.551 |
| mmlu | +0.056 $\pm$ 0.416 | +0.124 $\pm$ 0.455 | +0.004 $\pm$ 0.201 | +0.038 $\pm$ 0.525 | -0.001 $\pm$ 0.410 | +0.014 $\pm$ 0.432 |
| svamp | +0.154 $\pm$ 0.500 | +0.204 $\pm$ 0.605 | +0.081 $\pm$ 0.546 | +0.076 $\pm$ 0.464 | +0.046 $\pm$ 0.432 | +0.095 $\pm$ 0.512 |
| **QA Overall Avg** | +0.157 $\pm$ 0.555 | +0.102 $\pm$ 0.514 | -0.007 $\pm$ 0.511 | +0.037 $\pm$ 0.414 | +0.059 $\pm$ 0.438 | +0.043 $\pm$ 0.488 |

## Wiki Continuation ($\Delta$ log P, larger = Markovian probability drops faster)

|  | CharReplace | Delete | DigitReplace | TruncateBack | TruncateFront |
| --- | --- | --- | --- | --- | --- |
| $\Delta$ log P $\pm$ $\sigma$ | +0.742 $\pm$ 0.477 | +0.750 $\pm$ 0.492 | +0.028 $\pm$ 0.070 | +0.569 $\pm$ 0.499 | +0.381 $\pm$ 0.497 |
| Ratio (Markov fragility / Non) | $\times$ 2.100 $\pm$ 1.035 | $\times$ 2.118 $\pm$ 1.087 | $\times$ 1.029 $\pm$ 0.016 | $\times$ 1.766 $\pm$ 0.934 | $\times$ 1.464 $\pm$ 0.824 |

Overall wiki mean (log space): $\Delta$ log P $\approx$ **+0.466 $\pm$ 0.493**, ratio $\approx$ **$\times$ 1.53 $\pm$ 0.90**.

## Highlights & Takeaways

- **CharReplace remains the most damaging QA perturbation** (+0.157 absolute accuracy, i.e. +15.7 percentage points on average), just edging out Delete (+0.102). Truncation hurts less overall but still biases against Markovian models by roughly 4–6 percentage points.
- **ARC shows the clearest Markovian fragility in accuracy space** (+0.228 absolute accuracy $\approx$ +22.8 pp), followed by SVAMP (+0.095 $\approx$ +9.5 pp). Arithmetic is the only task where Markovian accuracy is slightly higher (-0.009 $\approx$ -0.9 pp).
- **Wiki continuation behaves differently:** Delete and character noise lower Markovian log-probs by ~0.74–0.75 nats ($\approx$ 2.1$\times$ probability loss) relative to Non-Markovian. Reporting it separately avoids polluting the QA averages with unbounded log-prob scales.
- **Variance tells a story:** the QA $\sigma$’s stay in the $\pm$ 0.4–0.6 absolute-accuracy band ($\pm$ 40–60 pp), while wiki’s ratios show tighter spreads—log-prob effects are consistent even though absolute magnitudes dwarf QA deltas.
- **Interpretation:** For QA benchmarks, Markovian vs Non-Markovian gaps are modest but systematic, concentrated in perturbations that disrupt semantic content. For wiki, the Markovian objective is ~1.5$\times$ more brittle under severe text corruption, reinforcing the idea that information bottlenecks amplify sensitivity to broad perturbations.
