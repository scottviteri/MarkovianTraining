# Table 1: Markovian vs Non-Markovian Perturbation Sensitivity Analysis

## Aggregated Results Across All Model Pairs

| Perturbation Type | Degree | Mean Effect Difference | Overall Consistency | Total Comparisons |
|-------------------|--------|----------------------|-------------------|-------------------|
| **Delete** | 20% | +0.1193 | 76.1% | 1,472 |
| | 40% | +0.2170 | 83.3% | 1,472 |
| | 60% | +0.2525 | 84.0% | 1,472 |
| | 80% | +0.2612 | 84.8% | 1,472 |
| | 100% | +0.3276 | 87.6% | 1,472 |
| **Truncate Front** | 20% | +0.0317 | 56.4% | 1,472 |
| | 40% | +0.0575 | 65.4% | 1,472 |
| | 60% | +0.0893 | 70.6% | 1,472 |
| | 80% | +0.1457 | 77.7% | 1,472 |
| | 100% | +0.3276 | 87.6% | 1,472 |
| **Truncate Back** | 20% | +0.0154 | 52.9% | 1,472 |
| | 40% | +0.0411 | 58.1% | 1,472 |
| | 60% | +0.1003 | 64.9% | 1,472 |
| | 80% | +0.1681 | 73.4% | 1,472 |
| | 100% | +0.3276 | 87.6% | 1,472 |
| **Character Replace** | 20% | +0.1271 | 76.3% | 1,472 |
| | 40% | +0.2245 | 85.9% | 1,472 |
| | 60% | +0.2485 | 86.5% | 1,472 |
| | 80% | +0.2534 | 86.9% | 1,472 |
| | 100% | +0.2548 | 86.9% | 1,472 |

## Detailed Results by Model Pair

| Pair | Perturbation | Degree | Mean Diff | Std Dev | Consistency | N Cases |
|------|-------------|--------|-----------|---------|-------------|---------|
| **Left** | Delete | 20% | +0.1103 | 0.1286 | 79.6% | 362 |
| | | 40% | +0.2359 | 0.1691 | 90.1% | 362 |
| | | 60% | +0.3156 | 0.1906 | 93.9% | 362 |
| | | 80% | +0.3456 | 0.1904 | 95.3% | 362 |
| | | 100% | +0.3704 | 0.1969 | 95.0% | 362 |
| | Truncate Front | 20% | +0.0099 | 0.0579 | 45.3% | 362 |
| | | 40% | +0.0408 | 0.0820 | 62.2% | 362 |
| | | 60% | +0.0918 | 0.1227 | 75.7% | 362 |
| | | 80% | +0.1707 | 0.1507 | 88.7% | 362 |
| | | 100% | +0.3704 | 0.1969 | 95.0% | 362 |
| | Truncate Back | 20% | +0.0673 | 0.1383 | 61.0% | 362 |
| | | 40% | +0.1649 | 0.1906 | 77.3% | 362 |
| | | 60% | +0.2837 | 0.2088 | 88.7% | 362 |
| | | 80% | +0.3441 | 0.2046 | 94.2% | 362 |
| | | 100% | +0.3704 | 0.1969 | 95.0% | 362 |
| | Character Replace | 20% | +0.1151 | 0.1286 | 79.3% | 362 |
| | | 40% | +0.2137 | 0.1668 | 87.6% | 362 |
| | | 60% | +0.2577 | 0.1818 | 89.8% | 362 |
| | | 80% | +0.2573 | 0.1811 | 90.3% | 362 |
| | | 100% | +0.2587 | 0.1854 | 89.8% | 362 |

---

**Table 1 Caption**: *Comprehensive analysis of perturbation sensitivity differences between Markovian and Non-Markovian language model training approaches across four distinct model pairs and four perturbation types.* The analysis is based on 5,888 total comparison points (1,472 per perturbation type) collected during training on Wikipedia continuation tasks. **Mean Effect Difference** represents the average difference in log-probability sensitivity (Markovian effect - Non-Markovian effect), where positive values indicate that Markovian models exhibit greater sensitivity to perturbations. **Consistency** measures the percentage of training instances where Markovian models showed higher perturbation sensitivity than their Non-Markovian counterparts. 

**Key Findings**: (1) **Delete perturbations** show the strongest and most consistent differences, with Markovian models demonstrating progressively higher sensitivity as perturbation severity increases (20%: +0.1193, 100%: +0.3276), achieving 87.6% consistency at maximum severity. (2) **Character replacement** exhibits robust sensitivity differences across all severities (20%-100%: +0.1271 to +0.2548) with high consistency (76.3%-86.9%). (3) **Truncation perturbations** (front/back) show more modest effects at lower severities but converge to similar high-severity effects (+0.3276 at 100%). (4) The pattern holds across all four independent model pairs (left, mid, right, riight), suggesting the observed differences are systematic rather than pair-specific. 

**Methodological Details**: Each model pair consisted of architecturally identical transformers trained with different reward calculation methods: Markovian models computed rewards based solely on current token predictions, while Non-Markovian models incorporated full sequence context. Perturbations were applied at five severity levels (20%, 40%, 60%, 80%, 100%) during training batch processing, with effects measured as changes in log-probability for correct completions. The batch size varied between pairs (6-8 examples) but remained consistent within each pair. **Statistical Significance**: The large sample sizes (N=362-458 per pair) and consistent directional effects across pairs provide strong evidence for systematic sensitivity differences between the two training paradigms, particularly for severe perturbations where consistency exceeds 85% across all perturbation types.