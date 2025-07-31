#!/bin/bash

# Example script showing how to plot KL divergence metrics

echo "Plotting KL Divergence Metrics"
echo "==============================="

# Change to src directory where the plotting script is located
cd src

# Plot summary metrics (focuses on KL divergences when moving_baseline is configured)
echo "1. Creating summary plot (focuses on KL metrics)..."
python plot_training_metrics.py --plot_summary --output_file ../kl_summary.png

# Plot all metrics including KL divergences
echo "2. Creating full metrics plot..."
python plot_training_metrics.py --output_file ../kl_full.png

# Example with specific file
echo "3. Example with specific log file:"
echo "   python plot_training_metrics.py --plot_summary --files results/your_experiment/log.jsonl"

# Example comparing multiple runs
echo "4. Example comparing multiple training runs:"
echo "   python plot_training_metrics.py --plot_summary --files run1/log.jsonl run2/log.jsonl"

echo ""
echo "Generated plots:"
echo "- kl_summary.png (key KL metrics when moving_baseline configured)"
echo "- kl_full.png (all metrics including KL divergences)"
echo ""
echo "Key metrics:"
echo "- Actor-Critic KL: Policy evolution relative to critic"
echo "- Actor-Reference KL: Policy drift from original model"
echo "- Weighted KL Penalty: Traditional PPO regularization"
echo "- Reference KL Penalty: Drift prevention penalty" 