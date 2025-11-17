#!/usr/bin/env python3
"""
Demo script for comparing answer extraction methods including LLM gold-standard.

This script demonstrates how to use the new LLM-based extraction as a gold-standard
to evaluate the performance of heuristic extraction methods.

Requirements:
    - Set ANTHROPIC_API_KEY environment variable to use LLM extraction
    - Model checkpoint or use --use_base_model flag

Usage:
    # With a trained model
    python compare_extraction_demo.py --model_path results/gsm8k/.../adapter_500

    # With base model (to test extraction only)
    python compare_extraction_demo.py --use_base_model --model_type mistral

    # Specify number of samples (recommended for LLM to control costs)
    python compare_extraction_demo.py --use_base_model --num_samples 20

    # Compare specific methods
    python compare_extraction_demo.py --use_base_model --methods simple anchor llm
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    load_model_for_evaluation,
    load_gsm8k_dataset,
    colored_print,
    Colors,
)
from evaluation import compare_extraction_methods


def main():
    parser = argparse.ArgumentParser(
        description="Compare answer extraction methods with LLM gold-standard"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: use base model)",
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use untrained base model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mistral",
        choices=["mistral", "llama", "llama3.2-1b", "phi", "qwen3", "gemma-3-small"],
        help="Model type to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50, use fewer for LLM to control costs)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Extraction methods to compare (default: auto-detect based on API key)",
    )
    parser.add_argument(
        "--answer_format",
        type=str,
        default="numeric",
        choices=["numeric", "A-D", "A-E"],
        help="Answer format for LLM extraction (default: numeric for GSM8K)",
    )
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        colored_print(
            "Warning",
            "ANTHROPIC_API_KEY not set - LLM gold-standard comparison will be skipped",
            Colors.YELLOW,
        )
        print("To use LLM comparison: export ANTHROPIC_API_KEY=your-key-here\n")
    else:
        colored_print(
            "API Key",
            "ANTHROPIC_API_KEY found - will include LLM gold-standard",
            Colors.GREEN,
        )

    # Load model
    colored_print("Setup", "Loading model...", Colors.CYAN)
    actor_model, critic_model, tokenizer, device = load_model_for_evaluation(
        model_path=args.model_path,
        use_base_model=args.use_base_model or args.model_path is None,
        model_type=args.model_type,
    )
    colored_print("Setup", "Model loaded successfully", Colors.GREEN)

    # Load test data (GSM8K)
    colored_print("Setup", f"Loading {args.num_samples} GSM8K test samples...", Colors.CYAN)
    test_data_iter = load_gsm8k_dataset(split="test")
    test_data = []
    for i, qa in enumerate(test_data_iter):
        if i >= args.num_samples:
            break
        test_data.append(qa)
    colored_print("Setup", f"Loaded {len(test_data)} test samples", Colors.GREEN)

    # Setup hyperparameters
    hyperparameters = {
        "task_type": "gsm8k",
        "model_type": args.model_type,
        "cot_length": 100,
        "temperature": 1.0,
        "batch_size": 8,
    }

    # Run comparison
    print("\n" + "=" * 70)
    colored_print("Extraction Comparison", "Starting comparison of extraction methods", Colors.BOLD)
    print("=" * 70 + "\n")

    if api_key:
        estimated_cost = (args.num_samples * 2 * 20) / 1_000_000 * 0.25  # Rough estimate
        colored_print(
            "Cost Estimate",
            f"Estimated API cost: ~${estimated_cost:.4f} for {args.num_samples} samples",
            Colors.CYAN,
        )
        print()

    results = compare_extraction_methods(
        actor_model=actor_model,
        critic_model=critic_model,
        tokenizer=tokenizer,
        device=device,
        test_data=test_data,
        hyperparameters=hyperparameters,
        methods=args.methods,
        batch_size=8,
        num_samples=None,  # Already limited test_data
        answer_format=args.answer_format,
    )

    # Additional analysis
    print("\n" + "=" * 70)
    colored_print("Detailed Analysis", "Extraction Method Performance", Colors.BOLD)
    print("=" * 70 + "\n")

    for method, (accuracy, detailed_results) in results.items():
        correct = sum(1 for r in detailed_results if r["correct"])
        incorrect = len(detailed_results) - correct

        print(f"{method.upper()} Method:")
        print(f"  ✓ Correct:   {correct:3d} / {len(detailed_results)} ({accuracy:.2%})")
        print(f"  ✗ Incorrect: {incorrect:3d} / {len(detailed_results)}")

        # Show example mistakes
        mistakes = [r for r in detailed_results if not r["correct"]]
        if mistakes:
            print(f"\n  Example mistakes:")
            for i, mistake in enumerate(mistakes[:2]):
                print(f"    {i+1}. Predicted: {mistake['predicted']}, Gold: {mistake['gold']}")
                print(f"       Generated: {mistake['generated_answer'][:80]}...")
        print()

    # Save detailed results if using LLM
    if "llm" in results:
        output_file = "extraction_comparison_results.json"
        import json

        with open(output_file, "w") as f:
            # Convert to JSON-serializable format
            json_results = {}
            for method, (accuracy, detailed) in results.items():
                json_results[method] = {
                    "accuracy": accuracy,
                    "num_samples": len(detailed),
                    "correct": sum(1 for r in detailed if r["correct"]),
                }
            json.dump(json_results, f, indent=2)

        colored_print(
            "Saved",
            f"Detailed results saved to {output_file}",
            Colors.GREEN,
        )


if __name__ == "__main__":
    main()

