#!/usr/bin/env python3
"""
Test different prompt variations for MCQ tasks to see which works best.
Uses existing checkpoints, no retraining required.
"""

import sys
sys.path.insert(0, 'src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import load_mmlu_dataset, load_aqua_dataset, colored_print, Colors
from tqdm import tqdm
import re
import argparse


def get_model_path(model_type: str) -> str:
    """Get HuggingFace model path."""
    model_paths = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
    }
    return model_paths[model_type]


def construct_prompt_variation(question: str, reasoning: str, variation: str) -> str:
    """Construct different prompt variations for testing.
    
    Args:
        question: The question text
        reasoning: The CoT reasoning
        variation: Which prompt variation to use
        
    Returns:
        Formatted prompt
    """
    if variation == "original":
        # Original prompt used in training
        return f"You will be given a multiple choice question. Use 150 tokens to think through the problem step-by-step, then select the correct answer. Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    elif variation == "explicit_letter":
        # Explicitly ask for just a letter
        return f"You will be given a multiple choice question. Use 150 tokens to think through the problem step-by-step, then output ONLY a single letter (A, B, C, or D) as your answer. Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    elif variation == "format_example":
        # Show the format by example
        return f"You will be given a multiple choice question. Think through the problem, then output your answer as a single letter. For example: 'Answer: B'. Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    elif variation == "short_instruction":
        # Very short and direct
        return f"Select the correct answer by outputting a single letter (A/B/C/D). Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    elif variation == "with_choices":
        # Include the choices even in Markovian mode
        # Extract choices from reasoning if they're there
        return f"Multiple choice question. Select one: A, B, C, or D. Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    elif variation == "natural":
        # More natural language
        return f"Based on your reasoning, which answer is correct? Reply with just the letter. Question: <Redacted>\nReasoning:{reasoning} Answer: "
    
    else:
        raise ValueError(f"Unknown variation: {variation}")


def extract_letter(text: str) -> str:
    """Extract first letter A-D from text. Returns 'X' if none found."""
    matches = re.findall(r"[A-D]", text.upper())
    return matches[0] if matches else "X"


def extract_letter_word_boundary(text: str) -> str:
    """Extract first letter A-D with word boundaries."""
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"\b([a-d])\b", text)
    if match:
        return match.group(1).upper()
    return "X"


def evaluate_with_prompt_variation(
    checkpoint_path: str,
    test_data,
    variation: str,
    stride: int = 10,
):
    """Evaluate a checkpoint with a specific prompt variation."""
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_model_path("llama")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Use stride
    test_data = test_data[::stride]
    
    correct_old = 0
    correct_wb = 0
    total = len(test_data)
    
    # First, generate CoT for all examples (use original prompt for CoT generation)
    cot_prompts = []
    for question, _ in test_data:
        prompt = f"You will be given a multiple choice question. Use 150 tokens to think through the problem step-by-step, then select the correct answer. Question: {question}\nReasoning:"
        cot_prompts.append(prompt)
    
    # Generate CoTs in batches
    batch_size = 16  # Larger batch for speed
    cot_texts = []
    
    for i in tqdm(range(0, len(cot_prompts), batch_size), desc=f"Generating CoT [{variation}]"):
        batch_prompts = cot_prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=150,
                min_new_tokens=150,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        batch_cot = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        cot_texts.extend(batch_cot)
    
    # Now generate answers using the prompt variation
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Generating Answers [{variation}]"):
        batch_data = test_data[i:i+batch_size]
        batch_cots = cot_texts[i:i+batch_size]
        
        # Construct prompts with variation
        answer_prompts = [
            construct_prompt_variation(q, cot, variation)
            for (q, _), cot in zip(batch_data, batch_cots)
        ]
        
        inputs = tokenizer(
            answer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Evaluate
        for (_, gold), gen in zip(batch_data, generated):
            pred_old = extract_letter(gen)
            pred_wb = extract_letter_word_boundary(gen)
            
            if pred_old == gold:
                correct_old += 1
            if pred_wb == gold:
                correct_wb += 1
    
    acc_old = correct_old / total
    acc_wb = correct_wb / total
    
    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return acc_old, acc_wb


def main():
    """Main evaluation."""
    
    parser = argparse.ArgumentParser(description="Test prompt variations on MCQ tasks")
    parser.add_argument("--task", choices=["mmlu", "aqua"], default="mmlu", help="Task to evaluate")
    parser.add_argument("--stride", type=int, default=100, help="Evaluate every Nth example")
    args = parser.parse_args()
    
    # Configure based on task
    if args.task == "mmlu":
        checkpoint_path = "results/mmlu/20251116_191617/adapter_1000"
        colored_print("Loading", "Loading MMLU test data...", Colors.CYAN)
        test_data = list(load_mmlu_dataset(split="test", subject=None))
    else:  # aqua
        checkpoint_path = "results/aqua/20251116_193803/adapter_1000"
        colored_print("Loading", "Loading AQuA test data...", Colors.CYAN)
        test_data = list(load_aqua_dataset(split="test"))
    
    colored_print("Loaded", f"{len(test_data)} examples (will use every {args.stride}th = {len(test_data)//args.stride})", Colors.GREEN)
    
    stride = args.stride
    
    # Test only most promising variations for speed
    variations = [
        "original",
        "explicit_letter",
        "short_instruction",
    ]
    
    results = {}
    
    for variation in variations:
        colored_print("Testing", f"Prompt variation: {variation}", Colors.BOLD)
        acc_old, acc_wb = evaluate_with_prompt_variation(
            checkpoint_path,
            test_data,
            variation,
            stride=stride
        )
        results[variation] = (acc_old, acc_wb)
        colored_print(
            variation,
            f"Old regex: {acc_old:.2%} | WB regex: {acc_wb:.2%}",
            Colors.GREEN if acc_wb > 0.20 else Colors.YELLOW
        )
        print()
    
    # Print summary
    print("\n" + "="*70)
    colored_print("Summary", "Prompt Variation Comparison", Colors.BOLD)
    print("="*70)
    print(f"{'Variation':<25} {'Old Regex':<15} {'WB Regex':<15}")
    print("-"*70)
    
    for variation in variations:
        acc_old, acc_wb = results[variation]
        print(f"{variation:<25} {acc_old:>13.2%} {acc_wb:>15.2%}")
    
    print("="*70)
    
    # Find best
    best_variation = max(results.items(), key=lambda x: x[1][1])
    colored_print(
        "Best Prompt",
        f"{best_variation[0]} with {best_variation[1][1]:.2%} accuracy (WB)",
        Colors.GREEN
    )


if __name__ == "__main__":
    main()

