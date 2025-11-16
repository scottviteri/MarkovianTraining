import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from utils import (
    construct_prompts,
    construct_baseline_prompts,
    find_latest_result,
    get_hyperparameters_from_log,
    get_model_paths_and_type,
)
import copy

def extract_answer(answer):
    if "=" in answer:
        answer = answer.split("=")[-1].strip()
    answer = answer.replace(",", "")
    try:
        matches = re.findall(r"-?\d+", answer.strip())
        if matches:
            answer = int(matches[0])
        else:
            answer = "[invalid]"
    except:
        answer = "[invalid]"
    return answer


def load_model(model_path, use_base_model=False, model_type="mistral"):
    """Load actor and critic models for evaluation."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_type == "llama3.2-1b":
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif model_type == "gpt2":
        model_name = "openai-community/gpt2"
    elif model_type == "tinystories":
        model_name = "roneneldan/TinyStories"
    elif model_type == "phi":
        model_name = "microsoft/Phi-3.5-mini-instruct"
    elif model_type == "phi-4":
        model_name = "microsoft/phi-4"
    elif model_type == "qwen3":
        model_name = "Qwen/Qwen3-4B"
    elif model_type == "qwen3-14b":
        model_name = "Qwen/Qwen3-14B"
    elif model_type == "gemma-3":
        model_name = "google/gemma-3-12b-it"
    elif model_type == "gemma-3-small":
        model_name = "google/gemma-3-1b-it"
    else:
        raise ValueError("model_type must be one of: 'mistral', 'llama', 'llama3.2-1b', 'gpt2', 'tinystories', 'phi', 'phi-4', 'qwen3', 'qwen3-14b', 'gemma-3', 'gemma-3-small'")

    # Check if model needs trust_remote_code
    trust_remote_code = model_type in ["phi", "phi-4", "gemma-3", "gemma-3-small"]
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    
    if use_base_model:
        # For base model evaluation, use same model for both actor and critic
        actor_model = critic_model = base_model
    else:
        # Create actor model with LoRA
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=True,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        actor_model = get_peft_model(base_model, peft_config)
        
        # Load checkpoint weights
        checkpoint = torch.load(model_path)
        actor_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create frozen critic model
        critic_model = copy.deepcopy(actor_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return actor_model, critic_model, tokenizer, device


def evaluate_model(
    actor_model,  
    critic_model, 
    tokenizer,
    device,
    test_data,
    hyperparameters,
    num_samples=None,
    batch_size=None,
    baseline_mode=False,
    baseline_thinking_tokens=None,
    baseline_temperature=None,
):
    """Evaluate model on GSM8K test set.
    
    Args:
        actor_model: Model for generating reasoning (with temperature) or baseline thinking
        critic_model: Frozen model for generating answers (deterministic)
        tokenizer: Tokenizer for both models
        device: torch device
        test_data: List of (question, answer) tuples
        hyperparameters: Configuration dictionary
        num_samples: Optional limit on number of samples to evaluate
        batch_size: Batch size for evaluation
        baseline_mode: If True, use standard baseline prompting
        baseline_thinking_tokens: Max thinking tokens for baseline (caps new tokens for stage 1)
        baseline_temperature: Temperature for baseline thinking generation
    """
    # Determine default eval batch size: floor(1.5x) of training batch size (defaults to 8 if absent)
    if batch_size is None:
        try:
            batch_size = max(1, int(hyperparameters.get("batch_size", 8) * 1.5))
        except Exception:
            batch_size = 12
    if num_samples:
        test_data = test_data[:num_samples]
    
    all_results = []
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        questions, answers = zip(*batch)
        
        # Create prompts for stage 1 (reasoning/thinking)
        if baseline_mode:
            prompts = [
                construct_baseline_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                )
                for q in questions
            ]
        else:
            prompts = [
                construct_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                )
                for q in questions
            ]
        
        # Tokenize
        tokenized_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        # 1. Generate CoT/Thinking using actor model (with temperature)
        with torch.no_grad():
            cot_outputs = actor_model.generate(
                input_ids=tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=(baseline_thinking_tokens if baseline_mode and baseline_thinking_tokens is not None else hyperparameters["cot_length"]),
                min_new_tokens=(baseline_thinking_tokens if baseline_mode and baseline_thinking_tokens is not None else hyperparameters["cot_length"]),
                do_sample=True,
                temperature=(baseline_temperature if baseline_mode and baseline_temperature is not None else hyperparameters["temperature"]),
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Decode CoT
        cot_texts = tokenizer.batch_decode(
            cot_outputs[:, tokenized_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 2. Generate answers using critic model (deterministic)
        # Honor Markovian flag: include question context when markovian=False
        include_question_in_eval = not hyperparameters.get("markovian", True)
        if baseline_mode:
            answer_prompts = [
                construct_baseline_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                    reasoning=r,
                    max_thinking_tokens=baseline_thinking_tokens,
                )
                for q, r in zip(questions, cot_texts)
            ]
        else:
            answer_prompts = [
                construct_prompts(
                    question=q,
                    hyperparameters=hyperparameters,
                    reasoning=r,
                    include_question=include_question_in_eval,
                )
                for q, r in zip(questions, cot_texts)
            ]
        
        tokenized_answer_inputs = tokenizer(
            answer_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            answer_outputs = critic_model.generate(
                input_ids=tokenized_answer_inputs.input_ids,
                attention_mask=tokenized_answer_inputs.attention_mask,
                max_new_tokens=10,  # Changed from 15 to 10 to match training
                do_sample=False,    # Deterministic
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and extract answers
        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, tokenized_answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        extracted_answers = [extract_answer(ans) for ans in generated_answers]
        
        # Check correctness
        for q, a, cot, gen_a, ext_a in zip(questions, answers, cot_texts, generated_answers, extracted_answers):
            correct_answer = extract_answer(a)
            is_correct = (ext_a == correct_answer)
            
            result = {
                "question": q,
                "correct_answer": correct_answer,
                "chain_of_thought": cot,
                "generated_answer": gen_a,
                "extracted_answer": ext_a,
                "is_correct": is_correct,
            }
            all_results.append(result)
            
            if is_correct:
                correct += 1
            total += 1
    
    accuracy = correct / total
    return accuracy, all_results


def find_checkpoint_with_index(model_dir, target_index=None):
    """
    Find checkpoint file matching the specified index or most recent if not specified.
    
    Args:
        model_dir: Directory to search in
        target_index: Specific training index to look for (e.g., 1000)
    
    Returns:
        str: Path to matching checkpoint file
    """
    checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    if target_index is not None:
        # Find files matching the target index
        matching_files = [f for f in checkpoint_files if f"_{target_index}_" in f]
        if matching_files:
            # Return most recent matching file
            return max(matching_files, key=os.path.getctime)
        raise FileNotFoundError(f"No checkpoint found with index {target_index}")
    
    # If no specific index requested, return most recent checkpoint
    return max(checkpoint_files, key=os.path.getctime)


def find_all_checkpoints(model_dir):
    """Find all checkpoint files in the directory."""
    checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    return sorted(checkpoint_files, key=os.path.getctime)  # Sort by creation time


def plot_evaluation_results(results: List[Dict], save_path: str):
    """
    Plot evaluation results and save to file.
    
    Args:
        results: List of result dictionaries with evaluation metrics
        save_path: Path to save the plot PNG
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('GSM8K Evaluation Results', fontsize=16)

    # Plot 1: Running Accuracy
    correct_cumsum = np.cumsum([1 if r['is_correct'] else 0 for r in results])
    indices = np.arange(1, len(results) + 1)
    running_accuracy = correct_cumsum / indices
    
    ax1.plot(indices, running_accuracy, 'b-', label='Running Accuracy')
    ax1.set_xlabel('Example Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Running Accuracy')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Question Length vs Correctness
    question_lengths = [len(r['question'].split()) for r in results]
    correct_lengths = [l for l, r in zip(question_lengths, results) if r['is_correct']]
    incorrect_lengths = [l for l, r in zip(question_lengths, results) if not r['is_correct']]
    
    ax2.hist([correct_lengths, incorrect_lengths], 
             label=['Correct', 'Incorrect'],
             bins=30,
             alpha=0.6)
    ax2.set_xlabel('Question Length (words)')
    ax2.set_ylabel('Count')
    ax2.set_title('Question Length Distribution by Correctness')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_over_batches(results_jsonl_path: str, save_path: str):
    """Plot accuracy vs. batch index from accumulated JSONL results and save one combined image.
    
    This generates a single plot across all recorded evaluations without smoothing.
    """
    if not os.path.exists(results_jsonl_path):
        return
    batch_to_entry = {}
    with open(results_jsonl_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            batch_idx = entry.get("batch_index")
            acc = entry.get("accuracy")
            if batch_idx is None or acc is None:
                continue
            # Keep the latest entry per batch index
            batch_to_entry[batch_idx] = acc
    if not batch_to_entry:
        return
    batch_indices = sorted(batch_to_entry.keys())
    accuracies = [batch_to_entry[i] for i in batch_indices]
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, accuracies, marker='o', linestyle='-', color='tab:blue')
    plt.title('GSM8K Accuracy vs Training Batch')
    plt.xlabel('Training Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(model_dir, checkpoint_path, model_type, accuracy, results, num_samples, batch_index_override=None):
    """Save results to file and generate plots.
    
    Args:
        model_dir: Directory where outputs should be written. Prefer a run directory (e.g., results/gsm8k/<timestamp>).
        checkpoint_path: Optional path to checkpoint file for inferring batch index.
        model_type: Model type string used in filenames.
        accuracy: Final accuracy value.
        results: Per-example evaluation results.
        num_samples: Optional number of evaluated samples.
        batch_index_override: If provided, force batch index in filenames/metadata (e.g., 0 for baseline).
    """
    # Create results entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": None,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,  # This is the critic model type
        "num_samples": num_samples,
        "detailed_results": results
    }
    
    # Determine batch index: prefer explicit override, then infer from checkpoint path
    if batch_index_override is not None:
        entry["batch_index"] = int(batch_index_override)
    elif checkpoint_path:
        basename = os.path.basename(checkpoint_path)
        # Support both new and old filename formats
        match = re.search(r'model_batch_(\d+)\.pt$', basename)
        if not match:
            match = re.search(r'model_(\d+)_', basename)
        if match:
            entry["batch_index"] = int(match.group(1))
    
    # Include model type in filenames
    model_type_suffix = f"_{model_type}"
    
    # Save JSONL results with model type in filename
    results_file = os.path.join(model_dir, f"gsm8k_results{model_type_suffix}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    
    # Update accuracy-over-batches plot in the run directory
    accuracy_plot_path = os.path.join(model_dir, "gsm8k_accuracy_over_batches.png")
    plot_accuracy_over_batches(results_file, accuracy_plot_path)
    print(f"Updated GSM8K accuracy plot at {accuracy_plot_path}")
    
    return results_file


def main(
    model_path=None,
    num_samples=None,
    batch_size=None,
    use_base_model=False,
    model_type=None,
    stride=1,
    training_index=None,
    all_checkpoints=False,
    cot_length=None,
    temperature=None,
    baseline=False,
    baseline_thinking_tokens=None,
    baseline_temperature=None,
):
    # Get model path(s) and type if not explicitly provided
    if not use_base_model:
        model_paths, inferred_model_type = get_model_paths_and_type(
            model_path, training_index, all_checkpoints
        )
        if model_type is None:
            model_type = inferred_model_type
            print(f"Inferred model type: {model_type}")
    else:
        model_paths = [None]  # For base model
        model_type = model_type or "mistral"

    for checkpoint_path in model_paths:
        if checkpoint_path:
            print(f"\nEvaluating checkpoint: {checkpoint_path}")
            hyperparameters = get_hyperparameters_from_log(os.path.dirname(checkpoint_path))
            if cot_length is not None:
                hyperparameters["cot_length"] = cot_length
            if temperature is not None:
                hyperparameters["temperature"] = temperature
        else:
            hyperparameters = {
                "model_type": model_type,
                "task_type": "gsm8k",
                "cot_length": cot_length or 150,
                "temperature": temperature or 1.0,
                "batch_size": 8,  # default training batch size for eval scaling
            }
        
        actor_model, critic_model, tokenizer, device = load_model(
            checkpoint_path, 
            use_base_model, 
            model_type
        )

        test_data = load_dataset("openai/gsm8k", "main", split="test")
        test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

        if stride > 1:
            test_data = test_data[::stride]
            print(f"Using stride={stride}, evaluating on {len(test_data)} examples")

        # Determine eval batch size default: 2x training batch size unless explicitly provided
        eval_bs = batch_size if batch_size is not None else 2 * int(hyperparameters.get("batch_size", 8))
        accuracy, results = evaluate_model(
            actor_model,  # For CoT generation
            critic_model,  # For answer generation
            tokenizer, 
            device, 
            test_data, 
            hyperparameters, 
            num_samples, 
            eval_bs,
            baseline_mode=baseline,
            baseline_thinking_tokens=baseline_thinking_tokens,
            baseline_temperature=baseline_temperature,
        )

        print(f"Accuracy: {accuracy:.2%}")

        # Save results to running file in model directory
        model_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "results/evaluations"
        os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
        results_file = save_results(
            model_dir,
            checkpoint_path,
            model_type,
            accuracy,
            results,
            num_samples,
            batch_index_override=(0 if baseline else None)
        )
        print(f"Results appended to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on GSM8K test set."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model weights (default: use latest result)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for evaluation (default: 2x training batch size)"
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use the base model without LoRA or loading weights",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"],
        default=None,
        help="Choose between Mistral and Llama 3.1 models (default: infer from model path)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every nth example in the test set",
    )
    parser.add_argument(
        "--training_index",
        type=int,
        help="Specific training index to evaluate (e.g., 1000)",
    )
    parser.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints in the directory",
    )
    parser.add_argument(
        "--cot_length",
        type=int,
        default=None,
        help="Override the chain-of-thought length for generation (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the temperature for generation (default: 1.0)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use standard baseline prompting (no specialized CoT phrasing)",
    )
    parser.add_argument(
        "--baseline_thinking_tokens",
        type=int,
        default=None,
        help="Max tokens for baseline thinking stage (caps new tokens before answer)",
    )
    parser.add_argument(
        "--baseline_temperature",
        type=float,
        default=None,
        help="Temperature used for baseline thinking generation",
    )
    args = parser.parse_args()

    try:
        main(
            args.model_path,
            args.num_samples,
            args.batch_size,
            args.use_base_model,
            args.model_type,
            args.stride,
            args.training_index,
            args.all_checkpoints,
            args.cot_length,
            args.temperature,
            args.baseline,
            args.baseline_thinking_tokens,
            args.baseline_temperature,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
