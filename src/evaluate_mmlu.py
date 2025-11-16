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
from typing import List, Dict, Tuple
from utils import construct_prompts, construct_baseline_prompts, find_latest_result
import copy


def extract_choice_letter(text: str) -> str:
    """
    Extract a single-choice letter (A-D) from model output.
    Returns one of 'A', 'B', 'C', 'D' or '[invalid]'.
    """
    match = re.search(r"\b([A-D])\b", text.strip())
    if match:
        return match.group(1)
    # Try lowercase
    match = re.search(r"\b([a-d])\b", text.strip())
    if match:
        return match.group(1).upper()
    return "[invalid]"


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

    if use_base_model or model_path is None:
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


def get_hyperparameters_from_log(model_dir):
    """Get hyperparameters from the first line of log.jsonl"""
    log_path = os.path.join(model_dir, "log.jsonl")
    try:
        with open(log_path, 'r') as f:
            hyperparameters = json.loads(f.readline().strip())
        return hyperparameters
    except Exception as e:
        print(f"Warning: Could not read hyperparameters from log file ({e})")
        # Fallback defaults for MMLU
        return {
            "model_type": "mistral",
            "task_type": "mmlu",
            "cot_length": 100,
            "temperature": 1.0,
            "batch_size": 12,
            "mmlu_split": "validation",
            "mmlu_subject": None,
        }


def get_model_paths_and_type(provided_path=None, target_index=None, all_checkpoints=False) -> Tuple[List[str], str]:
    """Get model path(s) and infer model type from log file, similar to GSM8K helper."""
    if provided_path:
        model_dir = os.path.dirname(provided_path)
    else:
        model_dir = find_latest_result()
        if not model_dir:
            raise FileNotFoundError("No results directory found")

    # Collect model paths
    if all_checkpoints:
        checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
        model_paths = sorted(checkpoint_files, key=os.path.getctime)
    else:
        # Find most recent checkpoint
        checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
        model_paths = [max(checkpoint_files, key=os.path.getctime)] if checkpoint_files else [None]

    # Read model type
    log_path = os.path.join(model_dir, "log.jsonl")
    try:
        with open(log_path, 'r') as f:
            hyperparameters = json.loads(f.readline().strip())
            model_type = hyperparameters.get("model_type", "mistral")
    except Exception as e:
        print(f"Warning: Could not read model type from log file ({e}), defaulting to mistral")
        model_type = "mistral"

    return model_paths, model_type


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
    """Evaluate model on MMLU using actor-critic or baseline prompting."""
    # Batch size default: 1.5x training bs
    if batch_size is None:
        try:
            batch_size = max(1, int(hyperparameters.get("batch_size", 12) * 1.5))
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

        # Stage 1: thinking / CoT
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

        tokenized_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

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

        cot_texts = tokenizer.batch_decode(
            cot_outputs[:, tokenized_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Stage 2: deterministic answer (single letter)
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
                    include_question=not hyperparameters.get("markovian", True),
                )
                for q, r in zip(questions, cot_texts)
            ]

        tokenized_answer_inputs = tokenizer(answer_prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            answer_outputs = critic_model.generate(
                input_ids=tokenized_answer_inputs.input_ids,
                attention_mask=tokenized_answer_inputs.attention_mask,
                max_new_tokens=3,  # expect a single letter
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, tokenized_answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        extracted_answers = [extract_choice_letter(ans) for ans in generated_answers]

        for q, correct_letter, cot, gen in zip(questions, answers, cot_texts, generated_answers):
            pred_letter = extract_choice_letter(gen)
            is_correct = (pred_letter == correct_letter)
            all_results.append({
                "question": q,
                "correct_answer": correct_letter,
                "chain_of_thought": cot,
                "generated_answer": gen,
                "extracted_answer": pred_letter,
                "is_correct": is_correct,
            })
            if is_correct:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_results


def plot_accuracy_over_batches(results_jsonl_path: str, save_path: str):
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
            batch_to_entry[batch_idx] = acc
    if not batch_to_entry:
        return
    batch_indices = sorted(batch_to_entry.keys())
    accuracies = [batch_to_entry[i] for i in batch_indices]
    plt.figure(figsize=(10, 5))
    plt.plot(batch_indices, accuracies, marker='o', linestyle='-', color='tab:green')
    plt.title('MMLU Accuracy vs Training Batch')
    plt.xlabel('Training Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(model_dir, checkpoint_path, model_type, accuracy, results, num_samples, batch_index_override=None):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": None,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,
        "num_samples": num_samples,
        "detailed_results": results
    }

    if batch_index_override is not None:
        entry["batch_index"] = int(batch_index_override)
    elif checkpoint_path:
        basename = os.path.basename(checkpoint_path)
        match = re.search(r'model_batch_(\d+)\.pt$', basename)
        if not match:
            match = re.search(r'model_(\d+)_', basename)
        if match:
            entry["batch_index"] = int(match.group(1))

    model_type_suffix = f"_{model_type}"
    results_file = os.path.join(model_dir, f"mmlu_results{model_type_suffix}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")

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
    if not use_base_model:
        model_paths, inferred_model_type = get_model_paths_and_type(model_path, training_index, all_checkpoints)
        if model_type is None:
            model_type = inferred_model_type
            print(f"Inferred model type: {model_type}")
    else:
        model_paths = [None]
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
                "task_type": "mmlu",
                "cot_length": cot_length or 100,
                "temperature": temperature or 1.0,
                "batch_size": 12,
                "mmlu_split": "validation",
                "mmlu_subject": None,
            }

        actor_model, critic_model, tokenizer, device = load_model(
            checkpoint_path,
            use_base_model,
            model_type
        )

        split = hyperparameters.get("mmlu_split", "validation")
        subject = hyperparameters.get("mmlu_subject", None)
        # Default to the aggregated 'all' config when no subject is specified
        ds = load_dataset("cais/mmlu", subject if subject else "all")
        split_name = "test" if split == "test" else "validation"
        d = ds[split_name]

        # Build Q/A pairs: question string with options; answer is correct letter
        test_data = []
        for ex in d:
            stem = ex["question"]
            choices = ex["choices"] if "choices" in ex else ex.get("options", [])
            # choices expected to be list of strings
            if not choices or len(choices) < 4:
                continue
            # Compose question with labeled options
            options_text = "\n".join([
                f"A. {choices[0]}",
                f"B. {choices[1]}",
                f"C. {choices[2]}",
                f"D. {choices[3]}",
            ])
            question_text = f"{stem}\n\nOptions:\n{options_text}"
            # Label index may be at 'answer' (0-3)
            if "answer" in ex and isinstance(ex["answer"], int):
                correct_letter = ["A", "B", "C", "D"][ex["answer"]]
            else:
                # Sometimes the dataset includes the string label
                correct_letter = str(ex.get("answer", "")).strip().upper()
                if correct_letter not in ["A", "B", "C", "D"]:
                    continue
            test_data.append((question_text, correct_letter))

        if stride > 1:
            test_data = test_data[::stride]
            print(f"Using stride={stride}, evaluating on {len(test_data)} examples")

        eval_bs = batch_size if batch_size is not None else 2 * int(hyperparameters.get("batch_size", 12))
        accuracy, results = evaluate_model(
            actor_model,
            critic_model,
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

        model_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "results/mmlu"
        os.makedirs(model_dir, exist_ok=True)
        results_file = save_results(
            model_dir,
            checkpoint_path,
            model_type,
            accuracy,
            results,
            num_samples,
            batch_index_override=(0 if baseline else None),
        )
        print(f"Results appended to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline or actor-critic on MMLU")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model weights (default: latest result)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--use_base_model", action="store_true", help="Use base model without adapters")
    parser.add_argument("--model_type", type=str, choices=["llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"], default=None)
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every nth example")
    parser.add_argument("--training_index", type=int, help="Specific training index to evaluate")
    parser.add_argument("--all_checkpoints", action="store_true", help="Evaluate all checkpoints in the directory")
    parser.add_argument("--cot_length", type=int, default=None, help="Override CoT length")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--baseline", action="store_true", help="Use standard baseline prompting")
    parser.add_argument("--baseline_thinking_tokens", type=int, default=None, help="Max tokens for baseline thinking stage")
    parser.add_argument("--baseline_temperature", type=float, default=None, help="Temperature for baseline thinking stage")
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


