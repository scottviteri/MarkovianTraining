import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model, PeftModel
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from utils import (
    construct_prompts,
    construct_baseline_prompts,
    find_latest_result,
    load_svamp_dataset,
    load_aqua_dataset,
    generate_question_answer_batches,
)
import copy


def extract_numeric_answer(answer: str):
    """Extract the intended numeric answer from a generated string.

    Heuristics (in order):
    1) If an 'Answer' anchor exists (e.g., 'Answer:' or 'answer'), extract the first integer after it.
    2) Else if an equals sign exists, extract the first integer after '='.
    3) Else extract the first integer in the text.
    """
    try:
        text = (answer or "").strip()
        # Normalize
        text = text.replace(",", "")

        # 1) Anchor on 'Answer' label (case-insensitive), optional colon
        m = re.search(r"(?i)answer\s*:?[\s\-]*", text)
        if m:
            after = text[m.end():]
            m_num = re.search(r"-?\d+", after)
            if m_num:
                return int(m_num.group(0))

        # 2) After equals sign
        if "=" in text:
            after_eq = text.split("=", 1)[1]
            m_num = re.search(r"-?\d+", after_eq)
            if m_num:
                return int(m_num.group(0))

        # 3) First integer anywhere
        m_num = re.search(r"-?\d+", text)
        if m_num:
            return int(m_num.group(0))

        return "[invalid]"
    except Exception:
        return "[invalid]"


def extract_choice_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.strip())
    if match:
        return match.group(1)
    match = re.search(r"\b([a-d])\b", text.strip())
    if match:
        return match.group(1).upper()
    return "[invalid]"


def load_model(model_path, use_base_model=False, model_type="mistral"):
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

    trust_remote_code = model_type in ["phi", "phi-4", "gemma-3", "gemma-3-small"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    if use_base_model or model_path is None:
        actor_model = critic_model = base_model
    else:
        # Support two formats:
        # 1) Legacy checkpoint file with model_state_dict
        # 2) PEFT adapter directory saved via save_pretrained
        if os.path.isdir(model_path):
            # Adapter directory
            actor_model = PeftModel.from_pretrained(
                base_model,
                model_path,
                is_trainable=False,
            )
            critic_model = copy.deepcopy(actor_model)
        else:
            # Legacy checkpoint
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules="all-linear",
            )
            actor_model = get_peft_model(base_model, peft_config)
            checkpoint = torch.load(model_path)
            actor_model.load_state_dict(checkpoint["model_state_dict"])
            critic_model = copy.deepcopy(actor_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return actor_model, critic_model, tokenizer, device


def get_hyperparameters_from_log(model_dir, default_task: str):
    log_path = os.path.join(model_dir, "log.jsonl")
    try:
        with open(log_path, 'r') as f:
            hyperparameters = json.loads(f.readline().strip())
        return hyperparameters
    except Exception as e:
        print(f"Warning: Could not read hyperparameters from log file ({e})")
        # Fallback defaults
        return {
            "model_type": "mistral",
            "task_type": default_task,
            "cot_length": 100,
            "temperature": 1.0,
            "batch_size": 12,
        }


def get_model_paths_and_type(provided_path=None, all_checkpoints=False) -> Tuple[List[str], str]:
    if provided_path:
        model_dir = os.path.dirname(provided_path)
    else:
        model_dir = find_latest_result()
        if not model_dir:
            raise FileNotFoundError("No results directory found")

    if all_checkpoints:
        checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
        if not checkpoint_files:
            model_paths = [None]
        else:
            model_paths = sorted(checkpoint_files, key=os.path.getctime)
    else:
        checkpoint_files = glob.glob(os.path.join(model_dir, "model*.pt"))
        model_paths = [max(checkpoint_files, key=os.path.getctime)] if checkpoint_files else [None]

    try:
        with open(os.path.join(model_dir, "log.jsonl"), 'r') as f:
            hyperparameters = json.loads(f.readline().strip())
            model_type = hyperparameters.get("model_type", "mistral")
    except Exception as e:
        print(f"Warning: Could not read model type from log file ({e}), defaulting to mistral")
        model_type = "mistral"

    return model_paths, model_type


def get_run_dir_from_path(path_hint: str = None) -> str:
    """Resolve a run directory from a file or directory hint.
    If None, uses the latest result directory.
    """
    if path_hint is None:
        run_dir = find_latest_result()
        if not run_dir:
            raise FileNotFoundError("No results directory found")
        return run_dir
    if os.path.isdir(path_hint):
        return path_hint
    return os.path.dirname(path_hint)


def list_adapter_dirs(run_dir: str) -> List[str]:
    """List adapter_* directories in a run directory, sorted by batch index."""
    adapters = glob.glob(os.path.join(run_dir, "adapter_*"))
    def batch_num(p):
        try:
            return int(os.path.basename(p).split("_")[-1])
        except Exception:
            return -1
    adapters = [a for a in adapters if os.path.isdir(a)]
    return sorted(adapters, key=batch_num)


def load_task_data(task_type: str, num_samples: int = None) -> List[Tuple[str, str]]:
    """Build a list of (question, answer) pairs for the given task."""
    data = []
    if task_type in ["arithmetic", "arithmetic-negative"]:
        # Generate synthetic arithmetic problems
        batch_size = num_samples or 200
        batch = next(
            generate_question_answer_batches(
                num_batches=1,
                batch_size=batch_size,
                task_type=task_type,
                tokenizer=None,
                hyperparameters={},
            )
        )
        data = batch
    elif task_type == "svamp":
        it = load_svamp_dataset(split="test")
        for i, qa in enumerate(it):
            data.append(qa)
            if num_samples and len(data) >= num_samples:
                break
    elif task_type == "aqua":
        it = load_aqua_dataset(split="test")
        for i, qa in enumerate(it):
            data.append(qa)
            if num_samples and len(data) >= num_samples:
                break
    else:
        raise ValueError(f"Unsupported task_type for reasoning eval: {task_type}")
    return data


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

    is_mcq = hyperparameters.get("task_type") in ["aqua"]

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i + batch_size]
        questions, answers = zip(*batch)

        # Stage 1: thinking
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

        # Stage 2: deterministic answer
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
                max_new_tokens=(3 if is_mcq else 10),
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, tokenized_answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        for q, a, cot, gen in zip(questions, answers, cot_texts, generated_answers):
            if is_mcq:
                pred = extract_choice_letter(gen)
                correct_answer = extract_choice_letter(a) if len(a) == 1 else extract_choice_letter(str(a))
                is_correct = (pred == correct_answer)
            else:
                pred = extract_numeric_answer(gen)
                correct_answer = extract_numeric_answer(a)
                is_correct = (pred == correct_answer)

            all_results.append({
                "question": q,
                "correct_answer": correct_answer,
                "chain_of_thought": cot,
                "generated_answer": gen,
                "extracted_answer": pred,
                "is_correct": is_correct,
            })
            if is_correct:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_results


def plot_accuracy_over_batches(results_jsonl_path: str, save_path: str, task_type: str):
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
    plt.plot(batch_indices, accuracies, marker='o', linestyle='-', color='tab:orange')
    plt.title(f'{task_type.upper()} Accuracy vs Training Batch')
    plt.xlabel('Training Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(model_dir, checkpoint_path, model_type, task_type, accuracy, results, num_samples, batch_index_override=None):
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": None,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,
        "task_type": task_type,
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
    results_file = os.path.join(model_dir, f"{task_type}_results{model_type_suffix}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")

    combined_plot_path = os.path.join(model_dir, f"combined_metrics_{task_type}.png")
    plot_accuracy_over_batches(results_file, combined_plot_path, task_type)
    print(f"Updated combined {task_type} accuracy plot at {combined_plot_path}")
    return results_file


def main(
    task_type: str,
    model_path=None,
    num_samples=None,
    batch_size=None,
    use_base_model=False,
    model_type=None,
    stride=1,
    cot_length=None,
    temperature=None,
    baseline=False,
    baseline_thinking_tokens=None,
    baseline_temperature=None,
    run_dir=None,
    all_adapters=False,
):
    supported = ["arithmetic", "arithmetic-negative", "svamp", "aqua"]
    if task_type not in supported:
        raise ValueError(f"task_type must be one of {supported}")

    # Determine evaluation targets: single model_path, or all adapters in a run_dir
    targets: List[Tuple[str, int]] = []  # (path, batch_index_override)
    if all_adapters:
        rd = get_run_dir_from_path(run_dir or model_path)
        adapter_dirs = list_adapter_dirs(rd)
        if not adapter_dirs:
            print(f"No adapters found in {rd}")
            return
        for ad in adapter_dirs:
            try:
                bn = int(os.path.basename(ad).split("_")[-1])
            except Exception:
                bn = None
            targets.append((ad, bn))
        # Infer model_type from run_dir log
        try:
            with open(os.path.join(rd, "log.jsonl"), 'r') as f:
                hp = json.loads(f.readline().strip())
                inferred_model_type = hp.get("model_type", "mistral")
        except Exception:
            inferred_model_type = "mistral"
        if model_type is None:
            model_type = inferred_model_type
            print(f"Inferred model type: {model_type}")
    else:
        if not use_base_model:
            model_paths, inferred_model_type = get_model_paths_and_type(model_path, all_checkpoints=False)
            if model_type is None:
                model_type = inferred_model_type
                print(f"Inferred model type: {model_type}")
        else:
            model_paths = [None]
            model_type = model_type or "mistral"
        for p in model_paths:
            targets.append((p, None))

    for checkpoint_path, batch_override in targets:
        if checkpoint_path:
            print(f"\nEvaluating checkpoint: {checkpoint_path}")
            run_base_dir = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
            # If adapter dir, its parent is the run directory
            if os.path.basename(run_base_dir).startswith("adapter_"):
                run_base_dir = os.path.dirname(run_base_dir)
            hyperparameters = get_hyperparameters_from_log(run_base_dir, default_task=task_type)
            hyperparameters["task_type"] = task_type
            if cot_length is not None:
                hyperparameters["cot_length"] = cot_length
            if temperature is not None:
                hyperparameters["temperature"] = temperature
        else:
            hyperparameters = {
                "model_type": model_type,
                "task_type": task_type,
                "cot_length": cot_length or 100,
                "temperature": temperature or 1.0,
                "batch_size": 12,
            }

        actor_model, critic_model, tokenizer, device = load_model(
            checkpoint_path,
            use_base_model,
            model_type
        )

        test_data = load_task_data(task_type, num_samples=None)
        if stride > 1:
            test_data = test_data[::stride]
            print(f"Using stride={stride}, evaluating on {len(test_data)} examples")
        if num_samples is not None:
            test_data = test_data[:num_samples]

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

        base_results_dir = os.path.dirname(checkpoint_path) if checkpoint_path else f"results/{task_type}"
        if checkpoint_path and os.path.isdir(checkpoint_path):
            # adapter dir -> save in its parent (run directory)
            base_results_dir = os.path.dirname(checkpoint_path)
        os.makedirs(base_results_dir, exist_ok=True)
        results_file = save_results(
            base_results_dir,
            checkpoint_path,
            model_type,
            task_type,
            accuracy,
            results,
            num_samples,
            batch_index_override=(0 if baseline else batch_override),
        )
        print(f"Results appended to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline or actor-critic on reasoning tasks")
    parser.add_argument("--task_type", type=str, required=True, choices=["arithmetic", "arithmetic-negative", "math", "svamp", "aqua"], help="Task to evaluate")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model weights (default: latest result)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--use_base_model", action="store_true", help="Use base model without adapters")
    parser.add_argument("--model_type", type=str, choices=["llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"], default=None)
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every nth example")
    parser.add_argument("--cot_length", type=int, default=None, help="Override CoT length")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--baseline", action="store_true", help="Use standard baseline prompting")
    parser.add_argument("--baseline_thinking_tokens", type=int, default=None, help="Max tokens for baseline thinking stage")
    parser.add_argument("--baseline_temperature", type=float, default=None, help="Temperature for baseline thinking stage")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory containing adapter_* folders to evaluate")
    parser.add_argument("--all_adapters", action="store_true", help="Evaluate all adapters in the specified run directory (or latest if not provided)")
    args = parser.parse_args()

    try:
        main(
            args.task_type,
            args.model_path,
            args.num_samples,
            args.batch_size,
            args.use_base_model,
            args.model_type,
            args.stride,
            args.cot_length,
            args.temperature,
            args.baseline,
            args.baseline_thinking_tokens,
            args.baseline_temperature,
            args.run_dir,
            args.all_adapters,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


