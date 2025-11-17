import argparse
import os
import json
from typing import Dict, List, Tuple

from datasets import load_dataset

from utils import (
    load_model as load_base_models,
    load_gsm8k_dataset,
    load_svamp_dataset,
    load_aqua_dataset,
    load_math_dataset,
    load_mathqa_dataset,
)

# Reuse evaluators' evaluate_model functions to ensure identical prompting behavior
from evaluation import evaluate_model_on_gsm8k as eval_gsm8k
from evaluate_mmlu import evaluate_model as eval_mmlu
from evaluate_reasoning import evaluate_model as eval_reasoning


def default_task_specs() -> Dict[str, Dict]:
    return {
        "gsm8k": {"cot_length": 100, "temperature": 1.0},
        "mmlu": {"cot_length": 150, "temperature": 1.0},
        "arithmetic": {"cot_length": 150, "temperature": 1.0},
        "arithmetic-negative": {"cot_length": 150, "temperature": 1.0},
        "svamp": {"cot_length": 150, "temperature": 1.0},
        "aqua": {"cot_length": 150, "temperature": 1.0},
        "math": {"cot_length": 150, "temperature": 1.0},
        "mathqa": {"cot_length": 150, "temperature": 1.0},
        "arc": {"cot_length": 150, "temperature": 1.0},
        "arc_easy": {"cot_length": 150, "temperature": 1.0},
        "arc_challenge": {"cot_length": 150, "temperature": 1.0},
    }


def load_task_data(task: str, num_samples: int | None, stride: int | None) -> List[Tuple[str, str]]:
    data: List[Tuple[str, str]] = []
    if task == "gsm8k":
        data = list(load_gsm8k_dataset(split="test", chunk_size=10_000))
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        split = "validation"
        d = ds[split]
        for ex in d:
            stem = ex["question"]
            choices = ex["choices"] if "choices" in ex else ex.get("options", [])
            if not choices or len(choices) < 4:
                continue
            options_text = "\n".join(
                [
                    f"A. {choices[0]}",
                    f"B. {choices[1]}",
                    f"C. {choices[2]}",
                    f"D. {choices[3]}",
                ]
            )
            question_text = f"{stem}\n\nOptions:\n{options_text}"
            if "answer" in ex and isinstance(ex["answer"], int):
                correct_letter = ["A", "B", "C", "D"][ex["answer"]]
            else:
                letter = str(ex.get("answer", "")).strip().upper()
                if letter not in ["A", "B", "C", "D"]:
                    continue
                correct_letter = letter
            data.append((question_text, correct_letter))
    elif task in ("arithmetic", "arithmetic-negative"):
        # Build synthetic pairs via the evaluator by requesting batches
        from utils import generate_question_answer_batches

        batch_size = 100
        total = num_samples or 200
        needed_batches = (total + batch_size - 1) // batch_size
        gen = generate_question_answer_batches(
            num_batches=needed_batches,
            batch_size=batch_size,
            task_type=task,
            tokenizer=None,
            hyperparameters={},
        )
        for _ in range(needed_batches):
            try:
                data.extend(next(gen))
            except StopIteration:
                break
        data = data[: total]
    elif task == "svamp":
        it = load_svamp_dataset(split="test")
        for qa in it:
            data.append(qa)
    elif task == "aqua":
        it = load_aqua_dataset(split="test")
        for qa in it:
            data.append(qa)
    elif task == "math":
        try:
            it = load_math_dataset(split="test")
        except Exception:
            it = load_math_dataset(split="validation")
        for qa in it:
            data.append(qa)
    elif task == "mathqa":
        it = load_mathqa_dataset(split="validation")
        for qa in it:
            data.append(qa)
    elif task in ("arc", "arc_easy", "arc_challenge"):
        # Map to ARC subsets
        subset = (
            "ARC-Challenge" if task in ("arc", "arc_challenge") else "ARC-Easy"
        )
        from utils import load_arc_dataset
        it = load_arc_dataset(split="validation", subset=subset)
        for qa in it:
            data.append(qa)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Apply stride and truncate to num_samples
    if stride and stride > 1:
        data = data[::stride]
    if num_samples is not None:
        data = data[:num_samples]
    return data


def run_baseline_for_task(
    task: str,
    model_type: str,
    use_base_model: bool,
    batch_size: int | None,
    num_samples: int | None,
    stride: int | None,
    overrides: Dict[str, float],
) -> Tuple[float, str]:
    specs = default_task_specs()[task].copy()
    specs.update({k: v for k, v in overrides.items() if v is not None})

    # Build hyperparameters dict expected by evaluators
    canonical_task = "arc" if task.startswith("arc") else task
    h: Dict[str, object] = {
        "model_type": model_type,
        "task_type": canonical_task,
        "cot_length": int(specs["cot_length"]),
        "temperature": float(specs["temperature"]),
        "batch_size": batch_size if batch_size is not None else 12,
        # Markovian default True (baseline answers use critic deterministically)
        "markovian": True,
    }

    # Load models/tokenizer/device via central utility
    actor_model, critic_model, tokenizer, device = load_base_models(model_type, h)

    # Get test data
    test_data = load_task_data(task, num_samples=num_samples, stride=stride)

    # Select evaluator
    if task == "gsm8k":
        accuracy, _ = eval_gsm8k(
            actor_model,
            critic_model,
            tokenizer,
            device,
            test_data,
            h,
            num_samples=num_samples,
            batch_size=(batch_size if batch_size is not None else int(h["batch_size"]) * 2),
            baseline_mode=True,
            baseline_thinking_tokens=int(h["cot_length"]),
            baseline_temperature=float(h["temperature"]),
        )
        out_dir = os.path.join("results", "evaluations")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"gsm8k_results_{model_type}.jsonl")
    elif task == "mmlu":
        accuracy, _ = eval_mmlu(
            actor_model,
            critic_model,
            tokenizer,
            device,
            test_data,
            h,
            num_samples=num_samples,
            batch_size=(batch_size if batch_size is not None else int(h["batch_size"]) * 2),
            baseline_mode=True,
            baseline_thinking_tokens=int(h["cot_length"]),
            baseline_temperature=float(h["temperature"]),
        )
        out_dir = os.path.join("results", "mmlu")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"mmlu_results_{model_type}.jsonl")
    else:
        # Reasoning evaluator for arithmetic-like and numeric/mc tasks
        accuracy, _ = eval_reasoning(
            actor_model,
            critic_model,
            tokenizer,
            device,
            test_data,
            h,
            num_samples=num_samples,
            batch_size=(batch_size if batch_size is not None else int(h["batch_size"]) * 2),
            baseline_mode=True,
            baseline_thinking_tokens=int(h["cot_length"]),
            baseline_temperature=float(h["temperature"]),
        )
        out_dir = os.path.join("results", canonical_task)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{task}_results_{model_type}.jsonl")

    # Append a simple JSONL row to per-task file (similar to evaluators)
    entry = {
        "accuracy": accuracy,
        "model_type": model_type,
        "task_type": task,
        "num_samples": len(test_data),
        "baseline": True,
        "cot_length": int(h["cot_length"]),
        "temperature": float(h["temperature"]),
    }
    with open(out_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return accuracy, out_file


def main():
    parser = argparse.ArgumentParser(description="Run baseline (natural prompt) evaluations across datasets.")
    parser.add_argument("--model_type", type=str, required=True, choices=[
        "llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"
    ])
    parser.add_argument("--use_base_model", action="store_true", help="Kept for API symmetry; models are loaded as base.")
    parser.add_argument("--tasks", type=str, nargs="*", default=[
        "gsm8k", "mmlu", "arithmetic", "arithmetic-negative", "svamp", "aqua", "arc_easy", "arc_challenge"
    ], help=(
        "Which tasks to evaluate. Use arc_easy or arc_challenge to select ARC subsets; "
        "you can pass any subset of tasks here."
    ))
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    # Optional global overrides
    parser.add_argument("--cot_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    overrides = {"cot_length": args.cot_length, "temperature": args.temperature}

    results: Dict[str, float] = {}
    for task in args.tasks:
        try:
            acc, out_file = run_baseline_for_task(
                task=task,
                model_type=args.model_type,
                use_base_model=args.use_base_model,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                stride=args.stride,
                overrides=overrides,
            )
            print(f"{task}: {acc:.2%} -> {out_file}")
            results[task] = acc
        except Exception as e:
            print(f"{task}: ERROR {e}")

    # Write a combined summary
    summary_path = os.path.join("results", f"baseline_summary_{args.model_type}.md")
    lines = [f"# Baseline (natural prompt) for {args.model_type}"]
    for task in args.tasks:
        if task in results:
            lines.append(f"- {task}: {results[task]:.2%}")
        else:
            lines.append(f"- {task}: error")
    os.makedirs("results", exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()


