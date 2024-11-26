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
from train import find_latest_result, initialize_model_and_optimizer
import glob


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
    # Create a minimal hyperparameters dict with required values
    hyperparameters = {
        "model_type": model_type,
        "lr": 1e-4,  # Default learning rate, won't be used in evaluation
        "gradient_accumulation_steps": 1,  # Won't be used in evaluation
    }

    # Initialize model using the same function as training
    model, _, tokenizer, device, _ = initialize_model_and_optimizer(
        model_type=model_type,
        hyperparameters=hyperparameters,
        checkpoint_path=None if use_base_model else model_path,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set model to evaluation mode and configure generation parameters
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def evaluate_model(
    model,
    tokenizer,
    device,
    test_data,
    num_samples=None,
    batch_size=16,
    model_type="mistral",
):
    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(test_data[:num_samples]), batch_size)):
        batch = test_data[i : i + batch_size]
        questions, answers = zip(*batch)

        # Generate prompts based on model type
        prompts = [
            (
                f"<|start_header_id|>user<|end_header_id|>Produce minimal text which will help you answer the question.<|eot_id|><|start_header_id|>assistant<|end_header_id|> Question: {q}\nReasoning:"
                if model_type == "llama"
                else f"{tokenizer.bos_token} Produce minimal text which will help you answer the question. {tokenizer.eos_token} Question: {q}\nReasoning:"
            )
            for q in questions
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                min_new_tokens=100,
                do_sample=False,
            )

        reasoning_tokens = outputs[:, inputs.input_ids.shape[1] :]

        # Use the provided batch processing function
        extracted_generated_answers = batch_process_answers(
            model, tokenizer, device, reasoning_tokens, answers, use_gsm8k=True
        )

        true_answers = [extract_answer(a) for a in answers]

        for q, true_ans, gen_ans in zip(
            questions, true_answers, extracted_generated_answers
        ):
            is_correct = gen_ans == true_ans
            correct += is_correct
            total += 1

            results.append(
                {
                    "question": q,
                    "true_answer": true_ans,
                    "generated_answer": gen_ans,
                    "is_correct": is_correct,
                }
            )

    accuracy = correct / total
    return accuracy, results


def batch_process_answers(
    model, tokenizer, device, reasoning_tokens, answers, use_gsm8k
):
    reasoning_text = tokenizer.batch_decode(reasoning_tokens, skip_special_tokens=True)

    full_prompts = [
        f"Reasoning: {r}\nAnswer: {a}" for r, a in zip(reasoning_text, answers)
    ]

    tokenized_full_prompts = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    extracted_generated_answers = None
    if use_gsm8k:
        partial_prompts = [f"Reasoning: {r}\nAnswer:" for r in reasoning_text]
        tokenized_partial_prompts = tokenizer(
            partial_prompts, padding=True, return_tensors="pt"
        ).to(device)
        max_answer_length = 15
        with torch.no_grad():
            generated_outputs = model.generate(
                input_ids=tokenized_partial_prompts.input_ids,
                attention_mask=tokenized_partial_prompts.attention_mask,
                max_new_tokens=max_answer_length,
                # min_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_answers = tokenizer.batch_decode(
            generated_outputs[:, -max_answer_length - 1 :], skip_special_tokens=True
        )
        selected_answers = [x.split("\nAnswer: ")[-1] for x in generated_answers]
        extracted_generated_answers = [extract_answer(ans) for ans in selected_answers]

    return extracted_generated_answers


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


def get_model_paths_and_type(provided_path=None, target_index=None, all_checkpoints=False):
    """Get model path(s) and infer model type from log file."""
    if provided_path:
        model_dir = os.path.dirname(provided_path)
    else:
        # Find most recent results directory
        results = glob.glob("results/gsm8k/*")
        if not results:
            raise FileNotFoundError("No GSM8K results directory found")
        model_dir = max(results, key=os.path.getctime)
    
    # Get model paths
    if all_checkpoints:
        model_paths = find_all_checkpoints(model_dir)
        print(f"Found {len(model_paths)} checkpoints")
    else:
        model_paths = [find_checkpoint_with_index(model_dir, target_index)]
    
    # Get model type from log.jsonl (same for all checkpoints in directory)
    log_path = os.path.join(model_dir, "log.jsonl")
    try:
        with open(log_path, 'r') as f:
            hyperparameters = json.loads(f.readline().strip())
            model_type = hyperparameters.get("model_type", "mistral")
    except Exception as e:
        print(f"Warning: Could not read model type from log file ({e}), defaulting to mistral")
        model_type = "mistral"
    
    return model_paths, model_type


def save_results(model_dir, checkpoint_path, model_type, accuracy, results, num_samples):
    """Save or append results to a running results file in the model directory."""
    results_file = os.path.join(model_dir, "gsm8k_results.jsonl")
    
    # Extract batch index from checkpoint filename
    batch_index = None
    if checkpoint_path:
        match = re.search(r'model_(\d+)_', os.path.basename(checkpoint_path))
        if match:
            batch_index = int(match.group(1))
    
    # Create results entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": batch_index,
        "accuracy": accuracy,
        "model_path": checkpoint_path,
        "model_type": model_type,
        "num_samples": num_samples,
        "detailed_results": results
    }
    
    # Append to results file
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")
    
    return results_file


def main(
    model_path=None,
    num_samples=None,
    batch_size=16,
    use_base_model=False,
    model_type=None,
    stride=1,
    training_index=None,
    all_checkpoints=False,
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
        
        model, tokenizer, device = load_model(checkpoint_path, use_base_model, model_type)
        
        test_data = load_dataset("openai/gsm8k", "main", split="test")
        test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

        if stride > 1:
            test_data = test_data[::stride]
            print(f"Using stride={stride}, evaluating on {len(test_data)} examples")

        accuracy, results = evaluate_model(
            model, tokenizer, device, test_data, num_samples, batch_size, model_type
        )

        print(f"Accuracy: {accuracy:.2%}")

        # Save results to running file in model directory
        model_dir = os.path.dirname(checkpoint_path) if checkpoint_path else "results/evaluations"
        results_file = save_results(
            model_dir,
            checkpoint_path,
            model_type,
            accuracy,
            results,
            num_samples
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
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use_base_model",
        action="store_true",
        help="Use the base model without LoRA or loading weights",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mistral", "llama"],
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
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
