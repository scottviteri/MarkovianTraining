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

    # Set model to evaluation mode and configure generation parameters
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None

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


def get_model_path(provided_path=None, target_index=None):
    """Get model path from provided path or find latest with optional index matching."""
    if provided_path:
        model_dir = os.path.dirname(provided_path)
        model_path = find_checkpoint_with_index(model_dir, target_index)
    else:
        # Find most recent results directory
        results = glob.glob("results/gsm8k/*")
        if not results:
            raise FileNotFoundError("No GSM8K results directory found")
        latest_dir = max(results, key=os.path.getctime)
        model_path = find_checkpoint_with_index(latest_dir, target_index)
    
    print(f"Using checkpoint: {model_path}")
    return model_path


def main(
    model_path=None,
    num_samples=None,
    batch_size=16,
    use_base_model=False,
    model_type=None,
    stride=1,
    training_index=None,
):
    # Get model path and type if not explicitly provided
    if not use_base_model:
        model_path, inferred_model_type = get_model_path(model_path, training_index)
        if model_type is None:
            model_type = inferred_model_type
            print("Inferred Model Type", model_type)
    else:
        model_type = model_type or "mistral"  # Default to mistral for base model

    model, tokenizer, device = load_model(model_path, use_base_model, model_type)

    test_data = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

    # Apply stride to test data
    if stride > 1:
        test_data = test_data[::stride]
        print(f"Using stride={stride}, evaluating on {len(test_data)} examples")

    accuracy, results = evaluate_model(
        model, tokenizer, device, test_data, num_samples, batch_size, model_type
    )

    print(f"Accuracy: {accuracy:.2%}")

    # Create results directory if it doesn't exist
    os.makedirs("results/evaluations", exist_ok=True)

    # Generate a timestamp-based filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/evaluations/gsm8k_eval_{model_type}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "results": results,
                "model_path": model_path,
                "model_type": model_type,
                "num_samples": num_samples,
            },
            f,
            indent=2,
        )

    print(f"Detailed results saved to {output_file}")


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
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
