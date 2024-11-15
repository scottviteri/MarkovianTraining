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
from train import find_latest_result


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
    # Load the base model and tokenizer based on model type
    if model_type == "mistral":
        base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        base_model = "meta-llama/Llama-3.2-3B-Instruct"
    else:
        raise ValueError("model_type must be either 'mistral' or 'llama'")

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if not use_base_model:
        # Define and apply the PEFT configuration
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear",
        )
        model = get_peft_model(model, peft_config)

        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main(model_path, num_samples, batch_size, use_base_model, model_type):
    model, tokenizer, device = load_model(model_path, use_base_model, model_type)

    test_data = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

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
        default=None,  # Remove default path
        help="Path to the trained model weights",
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
        default="mistral",
        help="Choose between Mistral and Llama 3.1 models",
    )
    args = parser.parse_args()

    # If no model path is provided, try to find the latest checkpoint
    if args.model_path is None:
        args.model_path = find_latest_result()

        if args.model_path is None:
            print("Error: No model checkpoint found in the checkpoints directory.")
            exit(1)

        # Optionally, detect model type from the path
        args.model_type = "mistral" if "mistral" in args.model_path.lower() else "llama"

    if not args.use_base_model and not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)

    main(
        args.model_path,
        args.num_samples,
        args.batch_size,
        args.use_base_model,
        args.model_type,
    )
