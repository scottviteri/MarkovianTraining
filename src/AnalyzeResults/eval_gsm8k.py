import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams


def extract_answer(answer):
    if "=" in answer:
        answer = answer.split("=")[-1].strip()
    answer = answer.replace(",", "")
    try:
        answer = re.findall(r"\d+", answer.strip())[-1]
        answer = int(answer)
    except:
        answer = "[invalid]"
    return answer


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize VLLM model with merged weights
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        trust_remote_code=True,
        dtype="bfloat16",
        load_format="safetensors",
    )

    return llm, tokenizer


def evaluate_model(llm, tokenizer, test_data, num_samples=None):
    correct = 0
    total = 0
    results = []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=400,
        stop=["</s>", "[/INST]"],
    )

    prompts = [
        f"Work through the following question step by step, concisely decomposing problems into subproblems.\nQuestion: {question}\nStepByStep:"
        for question, _ in test_data[:num_samples]
    ]

    outputs = llm.generate(prompts, sampling_params)

    for (question, answer), output in tqdm(
        zip(test_data[:num_samples], outputs), total=len(prompts)
    ):
        generated_text = output.outputs[0].text
        generated_answer = extract_answer(generated_text)
        true_answer = extract_answer(answer)

        is_correct = generated_answer == true_answer
        correct += is_correct
        total += 1

        results.append(
            {
                "question": question,
                "true_answer": true_answer,
                "generated_answer": generated_answer,
                "is_correct": is_correct,
                "full_generation": generated_text,
            }
        )

    accuracy = correct / total
    return accuracy, results


def main(model_path, num_samples):
    llm, tokenizer = load_model(model_path)

    test_data = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

    accuracy, results = evaluate_model(llm, tokenizer, test_data, num_samples)

    print(f"Accuracy: {accuracy:.2%}")

    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on GSM8K test set."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="SavedModels/PolicyGradientNormalized_GSM8K_latest.pt",
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)

    main(args.model_path, args.num_samples)
