import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm
import os


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

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def evaluate_model(model, tokenizer, device, test_data, num_samples=None):
    correct = 0
    total = 0
    results = []

    for question, answer in tqdm(test_data[:num_samples]):
        # Generate the chain of thought reasoning
        prompt = f"Work through the following question step by step, concisely decomposing problems into subproblems.\nQuestion: {question}\nStepByStep:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.0,
                do_sample=False,
            )

        reasoning_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate the potential answer given the reasoning
        full_prompt = f"Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.\nStepByStep: {reasoning_text} \nAnswer:"
        answer_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            answer_outputs = model.generate(
                **answer_inputs,
                max_new_tokens=15,
                do_sample=False,
            )

        generated_answer_text = tokenizer.decode(
            answer_outputs[0][answer_inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        generated_answer = extract_answer(generated_answer_text)
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
                "full_reasoning": reasoning_text,
                "generated_answer_text": generated_answer_text,
            }
        )

    accuracy = correct / total
    return accuracy, results


def main(model_path, num_samples):
    model, tokenizer, device = load_model(model_path)

    test_data = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

    accuracy, results = evaluate_model(model, tokenizer, device, test_data, num_samples)

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
