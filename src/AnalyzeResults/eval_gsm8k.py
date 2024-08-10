import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm


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
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def evaluate_model(model, tokenizer, device, test_data, num_samples=None):
    correct = 0
    total = 0
    results = []

    for question, answer in tqdm(test_data[:num_samples]):
        prompt = f"Work through the following question step by step, concisely decomposing problems into subproblems.\nQuestion: {question}\nStepByStep:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.0,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        "model_path", type=str, help="Path to the trained model weights"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    args = parser.parse_args()

    main(args.model_path, args.num_samples)
