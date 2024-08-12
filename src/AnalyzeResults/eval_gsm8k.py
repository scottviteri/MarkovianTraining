import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import argparse
import json
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model


def extract_answer(answer):
    if "=" in answer:
        answer = answer.split("=")[-1].strip()
    answer = answer.replace(",", "")
    try:
        answer = re.findall(r"\d+", answer.strip())[0]
        answer = int(answer)
    except:
        answer = "[invalid]"
    return answer


def load_model(model_path):
    # Load the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Define and apply the PEFT configuration
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, peft_config)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model, tokenizer, device


def evaluate_model(
    model, tokenizer, device, test_data, num_samples=None, batch_size=16
):
    correct = 0
    total = 0
    results = []

    for i in tqdm(range(0, len(test_data[:num_samples]), batch_size)):
        batch = test_data[i : i + batch_size]
        questions, answers = zip(*batch)

        # Generate the chain of thought reasoning
        prompts = [
            f"[INST] Produce minimal text which will help you answer the question.[/INST] Question: {q}\nReasoning:"
            for q in questions
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,
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


def main(model_path, num_samples, batch_size):
    model, tokenizer, device = load_model(model_path)

    test_data = load_dataset("openai/gsm8k", "main", split="test")
    test_data = [(q, a) for q, a in zip(test_data["question"], test_data["answer"])]

    accuracy, results = evaluate_model(
        model, tokenizer, device, test_data, num_samples, batch_size
    )

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
        default="/root/MarkovianTraining/SavedModels/PolicyGradientNormalized_GSM8K_latest.pt",
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
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)

    main(args.model_path, args.num_samples, args.batch_size)
