import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import numpy as np
import bitsandbytes
import json
import datetime


def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def generate_question_answer_batch(batch_size: int):
    qa_batch = []
    for _ in range(batch_size):
        numbers = [random.randint(1, 99) for _ in range(15)]
        qa_batch.append((" + ".join(map(str, numbers)), str(sum(numbers))))
    return qa_batch


def generate_question_answer_batches(num_batches: int, batch_size: int):
    return [generate_question_answer_batch(batch_size) for _ in range(num_batches)]


def generate_and_calculate_log_probs(model, frozen_model, tokenizer, device, questions, answers, reasoning_text=None):
    if reasoning_text is None:
        # Generate reasoning using the provided model (which could be trained or frozen)
        prompts = [
            f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST] Question: {q}\nStepByStep:"
            for q in questions
        ]
        tokenized_inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=400,
                min_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        reasoning = outputs[:, tokenized_inputs.input_ids.shape[1]:]
        reasoning_text = tokenizer.batch_decode(reasoning, skip_special_tokens=True)

    # Calculate log probs using the frozen model
    tokenized_answers = tokenizer(
        answers, padding=True, return_tensors="pt", add_special_tokens=False
    ).to(device)
    tokenized_cot_ans = tokenizer(
        [
            f"[INST] Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.[/INST] \nStepByStep:: {r} \nAnswer: {a}"
            for r, a in zip(reasoning_text, answers)
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = frozen_model(
            tokenized_cot_ans.input_ids,
            attention_mask=tokenized_cot_ans.attention_mask,
        )
        logits = outputs.logits

    answer_ids = tokenized_answers.input_ids
    log_probs = torch.nn.functional.log_softmax(
        logits[:, -answer_ids.shape[1] - 1 : -1, :], dim=-1
    )
    answer_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    avg_log_probs = (answer_log_probs * tokenized_answers.attention_mask).sum(
        dim=1
    ) / tokenized_answers.attention_mask.sum(dim=1)

    return reasoning_text, avg_log_probs


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"src/AnalyzeResults/PolicyGradientDictionary_{timestamp}.log"

    with open(filename, "w") as log_file:
        pass  # Empty file created

    model, tokenizer, device = load_mistral_model()
    frozen_model, _, _ = load_mistral_model()
    for param in frozen_model.parameters():
        param.requires_grad = False
    # Train the model to make q_cot_stub more likely
    learning_rate = 1e-4
    batch_size = 6
    gradient_accumulation_steps = 8  # Adjust this value as needed

    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    num_batches = 10000
    qa_batches = list(
        generate_question_answer_batches(num_batches=num_batches, batch_size=batch_size)
    )
    previous_losses = []

    # Log hyperparameters before starting the training loop
    with open(filename, "w") as log_file:
        hyperparameters = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "num_batches": num_batches,
        }
        json.dump({"hyperparameters": hyperparameters}, log_file)
        log_file.write("\n")  # Add a newline after the hyperparameters

    optimizer.zero_grad()  # Move this outside the loop

    for batch_index, qa_batch in enumerate(qa_batches):
        questions, answers = zip(*qa_batch)

        # Generate reasoning using the trained model, but calculate log probs using the frozen model
        reasoning_text, avg_log_probs = generate_and_calculate_log_probs(model, frozen_model, tokenizer, device, questions, answers)

        # Generate baseline reasoning using the frozen model, and calculate log probs using the frozen model
        baseline_reasoning_text, baseline_avg_log_probs = generate_and_calculate_log_probs(frozen_model, frozen_model, tokenizer, device, questions, answers)

        # Calculate the advantage (reward - baseline)
        advantage = avg_log_probs - baseline_avg_log_probs

        # Calculate loss for all examples in the batch
        tokenized_q_cot = tokenizer(
            [
                f"[INST] Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.[/INST] \nStepByStep:: {reasoning_text[i]}"
                for i in range(batch_size)
            ],
            padding=True,
            return_tensors="pt",
        ).to(device)
        q_cot_outputs = model(
            tokenized_q_cot.input_ids, tokenized_q_cot.attention_mask
        )
        q_cot_log_probs = torch.nn.functional.log_softmax(
            q_cot_outputs.logits[:, -401:-1, :], dim=-1
        )
        cot_log_probs = q_cot_log_probs.gather(
            2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Use advantage for policy gradient
        loss = (cot_log_probs.mean(dim=1) * -advantage).mean() / gradient_accumulation_steps
        loss.backward()

        # Only update weights after accumulating gradients
        if (batch_index + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update previous_losses and log results
        previous_losses.append(avg_log_probs.tolist())

        # Log only the first element of the batch
        q = questions[0]
        ans = answers[0]
        reasoning_text = reasoning_text[0]
        avg_log_prob = avg_log_probs[0].item()
        baseline_avg_log_prob = baseline_avg_log_probs[0].item()
        advantage = advantage[0].item()
        print(reasoning_text)
        print("Ans: ", ans, "Avg Log Prob: ", avg_log_prob)
        # Write progress to a file iteratively
        with open(filename, "a") as log_file:
            log_entry = {
                "Aggregate loss": loss.item() * gradient_accumulation_steps,
                "Batch Index": batch_index,
                "Prev Observation": f"Question: {q}",
                "Action": f"StepByStep: {reasoning_text}",
                "Observation": f"Answer: {ans}",
                "Reasoning Contains Answer": str(ans) in reasoning_text,
                "Avg Log Prob": avg_log_prob,
                "Baseline Avg Log Prob": baseline_avg_log_prob,
                "Advantage": advantage,
            }
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for each entry

    # Perform final optimization step for any remaining gradients
    if (batch_index + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()