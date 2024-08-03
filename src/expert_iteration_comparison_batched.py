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
    # model_name = "distilgpt2"
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


def calculate_threshold(previous_losses):
    if len(previous_losses) > 0:
        return min(2.0, np.mean(previous_losses) - np.std(previous_losses))
    return 2.0


if __name__ == "__main__":

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"src/AnalyzeResults/ExpertIterationDictionary_{timestamp}.log"

    with open(filename, "w") as log_file:
        pass  # Empty file created

    model, tokenizer, device = load_mistral_model()
    frozen_model, _, _ = load_mistral_model()
    for param in frozen_model.parameters():
        param.requires_grad = False
    # Train the model to make q_cot_stub more likely
    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=1e-5)
    num_batches = 10000
    batch_size = 8
    qa_batches = list(
        generate_question_answer_batches(num_batches=num_batches, batch_size=batch_size)
    )
    previous_losses = []

    for batch_index, qa_batch in enumerate(qa_batches):
        questions, answers = zip(*qa_batch)

        # Prepare inputs for generation (left-padded)
        prompts = [
            f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST] Question: {q}\nStepByStep:"
            for q in questions
        ]
        tokenized_inputs = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized_inputs.input_ids.to(device)
        attention_mask = tokenized_inputs.attention_mask.to(device)

        # Generate reasoning
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=400,
                min_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        model.train()

        # Extract only the newly generated tokens (reasoning)
        reasoning = outputs[:, input_ids.shape[1] :]
        reasoning_text = tokenizer.batch_decode(reasoning, skip_special_tokens=True)
        print(reasoning_text[0])

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
        # Calculate log probabilities for answers
        log_probs = torch.nn.functional.log_softmax(
            logits[:, -answer_ids.shape[1] - 1 : -1, :], dim=-1
        )
        answer_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
        print("log probs: ", answer_log_probs[0])
        # Calculate average log probability for each row, ignoring padded tokens
        avg_log_probs = (answer_log_probs * tokenized_answers.attention_mask).sum(
            dim=1
        ) / tokenized_answers.attention_mask.sum(dim=1)
        nll_losses = -avg_log_probs

        # Determine which examples to use for training
        threshold = calculate_threshold(
            [loss for batch in previous_losses for loss in batch]
        )
        train_mask = (nll_losses < threshold).float()

        if train_mask.sum() > 0:
            # Calculate loss only for selected examples
            tokenized_q_cot = tokenizer(
                [
                    f"[INST] Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.[/INST] \nStepByStep:: {reasoning_text[i]}"
                    for i in range(batch_size)
                    if train_mask[i] > 0.0
                ],
                padding=True,
                return_tensors="pt",
            ).to(device)
            optimizer.zero_grad()
            q_cot_outputs = model(
                tokenized_q_cot.input_ids, tokenized_q_cot.attention_mask
            )
            q_cot_log_probs = torch.nn.functional.log_softmax(
                q_cot_outputs.logits[:, -401:-1, :], dim=-1
            )
            cot_log_probs = q_cot_log_probs.gather(
                2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
            ).squeeze(-1)
            print("cot ave log prob: ", cot_log_probs[0].mean())
            loss = cot_log_probs.mean()
            loss.backward()
            optimizer.step()

        # Update previous_losses and log results
        previous_losses.append(nll_losses.tolist())

        # Log only the first element of the batch
        q = questions[0]
        ans = answers[0]
        reasoning_text = reasoning_text[0]
        nll_loss = nll_losses[0].item()
        print("Ans: ", ans, "NLL Loss: ", nll_loss)
        # Write progress to a file iteratively
        with open(filename, "a") as log_file:
            log_entry = {
                "Aggregate loss": nll_loss,
                "Batch Index": batch_index,
                "Prev Observation": f"Question: {q}",
                "Action": f"StepByStep: {reasoning_text}",
                "Observation": f"Answer: {ans}",
                "Reasoning Contains Answer": str(ans) in reasoning_text,
                "Training Example": bool(train_mask[0].item()),
            }
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for each entry
