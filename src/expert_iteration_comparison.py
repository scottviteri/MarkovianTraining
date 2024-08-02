import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import bitsandbytes
import json
import datetime


def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device


def generate_question_answer_pairs(num_pairs: int):
    for _ in range(num_pairs):
        numbers = [random.randint(1, 99) for _ in range(15)]
        question = " + ".join(map(str, numbers))
        answer = sum(numbers)
        yield (question, answer)


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
    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=1e-7)
    dataset = list(generate_question_answer_pairs(10000))
    previous_losses = []

    for i, qa in enumerate(dataset):
        q, ans = qa
        instructions = "Work through the following question step by step, concisely decomposing problems into subproblems."
        question = f"Question: {q}."
        cot_stub = "Reasoning:"
        prompt = f"[INST] {instructions} [/INST] {question} {cot_stub}"
        input_sequence = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = (input_sequence != tokenizer.pad_token_id).long()

        with torch.no_grad():
            output = model.generate(
                input_sequence,
                attention_mask=attention_mask,
                max_new_tokens=400,
                min_new_tokens=400,
                do_sample=True,
                temperature=1.0,
            )

        # Extract only the newly generated tokens
        reasoning = tokenizer.decode(
            output[0][input_sequence.shape[1] :], skip_special_tokens=True
        )
        predict = "Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction."
        q_cot_ans = f"[INST] {predict} [/INST] Question: {q} Reasoning: {reasoning} Answer: {ans}"
        # Tokenize the q_cot_ans sequence
        input_sequence = tokenizer.encode(q_cot_ans, return_tensors="pt").to(device)
        attention_mask = (input_sequence != tokenizer.pad_token_id).long()
        # Tokenize the answer separately
        answer_tokens = tokenizer.encode(
            str(ans), return_tensors="pt", add_special_tokens=False
        )[:, 1:].to(device)

        with torch.no_grad():
            # Get the model's output logits
            outputs = frozen_model(input_sequence, attention_mask=attention_mask)
            logits = outputs.logits

        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather the log probabilities of the answer tokens
        answer_log_probs = (
            log_probs[0, -len(answer_tokens[0]) - 1 : -1, :]
            .gather(1, answer_tokens[0].unsqueeze(-1))
            .squeeze(-1)
        )

        # Calculate the average log probability
        avg_log_prob = answer_log_probs.mean().item()
        nll_loss = -avg_log_prob

        if len(previous_losses) > 0:
            threshold = min(5.0, np.mean(previous_losses) - np.std(previous_losses))
        else:
            threshold = 5.0
        print(q_cot_ans)
        print("Token losses: ", -answer_log_probs)
        print(
            f"Answer: {ans} Threshold Loss: {threshold:.2f}, Current Loss: {nll_loss:.2f}"
        )

        if nll_loss < threshold:
            model.train()
            q_cot_stub = (
                f"[INST] {predict} [/INST] Question: {q} Reasoning: {reasoning} Answer:"
            )
            input_ids = tokenizer.encode(q_cot_stub, return_tensors="pt").to(device)
            outputs = model(input_ids)
            logits = outputs.logits
            start_pos = len(
                tokenizer.encode(
                    f"[INST] {predict} [/INST] Question: {q} Reasoning:",
                    add_special_tokens=False,
                )
            )
            log_probs = torch.nn.functional.log_softmax(
                logits[:, start_pos:-1, :], dim=-1
            )
            target_tokens = input_ids[
                :, start_pos + 1 :
            ]  # +1 to shift by one for next token prediction
            gathered_log_probs = log_probs.gather(
                2, target_tokens.unsqueeze(2)
            ).squeeze(2)
            loss = -gathered_log_probs.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Training loss: {loss.item()}")
            model.eval()

        # Write progress to a file iteratively
        with open(filename, "a") as log_file:
            log_entry = {
                "Aggregate loss": nll_loss,
                "Batch Index": len(previous_losses),
                "Prev Observation": f"Question: {q}",
                "Action": f"Reasoning: {reasoning}",
                "Observation": f"Answer: {ans}",
                "Reasoning Contains Answer": str(ans) in reasoning,
                "Training Example": str(nll_loss < threshold),
            }
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for each entry

        previous_losses.append(nll_loss)
