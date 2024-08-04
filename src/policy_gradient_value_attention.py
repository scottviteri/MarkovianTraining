import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralDecoderLayer,
)
import random
import numpy as np
import bitsandbytes
import json
import datetime


class ValueHead(torch.nn.Module):
    def __init__(self, hidden_size, pretrained_layer):
        super().__init__()

        config = MistralConfig(
            hidden_size=hidden_size,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )

        # Initialize with pretrained weights
        self.attention = MistralDecoderLayer(config, layer_idx=-1)
        self.attention.load_state_dict(pretrained_layer.state_dict(), strict=False)

        # Convert attention layers to bfloat16
        self.attention = self.attention.to(torch.bfloat16)

        # Linear layer for final value prediction
        self.linear = torch.nn.Linear(hidden_size, 1, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        # Ensure hidden_states are in bfloat16
        hidden_states = hidden_states.to(torch.bfloat16)

        # Generate position_ids
        batch_size, seq_length, _ = hidden_states.shape
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        ## Generate attention mask (assuming all tokens are valid)
        # attention_mask = torch.ones(
        #    (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
        # )

        attention_output = self.attention(
            hidden_states=hidden_states,
            # attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        last_token_hidden = attention_output[:, -1, :]
        return self.linear(last_token_hidden).squeeze(-1)


def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Add value head, initializing with the last pretrained layer
    pretrained_last_layer = model.model.layers[-1]
    model.value_head = ValueHead(4096, pretrained_last_layer)

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


def generate_reasoning(model, tokenizer, device, questions):
    prompts = [
        f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST]\nQuestion: {q}\nStepByStep:"
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
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    reasoning = outputs.sequences[:, tokenized_inputs.input_ids.shape[1] :]
    reasoning_text = tokenizer.batch_decode(reasoning, skip_special_tokens=True)

    # Return the second-to-last hidden state
    # last_hidden_state = outputs.hidden_states[-2]
    last_hidden_states = torch.cat(
        [x[-2] for x in outputs.hidden_states], dim=1
    ).detach()

    return reasoning_text, last_hidden_states


def calculate_log_probs(model, tokenizer, device, reasoning_text, answers):
    tokenized_cot_ans = tokenizer(
        [
            f"[INST] Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.[/INST]\nStepByStep:: {r} \nAnswer: {a}"
            for r, a in zip(reasoning_text, answers)
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    tokenized_answers = tokenizer(
        answers, padding=True, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        outputs = model(
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

    return avg_log_probs


def calculate_baseline(previous_advantages, window_size=100):
    if len(previous_advantages) < window_size:
        return np.mean(previous_advantages) if previous_advantages else 0
    return np.mean(previous_advantages[-window_size:])


def calculate_ppo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(current_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()


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
    batch_size = 4
    gradient_accumulation_steps = 8  # Adjust this value as needed
    use_ppo = True
    ppo_epsilon = 0.2

    optimizer = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    num_batches = 10000
    qa_batches = list(
        generate_question_answer_batches(num_batches=num_batches, batch_size=batch_size)
    )
    previous_advantages = []

    # Log hyperparameters before starting the training loop
    with open(filename, "w") as log_file:
        hyperparameters = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "num_batches": num_batches,
            "use_ppo": use_ppo,
            "ppo_epsilon": ppo_epsilon,
        }
        json.dump(hyperparameters, log_file)
        log_file.write("\n")  # Add a newline after the hyperparameters

    optimizer.zero_grad()  # Move this outside the loop

    for batch_index, qa_batch in enumerate(qa_batches):
        questions, answers = zip(*qa_batch)

        # Generate reasoning using the trained model
        reasoning_text, last_hidden_state = generate_reasoning(
            model, tokenizer, device, questions
        )

        # Calculate value prediction
        value_prediction = model.value_head(last_hidden_state[:, -400:, :])

        # Calculate log probs using the frozen model
        avg_log_probs = calculate_log_probs(
            frozen_model, tokenizer, device, reasoning_text, answers
        )

        # Generate baseline reasoning using the frozen model
        baseline_reasoning_text, baseline_last_hidden_state = generate_reasoning(
            frozen_model, tokenizer, device, questions
        )

        # Calculate baseline log probs
        baseline_avg_log_probs = calculate_log_probs(
            frozen_model, tokenizer, device, baseline_reasoning_text, answers
        )

        # Calculate the initial advantage and final advantage
        initial_advantage = avg_log_probs - baseline_avg_log_probs
        advantage = initial_advantage - value_prediction.detach()

        # Tokenize questions and reasoning together
        tokenized_q_cot = tokenizer(
            [
                f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST]\nQuestion: {q}\nStepByStep: {r}"
                for q, r in zip(questions, reasoning_text)
            ],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Calculate old log probabilities
        with torch.no_grad():
            old_q_cot_outputs = frozen_model(
                tokenized_q_cot.input_ids, tokenized_q_cot.attention_mask
            )
            old_q_cot_log_probs = torch.nn.functional.log_softmax(
                old_q_cot_outputs.logits[:, -401:-1, :], dim=-1
            )
            old_cot_log_probs = old_q_cot_log_probs.gather(
                2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
            ).squeeze(-1)

        # Calculate current log probabilities
        q_cot_outputs = model(tokenized_q_cot.input_ids, tokenized_q_cot.attention_mask)
        q_cot_log_probs = torch.nn.functional.log_softmax(
            q_cot_outputs.logits[:, -401:-1, :], dim=-1
        )
        current_cot_log_probs = q_cot_log_probs.gather(
            2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
        ).squeeze(-1)

        if use_ppo:
            # Use PPO loss
            ppo_ratio = torch.exp(
                current_cot_log_probs.mean(dim=1) - old_cot_log_probs.mean(dim=1)
            )
            clipped_ratio = torch.clamp(ppo_ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
            ppo_loss = (
                calculate_ppo_loss(
                    current_cot_log_probs.mean(dim=1),
                    old_cot_log_probs.mean(dim=1),
                    advantage,
                    epsilon=ppo_epsilon,
                )
                / gradient_accumulation_steps
            )
            value_loss = torch.abs(
                initial_advantage - value_prediction
            ).mean()  # or use torch.square()
            loss = ppo_loss + value_loss
        else:
            # Use original policy gradient loss
            policy_loss = (current_cot_log_probs.mean(dim=1) * -advantage).mean()
            value_loss = torch.abs(
                initial_advantage - value_prediction
            ).mean()  # or use torch.square()
            loss = (policy_loss + value_loss) / gradient_accumulation_steps

        loss.backward()

        # Only update weights after accumulating gradients
        if (batch_index + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update previous_advantages and log results
        previous_advantages.extend(initial_advantage.tolist())

        # Log only the first element of the batch
        q = questions[0]
        ans = answers[0]
        reasoning_text_first = reasoning_text[0]
        avg_log_prob = avg_log_probs[0].item()
        baseline_avg_log_prob = baseline_avg_log_probs[0].item()
        initial_advantage_value = initial_advantage[0].item()
        advantage_value = advantage[0].item()
        print(reasoning_text_first)
        print("Ans: ", ans, "Avg Log Prob: ", avg_log_prob)

        # Update log entry
        log_entry = {
            "Aggregate loss": loss.item() * gradient_accumulation_steps,
            "Batch Index": batch_index,
            "Prev Observation": f"Question: {q}",
            "Action": f"StepByStep: {reasoning_text_first}",
            "Observation": f"Answer: {ans}",
            "Reasoning Contains Answer": str(ans) in reasoning_text_first,
            "Avg Log Prob": avg_log_prob,
            "Baseline Avg Log Prob": baseline_avg_log_prob,
            "Initial Advantage": initial_advantage_value,
            "Advantage": advantage_value,
            "Value Prediction": value_prediction[0].item(),
            "Value Loss": value_loss.item(),
        }

        if use_ppo:
            log_entry.update(
                {
                    "PPO Ratio": ppo_ratio[0].item(),
                    "PPO Clipped Ratio": clipped_ratio[0].item(),
                }
            )

        # Write progress to a file iteratively
        with open(filename, "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for each entry

    # Perform final optimization step for any remaining gradients
    if (batch_index + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
