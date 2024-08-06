import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import bitsandbytes
import random
import numpy as np
import json
import copy
from torch.nn.utils import clip_grad_norm_

# Initialize the global variable at the module level
previous_normalized_rewards = []


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

    frozen_model = copy.deepcopy(model)
    for param in frozen_model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frozen_model.to(device)

    return model, frozen_model, tokenizer, device


def generate_question_answer_batch(batch_size: int):
    qa_batch = []
    for _ in range(batch_size):
        numbers = [random.randint(1, 99) for _ in range(15)]
        qa_batch.append((" + ".join(map(str, numbers)), str(sum(numbers))))
    return qa_batch


def generate_question_answer_batches(num_batches: int, batch_size: int):
    return [generate_question_answer_batch(batch_size) for _ in range(num_batches)]


def get_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def calculate_answer_log_probs(model, tokenizer, device, reasoning_tokens, answers):
    reasoning_text = tokenizer.batch_decode(reasoning_tokens, skip_special_tokens=True)

    full_prompts = [
        f"Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.\nStepByStep: {r} \nAnswer: {a}"
        for r, a in zip(reasoning_text, answers)
    ]

    tokenized_full_prompts = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Find the position of the last space before the answer
    # 28705 is space token, leading to the final space before the answer (which doesn't contain spaces)
    answer_start_positions = [
        (input_ids == 28705).nonzero(as_tuple=True)[0][-1].item() + 1
        for input_ids in tokenized_full_prompts.input_ids
    ]

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_full_prompts.input_ids,
            attention_mask=tokenized_full_prompts.attention_mask,
        )
        logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    answer_log_probs = [
        log_probs[i, start - 1 : -1]
        .gather(1, tokenized_full_prompts.input_ids[i, start:].unsqueeze(-1))
        .squeeze(-1)
        for i, start in enumerate(answer_start_positions)
    ]

    avg_log_probs = []
    for i, (mask, probs, start_idx, answer) in enumerate(
        zip(
            tokenized_full_prompts.attention_mask,
            answer_log_probs,
            answer_start_positions,
            answers,
        )
    ):
        answer_length = len(
            str(answer)
        )  # Convert to string in case answer is not already a string
        actual_tokens = mask[start_idx:].sum().item()
        assert (
            actual_tokens == answer_length
        ), f"Mismatch in answer length for index {i}. Expected {answer_length}, got {actual_tokens}"
        avg_log_prob = (probs * mask[start_idx:]).sum() / (actual_tokens + 1e-8)
        avg_log_probs.append(avg_log_prob)

    return torch.stack(avg_log_probs)


def calculate_ppo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(current_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()


def exponential_weighted_average(values, r):
    weights = np.array([r**i for i in range(len(values))])
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    return np.sum(weights * np.array(values))


def calculate_advantages(
    model,
    frozen_model,
    tokenizer,
    device,
    tokenized_inputs,
    outputs,
    baseline_outputs,
    answers,
    r,
    normalize_loss,
):
    global previous_normalized_rewards

    reasoning_tokens = outputs[:, tokenized_inputs.input_ids.shape[1] :]

    log_prob_ans_given_reasoning = calculate_answer_log_probs(
        frozen_model, tokenizer, device, reasoning_tokens, answers
    )

    if normalize_loss:
        baseline_reasoning_tokens = baseline_outputs[
            :, tokenized_inputs.input_ids.shape[1] :
        ]
        log_prob_ans_given_default_reasoning = calculate_answer_log_probs(
            frozen_model, tokenizer, device, baseline_reasoning_tokens, answers
        )
        normalized_reward = (
            log_prob_ans_given_reasoning - log_prob_ans_given_default_reasoning
        )
    else:
        normalized_reward = log_prob_ans_given_reasoning
        log_prob_ans_given_default_reasoning = None

    # Calculate advantage using exponentially weighted average of previous normalized rewards
    if len(previous_normalized_rewards) > 0:
        avg_previous_reward = exponential_weighted_average(
            previous_normalized_rewards, r
        )
        advantage = normalized_reward - avg_previous_reward
    else:
        advantage = normalized_reward

    # Update previous_normalized_rewards
    previous_normalized_rewards.extend(normalized_reward.detach().cpu().numpy())

    # Keep only the last 1000 rewards to limit memory usage
    if len(previous_normalized_rewards) > 1000:
        previous_normalized_rewards = previous_normalized_rewards[-1000:]

    return (
        advantage,
        reasoning_tokens,
        log_prob_ans_given_reasoning,
        log_prob_ans_given_default_reasoning,
        normalized_reward,
    )


def calculate_losses(
    unfrozen_avg_log_probs_reasoning_given_question,
    frozen_avg_log_probs_reasoning_given_question,
    advantage,
    use_ppo,
    ppo_epsilon,
):
    if use_ppo:
        ppo_ratio = torch.exp(
            unfrozen_avg_log_probs_reasoning_given_question
            - frozen_avg_log_probs_reasoning_given_question
        )
        clipped_ratio = torch.clamp(ppo_ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
        policy_loss = calculate_ppo_loss(
            unfrozen_avg_log_probs_reasoning_given_question,
            frozen_avg_log_probs_reasoning_given_question,
            advantage,
            epsilon=ppo_epsilon,
        )
    else:
        policy_loss = (
            unfrozen_avg_log_probs_reasoning_given_question * -advantage.detach()
        ).mean()
        ppo_ratio = None
        clipped_ratio = None

    total_loss = policy_loss

    return total_loss, policy_loss, ppo_ratio, clipped_ratio


def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"src/AnalyzeResults/PolicyGradientNormalized_{timestamp}.log"

    model, frozen_model, tokenizer, device = load_mistral_model()
    model_learning_rate = 1e-4
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=model_learning_rate
    )

    batch_size = 6
    normalize_loss = True 
    gradient_accumulation_steps = 8
    use_ppo = True 
    ppo_epsilon = 0.2
    r = 0  # Set the ratio for exponentially weighted average (adjust as needed)
    clip_grad_norm = True 

    num_batches = 10000
    qa_batches = list(
        generate_question_answer_batches(num_batches=num_batches, batch_size=batch_size)
    )

    with open(filename, "w") as log_file:
        hyperparameters = {
            "model_learning_rate": model_learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "num_batches": num_batches,
            "use_ppo": use_ppo,
            "ppo_epsilon": ppo_epsilon,
            "normalize_loss": normalize_loss,
        }
        json.dump(hyperparameters, log_file)
        log_file.write("\n")

    model_optimizer.zero_grad()

    for batch_index, qa_batch in enumerate(qa_batches):
        questions, answers = zip(*qa_batch)

        prompts = [
            f"Work through the following question step by step, concisely decomposing problems into subproblems.\nQuestion: {q}\nStepByStep:"
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
            baseline_outputs = frozen_model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=400,
                min_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        (
            advantage,
            reasoning_tokens,
            log_prob_ans_given_reasoning,
            log_prob_ans_given_default_reasoning,
            normalized_reward,
        ) = calculate_advantages(
            model,
            frozen_model,
            tokenizer,
            device,
            tokenized_inputs,
            outputs,
            baseline_outputs,
            answers,
            r,
            normalize_loss,
        )

        full_attention_mask = torch.cat(
            [tokenized_inputs.attention_mask, torch.ones_like(reasoning_tokens)], dim=1
        )
        unfrozen_outputs = model(input_ids=outputs, attention_mask=full_attention_mask)
        frozen_outputs = frozen_model(
            input_ids=outputs, attention_mask=full_attention_mask
        )
        unfrozen_logits = unfrozen_outputs.logits[
            :, tokenized_inputs.input_ids.shape[1] - 1 : -1, :
        ]
        frozen_logits = frozen_outputs.logits[
            :, tokenized_inputs.input_ids.shape[1] - 1 : -1, :
        ]

        unfrozen_log_probs = torch.nn.functional.log_softmax(unfrozen_logits, dim=-1)
        frozen_log_probs = torch.nn.functional.log_softmax(frozen_logits, dim=-1)

        unfrozen_token_log_probs = unfrozen_log_probs.gather(
            2, reasoning_tokens.unsqueeze(-1)
        ).squeeze(-1)
        frozen_token_log_probs = frozen_log_probs.gather(
            2, reasoning_tokens.unsqueeze(-1)
        ).squeeze(-1)

        unfrozen_avg_log_probs_reasoning_given_question = unfrozen_token_log_probs.mean(
            dim=1
        )
        frozen_avg_log_probs_reasoning_given_question = frozen_token_log_probs.mean(
            dim=1
        )

        total_loss, policy_loss, ppo_ratio, clipped_ratio = calculate_losses(
            unfrozen_avg_log_probs_reasoning_given_question,
            frozen_avg_log_probs_reasoning_given_question,
            advantage,
            use_ppo,
            ppo_epsilon,
        )

        loss = total_loss / gradient_accumulation_steps
        loss.backward()

        # Usage
        grad_norm = get_grad_norm(model.parameters())
        print(f"Current gradient norm: {grad_norm}")
        # Clip gradients to prevent extreme updates
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), 1.0)

        if (batch_index + 1) % gradient_accumulation_steps == 0:
            model_optimizer.step()
            model_optimizer.zero_grad()

        # Logging
        q = questions[0]
        ans = answers[0]
        reasoning_text_first = tokenizer.decode(
            reasoning_tokens[0], skip_special_tokens=True
        )
        avg_log_prob = log_prob_ans_given_reasoning[0].item()
        baseline_avg_log_prob = (
            log_prob_ans_given_default_reasoning[0].item()
            if log_prob_ans_given_default_reasoning is not None
            else None
        )
        advantage_value = advantage[0].item()
        print(reasoning_text_first)
        print("Ans: ", ans, "Avg Log Prob: ", avg_log_prob)

        log_entry = {
            "Aggregate loss": loss.item() * gradient_accumulation_steps,
            "Batch Index": batch_index,
            "Prev Observation": f"Question: {q}",
            "Action": f"StepByStep: {reasoning_text_first}",
            "Observation": f"Answer: {ans}",
            "Reasoning Contains Answer": str(ans) in reasoning_text_first,
            "Avg Log Prob": avg_log_prob,
            "Normalized Reward": normalized_reward[0].item(),
            "Advantage": advantage_value,
            "Policy Loss": policy_loss.item(),
            "Total Loss": total_loss.item(),
            "Grad Norm": grad_norm,
        }

        if normalize_loss and baseline_avg_log_prob is not None:
            log_entry["Baseline Avg Log Prob"] = baseline_avg_log_prob

        if use_ppo and ppo_ratio is not None:
            log_entry.update(
                {
                    "PPO Ratio": ppo_ratio[0].item(),
                    "PPO Clipped Ratio": clipped_ratio[0].item(),
                }
            )

        with open(filename, "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")


if __name__ == "__main__":
    # Set the normalize_loss parameter manually here
    train()
