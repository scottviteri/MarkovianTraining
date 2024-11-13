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
import argparse
from datasets import load_dataset
import re
import os
import glob

# Global variables
previous_normalized_rewards = []
previous_advantages = []


def load_model(model_type="mistral"):
    """Load either Mistral or Llama 3.1 model based on parameter."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Using 8B version
    else:
        raise ValueError("model_type must be either 'mistral' or 'llama'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Added for better device management
    )

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


def calculate_threshold(previous_advantages):
    if len(previous_advantages) > 15:
        return np.mean(previous_advantages) + np.std(previous_advantages)
    return float("inf")


def generate_question_answer_batch(batch_size: int, use_negative: bool = False):
    qa_batch = []
    for _ in range(batch_size):
        if use_negative:
            # Generate numbers between -99 and 99, excluding 0
            numbers = [random.randint(-99, 99) for _ in range(15)]
            numbers = [n for n in numbers if n != 0]  # Remove any zeros

            # Format each number, wrapping negatives in parentheses
            formatted_numbers = []
            for n in numbers:
                if n < 0:
                    formatted_numbers.append(f"({n})")
                else:
                    formatted_numbers.append(str(n))

            question = " + ".join(formatted_numbers)
            answer = str(sum(numbers))
        else:
            # Original positive-only logic
            numbers = [random.randint(1, 99) for _ in range(15)]
            question = " + ".join(map(str, numbers))
            answer = str(sum(numbers))

        qa_batch.append((question, answer))
    return qa_batch


def load_gsm8k_dataset():
    ds = load_dataset("openai/gsm8k", "main")
    questions = ds["train"]["question"]
    answers = list(map(lambda x: x[x.index("####") + 5 :], ds["train"]["answer"]))
    return list(zip(questions, answers))


def generate_question_answer_batches(
    num_batches: int,
    batch_size: int,
    use_gsm8k: bool,
    use_negative: bool = False,
    use_wiki: bool = False,
):
    """Generate batches of Q&A pairs from different sources."""
    if use_wiki:
        # Load Wikipedia dataset
        wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")

        # Create chunks and pairs
        qa_pairs = []
        for article in wiki_dataset:
            text = article["text"]
            chunks = [text[i : i + 200] for i in range(0, len(text), 200)]

            # Create pairs from adjacent chunks
            for i in range(0, len(chunks) - 1, 2):
                qa_pairs.append((chunks[i], chunks[i + 1]))

            if len(qa_pairs) >= num_batches * batch_size:
                break

        # Shuffle and create batches
        random.shuffle(qa_pairs)
        qa_pairs = qa_pairs[: num_batches * batch_size]
        return [
            qa_pairs[i : i + batch_size] for i in range(0, len(qa_pairs), batch_size)
        ]
    elif use_gsm8k:
        gsm8k_data = load_gsm8k_dataset()
        total_samples = len(gsm8k_data)
        if num_batches * batch_size > total_samples:
            print(
                f"Warning: Requested {num_batches * batch_size} samples, but GSM8K dataset only has {total_samples} samples."
            )
            print("Some samples will be repeated.")
            return [random.sample(gsm8k_data, batch_size) for _ in range(num_batches)]
        else:
            shuffled_data = random.sample(gsm8k_data, num_batches * batch_size)
            return [
                shuffled_data[i : i + batch_size]
                for i in range(0, len(shuffled_data), batch_size)
            ]
    else:
        return [
            generate_question_answer_batch(batch_size, use_negative)
            for _ in range(num_batches)
        ]


def get_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


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


def calculate_answer_log_probs(
    model,
    tokenizer,
    device,
    reasoning_tokens,
    answers,
    use_gsm8k,
    model_type,
    debug_index=None,
):
    """
    Calculate the log probabilities of the answers given the reasoning.

    Args:
        model: The language model to use for calculations.
        tokenizer: The tokenizer associated with the model.
        device: The device (CPU or GPU) to perform calculations on.
        reasoning_tokens (torch.Tensor): Tokenized reasoning text, shape [batch_size, seq_len].
        answers (List[str]): A list of answer strings, one for each item in the batch.
        use_gsm8k (bool): Whether to use GSM8K-specific processing.
        model_type (str): The type of model being used ('mistral' or 'llama').
        debug_index (int, optional): If set, enables debug output for this index.

    Returns:
        torch.Tensor: The average log probabilities of the answers, shape [batch_size].
        List[str] or None: Extracted generated answers if use_gsm8k is True, otherwise None.
    """
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
    # if llama, start at last index of colon (25), plus 1
    answer_start_positions = [
        (
            (input_ids == 28747) | (input_ids == 28705) | (input_ids == 29871)
            if model_type == "mistral"
            else (input_ids == 25)
        )
        .nonzero(as_tuple=True)[0][-1]
        .item()
        + 1
        for input_ids in tokenized_full_prompts.input_ids
    ]
    # if len(answers) == 1:
    assert (
        tokenizer.decode(
            tokenized_full_prompts.input_ids[0][answer_start_positions[0] :]
        ).strip()
        == answers[0].strip()
    )

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

    if debug_index is not None:
        # Debug output for token-by-token probabilities
        tokens = [
            tokenizer.decode([x])
            for x in tokenized_full_prompts.input_ids[
                0, answer_start_positions[0] :
            ].tolist()
        ]
        probs = answer_log_probs[0].tolist()
        print("\nToken-by-token log probabilities:")
        print([(tokens[i], probs[i]) for i in range(len(tokens))])

    avg_log_probs = list(map(lambda x: x.mean(), answer_log_probs))
    # print("Log Probs:", answer_log_probs[0])
    return torch.stack(avg_log_probs), extracted_generated_answers


# def calculate_ppo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2):
#    ratio = torch.exp(current_log_probs - old_log_probs)
#    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
#    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
#    return loss.mean()


def exponential_weighted_average(values, r):
    weights = np.array([r ** (len(values) - i) for i in range(len(values))])
    weights = weights / np.sum(weights)
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
    hyperparameters,
    use_gsm8k,
    model_type,
    debug_index=None,
):
    global previous_normalized_rewards, previous_advantages

    use_ei = hyperparameters["use_ei"]
    r = hyperparameters.get("r", None)
    normalize_loss = hyperparameters.get("normalize_loss", True)

    reasoning_tokens = outputs[:, tokenized_inputs.input_ids.shape[1] :]

    log_prob_ans_given_reasoning, extracted_generated_answers = (
        calculate_answer_log_probs(
            frozen_model,
            tokenizer,
            device,
            reasoning_tokens,
            answers,
            use_gsm8k,
            model_type,
            debug_index,
        )
    )

    if normalize_loss:
        baseline_reasoning_tokens = baseline_outputs[
            :, tokenized_inputs.input_ids.shape[1] :
        ]
        log_prob_ans_given_default_reasoning = calculate_answer_log_probs(
            frozen_model,
            tokenizer,
            device,
            baseline_reasoning_tokens,
            answers,
            use_gsm8k,
            model_type,
        )[0]
        normalized_reward = (
            log_prob_ans_given_reasoning - log_prob_ans_given_default_reasoning
        )
    else:
        normalized_reward = log_prob_ans_given_reasoning
        log_prob_ans_given_default_reasoning = None

    if len(previous_normalized_rewards) > 0 and r is not None:
        avg_previous_reward = exponential_weighted_average(
            previous_normalized_rewards, r
        )
        advantage = normalized_reward - avg_previous_reward
    else:
        advantage = normalized_reward

    previous_normalized_rewards.extend(normalized_reward.detach().float().cpu().numpy())
    previous_advantages.extend(advantage.detach().float().cpu().numpy())

    if len(previous_normalized_rewards) > 1000:
        previous_normalized_rewards = previous_normalized_rewards[-1000:]
        previous_advantages = previous_advantages[-1000:]

    if use_ei:
        threshold = calculate_threshold(previous_advantages)
        mask = (advantage > threshold).float()
        if not (hyperparameters["use_pg"] or hyperparameters["use_ppo"]):
            # If only use_ei is enabled, set advantage to 1 where mask is 1
            advantage = mask
        else:
            # When use_ei is combined with use_pg or use_ppo, zero out advantages below threshold
            advantage = advantage * mask

    return (
        advantage,
        reasoning_tokens,
        log_prob_ans_given_reasoning,
        log_prob_ans_given_default_reasoning,
        normalized_reward,
        extracted_generated_answers,
    )


def calculate_losses(
    unfrozen_avg_log_probs_reasoning_given_question,
    frozen_avg_log_probs_reasoning_given_question,
    advantage,
    hyperparameters,
):
    use_ei = hyperparameters["use_ei"]
    use_pg = hyperparameters["use_pg"]
    use_ppo = hyperparameters["use_ppo"]
    ppo_epsilon = hyperparameters.get("ppo_epsilon", 0.2)

    if use_ppo:
        ppo_ratio = torch.exp(
            unfrozen_avg_log_probs_reasoning_given_question
            - frozen_avg_log_probs_reasoning_given_question
        )
        clipped_ratio = torch.clamp(ppo_ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
        policy_loss = -torch.min(ppo_ratio * advantage, clipped_ratio * advantage)
        policy_loss = policy_loss.mean()
    elif use_ei or use_pg:
        # Standard Policy Gradient loss
        policy_loss = (
            -unfrozen_avg_log_probs_reasoning_given_question * advantage.detach()
        ).mean()
        ppo_ratio = None
        clipped_ratio = None
    else:
        raise ValueError("At least one of use_pg, use_ppo, or use_ei must be True.")

    total_loss = policy_loss

    return total_loss, policy_loss, ppo_ratio, clipped_ratio


def get_latest_checkpoint_and_log(dataset_type):
    model_save_path = f"SavedModels/PolicyGradientNormalized_{dataset_type}_latest.pt"
    log_pattern = f"src/AnalyzeResults/PolicyGradientNormalized_{dataset_type}_*.log"
    log_files = sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)

    if not os.path.exists(model_save_path) or not log_files:
        return None, None

    return model_save_path, log_files[0]


def load_training_state(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    last_line = json.loads(lines[-1])
    last_batch_index = last_line["Batch Index"]

    hyperparameters = json.loads(lines[0])

    return last_batch_index, hyperparameters


def load_previous_rewards_and_advantages(log_file):
    previous_normalized_rewards = []
    previous_advantages = []
    with open(log_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "Normalized Reward" in entry and "Advantage" in entry:
                previous_normalized_rewards.append(entry["Normalized Reward"])
                previous_advantages.append(entry["Advantage"])
    return previous_normalized_rewards, previous_advantages


def debug_single_datapoint(
    model, tokenizer, device, qa_pair, use_gsm8k, model_type, use_wiki, hyperparameters
):
    question, answer = qa_pair

    # Update prompts based on dataset type
    if use_wiki:
        prompt = (
            f"<|start_header_id|>user<|end_header_id|>Previous: {question}\nProvide a summary that captures key elements suggesting what comes next:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nSummary:"
            if model_type == "llama"
            else f"{question}\nProvide a summary that captures key elements suggesting what comes next: \nSummary:"
        )
    else:
        prompt = (
            f"<|start_header_id|>user<|end_header_id|>Produce minimal text which will help you answer the following question:  Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nReasoning:"
            if model_type == "llama"
            else f"{question}\nProduce minimal text which will help you answer the following question:  Question: {question} \nReasoning:"
        )

    tokenized_inputs = tokenizer(
        prompt,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=hyperparameters["cot_length"],
            min_new_tokens=hyperparameters["cot_length"],
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    reasoning_tokens = outputs[:, tokenized_inputs.input_ids.shape[1] :]
    reasoning_text = tokenizer.decode(reasoning_tokens[0], skip_special_tokens=True)

    log_prob_ans_given_reasoning, extracted_generated_answers = (
        calculate_answer_log_probs(
            model,
            tokenizer,
            device,
            reasoning_tokens,
            [answer],
            use_gsm8k,
            model_type,
            debug_index=0,
        )
    )

    true_answer = extract_answer(answer)
    generated_answer = (
        extracted_generated_answers[0] if extracted_generated_answers else None
    )
    is_correct = (
        generated_answer == true_answer if generated_answer is not None else False
    )

    print(f"Question: {question}")
    print(f"True Answer: {true_answer}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Is Correct: {is_correct}")
    print(f"Log Probability: {log_prob_ans_given_reasoning[0].item()}")
    print(f"Generated Reasoning: {reasoning_text}")


def tensor_to_python(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    elif isinstance(value, np.float32) or isinstance(value, np.float64):
        return float(value)
    elif isinstance(value, np.int32) or isinstance(value, np.int64):
        return int(value)
    return value


def train(
    use_gsm8k: bool,
    resume: bool,
    use_ei: bool,
    use_ppo: bool,
    use_pg: bool,
    model_type: str,
    use_negative: bool,
    use_wiki: bool = False,
    debug_index: int = None,
):
    global previous_normalized_rewards, previous_advantages

    # Update dataset type string
    dataset_type = "Wikipedia" if use_wiki else "GSM8K" if use_gsm8k else "Arithmetic"

    if resume:
        model_save_path, log_file = get_latest_checkpoint_and_log(dataset_type)
        if model_save_path is None or log_file is None:
            print("No checkpoint or log file found. Starting from scratch.")
            resume = False
        else:
            print(f"Resuming from checkpoint: {model_save_path}")
            print(f"Using log file: {log_file}")
            last_batch_index, hyperparameters = load_training_state(log_file)
            start_batch = last_batch_index + 1

            # Initialize previous_normalized_rewards and previous_advantages from the log file
            previous_normalized_rewards, previous_advantages = (
                load_previous_rewards_and_advantages(log_file)
            )
            print(
                f"Loaded {len(previous_normalized_rewards)} previous rewards and advantages"
            )

    if not resume:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"src/AnalyzeResults/PolicyGradientNormalized_{dataset_type}_{timestamp}.log"
        model_save_path = (
            f"SavedModels/PolicyGradientNormalized_{dataset_type}_latest.pt"
        )
        start_batch = 0

        # Set cot_length based on model type and dataset
        if model_type == "mistral":
            cot_length = 80 if use_gsm8k else 400  # 400 for arithmetic, 80 for GSM8K
        else:  # llama
            cot_length = 60 if use_gsm8k else 150  # 150 for arithmetic, 60 for GSM8K

        # Initialize hyperparameters
        hyperparameters = {
            "model_learning_rate": 0.0001,
            "batch_size": 10 if use_gsm8k else 6,
            "gradient_accumulation_steps": 32 if use_gsm8k else 8,
            "num_batches": 10000,
            "normalize_loss": True,
            "use_ppo": use_ppo,
            "ppo_epsilon": 0.2 if use_ppo else None,
            "r": 1.0 if use_ppo else None,
            "use_ei": use_ei,
            "use_pg": use_pg,
            "use_ppo": use_ppo,
            "cot_length": cot_length,
        }

    model, frozen_model, tokenizer, device = load_model(model_type)

    if resume:
        model.load_state_dict(torch.load(model_save_path))

    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=hyperparameters["model_learning_rate"]
    )

    batch_size = hyperparameters["batch_size"]
    normalize_loss = hyperparameters["normalize_loss"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]

    num_batches = hyperparameters["num_batches"]
    qa_batches = list(
        generate_question_answer_batches(
            num_batches=num_batches,
            batch_size=batch_size,
            use_gsm8k=use_gsm8k,
            use_negative=use_negative,
            use_wiki=use_wiki,
        )
    )

    if not resume:
        with open(log_file, "w") as f:
            json.dump(hyperparameters, f)
            f.write("\n")

    model_optimizer.zero_grad()

    for batch_index, qa_batch in enumerate(qa_batches[start_batch:], start=start_batch):
        questions, answers = zip(*qa_batch)
        print("\n")
        print("Batch Index:", batch_index)

        # Update prompts based on dataset type
        if use_wiki:
            prompts = [
                (
                    f"<|start_header_id|>user<|end_header_id|>Previous: {q}\nProvide a summary that captures key elements suggesting what comes next:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nSummary:"
                    if model_type == "llama"
                    else f"{q}\nProvide a summary that captures key elements suggesting what comes next: \nSummary:"
                )
                for q in questions
            ]
        else:
            prompts = [
                (
                    f"<|start_header_id|>user<|end_header_id|>Produce minimal text which will help you answer the following question:  Question: {q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nReasoning:"
                    if model_type == "llama"
                    else f"{q}\nProduce minimal text which will help you answer the following question:  Question: {q} \nReasoning:"
                )
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
                max_new_tokens=hyperparameters["cot_length"],
                min_new_tokens=hyperparameters["cot_length"],
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            baseline_outputs = frozen_model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=hyperparameters["cot_length"],
                min_new_tokens=hyperparameters["cot_length"],
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
            extracted_generated_answers,
        ) = calculate_advantages(
            model,
            frozen_model,
            tokenizer,
            device,
            tokenized_inputs,
            outputs,
            baseline_outputs,
            answers,
            hyperparameters,
            use_gsm8k,
            model_type,
            debug_index,
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

        reasoning_tokens = outputs[:, tokenized_inputs.input_ids.shape[1] : -1]
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
            hyperparameters,
        )
        loss = total_loss / gradient_accumulation_steps

        loss.backward()

        grad_norm = get_grad_norm(model.parameters())
        clip_grad_norm_(model.parameters(), 1.0)

        if (batch_index + 1) % gradient_accumulation_steps == 0:
            model_optimizer.step()
            model_optimizer.zero_grad()

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
        print("Question:", q)
        print("Reasoning:", reasoning_text_first)
        print("Answer:", ans, "Avg Log Prob:", avg_log_prob)
        if extracted_generated_answers is not None:
            print("Generated Answer:", extracted_generated_answers[0])

        if previous_advantages:
            mean_prev_advantage = np.mean(previous_advantages)
            std_prev_advantage = np.std(previous_advantages)
        else:
            mean_prev_advantage = None
            std_prev_advantage = None

        log_entry = {
            k: tensor_to_python(v)
            for k, v in {
                "Aggregate loss": loss.item() * gradient_accumulation_steps,
                "Batch Index": batch_index,
                "Prev Observation": f"Question: {q}",
                "Action": f"Reasoning: {reasoning_text_first}",
                "Observation": f"Answer: {ans}",
                "Reasoning Contains Answer": str(ans) in reasoning_text_first,
                "Avg Log Prob": avg_log_prob,
                "Normalized Reward": normalized_reward[0].item(),
                "Advantage": advantage_value,
                "Policy Loss": policy_loss.item(),
                "Total Loss": total_loss.item(),
                "Grad Norm": grad_norm,
                "Use EI": hyperparameters["use_ei"],
                "Mean Previous Advantage": mean_prev_advantage,
                "Std Previous Advantage": std_prev_advantage,
                "EI Threshold": (
                    calculate_threshold(previous_advantages)
                    if hyperparameters["use_ei"]
                    else None
                ),
            }.items()
        }

        if normalize_loss and baseline_avg_log_prob is not None:
            log_entry["Baseline Avg Log Prob"] = tensor_to_python(baseline_avg_log_prob)

        if hyperparameters["use_ppo"] and ppo_ratio is not None:
            log_entry.update(
                {
                    "PPO Ratio": tensor_to_python(ppo_ratio[0]),
                    "PPO Clipped Ratio": tensor_to_python(clipped_ratio[0]),
                }
            )

        if use_gsm8k and extracted_generated_answers is not None:
            true_answers = [extract_answer(answer) for answer in answers]
            correct_count = sum(
                gen_ans == true_ans
                for gen_ans, true_ans in zip(extracted_generated_answers, true_answers)
            )
            fraction_correct = correct_count / len(answers)

            true_answer = true_answers[0]
            log_entry.update(
                {
                    "Generated Answer": extracted_generated_answers[0],
                    "True Answer": true_answer,
                    "Is Correct": extracted_generated_answers[0] == true_answer,
                    "Fraction Correct": fraction_correct,
                }
            )

        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

        if batch_index % 200 == 0 and batch_index > 0:
            print(f"Saving model weights at batch {batch_index}")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # Create a new filename that includes the batch index
            model_save_path_with_batch = (
                f"SavedModels/PolicyGradientNormalized_{dataset_type}_latest.pt"
            )
            # Save the model with the batch index in the filename
            torch.save(model.state_dict(), model_save_path_with_batch)


def main(
    use_gsm8k: bool,
    resume: bool,
    debug_index: int = None,
    use_ei: bool = False,
    use_ppo: bool = False,
    use_pg: bool = False,
    model_type: str = "mistral",
    use_negative: bool = False,
    use_wiki: bool = False,
):
    # Update dataset type
    dataset_type = "Wikipedia" if use_wiki else "GSM8K" if use_gsm8k else "Arithmetic"

    if debug_index is not None:
        model_save_path = (
            f"SavedModels/PolicyGradientNormalized_{dataset_type}_latest.pt"
        )
        if not os.path.exists(model_save_path):
            print(f"Error: Model file not found at {model_save_path}")
            return

        model, frozen_model, tokenizer, device = load_model(model_type)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        # Set cot_length based on model type and dataset
        if model_type == "mistral":
            cot_length = (
                80 if use_gsm8k else 400
            )  # 400 for arithmetic/wiki, 80 for GSM8K
        else:  # llama
            cot_length = (
                60 if use_gsm8k else 150
            )  # 150 for arithmetic/wiki, 60 for GSM8K

        hyperparameters = {
            "cot_length": cot_length,
            # Add other hyperparameters as needed
        }

        if use_wiki:
            # Load Wikipedia dataset
            wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
            # Get a single article and create a QA pair
            article = wiki_dataset[debug_index]["text"]
            chunks = [article[i : i + 200] for i in range(0, len(article), 200)]
            if len(chunks) >= 2:
                qa_pair = (chunks[0], chunks[1])
            else:
                print("Error: Article too short to create QA pair")
                return
        elif use_gsm8k:
            gsm8k_data = load_gsm8k_dataset()
            if debug_index >= len(gsm8k_data):
                print(
                    f"Error: Debug index {debug_index} is out of range. Max index is {len(gsm8k_data) - 1}"
                )
                return
            qa_pair = gsm8k_data[debug_index]
        else:
            qa_pair = generate_question_answer_batch(1, use_negative)[0]

        debug_single_datapoint(
            model,
            tokenizer,
            device,
            qa_pair,
            use_gsm8k,
            model_type,
            use_wiki,
            hyperparameters,
        )
        return

    train(
        use_gsm8k,
        resume,
        use_ei,
        use_ppo,
        use_pg,
        model_type,
        use_negative,
        use_wiki,
        debug_index,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model on arithmetic, GSM8K, or Wikipedia dataset."
    )
    parser.add_argument(
        "--use_gsm8k",
        action="store_true",
        default=False,
        help="Use GSM8K dataset instead of arithmetic questions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the last checkpoint",
    )
    parser.add_argument(
        "--debug_index",
        type=int,
        default=None,
        help="Index of the datapoint to debug (GSM8K only, random for arithmetic)",
    )
    parser.add_argument(
        "--use_ei",
        action="store_true",
        default=False,
        help="Use Expert Iteration",
    )
    parser.add_argument(
        "--use_ppo",
        action="store_true",
        default=False,
        help="Use Proximal Policy Optimization",
    )
    parser.add_argument(
        "--use_pg",
        action="store_true",
        default=False,
        help="Use Policy Gradient",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mistral", "llama"],
        default="mistral",
        help="Choose between Mistral and Llama 3.1 models",
    )
    parser.add_argument(
        "--use_negative",
        action="store_true",
        default=False,
        help="Include negative numbers in arithmetic questions",
    )
    parser.add_argument(
        "--use_wiki",
        action="store_true",
        default=False,
        help="Use Wikipedia dataset for training",
    )
    args = parser.parse_args()

    if not (args.use_ei or args.use_pg or args.use_ppo):
        raise ValueError(
            "At least one of --use_ei, --use_pg, or --use_ppo must be specified."
        )

    main(
        use_gsm8k=args.use_gsm8k,
        resume=args.resume,
        debug_index=args.debug_index,
        use_ei=args.use_ei,
        use_ppo=args.use_ppo,
        use_pg=args.use_pg,
        model_type=args.model_type,
        use_negative=args.use_negative,
        use_wiki=args.use_wiki,
    )
