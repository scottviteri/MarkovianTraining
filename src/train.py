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
import subprocess

# Global variables
previous_normalized_rewards = []
previous_advantages = []


def find_latest_result(return_log=False):
    """
    Find the most recent result file across all tasks and model types.

    Args:
        return_log (bool): If True, return the log file instead of the model checkpoint

    Returns:
        str: Path to the most recent result file, or None if no result found
    """
    results_dir = "results"

    # Collect all potential result files
    result_files = []

    # Walk through the results directory
    for task_dir in os.listdir(results_dir):
        task_path = os.path.join(results_dir, task_dir)
        if os.path.isdir(task_path):
            for timestamp_dir in os.listdir(task_path):
                full_timestamp_path = os.path.join(task_path, timestamp_dir)

                # Check for model checkpoint or log file
                if return_log:
                    log_path = os.path.join(full_timestamp_path, "log.jsonl")
                    file_path = log_path
                else:
                    model_path = os.path.join(full_timestamp_path, "model.pt")
                    file_path = model_path

                if os.path.exists(file_path):
                    result_files.append(
                        (
                            os.path.getmtime(full_timestamp_path),
                            file_path,
                        )
                    )

    # Sort by timestamp, most recent first
    if result_files:
        return sorted(result_files, key=lambda x: x[0], reverse=True)[0][1]

    return None


# Add at the top of the file with other imports
class Colors:
    """ANSI color codes"""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_batch_delimiter():
    """Print a visually distinct line between batches"""
    print("\n" + "=" * 80)


def colored_print(
    label: str, text: str, color: str = Colors.BLUE, inline: bool = False
):
    """Print text with colored label, optionally on same line."""
    if inline:
        print(f"\n{color}{label}{Colors.END} {text}", end="")
    else:
        print(f"\n{color}{label}{Colors.END}")
        print(repr(text))


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


def generate_arithmetic_pairs(task_type: str, num_examples: int = 1000):
    """Lazily generate arithmetic QA pairs with shuffling within chunks."""
    qa_pairs = []
    for _ in range(num_examples):
        if task_type == "arithmetic-negative":
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
        else:  # regular arithmetic
            numbers = [random.randint(1, 99) for _ in range(15)]
            question = " + ".join(map(str, numbers))
            answer = str(sum(numbers))
        qa_pairs.append((question, answer))

    random.shuffle(qa_pairs)
    return qa_pairs


def load_gsm8k_dataset(chunk_size: int = 1000):
    """Lazily load GSM8K dataset in chunks."""
    ds = load_dataset("openai/gsm8k", "main")
    questions = ds["train"]["question"]
    answers = list(map(lambda x: x[x.index("####") + 5 :], ds["train"]["answer"]))
    qa_pairs = list(zip(questions, answers))

    for i in range(0, len(qa_pairs), chunk_size):
        chunk = qa_pairs[i : i + chunk_size]
        random.shuffle(chunk)
        yield from chunk


def get_model_specific_tokens(model_type):
    """
    Return model-specific tokens for prompt construction.

    !!! DO NOT MODIFY THIS FUNCTION!!!
    These tokens are specific to each model's training format and changing them will break the model.
    """
    if model_type == "mistral":
        return {
            "inst_start": "[INST]",
            "inst_end": "[/INST]",
        }
    else:  # llama
        return {
            "inst_start": "",
            "inst_end": "",
        }


def construct_prompts(task_type, question, model_type, hyperparameters, reasoning=None):
    """
    Construct both partial and full prompts.
    Returns (partial_prompt, full_prompt). full_prompt will be None if reasoning or answer is None.

    partial_prompt: Used for generating reasoning, includes all formatting except answer
    full_prompt: Used for calculating log probs, includes everything including answer

    Args:
        task_type: Type of task ('arithmetic', 'wiki_compression', 'wiki_continuation', 'gsm8k')
        question: The input question or text
        model_type: Type of model ('mistral' or 'llama')
        hyperparameters: Dictionary containing model hyperparameters
        reasoning: Optional reasoning text to include
    """

    tokens = get_model_specific_tokens(model_type)

    # Construct base prompt
    if task_type == "wiki_compression":
        base_prompt = (
            f"The following text is the {hyperparameters['target_length']} characters, which you will need to reconstruct."
            f"You can write {hyperparameters['cot_length']} tokens as your memory during the reconstruction. "
            f"Feel free to be creative!\n\nFull Text:"
        )
        prompt_type = "Compression:"
    elif task_type == "wiki_continuation":
        base_prompt = (
            f"Given this opening text from an article, write whatever "
            f"{hyperparameters['cot_length']} tokens you suspect might help you "
            f"predict the next {hyperparameters['target_length']} tokens. Be creative!\n\nOpening text:"
        )
        prompt_type = "Helpful Text:"
    else:  # arithmetic/gsm8k
        base_prompt = f"You will be given an arithmetic problem, which you have {hyperparameters['cot_length']} tokens to work through step-by-step. Question:"
        prompt_type = "Reasoning:"

    # Construct initial prompt with model-specific tokens
    if reasoning is None:
        return f"{tokens['inst_start']} {base_prompt} {question} {tokens['inst_end']}\n{prompt_type}"

    base_with_type = f"{tokens['inst_start']} {base_prompt} <Redacted> {tokens['inst_end']}\n{prompt_type}"

    # Add model-specific answer header to partial prompt
    if model_type == "mistral":
        partial_prompt = base_with_type + reasoning + " Answer: "
    else:  # llama -- maybe the space afterwards is important
        partial_prompt = base_with_type + reasoning + f" Answer: "
    return partial_prompt


def get_text_with_token_length(
    text: str, desired_tokens: int, tokenizer
) -> tuple[str, int]:
    """
    Binary search to find text that tokenizes to desired number of tokens.
    Returns (text_chunk, actual_token_count) or (None, 0) if text is too short.
    """
    # Initial check
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(tokens) < desired_tokens:
        return None, 0

    # Binary search for correct length
    left, right = 1, len(text)
    best_text = None
    best_count = 0

    while left <= right:
        mid = (left + right) // 2
        chunk = text[:mid]
        tokens = tokenizer(chunk, return_tensors="pt").input_ids[0]
        token_count = len(tokens)

        if token_count == desired_tokens:
            return chunk, token_count
        elif token_count < desired_tokens:
            left = mid + 1
            # Save this as best if it's closer than previous best
            if abs(token_count - desired_tokens) < abs(best_count - desired_tokens):
                best_text = chunk
                best_count = token_count
        else:
            right = mid - 1
            # Save this as best if it's closer than previous best
            if abs(token_count - desired_tokens) < abs(best_count - desired_tokens):
                best_text = chunk
                best_count = token_count

    return best_text, best_count


def generate_question_answer_batches(
    num_batches: int,
    batch_size: int,
    task_type: str,
    tokenizer,
    hyperparameters: dict = None,
):
    """Generate batches of Q&A pairs lazily."""
    total_examples_needed = num_batches * batch_size
    qa_pairs = []
    chunk_size = 100  # Reduced from 1000 to 100

    if task_type in ["wiki_compression", "wiki_continuation"]:
        wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
        article_idx = 0

        while len(qa_pairs) < chunk_size:
            if article_idx >= len(wiki_dataset):
                print(
                    f"Warning: Reached end of Wikipedia dataset after {len(qa_pairs)} examples"
                )
                break

            article = wiki_dataset[article_idx]["text"]
            article_idx += 1

            if task_type == "wiki_compression":
                chunk, token_count = get_text_with_token_length(
                    article, hyperparameters["target_length"], tokenizer
                )
                if chunk is not None:
                    qa_pairs.append((chunk, chunk))
            else:  # wiki_continuation
                q_chunk, q_count = get_text_with_token_length(
                    article, hyperparameters["question_length"], tokenizer
                )
                if q_chunk is None:
                    continue

                # Skip if question contains "Answer:"
                if "Answer:" in q_chunk:
                    continue

                remaining_text = article[len(q_chunk) :]
                a_chunk, a_count = get_text_with_token_length(
                    remaining_text, hyperparameters["target_length"], tokenizer
                )
                if a_chunk is not None:
                    # Skip if answer contains "Answer:"
                    if "Answer:" in a_chunk:
                        continue
                    qa_pairs.append((q_chunk, a_chunk))

            # If we have enough pairs, shuffle and yield batches
            if len(qa_pairs) >= chunk_size:
                random.shuffle(qa_pairs)
                for i in range(0, len(qa_pairs), batch_size):
                    yield qa_pairs[i : i + batch_size]
                qa_pairs = []  # Reset for next chunk

    elif task_type == "gsm8k":
        gsm8k_iter = load_gsm8k_dataset(chunk_size)
        current_chunk = []

        for qa_pair in gsm8k_iter:
            current_chunk.append(qa_pair)
            if len(current_chunk) >= chunk_size:
                random.shuffle(current_chunk)
                for i in range(0, len(current_chunk), batch_size):
                    yield current_chunk[i : i + batch_size]
                current_chunk = []

    else:  # arithmetic tasks
        while True:
            qa_pairs = generate_arithmetic_pairs(task_type, chunk_size)
            for i in range(0, len(qa_pairs), batch_size):
                yield qa_pairs[i : i + batch_size]


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
    questions,
    reasoning_tokens,
    answers,
    task_type,
    model_type,
    hyperparameters,
):
    """Calculate the log probabilities of the answers given the reasoning."""
    reasoning_text = tokenizer.batch_decode(reasoning_tokens, skip_special_tokens=True)
    # Update full prompts based on dataset type
    partial_prompts = [
        construct_prompts(
            task_type=task_type,
            question=q,
            model_type=model_type,
            hyperparameters=hyperparameters,
            reasoning=r,
        )
        for q, r in zip(questions, reasoning_text)
    ]
    full_prompts = [x + y for x, y in zip(partial_prompts, answers)]
    tokenized_full_prompts = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    extracted_generated_answers = None
    if task_type == "gsm8k":
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
        selected_answers = [x.split("\n")[-1] for x in generated_answers]  # check this
        extracted_generated_answers = [extract_answer(ans) for ans in selected_answers]
    # if llama, find last occurrence of [16533, 25] and use 25's index + 1
    answer_start_positions = []
    for input_ids in tokenized_full_prompts.input_ids:
        if model_type == "mistral":
            matching_indices = (
                (input_ids[:-1] == 26307)
                & (
                    (input_ids[1:] == 28747)
                    | (input_ids[1:] == 28705)
                    | (input_ids[1:] == 29871)
                )
            ).nonzero(as_tuple=True)[0]
            pos = matching_indices[-1].item() + 2
        else:  # llama
            # could potentially do
            # tokenizer.encode(" [Answer] ")
            # [128000, 510 =[, 16533=Answer, 60=], 220] <- unnecessary, just remove Answer: from dataset
            matching_indices = (
                ((input_ids[:-1] == 16533) | (input_ids[:-1] == 22559))
                & (input_ids[1:] == 25)
            ).nonzero(as_tuple=True)[0]
            pos = matching_indices[-1].item() + 2  # to account for Answer:
        answer_start_positions.append(pos)

    # Assert that the decoded tokens after each start position match the expected answers
    for i in range(len(answers)):
        decoded_answer = tokenizer.decode(
            tokenized_full_prompts.input_ids[i][answer_start_positions[i] :]
        ).strip()
        expected_answer = answers[i].strip()
        if (
            decoded_answer[:3] != expected_answer[:3]
            or decoded_answer[-3:] != expected_answer[-3:]
        ):
            colored_print("Answer mismatch at index", str(i), Colors.RED)
            # print(f"Decoded:  {decoded_answer}")
            # print(f"Expected: {expected_answer}")
        # assert decoded_answer == expected_answer, f"Answer mismatch at index {i}"

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
    questions,
    answers,
    hyperparameters,
    task_type,
    model_type,
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
            questions,
            reasoning_tokens,
            answers,
            task_type,
            model_type,
            hyperparameters,
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
            questions,
            baseline_reasoning_tokens,
            answers,
            task_type,
            model_type,
            hyperparameters,
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
        # Get number of non-zero advantages
        num_active = torch.sum(advantage != 0).item()
        if num_active > 0:
            policy_loss = (
                policy_loss.sum() / num_active
            )  # Average only over active samples
        else:
            policy_loss = (
                policy_loss.sum() * 0.0
            )  # Return zero loss if no active samples
    elif use_ei or use_pg:
        # Standard Policy Gradient loss
        policy_loss = (
            -unfrozen_avg_log_probs_reasoning_given_question * advantage.detach()
        )
        num_active = torch.sum(advantage != 0).item()
        if num_active > 0:
            policy_loss = policy_loss.sum() / num_active
        else:
            policy_loss = policy_loss.sum() * 0.0
        ppo_ratio = None
        clipped_ratio = None
    else:
        raise ValueError("At least one of use_pg, use_ppo, or use_ei must be True.")

    total_loss = policy_loss

    return total_loss, policy_loss, ppo_ratio, clipped_ratio, num_active


def get_latest_result_and_log(dataset_type):
    results_dir = f"results/{dataset_type}"
    if not os.path.exists(results_dir):
        return None, None

    # Get all subdirectories (timestamps) in the results directory
    results_folders = sorted(
        [
            os.path.join(results_dir, d)
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ],
        key=os.path.getmtime,
        reverse=True,
    )

    if not results_folders:
        return None, None

    latest_result_folder = results_folders[0]
    model_save_path = os.path.join(latest_result_folder, "model")
    log_file = os.path.join(latest_result_folder, "log.jsonl")

    if not os.path.exists(model_save_path) or not os.path.exists(log_file):
        return None, None

    return model_save_path, log_file


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


def get_default_hyperparameters(
    task_type: str, model_type: str, training_methods: dict
):
    """
    Get default hyperparameters based on task, model, and training methods.

    Args:
        task_type: Type of task (e.g., 'gsm8k', 'wiki_compression')
        model_type: Type of model (e.g., 'llama', 'mistral')
        training_methods: Dictionary of training method flags
            {
                'use_ppo': bool,
                'use_ei': bool,
                'use_pg': bool
            }

    Returns:
        Dictionary of default hyperparameters
    """
    # Base default hyperparameters
    defaults = {
        "model_learning_rate": 0.0001,
        "num_batches": 10000,
        "normalize_loss": True,
    }

    # Task-specific batch sizes and gradient accumulation
    batch_size_defaults = {
        "llama": {
            "gsm8k": 10,
            "wiki_compression": 6,
            "wiki_continuation": 6,
            "arithmetic": 2,  # Added default for arithmetic
            "arithmetic_negative": 8,  # Added default for negative arithmetic
            "default": 6,
        },
        "mistral": {
            "gsm8k": 10,
            "wiki_compression": 2,
            "wiki_continuation": 2,
            "arithmetic": 8,  # Added default for arithmetic
            "arithmetic_negative": 8,  # Added default for negative arithmetic
            "default": 2,
        },
    }

    # Gradient accumulation steps
    grad_accumulation_defaults = {
        "gsm8k": 32,
        "wiki_compression": 8,
        "wiki_continuation": 8,
        "arithmetic": 8,  # Added default for arithmetic
        "arithmetic_negative": 8,  # Added default for negative arithmetic
        "default": 8,
    }

    # Chain of thought length defaults
    cot_length_defaults = {
        "llama": {
            "gsm8k": 60,
            "wiki_compression": 150,
            "wiki_continuation": 150,
            "arithmetic": 250,
            "arithmetic_negative": 250,
            "default": 150,
        },
        "mistral": {
            "gsm8k": 80,
            "wiki_compression": 200,
            "wiki_continuation": 200,
            "arithmetic": 400,
            "arithmetic_negative": 400,
            "default": 400,
        },
    }

    # Get specific or default values
    defaults["batch_size"] = batch_size_defaults.get(model_type, {}).get(
        task_type, batch_size_defaults[model_type]["default"]
    )

    defaults["gradient_accumulation_steps"] = grad_accumulation_defaults.get(
        task_type, grad_accumulation_defaults["default"]
    )

    defaults["cot_length"] = cot_length_defaults.get(model_type, {}).get(
        task_type, cot_length_defaults[model_type]["default"]
    )

    # Task-specific length parameters
    if task_type in ["wiki_compression", "wiki_continuation"]:
        defaults["question_length"] = 500
        defaults["target_length"] = 500

    # Training method specific parameters
    if training_methods.get("use_ppo", False):
        defaults["ppo_epsilon"] = 0.2
        defaults["r"] = 1.0
    else:
        defaults["ppo_epsilon"] = None
        defaults["r"] = None

    # Add training method flags
    defaults.update(training_methods)

    return defaults


def print_debug_info(
    task_type,
    q,
    reasoning_text_first,
    ans,
    avg_log_prob,
    extracted_generated_answers=None,
):
    """Print debug information with consistent coloring and formatting."""
    if task_type == "wiki_compression":
        colored_print("Full Text:", q, Colors.BLUE)
        colored_print("Compression:", reasoning_text_first, Colors.YELLOW)
    elif task_type == "wiki_continuation":
        colored_print("Context:", q, Colors.BLUE)
        colored_print("Helpful Text:", reasoning_text_first, Colors.YELLOW)
    else:  # arithmetic or gsm8k
        colored_print("Question:", q, Colors.BLUE)
        colored_print("Reasoning:", reasoning_text_first, Colors.YELLOW)

    colored_print("Answer:", ans, Colors.GREEN)
    colored_print("Avg Log Prob:", str(avg_log_prob), Colors.BOLD, inline=True)

    if extracted_generated_answers is not None:
        colored_print("Generated Answer:", extracted_generated_answers[0], Colors.RED)


def train(
    task_type: str,
    resume: bool,
    use_ei: bool,
    use_ppo: bool,
    use_pg: bool,
    model_type: str,
    hyperparameters: dict,
):
    """Train the model with the specified configuration."""
    global previous_normalized_rewards, previous_advantages

    # Create a results directory with timestamp
    results_dir = os.path.join(
        "results", task_type, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(results_dir, exist_ok=True)

    # Define paths for model and log file
    model_save_path = os.path.join(results_dir, "model")
    log_file = os.path.join(results_dir, "log.jsonl")

    # If resuming, load the latest checkpoint and log file
    if resume:
        latest_checkpoint, latest_log = get_latest_result_and_log(task_type)
        if latest_checkpoint and latest_log:
            # Load model and optimizer states
            checkpoint = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load previous training state
            start_batch, hyperparameters = load_training_state(latest_log)

            # Load previous rewards and advantages for EI
            previous_normalized_rewards, previous_advantages = (
                load_previous_rewards_and_advantages(latest_log)
            )
        else:
            start_batch = 0
    else:
        start_batch = 0
        previous_normalized_rewards = []
        previous_advantages = []
        # Write hyperparameters as the first line in the log file
        with open(log_file, "w") as f:
            json.dump(hyperparameters, f)
            f.write("\n")

    model, frozen_model, tokenizer, device = load_model(model_type)

    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=hyperparameters["model_learning_rate"]
    )

    # Create generator instead of materializing all batches
    qa_generator = generate_question_answer_batches(
        num_batches=hyperparameters["num_batches"],
        batch_size=hyperparameters["batch_size"],
        task_type=task_type,
        tokenizer=tokenizer,
        hyperparameters=hyperparameters,
    )

    model_optimizer.zero_grad()

    # Iterate over generator directly
    for batch_index in range(start_batch, hyperparameters["num_batches"]):
        print_batch_delimiter()
        colored_print("Batch:", str(batch_index), Colors.BOLD, inline=True)
        try:
            qa_batch = next(qa_generator)
        except StopIteration:
            print("Reached end of dataset")
            break

        questions, answers = zip(*qa_batch)

        # Use construct_prompts for creating prompts
        prompts = [
            construct_prompts(
                task_type=task_type,
                question=q,
                model_type=model_type,
                hyperparameters=hyperparameters,
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
            questions,
            answers,
            hyperparameters,
            task_type,
            model_type,
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

        total_loss, policy_loss, ppo_ratio, clipped_ratio, num_active = (
            calculate_losses(
                unfrozen_avg_log_probs_reasoning_given_question,
                frozen_avg_log_probs_reasoning_given_question,
                advantage,
                hyperparameters,
            )
        )

        # Only accumulate gradients if we have active samples
        if num_active > 0:
            loss = total_loss / hyperparameters["gradient_accumulation_steps"]
            loss.backward()

            grad_norm = get_grad_norm(model.parameters())
            clip_grad_norm_(model.parameters(), 1.0)
        else:
            loss = torch.tensor(0.0)
            grad_norm = 0.0

        # Update logging to include fraction of active samples
        fraction_active = num_active / hyperparameters["batch_size"]

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

        print_debug_info(
            task_type,
            q,
            reasoning_text_first,
            ans,
            avg_log_prob,
            extracted_generated_answers,
        )

        if previous_advantages:
            mean_prev_advantage = np.mean(previous_advantages)
            std_prev_advantage = np.std(previous_advantages)
        else:
            mean_prev_advantage = None
            std_prev_advantage = None

        log_entry = {
            k: tensor_to_python(v)
            for k, v in {
                "Aggregate loss": loss.item()
                * hyperparameters["gradient_accumulation_steps"],
                "Batch Index": batch_index,
                "Prev Observation": f"{'Context' if task_type in ['wiki_compression', 'wiki_continuation'] else 'Question'}: {q}",
                "Action": f"{'Helpful Text' if task_type in ['wiki_compression', 'wiki_continuation'] else 'Reasoning'}: {reasoning_text_first}",
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
                "Fraction Active Samples": fraction_active,
                "Num Active Samples": num_active,
                "Batch Size": hyperparameters["batch_size"],
            }.items()
        }

        if baseline_avg_log_prob is not None:
            log_entry["Baseline Avg Log Prob"] = tensor_to_python(baseline_avg_log_prob)

        if hyperparameters["use_ppo"] and ppo_ratio is not None:
            log_entry.update(
                {
                    "PPO Ratio": tensor_to_python(ppo_ratio[0]),
                    "PPO Clipped Ratio": tensor_to_python(clipped_ratio[0]),
                }
            )

        if task_type == "gsm8k" and extracted_generated_answers is not None:
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

            # Save model weights
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model_optimizer.state_dict(),
                    "batch_index": batch_index,
                    "hyperparameters": hyperparameters,
                },
                model_save_path,
            )

        # Periodically plot training metrics (every 10 batches)
        if batch_index > 0 and batch_index % 10 == 0:
            try:
                # Call plot_training_metrics.py with the current log file
                subprocess.run(
                    [
                        "python",
                        "src/plot_training_metrics.py",
                        "--log_file",
                        log_file,
                        "--window_size",
                        "10",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error plotting metrics: {e}")


def main(
    task_type: str,
    resume: bool,
    use_ei: bool = False,
    use_ppo: bool = False,
    use_pg: bool = False,
    model_type: str = "llama",
):
    """Main entry point with command-line parameter handling."""
    # 1. Get default hyperparameters based on task, model, and training methods
    hyperparameters = get_default_hyperparameters(
        task_type=task_type,
        model_type=model_type,
        training_methods={"use_ppo": use_ppo, "use_ei": use_ei, "use_pg": use_pg},
    )

    # 3. Validate training method selection
    if not (use_ei or use_pg or use_ppo):
        raise ValueError(
            "At least one of --use_ei, --use_pg, or --use_ppo must be specified."
        )

    # 4. Call train with fully prepared hyperparameters
    train(
        task_type=task_type,
        resume=resume,
        use_ei=use_ei,
        use_ppo=use_ppo,
        use_pg=use_pg,
        model_type=model_type,
        hyperparameters=hyperparameters,
    )


def get_latest_log_file():
    """
    Find the most recent log file in the results directory.
    Searches across all task subdirectories.
    """
    results_dir = "results"

    # Find all log files recursively
    log_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "log.jsonl" or file.endswith(".log"):
                log_file_path = os.path.join(root, file)
                log_files.append(log_file_path)

    if not log_files:
        raise FileNotFoundError("No log files found in results directory.")

    # Return the most recently modified log file
    return max(log_files, key=os.path.getmtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on various tasks.")
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            "arithmetic",
            "arithmetic_negative",
            "gsm8k",
            "wiki_compression",
            "wiki_continuation",
        ],
        default="arithmetic",
        help="Type of task to train on",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from the last checkpoint",
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
        choices=["llama", "mistral"],
        default="llama",
        help="Choose between Llama and Mistral models",
    )

    args = parser.parse_args()

    if not (args.use_ei or args.use_pg or args.use_ppo):
        raise ValueError(
            "At least one of --use_ei, --use_pg, or --use_ppo must be specified."
        )

    main(
        task_type=args.task_type,
        resume=args.resume,
        use_ei=args.use_ei,
        use_ppo=args.use_ppo,
        use_pg=args.use_pg,
        model_type=args.model_type,
    )
