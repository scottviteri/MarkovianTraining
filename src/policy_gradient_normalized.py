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
    print("\n" + "="*80)

def colored_print(label: str, text: str, color: str = Colors.BLUE):
    """Print text with colored label, adding newline before and keeping text raw."""
    print(f"\n{color}{label}{Colors.END}")
    print(repr(text))

def load_model(model_type="mistral"):
    """Load either Mistral or Llama 3.1 model based on parameter."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Using 8B version
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


def generate_question_answer_batch(batch_size: int, task_type: str):
    """Generate a batch of arithmetic questions and answers."""
    qa_batch = []
    for _ in range(batch_size):
        if task_type == 'arithmetic-negative':
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

        qa_batch.append((question, answer))
    return qa_batch


def load_gsm8k_dataset():
    ds = load_dataset("openai/gsm8k", "main")
    questions = ds["train"]["question"]
    answers = list(map(lambda x: x[x.index("####") + 5 :], ds["train"]["answer"]))
    return list(zip(questions, answers))


def get_model_specific_tokens(model_type):
    """Return model-specific tokens for prompt construction."""
    if model_type == "mistral":
        return {
            "inst_start": "]",
            "inst_end": "]",
            "user_start": "",
            "user_end": "",
            "assistant_start": "",
            "assistant_end": "",
        }
    else:  # llama
        return {
            "inst_start": "<start_header_id>user<|end_header_id|>",
            "inst_end": "<|eot_id|><start_header_id>assistant<|end_header_id|>",
            "user_start": "",
            "user_end": "",
            "assistant_start": "",
            "assistant_end": "",
        }


def construct_prompt(task_type, question, model_type, hyperparameters):
    """Construct prompt based on task type and model."""
    tokens = get_model_specific_tokens(model_type)
    
    if task_type == "wiki_compression":
        base_prompt = (
            f"The following text is the {hyperparameters['target_length']} tokens you need to reconstruct. "
            f"Write {hyperparameters['cot_length']} tokens that will help you reconstruct it. "
            f"Be concise and focus on key information!\n\nText to reconstruct: {question}"
        )
    elif task_type == "wiki_continuation":
        # Preserve old wiki prompt
        base_prompt = (
            f"Given this opening text from an article, write whatever "
            f"{hyperparameters['cot_length']} tokens you suspect might help you "
            f"predict the next {hyperparameters['target_length']} tokens. Be creative!\n\nOpening text: {question}"
        )
    else:  # arithmetic/gsm8k
        base_prompt = f"Produce minimal text which will help you answer this question. Question: {question}"

    # Construct full prompt with model-specific tokens
    if model_type == "mistral":
        return f"{tokens['inst_start']} {base_prompt} {tokens['inst_end']}\nReasoning:"
    else:  # llama
        return f"{tokens['inst_start']} {base_prompt} {tokens['inst_end']}\nReasoning:"


def get_text_with_token_length(text: str, desired_tokens: int, tokenizer) -> tuple[str, int]:
    """
    Binary search to find text that tokenizes to desired number of tokens.
    Returns (text_chunk, actual_token_count) or (None, 0) if text is too short.
    """
    # Initial guess based on 4 chars per token
    chars = desired_tokens * 4
    if len(text) < chars:
        return None, 0
        
    # Binary search for correct length
    left, right = 1, len(text)
    best_text = None
    best_count = 0
    
    while left <= right:
        mid = (left + right) // 2
        chunk = text[:mid]
        tokens = tokenizer(chunk, return_tensors="pt")
        token_count = len(tokens.input_ids[0])
        
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
    tokenizer,  # Need to pass tokenizer
    hyperparameters: dict = None,
):
    """Generate batches of Q&A pairs from different sources."""
    if task_type in ['wiki_compression', 'wiki_continuation']:
        # Load Wikipedia dataset
        wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
        qa_pairs = []
        
        question_tokens = hyperparameters.get('question_length', 500)
        target_tokens = hyperparameters.get('target_length', 500)
        
        article_idx = 0
        while len(qa_pairs) < num_batches * batch_size:
            if article_idx >= len(wiki_dataset):
                print(f"Warning: Reached end of Wikipedia dataset after {len(qa_pairs)} examples")
                break
                
            article = wiki_dataset[article_idx]["text"]
            article_idx += 1
            
            if task_type == 'wiki_compression':
                # For compression task, get text of exact token length
                text_chunk, actual_tokens = get_text_with_token_length(
                    article, target_tokens, tokenizer
                )
                if text_chunk is not None:
                    qa_pairs.append((text_chunk, text_chunk))
                    
            else:  # wiki_continuation
                # Get question and answer chunks
                q_chunk, q_tokens = get_text_with_token_length(
                    article, question_tokens, tokenizer
                )
                if q_chunk is None:
                    continue
                    
                remaining_text = article[len(q_chunk):]
                a_chunk, a_tokens = get_text_with_token_length(
                    remaining_text, target_tokens, tokenizer
                )
                if a_chunk is not None:
                    qa_pairs.append((q_chunk, a_chunk))

        random.shuffle(qa_pairs)
        return [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]
    
    elif task_type == 'gsm8k':
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
    else:  # arithmetic or arithmetic-negative
        return [
            generate_question_answer_batch(batch_size, task_type)
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
    task_type,
    model_type,
):
    """Calculate the log probabilities of the answers given the reasoning."""
    reasoning_text = tokenizer.batch_decode(reasoning_tokens, skip_special_tokens=True)

    # Update full prompts based on dataset type
    if task_type in ['wiki_compression', 'wiki_continuation']:
        full_prompts = [
            f"Helpful Text: {r}\nAnswer: {a}" for r, a in zip(reasoning_text, answers)
        ]
        partial_prompts = [f"Helpful Text: {r}\nAnswer:" for r in reasoning_text]
    else:
        full_prompts = [
            f"Reasoning: {r}\nAnswer: {a}" for r, a in zip(reasoning_text, answers)
        ]
        partial_prompts = [f"Reasoning: {r}\nAnswer:" for r in reasoning_text]

    tokenized_full_prompts = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    extracted_generated_answers = None
    if task_type == 'gsm8k':
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
    # if llama, find last occurrence of [16533, 25] and use 25's index + 1
    answer_start_positions = []
    for input_ids in tokenized_full_prompts.input_ids:
        if model_type == "mistral":
            pos = (
                (input_ids == 28747) | (input_ids == 28705) | (input_ids == 29871)
            ).nonzero(as_tuple=True)[0][-1].item() + 1
        else:  # llama
            # Find positions of both tokens
            token_positions = (input_ids == 16533).nonzero(as_tuple=True)[0]
            colon_positions = (input_ids == 25).nonzero(as_tuple=True)[0]

            # Find the last pair where 16533 is followed by 25
            last_pair_pos = None
            for token_pos in reversed(token_positions.tolist()):
                # Look for a colon after this token
                following_colons = colon_positions[colon_positions > token_pos]
                if len(following_colons) > 0 and following_colons[0] == token_pos + 1:
                    last_pair_pos = following_colons[0]
                    break

            if last_pair_pos is None:
                raise ValueError("Could not find required token sequence [16533, 25]")

            pos = last_pair_pos + 1
        answer_start_positions.append(pos)

    # Assert that the decoded tokens after each start position match the expected answers
    for i in range(len(answers)):
        decoded_answer = tokenizer.decode(
            tokenized_full_prompts.input_ids[i][answer_start_positions[i] :]
        ).strip()
        expected_answer = answers[i].strip()
        if decoded_answer != expected_answer:
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
            reasoning_tokens,
            answers,
            task_type,
            model_type,
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
            task_type,
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


def get_default_hyperparameters(task_type: str, model_type: str, use_ppo: bool, use_ei: bool, use_pg: bool) -> dict:
    """Get default hyperparameters based on task and model configuration."""
    defaults = {
        "batch_size": {
            "gsm8k": 10,
            "default": 6
        },
        "gradient_accumulation_steps": {
            "gsm8k": 32,
            "default": 8
        },
        "cot_length": {
            "llama": {
                "gsm8k": 60,
                "wiki_compression": 100,
                "wiki_continuation": 100,
                "default": 150
            },
            "mistral": {
                "gsm8k": 80,
                "default": 400
            }
        }
    }

    # Get task-specific or default values
    batch_size = defaults["batch_size"].get(task_type, defaults["batch_size"]["default"])
    grad_steps = defaults["gradient_accumulation_steps"].get(task_type, 
                                                           defaults["gradient_accumulation_steps"]["default"])
    
    # Get model and task specific CoT length
    model_defaults = defaults["cot_length"][model_type]
    cot_length = model_defaults.get(task_type, model_defaults["default"])

    return {
        "model_learning_rate": 0.0001,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_steps,
        "num_batches": 10000,
        "cot_length": cot_length,
        "question_length": 500,
        "target_length": 500,
        "normalize_loss": True,
        # PPO specific parameters
        "ppo_epsilon": 0.2 if use_ppo else None,
        "r": 1.0 if use_ppo else None,
        # Training method flags
        "use_ppo": use_ppo,
        "use_ei": use_ei,
        "use_pg": use_pg,
    }

def print_debug_info(task_type, q, reasoning_text_first, ans, avg_log_prob, extracted_generated_answers=None):
    """Print debug information with consistent coloring and formatting."""
    print_batch_delimiter()
    
    if task_type == 'wiki_compression':
        colored_print("Full Text", q, Colors.BLUE)
        colored_print("Compression", reasoning_text_first, Colors.YELLOW)
    elif task_type == 'wiki_continuation':
        colored_print("Context", q, Colors.BLUE)
        colored_print("Helpful Text", reasoning_text_first, Colors.YELLOW)
    else:  # arithmetic or gsm8k
        colored_print("Question", q, Colors.BLUE)
        colored_print("Reasoning", reasoning_text_first, Colors.YELLOW)
    
    colored_print("Answer", ans, Colors.GREEN)
    colored_print("Avg Log Prob", str(avg_log_prob), Colors.BOLD)
    
    if extracted_generated_answers is not None:
        colored_print("Generated Answer", repr(extracted_generated_answers[0]), Colors.RED)

def train(
    task_type: str,
    resume: bool,
    use_ei: bool,
    use_ppo: bool,
    use_pg: bool,
    model_type: str,
    hyperparameters: dict = None,
):
    """Train the model with the specified configuration."""
    global previous_normalized_rewards, previous_advantages
    
    # Update dataset type for logging
    dataset_type = task_type.replace('_', '-').title()
    
    # Initialize paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = f"checkpoints/{dataset_type}/{timestamp}"
    os.makedirs(base_path, exist_ok=True)
    model_save_path = f"{base_path}/model"
    log_file = f"{base_path}/log.jsonl"

    if resume:
        model_save_path, log_file = get_latest_checkpoint_and_log(dataset_type)
        with open(log_file, "r") as f:
            lines = f.readlines()
            hyperparameters = json.loads(lines[0])["hyperparameters"]
            
            # Repopulate previous values from log
            previous_normalized_rewards = []
            previous_advantages = []
            for line in lines[1:]:  # Skip hyperparameters line
                log_entry = json.loads(line)
                if "normalized_rewards" in log_entry:
                    previous_normalized_rewards.extend(log_entry["normalized_rewards"])
                if "advantages" in log_entry:
                    previous_advantages.extend(log_entry["advantages"])
            
            start_batch = len(lines) - 1
            print(f"Resuming from batch {start_batch}")
            print(f"Loaded {len(previous_normalized_rewards)} previous rewards")
            print(f"Loaded {len(previous_advantages)} previous advantages")
    else:
        start_batch = 0
        previous_normalized_rewards = []
        previous_advantages = []
        # Get default hyperparameters
        defaults = get_default_hyperparameters(
            task_type=task_type,
            model_type=model_type,
            use_ppo=use_ppo,
            use_ei=use_ei,
            use_pg=use_pg
        )
        
        # Override defaults with any provided hyperparameters
        hyperparameters = {**defaults, **(hyperparameters or {})}

        # Initialize logging
        with open(log_file, "w") as f:
            json.dump({"hyperparameters": hyperparameters}, f)
            f.write("\n")

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
            num_batches=hyperparameters["num_batches"],
            batch_size=hyperparameters["batch_size"],
            task_type=task_type,
            tokenizer=tokenizer,  # Pass tokenizer
            hyperparameters=hyperparameters,
        )
    )

    model_optimizer.zero_grad()

    for batch_index, qa_batch in enumerate(qa_batches[start_batch:], start=start_batch):
        print_batch_delimiter()
        colored_print("Batch:", str(batch_index), Colors.BOLD)
        
        questions, answers = zip(*qa_batch)
        prompts = [
            construct_prompt(
                task_type,
                q,
                model_type,
                hyperparameters
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
            loss = total_loss / gradient_accumulation_steps
            loss.backward()

            grad_norm = get_grad_norm(model.parameters())
            clip_grad_norm_(model.parameters(), 1.0)
        else:
            loss = torch.tensor(0.0)
            grad_norm = 0.0

        # Update logging to include fraction of active samples
        fraction_active = num_active / batch_size

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

        print_debug_info(task_type, q, reasoning_text_first, ans, avg_log_prob, extracted_generated_answers)

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
                "Batch Size": batch_size,
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

        if task_type == 'gsm8k' and extracted_generated_answers is not None:
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
    task_type: str,
    resume: bool,
    use_ei: bool = False,
    use_ppo: bool = False,
    use_pg: bool = False,
    model_type: str = "llama",
    cot_length: int = None,
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    normalize_loss: bool = None,
):
    """Main entry point with command-line parameter handling."""
    hyperparameters = {}
    if cot_length is not None:
        hyperparameters["cot_length"] = cot_length
    if batch_size is not None:
        hyperparameters["batch_size"] = batch_size
    if gradient_accumulation_steps is not None:
        hyperparameters["gradient_accumulation_steps"] = gradient_accumulation_steps
    if normalize_loss is not None:
        hyperparameters["normalize_loss"] = normalize_loss

    train(
        task_type=task_type,
        resume=resume,
        use_ei=use_ei,
        use_ppo=use_ppo,
        use_pg=use_pg,
        model_type=model_type,
        hyperparameters=hyperparameters if hyperparameters else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model on various tasks."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            'arithmetic',
            'arithmetic-negative',
            'gsm8k',
            'wiki_compression',
            'wiki_continuation'
        ],
        default='arithmetic',
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
    # Hyperparameter overrides
    parser.add_argument(
        "--cot_length",
        type=int,
        default=None,
        help="Length of chain-of-thought reasoning",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--normalize_loss",
        type=bool,
        default=None,
        help="Whether to normalize the loss (default: True)",
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
        cot_length=args.cot_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        normalize_loss=args.normalize_loss,
    )
