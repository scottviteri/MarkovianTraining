import datetime
import torch
from torch import nn
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
from src.constants import MISTRAL_INST_START, MISTRAL_INST_END
from src.constants import EI_SKIP_INITIAL
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


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
    CYAN = "\033[96m"
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


def calculate_threshold(previous_advantages, fixed_threshold=None):
    """
    Calculate threshold for expert iteration.

    Args:
        previous_advantages: List of previous advantage values
        fixed_threshold: If provided, use this fixed value instead of statistical threshold

    Returns:
        float: Threshold value (inf if not enough previous advantages)
    """
    if len(previous_advantages) <= EI_SKIP_INITIAL:
        return float("inf")

    if fixed_threshold is not None:
        return fixed_threshold

    return np.mean(previous_advantages) + np.std(previous_advantages)


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
    """Return model-specific tokens for prompt construction."""
    if model_type == "mistral":
        return {
            "inst_start": MISTRAL_INST_START,
            "inst_end": MISTRAL_INST_END,
        }
    else:  # llama
        return {
            "inst_start": "",
            "inst_end": "",
        }


def construct_prompts(
    question: str,
    hyperparameters: Dict[str, Any],
    reasoning: Optional[str] = None
) -> str:
    """
    Construct prompt for model input.
    
    Args:
        question: The input question or text
        hyperparameters: Dictionary containing model and task configuration
        reasoning: Optional reasoning text to include
        
    Returns:
        str: Formatted prompt
    """
    model_type = hyperparameters["model_type"]
    task_type = hyperparameters["task_type"]

    tokens = get_model_specific_tokens(model_type)

    # Construct base prompt
    if task_type == "wiki_compression":
        base_prompt = (
            f"You will need to reconstruct the following {hyperparameters['target_length']} tokens, which you will need to reconstruct given {hyperparameters['cot_length']} memory tokens which you can write for yourself."
            f"Feel free to be creative in your chosen compression strategy!\n\nFull Text:"
        )
        prompt_type = "Compression:"
    elif task_type == "wiki_continuation":
        base_prompt = (
            "You will need to predict the next {hyperparameters['target_length']} tokens which follow the provided passage."
            f"You can write {hyperparameters['cot_length']} thinking tokens which will be your sole context for prediction."
            f"Feel free to be creative in your thinking strategy!\n\nOpening text:"
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
    frozen_model,
    tokenizer,
    device,
    questions,
    reasoning,
    answers,
    hyperparameters,
):
    """Calculate the log probabilities of the answers given the reasoning.
    
    Args:
        frozen_model: The critic model (frozen)
        questions: List of question strings
        reasoning: List of reasoning strings (from either actor or critic)
        answers: List of answer strings
        
    Returns:
        tuple: (
            mean_answer_logprobs,  # Average log prob of each answer token
            answer_logprobs,       # Full sequence of answer token log probs
            extracted_answers      # Only for GSM8K: extracted numerical answers
        )
    """
    # Create prompts with reasoning
    partial_prompts = [
        construct_prompts(
            question=q,
            hyperparameters=hyperparameters,
            reasoning=r,
        )
        for q, r in zip(questions, reasoning)
    ]
    
    # Add answers to create full prompts
    q_r_a_prompts = [x + y for x, y in zip(partial_prompts, answers)]
    
    # Tokenize full prompts
    q_r_a_tokens = tokenizer(
        q_r_a_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # For GSM8K, we also generate answers to extract numerical values
    extracted_generated_answers = None
    if hyperparameters["task_type"] == "gsm8k":
        # Tokenize partial prompts (without answers) for generation
        q_r_tokens = tokenizer(
            partial_prompts, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        # Generate answer tokens
        max_answer_length = 15
        with torch.no_grad():
            q_r_a_generated = frozen_model.generate(
                input_ids=q_r_tokens.input_ids,
                attention_mask=q_r_tokens.attention_mask,
                max_new_tokens=max_answer_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and extract numerical answers
        generated_answers = tokenizer.batch_decode(
            q_r_a_generated[:, -max_answer_length - 1 :], 
            skip_special_tokens=True
        )
        selected_answers = [x.split("\n")[-1] for x in generated_answers]
        extracted_generated_answers = [extract_answer(ans) for ans in selected_answers]

    # Find the starting positions of answers in the full prompts
    answer_start_positions = []
    for input_ids in q_r_a_tokens.input_ids:
        if hyperparameters["model_type"] == "mistral":
            # Find "Answer:" token sequence
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
            matching_indices = (
                ((input_ids[:-1] == 16533) | (input_ids[:-1] == 22559))
                & (input_ids[1:] == 25)
            ).nonzero(as_tuple=True)[0]
            pos = matching_indices[-1].item() + 2
        answer_start_positions.append(pos)

    # Verify answer positions are correct
    for i in range(len(answers)):
        decoded_answer = tokenizer.decode(
            q_r_a_tokens.input_ids[i][answer_start_positions[i] :]
        ).strip()
        expected_answer = answers[i].strip()
        if (
            decoded_answer[:3] != expected_answer[:3]
            or decoded_answer[-3:] != expected_answer[-3:]
        ):
            colored_print("Answer mismatch at index", str(i), Colors.RED)

    # Calculate log probabilities
    with torch.no_grad():
        q_r_a_critic_logits = frozen_model(
            input_ids=q_r_a_tokens.input_ids,
            attention_mask=q_r_a_tokens.attention_mask,
        ).logits

    # Convert to log probabilities
    q_r_a_logprobs = torch.nn.functional.log_softmax(q_r_a_critic_logits, dim=-1)
    
    # Get log probs for each answer token
    answer_logprobs = [
        q_r_a_logprobs[i, start - 1 : -1]
        .gather(1, q_r_a_tokens.input_ids[i, start:].unsqueeze(-1))
        .squeeze(-1)
        for i, start in enumerate(answer_start_positions)
    ]

    # Calculate mean log prob per answer
    mean_answer_logprobs = torch.stack([x.mean() for x in answer_logprobs])

    return mean_answer_logprobs, extracted_generated_answers


def exponential_weighted_average(values, r):
    weights = np.array([r ** (len(values) - i) for i in range(len(values))])
    weights = weights / np.sum(weights)
    return np.sum(weights * np.array(values))


@dataclass
class ReasoningOutput:
    """Holds the output from reasoning generation"""
    actor_reasoning: List[str]
    critic_reasoning: List[str]
    R_mean_actor_logprobs: torch.Tensor
    R_mean_critic_logprobs: torch.Tensor
    kl: torch.Tensor

@dataclass
class AdvantageOutput:
    """Holds the output from advantage calculation"""
    advantages: torch.Tensor
    normalized_rewards: torch.Tensor
    extracted_answers: Optional[List[Any]]

@dataclass
class TrainingState:
    """Holds the state of the training process"""
    batch_index: int
    previous_normalized_rewards: List[float]
    previous_advantages: List[float]
    grad_accum_count: int
    
    # Models and optimization
    actor_model: nn.Module
    critic_model: nn.Module
    actor_optimizer: torch.optim.Optimizer
    tokenizer: Any
    device: torch.device
    
    # Paths and logging
    model_save_path: str
    log_file: str
    
    # Configuration
    hyperparameters: Dict[str, Any]
    
    @classmethod
    def initialize(cls, task_type: str, resume: bool, model_type: str, hyperparameters: dict):
        """Factory method to create a new TrainingState"""
        model_save_path, log_file, start_batch, prev_rewards, prev_advantages = setup_training_environment(
            task_type, resume
        )
        
        actor_model, critic_model, tokenizer, device, actor_optimizer = initialize_model_and_optimizer(
            model_type, hyperparameters
        )
        
        return cls(
            batch_index=start_batch,
            previous_normalized_rewards=prev_rewards,
            previous_advantages=prev_advantages,
            grad_accum_count=0,
            actor_model=actor_model,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            tokenizer=tokenizer,
            device=device,
            model_save_path=model_save_path,
            log_file=log_file,
            hyperparameters=hyperparameters
        )

def generate_reasoning_and_kl(state: TrainingState, questions: List[str]) -> ReasoningOutput:
    """Generate reasoning from both models and calculate KL divergence."""
    # Create prompts for each question
    prompts = [
        construct_prompts(
            question=q,
            hyperparameters=state.hyperparameters,
        )
        for q in questions
    ]

    # Tokenize inputs
    tokenized_inputs = state.tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(state.device)

    # Generate reasoning tokens from both models
    with torch.no_grad():
        # Actor (unfrozen) generates reasoning
        q_R_tokens = state.actor_model.generate(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=state.hyperparameters["cot_length"],
            min_new_tokens=state.hyperparameters["cot_length"],
            do_sample=True,
            temperature=state.hyperparameters["temperature"],
            pad_token_id=state.tokenizer.pad_token_id,
        )
        # Critic (frozen) generates reasoning
        q_r_tokens = state.critic_model.generate(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=state.hyperparameters["cot_length"],
            min_new_tokens=state.hyperparameters["cot_length"],
            do_sample=True,
            temperature=state.hyperparameters["temperature"],
            pad_token_id=state.tokenizer.pad_token_id,
        )

    # Get logits from both models on actor's reasoning
    q_R_actor_logits = state.actor_model(q_R_tokens).logits
    q_R_critic_logits = state.critic_model(q_R_tokens).logits

    # Calculate log probabilities and KL
    R_actor_logprobs = q_R_actor_logits[:,-state.hyperparameters["cot_length"]-1:-1,:].log_softmax(dim=-1)
    R_critic_logprobs = q_R_critic_logits[:,-state.hyperparameters["cot_length"]-1:-1,:].log_softmax(dim=-1)

    R_mean_actor_logprobs = R_actor_logprobs.gather(
        2, 
        q_R_tokens[:,-state.hyperparameters["cot_length"]:].unsqueeze(-1)
    ).squeeze(-1).mean(dim=1)
    
    R_mean_critic_logprobs = R_critic_logprobs.gather(
        2, 
        q_R_tokens[:,-state.hyperparameters["cot_length"]:].unsqueeze(-1)
    ).squeeze(-1).mean(dim=1)

    kl = calculate_mean_kl(q_R_actor_logits, q_R_critic_logits, state.hyperparameters["cot_length"])

    # Decode reasoning text
    actor_reasoning = state.tokenizer.batch_decode(q_R_tokens, skip_special_tokens=True)
    critic_reasoning = state.tokenizer.batch_decode(q_r_tokens, skip_special_tokens=True)

    return ReasoningOutput(
        actor_reasoning=actor_reasoning,
        critic_reasoning=critic_reasoning,
        R_mean_actor_logprobs=R_mean_actor_logprobs,
        R_mean_critic_logprobs=R_mean_critic_logprobs,
        kl=kl
    )

def calculate_advantages(
    state: TrainingState,
    questions: List[str],
    answers: List[str],
    reasoning_output: ReasoningOutput,
) -> AdvantageOutput:
    """Calculate advantages by comparing answer probabilities under different reasoning."""
    
    # Calculate log probs of answers given actor's reasoning
    actor_answer_logprobs, extracted_answers = calculate_answer_log_probs(
        state.critic_model,
        state.tokenizer,
        state.device,
        questions,
        reasoning_output.actor_reasoning,
        answers,
        state.hyperparameters,
    )

    if state.hyperparameters.get("normalize_loss", True):
        # Calculate log probs of answers given critic's reasoning (baseline)
        critic_answer_logprobs,_ = calculate_answer_log_probs(
            state.critic_model,
            state.tokenizer,
            state.device,
            questions,
            reasoning_output.critic_reasoning,
            answers,
            state.hyperparameters,
        )
        # Normalize reward as improvement over baseline
        normalized_rewards = actor_answer_logprobs - critic_answer_logprobs
    else:
        normalized_rewards = actor_answer_logprobs

    # Calculate advantage using exponential moving average baseline
    r = state.hyperparameters.get("r", None)
    if len(state.previous_normalized_rewards) > 0 and r is not None:
        value = exponential_weighted_average(state.previous_normalized_rewards, r)
        advantages = normalized_rewards - value
    else:
        advantages = normalized_rewards

    return AdvantageOutput(
        advantages=advantages,
        normalized_rewards=normalized_rewards,
        extracted_answers=extracted_answers
    )


def calculate_losses(
    kl,
    R_mean_actor_logprobs,
    R_mean_critic_logprobs,
    advantages,
    previous_advantages,
    hyperparameters
):
    """Calculate training losses using specified methods (PG/PPO/EI).
    
    Args:
        kl: KL divergence between actor and critic distributions
        R_mean_actor_logprobs: Mean log probs of actor's reasoning under actor
        R_mean_critic_logprobs: Mean log probs of actor's reasoning under critic
        advantages: Advantage values for actor's reasoning
        previous_advantages: History of advantages for EI threshold
        hyperparameters: Training configuration
        
    Returns:
        tuple: (
            losses,         # Final loss values for backprop
            training_mask,  # Binary mask for active training examples (EI)
            metrics        # Dictionary of metrics for logging
        )
    """
    use_ppo = hyperparameters["use_ppo"]
    ppo_epsilon = hyperparameters.get("ppo_epsilon", 0.2)
    kl_penalty = hyperparameters.get("kl_penalty", None)

    # Initialize metrics dictionary
    metrics = {}

    # Base policy gradient loss
    pg_losses = -R_mean_actor_logprobs * advantages.detach()
    metrics['pg_losses'] = pg_losses
    losses = pg_losses

    # Add KL penalty if specified
    weighted_kl = None
    if kl_penalty is not None:
        weighted_kl = kl_penalty * kl
        losses = losses + weighted_kl
        metrics['weighted_kl'] = weighted_kl

    # Apply PPO if specified
    prob_ratios = None
    clipped_ratios = None
    if use_ppo:
        # Calculate probability ratio between actor and critic
        prob_ratios = torch.exp(R_mean_actor_logprobs - R_mean_critic_logprobs)
        # Clip probability ratios
        clipped_ratios = torch.clamp(prob_ratios, 1 - ppo_epsilon, 1 + ppo_epsilon)
        # Take minimum of clipped and unclipped objectives
        losses = -torch.min(
            prob_ratios * advantages,
            clipped_ratios * advantages
        )
        metrics['prob_ratios'] = prob_ratios
        metrics['clipped_ratios'] = clipped_ratios

    # Apply Expert Iteration mask if specified
    training_mask = None
    if hyperparameters.get("use_ei", False):
        threshold = calculate_threshold(previous_advantages)
        training_mask = (advantages > threshold).float()
        metrics['ei_threshold'] = threshold
        metrics['ei_mask'] = training_mask

    return losses, training_mask, metrics


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
    task_type: str,
    model_type: str,
    training_methods: dict,
    cot_length: int,
    r: float,
    temperature: float,
    question_length: int,
    target_length: int,
    shrink_cot: Union[bool, int],
    ei_threshold: float,
    gradient_accumulation_steps: int,
    kl_penalty: float = None,
):
    defaults = {
        "task_type": task_type,
        "model_type": model_type,
        "model_learning_rate": 0.0001,
        "num_batches": 10000,
        "normalize_loss": True,
        "shrink_cot": shrink_cot,
        "ei_threshold": ei_threshold,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "kl_penalty": kl_penalty,
        "ppo_epsilon": 0.2 if training_methods.get("use_ppo", False) else None,
    }

    # Task-specific batch sizes and gradient accumulation
    batch_size_defaults = {
        "llama": {
            "gsm8k": 10,
            "wiki_compression": 2,
            "wiki_continuation": 2,
            "arithmetic": 4,
            "arithmetic_negative": 4,
            "default": 6,
        },
        "mistral": {
            "gsm8k": 10,
            "wiki_compression": 1,
            "wiki_continuation": 1,
            "arithmetic": 6,
            "arithmetic_negative": 6,
            "default": 2,
        },
    }

    # Chain of thought length defaults
    cot_length_defaults = {
        "llama": {
            "gsm8k": 60,
            "wiki_compression": 150,
            "wiki_continuation": 150,
            "arithmetic": 120,
            "arithmetic_negative": 120,
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

    defaults["cot_length"] = cot_length

    # Task-specific length parameters
    if task_type in ["wiki_compression", "wiki_continuation"]:
        defaults["question_length"] = question_length
        defaults["target_length"] = target_length

    # Set the discount factor 'r'
    defaults["r"] = r  # Use the provided value

    # Training method specific parameters
    if training_methods.get("use_ppo", False):
        defaults["ppo_epsilon"] = 0.2
    else:
        defaults["ppo_epsilon"] = None

    # Add training method flags
    defaults.update(training_methods)

    # Add temperature to defaults
    defaults["temperature"] = temperature

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


def should_decrease_cot_length(recent_log_probs, threshold=-0.5, window_size=10):
    """
    Check if we should decrease the cot_length based on recent log probabilities.

    Args:
        recent_log_probs: List of recent log probabilities
        threshold: Threshold value for log probabilities (default: -0.5)
        window_size: Number of consecutive values needed above threshold (default: 10)

    Returns:
        bool: True if cot_length should be decreased
    """
    if len(recent_log_probs) < window_size:
        return False

    # Check last window_size values
    recent_window = recent_log_probs[-window_size:]
    return all(prob > threshold for prob in recent_window)


def setup_training_environment(task_type, resume):
    """Set up the results directory and load checkpoints if resuming."""
    results_dir = os.path.join(
        "results", task_type, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(results_dir, exist_ok=True)

    model_save_path = os.path.join(results_dir, "model")
    log_file = os.path.join(results_dir, "log.jsonl")

    if resume:
        latest_checkpoint, latest_log = get_latest_result_and_log(task_type)
        if latest_checkpoint and latest_log:
            checkpoint = torch.load(latest_checkpoint)
            start_batch, hyperparameters = load_training_state(latest_log)
            previous_normalized_rewards, previous_advantages = (
                load_previous_rewards_and_advantages(latest_log)
            )
        else:
            start_batch = 0
    else:
        start_batch = 0
        previous_normalized_rewards = []
        previous_advantages = []
        with open(log_file, "w") as f:
            json.dump(hyperparameters, f)
            f.write("\n")

    return model_save_path, log_file, start_batch, previous_normalized_rewards, previous_advantages


def initialize_model_and_optimizer(model_type, hyperparameters):
    """Initialize the model, frozen model, tokenizer, device, and optimizer."""
    model, frozen_model, tokenizer, device = load_model(model_type)
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=hyperparameters["model_learning_rate"]
    )
    return model, frozen_model, tokenizer, device, model_optimizer


def calculate_mean_kl(q_R_actor_logits, q_R_critic_logits, cot_length):
    """Calculate mean KL divergence between actor and critic distributions."""
    actor_logprobs = q_R_actor_logits[:,-cot_length:,:].log_softmax(dim=-1)
    critic_logprobs = q_R_critic_logits[:,-cot_length:,:].log_softmax(dim=-1)
    return (torch.exp(actor_logprobs) * (actor_logprobs - critic_logprobs)).sum(dim=-1).mean(dim=1)

@dataclass
class BatchData:
    """Holds data for a single training batch"""
    questions: List[str]
    answers: List[str]
    actor_reasoning: List[str]
    critic_reasoning: List[str]
    R_mean_actor_logprobs: torch.Tensor
    R_mean_critic_logprobs: torch.Tensor
    kl: torch.Tensor
    advantages: torch.Tensor
    normalized_rewards: torch.Tensor
    losses: torch.Tensor
    training_mask: Optional[torch.Tensor]
    metrics: Dict[str, Any]

@dataclass
class LogMetrics:
    """Holds metrics for logging"""
    loss: float
    pg_loss: float
    kl_penalty: Optional[float]
    ppo_ratio: Optional[float]
    ppo_clipped_ratio: Optional[float]
    advantage: float
    normalized_reward: float
    gradient_norm: float
    num_active: int
    fraction_active: float
    ei_threshold: Optional[float]
    mean_prev_advantage: Optional[float]
    std_prev_advantage: Optional[float]

    @classmethod
    def from_batch(
        cls,
        batch_data: BatchData,
        grad_norm: float,
        grad_accum_count: int,
        previous_advantages: List[float],
        batch_size: int,
    ):
        """Create LogMetrics from batch data and training state"""
        num_active = batch_data.training_mask.sum().item() if batch_data.training_mask is not None else len(batch_data.losses)
        
        return cls(
            loss=batch_data.losses.mean().item(),
            pg_loss=batch_data.metrics['pg_losses'][0].item(),
            kl_penalty=batch_data.metrics.get('weighted_kl', [0])[0].item() if batch_data.metrics.get('weighted_kl') is not None else None,
            ppo_ratio=batch_data.metrics.get('prob_ratios', [0])[0].item() if batch_data.metrics.get('prob_ratios') is not None else None,
            ppo_clipped_ratio=batch_data.metrics.get('clipped_ratios', [0])[0].item() if batch_data.metrics.get('clipped_ratios') is not None else None,
            advantage=batch_data.advantages[0].item(),
            normalized_reward=batch_data.normalized_rewards[0].item(),
            gradient_norm=grad_norm / grad_accum_count,
            num_active=num_active,
            fraction_active=num_active / batch_size,
            ei_threshold=batch_data.metrics.get('ei_threshold', None),
            mean_prev_advantage=np.mean(previous_advantages) if previous_advantages else None,
            std_prev_advantage=np.std(previous_advantages) if previous_advantages else None,
        )

def log_batch_results(
    state: TrainingState,
    batch_data: BatchData,
    metrics: LogMetrics,
):
    """Log training results for current batch"""
    # Print debug information
    q = batch_data.questions[0]
    a = batch_data.answers[0]
    actor_reasoning_text = batch_data.actor_reasoning[0]
    critic_reasoning_text = batch_data.critic_reasoning[0]

    if state.hyperparameters["task_type"] in ["wiki_compression", "wiki_continuation"]:
        colored_print("Context:", q, Colors.BLUE)
        colored_print("Actor Reasoning:", actor_reasoning_text, Colors.YELLOW)
        colored_print("Critic Reasoning:", critic_reasoning_text, Colors.CYAN)
    else:  # arithmetic or gsm8k
        colored_print("Question:", q, Colors.BLUE)
        colored_print("Actor Reasoning:", actor_reasoning_text, Colors.YELLOW)
        colored_print("Critic Reasoning:", critic_reasoning_text, Colors.CYAN)

    colored_print("Answer:", a, Colors.GREEN)
    colored_print("Advantage:", f"{metrics.advantage:.4f}", Colors.BOLD, inline=True)

    # Create log entry
    log_entry = {
        "Batch Index": state.batch_index,
        "Task Type": state.hyperparameters["task_type"],
        "Example": {
            "Question": q,
            "Actor Reasoning": actor_reasoning_text,
            "Critic Reasoning": critic_reasoning_text,
            "Answer": a,
            "Contains Answer": str(a) in actor_reasoning_text,
        },
        "Training Metrics": {
            "Loss": metrics.loss,
            "Policy Gradient Loss": metrics.pg_loss,
            "KL Penalty": metrics.kl_penalty,
            "PPO Ratio": metrics.ppo_ratio,
            "PPO Clipped Ratio": metrics.ppo_clipped_ratio,
            "Advantage": metrics.advantage,
            "Normalized Reward": metrics.normalized_reward,
            "Gradient Norm": metrics.gradient_norm,
            "Active Samples": {
                "Count": metrics.num_active,
                "Fraction": metrics.fraction_active,
            },
        },
        "EI Metrics": {
            "Use EI": state.hyperparameters["use_ei"],
            "Mean Previous Advantage": metrics.mean_prev_advantage,
            "Std Previous Advantage": metrics.std_prev_advantage,
            "Threshold": metrics.ei_threshold,
        },
        "Hyperparameters": {
            "Batch Size": state.hyperparameters["batch_size"],
            "CoT Length": state.hyperparameters["cot_length"],
            "Temperature": state.hyperparameters["temperature"],
        }
    }

    # Write to log file
    with open(state.log_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

def save_checkpoint(state: TrainingState):
    """Save model checkpoint"""
    colored_print("Checkpoint", f"Saving model at batch {state.batch_index}", Colors.BOLD)
    torch.save(
        {
            "model_state_dict": state.actor_model.state_dict(),
            "optimizer_state_dict": state.actor_optimizer.state_dict(),
            "batch_index": state.batch_index,
            "hyperparameters": state.hyperparameters,
        },
        state.model_save_path,
    )

def process_batch(state: TrainingState, qa_batch: List[Tuple[str, str]]) -> BatchData:
    """Process a single batch of data"""
    questions, answers = zip(*qa_batch)
    
    # Generate reasoning from both models and compute KL
    reasoning_output = generate_reasoning_and_kl(state, questions)
    
    # Calculate advantages
    advantage_output = calculate_advantages(
        state,
        questions,
        answers,
        reasoning_output,
    )
    
    # Calculate losses
    losses, training_mask, metrics = calculate_losses(
        reasoning_output.kl,
        reasoning_output.R_mean_actor_logprobs,
        reasoning_output.R_mean_critic_logprobs,
        advantage_output.advantages,
        state.previous_advantages,
        state.hyperparameters,
    )
    
    return BatchData(
        questions=questions,
        answers=answers,
        actor_reasoning=reasoning_output.actor_reasoning,
        critic_reasoning=reasoning_output.critic_reasoning,
        R_mean_actor_logprobs=reasoning_output.R_mean_actor_logprobs,
        R_mean_critic_logprobs=reasoning_output.R_mean_critic_logprobs,
        kl=reasoning_output.kl,
        advantages=advantage_output.advantages,
        normalized_rewards=advantage_output.normalized_rewards,
        losses=losses,
        training_mask=training_mask,
        metrics=metrics
    )

def update_model(state: TrainingState, batch_data: BatchData) -> float:
    """Perform model update and return gradient norm"""
    num_active = batch_data.training_mask.sum().item() if batch_data.training_mask is not None else len(batch_data.losses)
    state.grad_accum_count += num_active
    
    if num_active > 0:
        loss = (batch_data.losses * (batch_data.training_mask if batch_data.training_mask is not None else 1.0)).sum()
        loss.backward()
    
    grad_norm = get_grad_norm(state.actor_model.parameters())
    
    if state.grad_accum_count >= state.hyperparameters["gradient_accumulation_steps"]:
        for p in state.actor_model.parameters():
            if p.grad is not None:
                p.grad.data.div_(state.grad_accum_count)
        
        clip_grad_norm_(state.actor_model.parameters(), 1.0)
        state.actor_optimizer.step()
        state.actor_optimizer.zero_grad()
        state.grad_accum_count = 0
    
    return grad_norm

def train(task_type: str, resume: bool, model_type: str, hyperparameters: dict):
    """Main training loop"""
    state = TrainingState.initialize(task_type, resume, model_type, hyperparameters)
    
    qa_generator = generate_question_answer_batches(
        num_batches=hyperparameters["num_batches"],
        batch_size=hyperparameters["batch_size"],
        task_type=task_type,
        tokenizer=state.tokenizer,
        hyperparameters=hyperparameters,
    )
    
    for batch_index in range(state.batch_index, hyperparameters["num_batches"]):
        state.batch_index = batch_index
        print_batch_delimiter()
        colored_print("Batch:", str(batch_index), Colors.BOLD, inline=True)
        
        try:
            qa_batch = next(qa_generator)
        except StopIteration:
            print("Reached end of dataset")
            break
            
        batch_data = process_batch(state, qa_batch)
        grad_norm = update_model(state, batch_data)
        
        # Update history
        state.previous_normalized_rewards.extend(batch_data.normalized_rewards.detach().float().cpu().numpy())
        state.previous_advantages.extend(batch_data.advantages.detach().float().cpu().numpy())
        
        # Log results
        metrics = LogMetrics.from_batch(
            batch_data,
            grad_norm,
            state.grad_accum_count,
            state.previous_advantages,
            state.hyperparameters["batch_size"]
        )
        log_batch_results(state, batch_data, metrics)
        
        # Save checkpoint periodically
        if batch_index % 1000 == 0 and batch_index > 0:
            save_checkpoint(state)


def main(
    task_type: str,
    resume: bool,
    use_ei: bool,
    use_ppo: bool,
    use_pg: bool,
    model_type: str,
    cot_length: int,
    r: float,
    temperature: float,
    question_length: int,
    target_length: int,
    shrink_cot: Union[bool, int],
    ei_threshold: float,
    gradient_accumulation_steps: int,
    kl_penalty: float = None,
):
    """Main entry point with command-line parameter handling."""
    # Get default hyperparameters
    hyperparameters = get_default_hyperparameters(
        task_type=task_type,
        model_type=model_type,
        training_methods={"use_ppo": use_ppo, "use_ei": use_ei, "use_pg": use_pg},
        cot_length=cot_length,
        r=r,
        temperature=temperature,
        question_length=question_length,
        target_length=target_length,
        shrink_cot=shrink_cot,
        ei_threshold=ei_threshold,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kl_penalty=kl_penalty,
    )

    # Validate training method selection
    if not (use_ei or use_pg or use_ppo):
        raise ValueError(
            "At least one of --use_ei, --use_pg, or --use_ppo must be specified."
        )

    # Call the training function
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
        type=float,
        nargs="?",  # Makes the argument optional
        const=None,  # Value if flag is present but no value given
        default=False,  # Value if flag is not present
        help="Use Expert Iteration. Optionally specify a fixed threshold value.",
    )
    parser.add_argument(
        "--use_ppo",
        action="store_true",
        default=False,
        help="Use PPO with clipping",
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
    parser.add_argument(
        "--cot_length",
        type=int,
        default=50,
        help="Chain of thought length (overrides default based on task and model)",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.9,
        help="Discount factor for the exponential weighted average (overrides default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--question_length",
        type=int,
        default=50,
        help="Length of question/context for wiki tasks (default: 200)",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=50,
        help="Length of target/continuation for wiki tasks (default: 200)",
    )
    parser.add_argument(
        "--shrink_cot",
        type=float,
        nargs="?",  # Makes the argument optional
        const=True,  # Value if flag is present but no value given
        default=None,  # Value if flag is not present
        help="Enable CoT length reduction. If number provided, linearly decrease until that batch.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over (default: 8)",
    )
    parser.add_argument(
        "--kl_penalty",
        type=float,
        default=None,
        help="KL penalty coefficient. If specified, adds k*KL to the loss.",
    )

    args = parser.parse_args()

    # Convert use_ei argument to bool/float for main
    use_ei = bool(
        args.use_ei is not False
    )  # True if any value (including None) provided
    ei_threshold = args.use_ei if isinstance(args.use_ei, float) else None

    # Convert use_ppo argument to bool/float for main
    use_ppo = bool(
        args.use_ppo is not False
    )  # True if any value (including None) provided
    ppo_kl_coef = args.use_ppo if isinstance(args.use_ppo, float) else None

    # Convert shrink_cot argument
    shrink_cot = args.shrink_cot
    if isinstance(shrink_cot, float):
        if shrink_cot.is_integer():
            shrink_cot = int(shrink_cot)  # Convert to int if it's a whole number
        else:
            raise ValueError("--shrink_cot value must be a whole number if provided")

    main(
        task_type=args.task_type,
        resume=args.resume,
        use_ei=use_ei,
        use_ppo=use_ppo,
        use_pg=args.use_pg,
        model_type=args.model_type,
        cot_length=args.cot_length,
        r=args.r,
        temperature=args.temperature,
        question_length=args.question_length,
        target_length=args.target_length,
        shrink_cot=shrink_cot,
        ei_threshold=ei_threshold,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kl_penalty=args.kl_penalty,
    )
