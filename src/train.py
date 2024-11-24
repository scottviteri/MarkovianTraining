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
from constants import MISTRAL_INST_START, MISTRAL_INST_END, EI_SKIP_INITIAL
from constants import EI_SKIP_INITIAL
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm


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


def calculate_threshold(previous_advantages, ei_std_multiplier):
    """
    Calculate threshold for expert iteration.

    Args:
        previous_advantages: List of previous advantage values
        ei_std_multiplier: Number of standard deviations above mean for threshold

    Returns:
        float: Threshold value (inf if not enough previous advantages)
    """
    if len(previous_advantages) <= EI_SKIP_INITIAL:
        return float("inf")

    return np.mean(previous_advantages) + ei_std_multiplier * np.std(
        previous_advantages
    )


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
    question: str, hyperparameters: Dict[str, Any], reasoning: Optional[str] = None
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
            f"You will need to predict the next {hyperparameters['target_length']} tokens which follow the provided passage."
            f"You can write {hyperparameters['cot_length']} thinking tokens which will be your sole context for prediction."
            f"Feel free to be creative in your thinking strategy!\n\nOpening text:"
        )
        prompt_type = "Helpful Text:"
    elif task_type == "arithmetic":
        base_prompt = f"You will be given an arithmetic problem, which you have {hyperparameters['cot_length']} tokens to work through step-by-step. Question:"
        prompt_type = "Reasoning:"
    elif task_type == "gsm8k":
        base_prompt = f"You will be given a reasoning problem, which you have {hyperparameters['cot_length']} tokens to work through step-by-step. Question:"
        prompt_type = "Reasoning:"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    # Construct initial prompt with model-specific tokens
    if reasoning is None:
        return f"{tokens['inst_start']} {base_prompt} {question} {tokens['inst_end']}\n{prompt_type}"

    base_with_type = f"{tokens['inst_start']} {base_prompt} <Redacted> {tokens['inst_end']}\n{prompt_type}"

    # Add model-specific answer header to partial prompt
    return base_with_type + reasoning + f" Answer: "


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
    chunk_size: int = 5000,
):
    """Generate batches of Q&A pairs lazily."""
    if task_type == "gsm8k":
        # Load GSM8K dataset
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")["train"]
        dataset_size = len(dataset)
        
        # Create indices for all batches
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)
        
        # Yield batches
        for batch_start in range(0, num_batches * batch_size, batch_size):
            # Wrap around if we reach the end of the dataset
            batch_indices = [all_indices[i % dataset_size] for i in range(batch_start, batch_start + batch_size)]
            
            # Get batch of examples
            batch = []
            for idx in batch_indices:
                example = dataset[idx]
                question = example["question"]
                answer = example["answer"]
                batch.append((question, answer))
            
            yield batch
            
    elif task_type in ["wiki_compression", "wiki_continuation"]:
        print("Loading Wikipedia dataset...")
        wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
        article_idx = 0
        articles_examined = 0
        qa_pairs = []
        
        pbar = tqdm(total=chunk_size, desc="Collecting examples")
        last_qa_pairs_len = 0

        while len(qa_pairs) < chunk_size:
            if article_idx >= len(wiki_dataset):
                print("\nReached end of dataset!")
                break
            
            article = wiki_dataset[article_idx]
            article_idx += 1
            articles_examined += 1
            
            text = article['text']
            tokens = tokenizer(text, truncation=False, return_tensors="pt")
            token_length = tokens.input_ids.size(1)
            
            # Calculate required total length based on task type
            if "question_length" in hyperparameters and "target_length" in hyperparameters:
                required_length = hyperparameters["question_length"] + hyperparameters["target_length"]
            else:
                required_length = hyperparameters.get("target_length", 0)
            
            if token_length < required_length:
                continue
            
            if "question_length" in hyperparameters and "target_length" in hyperparameters:
                # Get question chunk
                question_chunk, actual_q_tokens = get_text_with_token_length(
                    text, 
                    hyperparameters["question_length"], 
                    tokenizer
                )
                
                if question_chunk is None:
                    continue
                
                # Get remaining text after question chunk
                remaining_text = text[len(question_chunk):]
                
                # Get target chunk from remaining text
                target_chunk, actual_t_tokens = get_text_with_token_length(
                    remaining_text,
                    hyperparameters["target_length"],
                    tokenizer
                )
                
                if target_chunk is None:
                    continue
                    
                qa_pairs.append((question_chunk, target_chunk))
                
            else:
                # Single chunk mode (for base model analysis)
                text_chunk, actual_tokens = get_text_with_token_length(
                    text, 
                    hyperparameters["target_length"], 
                    tokenizer
                )
                
                if text_chunk is None:
                    continue
                    
                qa_pairs.append((text_chunk, ""))

            # Update progress bar only when we've added new pairs
            new_pairs = len(qa_pairs) - last_qa_pairs_len
            if new_pairs > 0:
                pbar.update(new_pairs)
                last_qa_pairs_len = len(qa_pairs)

        pbar.close()
        print(f"\nFinished collecting examples. "
              f"Examined {articles_examined} articles to find {len(qa_pairs)} valid examples.")

        # Yield batches from collected pairs
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            yield batch


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
        q_r_tokens = tokenizer(partial_prompts, padding=True, return_tensors="pt").to(
            device
        )

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
            q_r_a_generated[:, -max_answer_length - 1 :], skip_special_tokens=True
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
    actor_answer_logprobs: torch.Tensor
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
    def initialize(
        cls, task_type: str, resume: bool, model_type: str, hyperparameters: dict
    ):
        """Factory method to create a new TrainingState"""
        (
            model_save_path,
            log_file,
            start_batch,
            prev_rewards,
            prev_advantages,
            updated_hyperparameters,  # Receive the updated hyperparameters
        ) = setup_training_environment(task_type, resume, hyperparameters)

        actor_model, critic_model, tokenizer, device, actor_optimizer = (
            initialize_model_and_optimizer(
                model_type,
                updated_hyperparameters,
                checkpoint_path=model_save_path if resume else None,
            )
        )
        critic_model.generation_config.temperature = None
        critic_model.generation_config.top_p = None

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
            hyperparameters=updated_hyperparameters,  # Use the updated hyperparameters
        )


def generate_reasoning_and_kl(
    state: TrainingState, questions: List[str]
) -> ReasoningOutput:
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
            do_sample=False,
            pad_token_id=state.tokenizer.pad_token_id,
        )

    # Get logits from both models on actor's reasoning
    q_R_actor_logits = (
        state.actor_model(q_R_tokens).logits / state.hyperparameters["temperature"]
    )
    q_R_critic_logits = (
        state.critic_model(q_R_tokens).logits / state.hyperparameters["temperature"]
    )

    # Calculate log probabilities and KL
    R_actor_logprobs = q_R_actor_logits[
        :, -state.hyperparameters["cot_length"] - 1 : -1, :
    ].log_softmax(dim=-1)
    R_critic_logprobs = q_R_critic_logits[
        :, -state.hyperparameters["cot_length"] - 1 : -1, :
    ].log_softmax(dim=-1)

    R_mean_actor_logprobs = (
        R_actor_logprobs.gather(
            2, q_R_tokens[:, -state.hyperparameters["cot_length"] :].unsqueeze(-1)
        )
        .squeeze(-1)
        .mean(dim=1)
    )

    R_mean_critic_logprobs = (
        R_critic_logprobs.gather(
            2, q_R_tokens[:, -state.hyperparameters["cot_length"] :].unsqueeze(-1)
        )
        .squeeze(-1)
        .mean(dim=1)
    )

    kl = calculate_mean_kl(
        q_R_actor_logits, q_R_critic_logits, state.hyperparameters["cot_length"]
    )

    # Decode reasoning text
    actor_reasoning = state.tokenizer.batch_decode(
        q_R_tokens[:, -state.hyperparameters["cot_length"] :], skip_special_tokens=True
    )
    critic_reasoning = state.tokenizer.batch_decode(
        q_r_tokens[:, -state.hyperparameters["cot_length"] :], skip_special_tokens=True
    )

    return ReasoningOutput(
        actor_reasoning=actor_reasoning,
        critic_reasoning=critic_reasoning,
        R_mean_actor_logprobs=R_mean_actor_logprobs,
        R_mean_critic_logprobs=R_mean_critic_logprobs,
        kl=kl,
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
        critic_answer_logprobs, _ = calculate_answer_log_probs(
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
        actor_answer_logprobs=actor_answer_logprobs,
        extracted_answers=extracted_answers,
    )


def calculate_losses(
    kl,
    R_mean_actor_logprobs,
    R_mean_critic_logprobs,
    advantages,
    normalized_rewards,
    previous_advantages,
    previous_normalized_rewards,
    hyperparameters,
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
    metrics["pg_losses"] = pg_losses
    losses = pg_losses

    # Add KL penalty if specified
    weighted_kl = None
    if kl_penalty is not None:
        weighted_kl = kl_penalty * kl
        losses = losses + weighted_kl
        metrics["weighted_kl"] = weighted_kl

    # Apply PPO if specified
    prob_ratios = torch.exp(R_mean_actor_logprobs - R_mean_critic_logprobs)
    clipped_ratios = torch.clamp(prob_ratios, 1 - ppo_epsilon, 1 + ppo_epsilon)
    metrics["prob_ratios"] = prob_ratios
    metrics["clipped_ratios"] = clipped_ratios
    if use_ppo:
        losses = -torch.min(prob_ratios * advantages, clipped_ratios * advantages)
    # Apply Expert Iteration mask if specified
    training_mask = None
    if hyperparameters["use_ei"] is not None:
        threshold = calculate_threshold(
            previous_normalized_rewards, hyperparameters["use_ei"]
        )
        training_mask = (normalized_rewards > threshold).float()
        metrics["ei_threshold"] = threshold
        metrics["ei_mask"] = training_mask

    return (
        -R_mean_actor_logprobs if hyperparameters["flatten"] else losses,
        training_mask,
        metrics,
    )


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

    return last_batch_index + 1, hyperparameters


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


def setup_training_environment(task_type, resume, hyperparameters):
    """Set up the results directory and load checkpoints if resuming."""
    if resume:
        latest_checkpoint_path, latest_log_path = get_latest_result_and_log(task_type)
        if latest_checkpoint_path is None or latest_log_path is None:
            raise ValueError(f"No previous run found for task type: {task_type}")

        start_batch, hyperparameters = load_training_state(latest_log_path)
        previous_normalized_rewards, previous_advantages = (
            load_previous_rewards_and_advantages(latest_log_path)
        )
        # Use the same directory as the loaded checkpoint
        results_dir = os.path.dirname(latest_checkpoint_path)
        model_save_path = latest_checkpoint_path
        log_file = latest_log_path
    else:
        results_dir = os.path.join(
            "results", task_type, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(results_dir, exist_ok=True)
        model_save_path = os.path.join(results_dir, "model")
        log_file = os.path.join(results_dir, "log.jsonl")
        start_batch = 0
        previous_normalized_rewards = []
        previous_advantages = []
        with open(log_file, "w") as f:
            json.dump(hyperparameters, f)
            f.write("\n")

    return (
        model_save_path,
        log_file,
        start_batch,
        previous_normalized_rewards,
        previous_advantages,
        hyperparameters,  # Now returning the potentially updated hyperparameters
    )


def initialize_model_and_optimizer(model_type, hyperparameters, checkpoint_path=None):
    """Initialize the model, frozen model, tokenizer, device, and optimizer."""
    model, frozen_model, tokenizer, device = load_model(model_type)
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=hyperparameters["lr"]
    )

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, frozen_model, tokenizer, device, model_optimizer


def calculate_mean_kl(q_R_actor_logits, q_R_critic_logits, cot_length):
    """Calculate mean KL divergence between actor and critic distributions."""
    actor_logprobs = q_R_actor_logits[:, -cot_length:, :].log_softmax(dim=-1)
    critic_logprobs = q_R_critic_logits[:, -cot_length:, :].log_softmax(dim=-1)
    return (
        (torch.exp(actor_logprobs) * (actor_logprobs - critic_logprobs))
        .sum(dim=-1)
        .mean(dim=1)
    )


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
    actor_answer_logprobs: (
        torch.Tensor
    )  # Changed from advantage_output to just the tensor we need
    losses: torch.Tensor
    training_mask: Optional[torch.Tensor]
    metrics: Dict[str, Any]


@dataclass
class LogMetrics:
    """Holds metrics for logging"""

    loss: float
    pg_loss: float
    actor_logprobs: float
    actor_answer_logprobs: float
    critic_answer_logprobs: float
    kl: float
    weighted_kl: Optional[float]
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
        num_active = (
            batch_data.training_mask.sum().item()
            if batch_data.training_mask is not None
            else len(batch_data.losses)
        )

        # Get PPO metrics
        ppo_ratio = batch_data.metrics.get("prob_ratios", [None])[0]
        ppo_clipped_ratio = batch_data.metrics.get("clipped_ratios", [None])[0]

        # Convert to float if they exist
        ppo_ratio = float(ppo_ratio.item()) if ppo_ratio is not None else None
        ppo_clipped_ratio = (
            float(ppo_clipped_ratio.item()) if ppo_clipped_ratio is not None else None
        )

        # Get KL values
        raw_kl = batch_data.kl[0].item()
        weighted_kl = batch_data.metrics.get("weighted_kl", [None])[0]
        weighted_kl = float(weighted_kl.item()) if weighted_kl is not None else None

        return cls(
            loss=batch_data.losses.mean().item(),
            pg_loss=batch_data.metrics["pg_losses"][0].item(),
            actor_logprobs=batch_data.R_mean_actor_logprobs[0].item(),
            actor_answer_logprobs=batch_data.actor_answer_logprobs[
                0
            ].item(),  # Updated this line
            critic_answer_logprobs=batch_data.critic_answer_logprobs[
                0
            ].item(),  # Updated this line
            kl=raw_kl,
            weighted_kl=weighted_kl,
            ppo_ratio=ppo_ratio,
            ppo_clipped_ratio=ppo_clipped_ratio,
            advantage=batch_data.advantages[0].item(),
            normalized_reward=batch_data.normalized_rewards[0].item(),
            gradient_norm=grad_norm,
            num_active=num_active,
            fraction_active=num_active / batch_size,
            ei_threshold=batch_data.metrics.get("ei_threshold", None),
            mean_prev_advantage=(
                np.mean(previous_advantages) if previous_advantages else None
            ),
            std_prev_advantage=(
                np.std(previous_advantages) if previous_advantages else None
            ),
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

    # Calculate fraction of answers contained in reasoning across batch
    contains_answer_fraction = sum(
        answer in reasoning
        for answer, reasoning in zip(batch_data.answers, batch_data.actor_reasoning)
    ) / len(batch_data.answers)

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

    # Determine which KL value to log
    kl_to_log = metrics.weighted_kl if metrics.weighted_kl is not None else metrics.kl
    kl_label = "Weighted KL" if metrics.weighted_kl is not None else "KL"

    log_entry = {
        "Batch Index": int(state.batch_index),
        "Task Type": state.hyperparameters["task_type"],
        "Example": {
            "Question": q,
            "Actor Reasoning": actor_reasoning_text,
            "Critic Reasoning": critic_reasoning_text,
            "Answer": a,
            "Contains Answer": contains_answer_fraction,
        },
        "Training Metrics": {
            "Loss": float(metrics.loss),
            "Policy Gradient Loss": float(metrics.pg_loss),
            "Actor Log Probs": float(metrics.actor_logprobs),
            "Actor Answer Log Probs": float(metrics.actor_answer_logprobs),
            "Critic Answer Log Probs": float(metrics.critic_answer_logprobs),
            "KL": float(kl_to_log),
            "KL Type": kl_label,
            "PPO Ratio": (
                float(metrics.ppo_ratio) if metrics.ppo_ratio is not None else None
            ),
            "PPO Clipped Ratio": (
                float(metrics.ppo_clipped_ratio)
                if metrics.ppo_clipped_ratio is not None
                else None
            ),
            "Advantage": float(metrics.advantage),
            "Normalized Reward": float(metrics.normalized_reward),
            "Gradient Norm": float(metrics.gradient_norm),
            "Active Samples": {
                "Count": int(metrics.num_active),
                "Fraction": float(metrics.fraction_active),
            },
        },
        "EI Metrics": {
            "Use EI": (
                float(state.hyperparameters["use_ei"])
                if state.hyperparameters["use_ei"] is not None
                else None
            ),
            "Mean Previous Advantage": (
                float(metrics.mean_prev_advantage)
                if metrics.mean_prev_advantage is not None
                else None
            ),
            "Std Previous Advantage": (
                float(metrics.std_prev_advantage)
                if metrics.std_prev_advantage is not None
                else None
            ),
            "Threshold": (
                float(metrics.ei_threshold)
                if metrics.ei_threshold is not None
                else None
            ),
        },
        "Hyperparameters": {
            "Batch Size": int(state.hyperparameters["batch_size"]),
            "CoT Length": int(state.hyperparameters["cot_length"]),
            "Temperature": float(state.hyperparameters["temperature"]),
        },
    }

    # Write to log file
    with open(state.log_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")


def save_checkpoint(state: TrainingState):
    """Save model checkpoint"""
    colored_print(
        "Checkpoint", f"Saving model at batch {state.batch_index}", Colors.BOLD
    )
    torch.save(
        {
            "model_state_dict": state.actor_model.state_dict(),
            "optimizer_state_dict": state.actor_optimizer.state_dict(),
            "batch_index": state.batch_index,
            "hyperparameters": state.hyperparameters,
        },
        state.model_save_path + ".pt"
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
        advantage_output.normalized_rewards,
        state.previous_advantages,
        state.previous_normalized_rewards,
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
        actor_answer_logprobs=advantage_output.actor_answer_logprobs,  # Just pass the tensor
        losses=losses,
        training_mask=training_mask,
        metrics=metrics,
    )


def update_model(state: TrainingState, batch_data: BatchData) -> float:
    """Perform model update and return gradient norm"""
    num_active = (
        batch_data.training_mask.sum().item()
        if batch_data.training_mask is not None
        else len(batch_data.losses)
    )
    state.grad_accum_count += num_active

    if num_active > 0:
        loss = (
            batch_data.losses
            * (
                batch_data.training_mask
                if batch_data.training_mask is not None
                else 1.0
            )
        ).sum()
        loss.backward()

    if state.grad_accum_count > 0:
        grad_norm = (
            get_grad_norm(state.actor_model.parameters()) / state.grad_accum_count
        )
    else:
        grad_norm = 0

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

    # Get dataset size for tracking full passes
    if task_type == "gsm8k":
        dataset_size = len(load_dataset("openai/gsm8k", "main")["train"])
        checkpoint_frequency = 500
    else:
        dataset_size = float('inf')  # For generated datasets
        checkpoint_frequency = 1000
    
    batches_per_epoch = dataset_size // hyperparameters["batch_size"]
    completed_epochs = 0

    qa_generator = generate_question_answer_batches(
        num_batches=hyperparameters["num_batches"],
        batch_size=hyperparameters["batch_size"],
        task_type=task_type,
        tokenizer=state.tokenizer,
        hyperparameters=hyperparameters,
    )

    try:
        for batch_index in range(state.batch_index, hyperparameters["num_batches"]):
            state.batch_index = batch_index
            print_batch_delimiter()
            colored_print("Batch:", str(batch_index), Colors.BOLD, inline=True)

            try:
                qa_batch = next(qa_generator)
            except StopIteration:
                # Reset generator if we need more batches
                if batch_index < hyperparameters["num_batches"] - 1:
                    qa_generator = generate_question_answer_batches(
                        num_batches=hyperparameters["num_batches"] - batch_index,
                        batch_size=hyperparameters["batch_size"],
                        task_type=task_type,
                        tokenizer=state.tokenizer,
                        hyperparameters=hyperparameters,
                    )
                    qa_batch = next(qa_generator)
                    completed_epochs += 1
                    print(f"\nCompleted epoch {completed_epochs}, restarting dataset")
                else:
                    print("\nReached end of training")
                    save_checkpoint(state)
                    break

            batch_data = process_batch(state, qa_batch)
            grad_norm = update_model(state, batch_data)

            # Update history and log as before
            state.previous_normalized_rewards.extend(
                batch_data.normalized_rewards.detach().float().cpu().numpy()
            )
            state.previous_advantages.extend(
                batch_data.advantages.detach().float().cpu().numpy()
            )

            metrics = LogMetrics.from_batch(
                batch_data,
                grad_norm,
                state.grad_accum_count,
                state.previous_advantages,
                state.hyperparameters["batch_size"],
                critic_answer_logprobs=batch_data.R_mean_critic_logprobs.mean().item(),
            )
            log_batch_results(state, batch_data, metrics)

            # Save checkpoint periodically regardless of epochs
            if batch_index % checkpoint_frequency == 0 and batch_index > 0:
                if task_type == "gsm8k":
                    # Save with unique timestamp for GSM8K
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = f"{state.model_save_path}_{batch_index}_{timestamp}.pt"
                else:
                    checkpoint_path = f"{state.model_save_path}.pt"
                
                colored_print(
                    "Checkpoint", f"Saving model at batch {state.batch_index}", Colors.BOLD
                )
                torch.save(
                    {
                        "model_state_dict": state.actor_model.state_dict(),
                        "optimizer_state_dict": state.actor_optimizer.state_dict(),
                        "batch_index": state.batch_index,
                        "hyperparameters": state.hyperparameters,
                    },
                    checkpoint_path
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if completed_epochs > 0:
            print("Saving checkpoint as at least one epoch was completed")
            save_checkpoint(state)
        else:
            print("No checkpoint saved (no full epochs completed)")
        return


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


@dataclass
class TrainingConfig:
    """Configuration for training run"""

    task_type: str
    model_type: str
    resume: bool
    use_ei: float
    use_ppo: bool
    cot_length: int
    r: float
    temperature: float
    question_length: int
    target_length: int
    shrink_cot: Union[bool, int, None]
    gradient_accumulation_steps: int
    kl_penalty: Optional[float]
    batch_size: int
    normalize_loss: bool
    flatten: bool
    lr: float
    num_batches: int
    ppo_epsilon: float

    @classmethod
    def from_args(cls, args):
        """Create config from parsed command line arguments"""
        # Convert shrink_cot argument
        shrink_cot = args.shrink_cot
        if isinstance(shrink_cot, float):
            if shrink_cot.is_integer():
                shrink_cot = int(shrink_cot)
            else:
                raise ValueError(
                    "--shrink_cot value must be a whole number if provided"
                )

        # Create config with all arguments
        return cls(
            task_type=args.task_type,
            model_type=args.model_type,
            resume=args.resume,
            use_ei=args.use_ei,
            use_ppo=args.use_ppo,
            cot_length=args.cot_length,
            r=args.r,
            temperature=args.temperature,
            question_length=args.question_length,
            target_length=args.target_length,
            shrink_cot=shrink_cot,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kl_penalty=args.kl_penalty,
            batch_size=args.batch_size,
            normalize_loss=args.normalize_loss,
            flatten=args.flatten,
            lr=args.lr,
            num_batches=args.num_batches,
            ppo_epsilon=args.ppo_epsilon,
        )


def main(config: TrainingConfig):
    """Main entry point with configuration object"""
    train(
        task_type=config.task_type,
        resume=config.resume,
        model_type=config.model_type,
        hyperparameters=asdict(config),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model on various tasks.")

    # Arguments with defaults
    parser.add_argument(
        "--task_type",
        type=str,
        default="wiki_continuation",
        choices=[
            "arithmetic",
            "arithmetic_negative",
            "gsm8k",
            "wiki_compression",
            "wiki_continuation",
        ],
        help="Task type (default: wiki_continuation)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["llama", "mistral"],
        help="Model type (default: llama)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use_ei",
        type=float,
        default=None,
        help="Use Expert Iteration with specified number of standard deviations",
    )
    parser.add_argument("--use_ppo", action="store_true")
    parser.add_argument("--cot_length", type=int, default=50)
    parser.add_argument("--r", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--question_length", type=int, default=50)
    parser.add_argument("--target_length", type=int, default=50)
    parser.add_argument("--shrink_cot", type=float, nargs="?", const=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--kl_penalty", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--normalize_loss", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_batches", type=int, default=100000)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)

    args = parser.parse_args()
    config = TrainingConfig.from_args(args)
    main(config)
