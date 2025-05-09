import os
from typing import Dict, Any, Optional
from peft import LoraConfig, get_peft_model, PeftModel
from constants import (
    MISTRAL_INST_START, 
    MISTRAL_INST_END, 
    PHI4_IM_START, 
    PHI4_IM_SEP, 
    PHI4_IM_END,
    GEMMA3_BOS,
    GEMMA3_START_OF_TURN,
    GEMMA3_END_OF_TURN
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import hashlib
import re
import glob
import numpy as np
from constants import EI_SKIP_INITIAL
from datasets import load_dataset
from tqdm import tqdm

class Colors:
    """ANSI color codes"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    END = "\033[0m"


def colored_print(
    label: str, text: str, color: str = Colors.BLUE, inline: bool = False
):
    """Print text with colored label, optionally on same line."""
    if inline:
        print(f"\n{color}{label}{Colors.END} {text}", end="")
    else:
        print(f"\n{color}{label}{Colors.END}")
        print(repr(text))

def find_latest_result():
    """
    Find the most recent result directory across all tasks and model types.

    Returns:
        str: Path to the most recent result directory, or None if no results found
    """
    results_dir = "results"

    # Collect all result directories with their timestamps
    result_dirs = []

    # Walk through the results directory
    for task_dir in os.listdir(results_dir):
        task_path = os.path.join(results_dir, task_dir)
        if os.path.isdir(task_path):
            for timestamp_dir in os.listdir(task_path):
                full_timestamp_path = os.path.join(task_path, timestamp_dir)
                if os.path.isdir(full_timestamp_path):
                    result_dirs.append(
                        (
                            os.path.getmtime(full_timestamp_path),
                            full_timestamp_path,
                        )
                    )

    # Sort by timestamp, most recent first
    if result_dirs:
        return sorted(result_dirs, key=lambda x: x[0], reverse=True)[0][1]

    return None

def find_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint file in a directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_batch_*.pt"))
    if not checkpoint_files:
        # Fall back to the old format
        if os.path.exists(os.path.join(checkpoint_dir, "model.pt")):
            return os.path.join(checkpoint_dir, "model.pt")
        return None
    
    # Sort by batch number (extract from filename)
    return max(checkpoint_files, key=lambda f: int(re.search(r'model_batch_(\d+)\.pt', f).group(1)))


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

def get_model_specific_tokens(model_type):
    """Return model-specific tokens for prompt construction."""
    if model_type == "mistral":
        return {
            "inst_start": MISTRAL_INST_START,
            "inst_end": MISTRAL_INST_END,
            "format_type": "mistral"
        }
    elif model_type == "phi-4":
        return {
            "im_start": PHI4_IM_START,
            "im_sep": PHI4_IM_SEP, 
            "im_end": PHI4_IM_END,
            "format_type": "phi-4"
        }
    elif model_type in ["gemma-3", "gemma-3-small"]:
        return {
            "bos": GEMMA3_BOS,
            "start_of_turn": GEMMA3_START_OF_TURN,
            "end_of_turn": GEMMA3_END_OF_TURN,
            "format_type": "gemma-3"
        }
    else:  # llama, gpt2, tinystories, phi
        return {
            "inst_start": "",
            "inst_end": "",
            "format_type": "standard"
        }

def construct_prompts(
    question: str, hyperparameters: Dict[str, Any], reasoning: Optional[str] = None, include_question: bool = False
) -> str:
    """
    Construct prompt for model input.

    Args:
        question: The input question or text
        hyperparameters: Dictionary containing model and task configuration
        reasoning: Optional reasoning text to include
        include_question: Whether to include the question when reasoning is provided (otherwise uses <Redacted>)

    Returns:
        str: Formatted prompt
    """
    model_type = hyperparameters["model_type"]
    task_type = hyperparameters["task_type"]

    tokens = get_model_specific_tokens(model_type)
    format_type = tokens["format_type"]

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
    if format_type == "phi-4":
        system_msg = "You are a helpful AI assistant that solves problems step by step."
        
        if reasoning is None:
            # Initial prompt without reasoning
            return (
                f"{tokens['im_start']}system{tokens['im_sep']}\n{system_msg}{tokens['im_end']}\n"
                f"{tokens['im_start']}user{tokens['im_sep']}\n{base_prompt} {question}{tokens['im_end']}\n"
                f"{tokens['im_start']}assistant{tokens['im_sep']}\n{prompt_type}"
            )
        else:
            # Prompt with reasoning (for generating/evaluating the answer)
            question_placeholder = question if include_question else "<Redacted>"
            return (
                f"{tokens['im_start']}system{tokens['im_sep']}\n{system_msg}{tokens['im_end']}\n"
                f"{tokens['im_start']}user{tokens['im_sep']}\n{base_prompt} {question_placeholder}{tokens['im_end']}\n"
                f"{tokens['im_start']}assistant{tokens['im_sep']}\n{prompt_type}{reasoning} Answer: "
            )
    elif format_type == "gemma-3":
        if reasoning is None:
            # Initial prompt without reasoning
            return (
                f"{tokens['bos']}{tokens['start_of_turn']}user\n"
                f"{base_prompt} {question}{tokens['end_of_turn']}\n"
                f"{tokens['start_of_turn']}model\n"
                f"{prompt_type}"
            )
        else:
            # Prompt with reasoning (for generating/evaluating the answer)
            question_placeholder = question if include_question else "<Redacted>"
            return (
                f"{tokens['bos']}{tokens['start_of_turn']}user\n"
                f"{base_prompt} {question_placeholder}{tokens['end_of_turn']}\n"
                f"{tokens['start_of_turn']}model\n"
                f"{prompt_type}{reasoning} Answer: "
            )
    elif format_type == "mistral":
        if reasoning is None:
            return f"{tokens['inst_start']} {base_prompt} {question} {tokens['inst_end']}\n{prompt_type}"
        else:
            # Include the actual question or use <Redacted> placeholder
            question_placeholder = question if include_question else "<Redacted>"
            base_with_type = f"{tokens['inst_start']} {base_prompt} {question_placeholder} {tokens['inst_end']}\n{prompt_type}"
            # Add answer header to partial prompt
            return base_with_type + reasoning + f" Answer: "
    else:  # standard format (no special tokens)
        if reasoning is None:
            return f"{base_prompt} {question}\n{prompt_type}"
        else:
            # Include the actual question or use <Redacted> placeholder
            question_placeholder = question if include_question else "<Redacted>"
            base_with_type = f"{base_prompt} {question_placeholder}\n{prompt_type}"
            # Add answer header to partial prompt
            return base_with_type + reasoning + f" Answer: "

def load_model(model_type, hyperparameters=None):
    """Load either Mistral, Llama, GPT2, TinyStories, Phi, Phi-4, or Gemma 3 model based on parameter."""
    if model_type == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_type == "llama":
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_type == "gpt2":
        model_name = "openai-community/gpt2"
    elif model_type == "tinystories":
        model_name = "roneneldan/TinyStories"
    elif model_type == "phi":
        model_name = "microsoft/Phi-3.5-mini-instruct"
    elif model_type == "phi-4":
        model_name = "microsoft/phi-4"
    elif model_type == "gemma-3":
        model_name = "google/gemma-3-12b-it"
    elif model_type == "gemma-3-small":
        model_name = "google/gemma-3-1b-it"
    else:
        raise ValueError("model_type must be either 'mistral', 'llama', 'gpt2', 'tinystories', 'phi', 'phi-4', 'gemma-3', 'gemma-3-small', or 'gemma-3-12b-it'")

    # Common settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trust_remote_code = model_type in ["phi", "phi-4", "gemma-3", "gemma-3-small"]
    
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load actor model with LoRA for training
    colored_print("Loading Model", f"Loading {model_name} for {model_type}", Colors.BOLD)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )
    
    # Apply any model-specific patches
    model = apply_model_specific_patches(model, model_type)
    
    colored_print("Model Info", f"Base model loaded: {type(model).__name__}", Colors.BLUE)
    
    # Print base model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    colored_print("Model Params", f"Before LoRA: Total: {total_params:,}, Trainable: {trainable_params:,}", Colors.BLUE)

    # Create LoRA config for actor model
    # Get LoRA parameters from hyperparameters
    lora_rank = hyperparameters.get("lora_rank", 8) if hyperparameters else 8
    lora_alpha = hyperparameters.get("lora_alpha", 16) if hyperparameters else 16
    colored_print("LoRA Config", f"Using rank={lora_rank}, alpha={lora_alpha}", Colors.CYAN)
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    
    # Use our improved PEFT model initialization 
    model = create_peft_model_with_adapter(model, peft_config)
    
    # Load critic model separately (no LoRA needed)
    # This avoids OOM from deepcopy while keeping the model architecture intact
    frozen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )
    
    # Apply same patches to critic model
    frozen_model = apply_model_specific_patches(frozen_model, model_type)

    # Ensure all parameters are frozen in critic model
    for param in frozen_model.parameters():
        param.requires_grad = False

    # Move models to device
    model.to(device)
    frozen_model.to(device)

    return model, frozen_model, tokenizer, device


def print_batch_delimiter():
    """Print a delimiter between training batches."""
    print("\n" + "=" * 80 + "\n")


def print_grpo_overview(hyperparameters):
    """Print an overview of GRPO (Group-Relative Policy Optimization) settings when enabled."""
    parallel_samples = hyperparameters.get("parallel_samples", 1)
    
    if parallel_samples <= 1:
        return  # No overview needed for standard mode
    
    colored_print("GRPO Overview", f"Parallel sampling is enabled with {parallel_samples} samples per question", Colors.BOLD)
    colored_print("GRPO Mode", "Using group means as baselines for advantage calculation", Colors.CYAN)
    colored_print("GRPO Info", "This implements a form of Group-Relative Policy Optimization", Colors.CYAN)
    colored_print("GRPO Info", "Each batch element generates multiple reasoning paths in parallel", Colors.CYAN)
    colored_print("GRPO Info", "Advantages are calculated relative to the mean reward of each group", Colors.CYAN)
    print("\n" + "-" * 80 + "\n")


def get_model_hash(model):
    """Get a comprehensive hash of all model parameters and structure.
    
    This creates a single hash that uniquely identifies the entire model state,
    including parameter values and structure.
    
    Args:
        model: The model to hash
        
    Returns:
        str: Hexadecimal hash string
    """
    # Create a new blake2b hash object
    full_hash = hashlib.blake2b()
    
    # Get the model's state dict which contains all parameters and buffers
    state_dict = model.state_dict()
    
    # Sort keys to ensure consistent ordering
    for k in sorted(state_dict.keys()):
        t = state_dict[k]
        # Update hash with parameter name
        full_hash.update(k.encode())
        
        # Ensure tensor is on CPU and convert to numpy
        if t.dtype == torch.bfloat16:
            # Handle bfloat16 by converting to float32 first
            t_numpy = t.cpu().to(torch.float32).numpy()
        else:
            t_numpy = t.cpu().numpy() 
        
        # Update hash with parameter values
        full_hash.update(t_numpy.tobytes())
    
    # Return hash as hex string
    return full_hash.hexdigest()


def calculate_threshold(previous_advantages, ei_std_multiplier, batch_index=None):
    """
    Calculate threshold for expert iteration.

    Args:
        previous_advantages: List of previous advantage values
        ei_std_multiplier: Number of standard deviations above mean for threshold
        batch_index: Current batch index (used to determine if we're in initial skip period)

    Returns:
        float: Threshold value (inf if we're in the initial skip period)
    """
    # If batch_index is provided, use it to determine if we're in initial skip period
    if batch_index is not None and batch_index < EI_SKIP_INITIAL:
        colored_print("EI Threshold", f"In initial skip period (batch {batch_index} < {EI_SKIP_INITIAL}), returning inf", Colors.YELLOW)
        return float("inf")

    # Fall back to previous length-based check if batch_index not provided
    if batch_index is None and len(previous_advantages) <= EI_SKIP_INITIAL:
        colored_print("EI Threshold", f"Not enough previous advantages ({len(previous_advantages)} ≤ {EI_SKIP_INITIAL}), returning inf", Colors.YELLOW)
        return float("inf")

    threshold = np.mean(previous_advantages) + ei_std_multiplier * np.std(previous_advantages)
    colored_print("EI Threshold", f"Calculated threshold: {threshold:.4f} (mean: {np.mean(previous_advantages):.4f}, std: {np.std(previous_advantages):.4f})", Colors.CYAN)
    return threshold


def load_gsm8k_dataset(chunk_size: int = 1000, split: str = "train"):
    """
    Lazily load GSM8K dataset in chunks.
    
    Args:
        chunk_size: Number of examples to yield at a time
        split: Dataset split to use ("train" or "test")
    """
    ds = load_dataset("openai/gsm8k", "main")
    questions = ds[split]["question"]
    answers = list(map(lambda x: x[x.index("####") + 5 :], ds[split]["answer"]))
    qa_pairs = list(zip(questions, answers))

    for i in range(0, len(qa_pairs), chunk_size):
        chunk = qa_pairs[i : i + chunk_size]
        if split == "train":  # Only shuffle training data
            random.shuffle(chunk)
        yield from chunk


def extract_answer(answer):
    """Extract answer from GSM8k-style answer string."""
    if "####" in answer:
        return answer[answer.index("####") + 5:].strip()
    return answer.strip()


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


def get_grad_norm(parameters):
    """Calculate the gradient norm of all parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def is_lora_param(param_name):
    """Check if a parameter name belongs to a LoRA adapter."""
    # Common patterns in LoRA parameter names
    lora_patterns = [
        "lora_A", "lora_B", "lora_embedding", 
        "adapter", "peft", "_adapter"
    ]
    return any(pattern in param_name for pattern in lora_patterns)


def load_mmlu_dataset(chunk_size: int = 1000, split: str = "validation", subject: str = None):
    """
    Load MMLU dataset with optional subject filtering.
    
    Args:
        chunk_size: Number of examples to process at a time
        split: Dataset split ("train", "validation", or "test")
        subject: Specific subject to filter on (None for all subjects)
        
    Returns:
        Iterator yielding (question, answer) pairs
    """
    from datasets import concatenate_datasets
    
    # Load MMLU dataset
    mmlu_data = load_dataset("cais/mmlu", "all", split=split)
    
    # Filter by subject if specified
    if subject is not None:
        mmlu_data = mmlu_data.filter(lambda example: example["subject"] == subject)
    
    # Format questions
    formatted_data = []
    for item in mmlu_data:
        # Extract question components
        question = item["question"]
        choices = [item["choices"][i] for i in range(4)]  # A, B, C, D choices
        
        # Format as multiple choice
        formatted_question = f"{question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
        
        # Get answer (0-indexed in dataset, convert to A, B, C, D)
        answer_idx = item["answer"]
        answer = chr(65 + answer_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        formatted_data.append((formatted_question, answer))
    
    # Shuffle data for training
    if split == "train" or split == "validation":
        random.shuffle(formatted_data)
    
    # Yield data in chunks
    for i in range(0, len(formatted_data), chunk_size):
        yield from formatted_data[i:i+chunk_size]


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


def generate_question_answer_batches(
    num_batches: int,
    batch_size: int,
    task_type: str,
    tokenizer,
    hyperparameters: dict = None,
    chunk_size: int = 500,
):
    """Generate batches of Q&A pairs lazily."""
    # If debug_repeat_datapoint mode is enabled, generate a single batch and repeat it
    if hyperparameters.get("debug_repeat_datapoint", False):
        colored_print("Debug Mode", "Training on the same datapoint repeatedly", Colors.RED)
        
        # Generate a single batch based on task type
        debug_batch = None
        
        if task_type in ["arithmetic", "arithmetic_negative"]:
            debug_batch = generate_arithmetic_pairs(task_type, num_examples=batch_size)
        elif task_type == "gsm8k":
            dataset_iter = load_gsm8k_dataset(chunk_size=chunk_size)
            debug_batch = []
            for _ in range(batch_size):
                try:
                    qa_pair = next(dataset_iter)
                    debug_batch.append(qa_pair)
                except StopIteration:
                    dataset_iter = load_gsm8k_dataset(chunk_size=chunk_size)
                    qa_pair = next(dataset_iter)
                    debug_batch.append(qa_pair)
        elif task_type == "mmlu":
            subject = hyperparameters.get("mmlu_subject", None)
            split = hyperparameters.get("mmlu_split", "validation")
            dataset_iter = load_mmlu_dataset(chunk_size=chunk_size, split=split, subject=subject)
            debug_batch = []
            for _ in range(batch_size):
                try:
                    qa_pair = next(dataset_iter)
                    debug_batch.append(qa_pair)
                except StopIteration:
                    dataset_iter = load_mmlu_dataset(chunk_size=chunk_size, split=split, subject=subject)
                    qa_pair = next(dataset_iter)
                    debug_batch.append(qa_pair)
        elif task_type in ["wiki_compression", "wiki_continuation"]:
            print("Loading Wikipedia dataset...")
            wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
            article_idx = 0
            articles_examined = 0
            qa_pairs = []
            
            # Define indices to skip (from 1900*8 to 2800*8)
            skip_start = 15200  # 1900 * 8
            skip_end = 22400    # 2800 * 8
            print(f"Will skip wiki articles from index {skip_start} to {skip_end}")
            
            # Keep track of total examples used across all batches
            total_examples_used = 0
            # Set target number for progress tracking
            examples_target = batch_size  # Only need one batch for debug mode
            
            # We only need to collect one batch worth of examples
            pbar = tqdm(total=batch_size, desc="Collecting examples for debug mode")
            last_qa_pairs_len = len(qa_pairs)
            
            # Collect enough examples for one batch
            while len(qa_pairs) < batch_size:
                if article_idx >= len(wiki_dataset):
                    print("\nReached end of dataset! Wrapping around to beginning.")
                    article_idx = 0
                
                # Skip indices in the specified range
                if skip_start <= article_idx < skip_end:
                    article_idx = skip_end
                    continue
                    
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
                    
                # Check if we've collected enough examples
                if len(qa_pairs) >= batch_size:
                    break

            pbar.close()
            print(f"\nFinished collecting examples for debug mode. "
                  f"Examined {articles_examined} articles to find {len(qa_pairs)} valid examples.")
            
            # Create the debug batch
            debug_batch = qa_pairs[:batch_size]
            
            # Now yield the same debug batch for all requested batches
            for _ in range(num_batches):
                yield debug_batch
        
        # For non-wiki tasks, check if we have a valid debug_batch and yield it if so
        if debug_batch is not None and task_type not in ["wiki_compression", "wiki_continuation"]:
            print(f"Created debug batch for {task_type}, will use it for all {num_batches} batches")
            for _ in range(num_batches):
                yield debug_batch
        
        # Return immediately after debug batch creation without proceeding to regular data generation
        return
    
    # Regular (non-debug) data generation continues below
    if task_type in ["arithmetic", "arithmetic_negative"]:
        # For arithmetic, generate chunks of data as needed
        for batch_idx in range(num_batches):
            # Generate a new batch of arithmetic problems
            qa_pairs = generate_arithmetic_pairs(task_type, num_examples=batch_size)
            yield qa_pairs
            
    elif task_type == "gsm8k":
        # Use load_gsm8k_dataset directly which already processes answers correctly
        dataset_iter = load_gsm8k_dataset(chunk_size=chunk_size)
        for batch_start in range(0, num_batches * batch_size, batch_size):
            batch = []
            for _ in range(batch_size):
                try:
                    qa_pair = next(dataset_iter)
                    batch.append(qa_pair)
                except StopIteration:
                    # Reset iterator if we run out of data
                    dataset_iter = load_gsm8k_dataset(chunk_size=chunk_size)
                    qa_pair = next(dataset_iter)
                    batch.append(qa_pair)
            yield batch
            
    elif task_type == "mmlu":
        # Get optional subject filter from hyperparameters
        subject = hyperparameters.get("mmlu_subject", None)
        split = hyperparameters.get("mmlu_split", "validation")
        
        # Use the MMLU dataset loader
        dataset_iter = load_mmlu_dataset(chunk_size=chunk_size, split=split, subject=subject)
        
        for batch_start in range(0, num_batches * batch_size, batch_size):
            batch = []
            for _ in range(batch_size):
                try:
                    qa_pair = next(dataset_iter)
                    batch.append(qa_pair)
                except StopIteration:
                    # Reset iterator if we run out of data
                    dataset_iter = load_mmlu_dataset(chunk_size=chunk_size, split=split, subject=subject)
                    qa_pair = next(dataset_iter)
                    batch.append(qa_pair)
            yield batch
            
    elif task_type in ["wiki_compression", "wiki_continuation"]:
        print("Loading Wikipedia dataset...")
        wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train")
        article_idx = 0
        articles_examined = 0
        qa_pairs = []
        
        # Define indices to skip (from 1900*8 to 2800*8)
        skip_start = 15200  # 1900 * 8
        skip_end = 22400    # 2800 * 8
        print(f"Will skip wiki articles from index {skip_start} to {skip_end}")
        
        # Keep track of total examples used across all batches
        total_examples_used = 0
        # Set target number for progress tracking
        examples_target = num_batches * batch_size
        
        for batch_start in range(0, num_batches * batch_size, batch_size):
            # Check if we need to collect more examples
            if len(qa_pairs) < batch_size:
                pbar = tqdm(total=min(chunk_size, examples_target - total_examples_used), 
                           desc=f"Collecting examples (batch {batch_start//batch_size + 1}/{num_batches})")
                last_qa_pairs_len = len(qa_pairs)
                
                # Collect more examples
                while len(qa_pairs) < chunk_size:
                    if article_idx >= len(wiki_dataset):
                        print("\nReached end of dataset! Wrapping around to beginning.")
                        article_idx = 0
                    
                    # Skip indices in the specified range
                    if skip_start <= article_idx < skip_end:
                        article_idx = skip_end
                        continue
                        
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
                        
                    # Check if we've collected enough examples
                    if len(qa_pairs) >= chunk_size:
                        break

                pbar.close()
                print(f"\nFinished collecting examples for batch {batch_start//batch_size + 1}/{num_batches}. "
                      f"Examined {articles_examined} articles to find {len(qa_pairs)} valid examples.")
                
                # Shuffle the collected pairs
                random.shuffle(qa_pairs)
            
            # Extract batch_size examples
            batch = qa_pairs[:batch_size]
            # Remove used examples
            qa_pairs = qa_pairs[batch_size:]
            total_examples_used += len(batch)
            
            yield batch


def get_full_weight_snapshot(model):
    """Create a detailed snapshot of model weights for comparison later.
    
    Args:
        model: The model to snapshot
        
    Returns:
        dict: A dictionary of parameter/buffer information for later comparison
    """
    snapshot = {}
    
    # Capture all parameters
    for name, param in model.named_parameters():
        full_name = "param:" + name
        
        # Get parameter data
        param_data = param.data.detach().cpu()
        
        # Convert BFloat16 tensors to float32 before hashing
        if param_data.dtype == torch.bfloat16:
            param_data_for_hash = param_data.to(torch.float32)
        else:
            param_data_for_hash = param_data
        
        # Calculate hash and stats
        param_hash = hash(param_data_for_hash.numpy().tobytes())
        
        # Convert to float32 for statistics calculation
        param_float = param_data.to(torch.float32)
        
        # Store basic stats and hash
        snapshot[full_name] = {
            'hash': param_hash,
            'mean': float(param_float.mean().item()),
            'std': float(param_float.std().item()),
            'min': float(param_float.min().item()),
            'max': float(param_float.max().item()),
        }
    
    # Capture all buffers
    for name, buffer in model.named_buffers():
        full_name = "buffer:" + name
        
        # Get buffer data
        buffer_data = buffer.data.detach().cpu()
        
        # Convert BFloat16 tensors to float32 before hashing
        if buffer_data.dtype == torch.bfloat16:
            buffer_data_for_hash = buffer_data.to(torch.float32)
        else:
            buffer_data_for_hash = buffer_data
        
        # Calculate hash and stats
        try:
            buffer_hash = hash(buffer_data_for_hash.numpy().tobytes())
            
            # Only calculate stats if buffer is not empty
            if buffer_data.numel() > 0:
                # Convert to float32 for statistics calculation
                buffer_float = buffer_data.to(torch.float32)
                
                # Store basic stats and hash
                snapshot[full_name] = {
                    'hash': buffer_hash,
                    'mean': float(buffer_float.mean().item()),
                    'std': float(buffer_float.std().item()),
                    'min': float(buffer_float.min().item()),
                    'max': float(buffer_float.max().item()),
                }
            else:
                # For empty buffers, just store the hash
                snapshot[full_name] = {
                    'hash': buffer_hash,
                    'empty': True,
                }
        except Exception as e:
            # Some buffers might not be hashable, store info instead
            snapshot[full_name] = {
                'hash': hash(str(buffer_data)),
                'shape': str(buffer_data.shape),
                'dtype': str(buffer_data.dtype),
                'error': str(e),
            }
    
    # Capture module states
    for name, module in model.named_modules():
        # Check training mode
        training_name = "module_training:" + name
        snapshot[training_name] = {
            'value': int(module.training),
            'hash': hash(int(module.training)),
        }
        
        # Check active adapter if exists
        if hasattr(module, 'active_adapter'):
            adapter_name = "module_adapter:" + name
            snapshot[adapter_name] = {
                'value': module.active_adapter,
                'hash': hash(str(module.active_adapter)),
            }
        
        # Check LoRA-specific properties
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check rank
            if hasattr(module, 'r'):
                rank_name = f"lora_rank:{name}"
                snapshot[rank_name] = {
                    'value': module.r,
                    'hash': hash(module.r),
                }
                
            # Check alpha
            if hasattr(module, 'lora_alpha'):
                alpha_name = f"lora_alpha:{name}"
                snapshot[alpha_name] = {
                    'value': module.lora_alpha,
                    'hash': hash(module.lora_alpha),
                }
    
    return snapshot

def verify_all_frozen_weights(model, weight_snapshot, full_check=False):
    """Verify that all weights in the model haven't changed.
    
    Args:
        model: The model to check
        weight_snapshot: Dictionary from get_full_weight_snapshot with initial values
        full_check: If True, compare every value rather than just hash/stats
        
    Returns:
        tuple: (is_unchanged, changed_params, total_changed_values, total_values)
    """
    # If weight_snapshot is a string, it's a hash from get_model_hash
    if isinstance(weight_snapshot, str):
        # Create current hash
        current_hash = get_model_hash(model)
        
        # Compare hashes
        if current_hash == weight_snapshot:
            colored_print("Frozen Check", "Model hash verification successful - all weights unchanged", Colors.GREEN)
            return True, [], 0, 0
        else:
            colored_print("WARNING", "Model hash verification failed! Some weights have changed", Colors.RED)
            return False, ["Model hash mismatch"], 1, 1
    
    # Original parameter-by-parameter comparison for backward compatibility
    changed_params = []
    changed_values = 0
    total_values = 0
    
    # First check all parameters
    for name, param in model.named_parameters():
        full_name = "param:" + name
        if full_name not in weight_snapshot:
            changed_params.append((full_name, 'New parameter not in snapshot'))
            continue
            
        # Get current parameter data
        param_data = param.data.detach().cpu()
        total_values += param_data.numel()
        
        # Convert BFloat16 tensors to float32 before hashing
        if param_data.dtype == torch.bfloat16:
            param_data_for_hash = param_data.to(torch.float32)
        else:
            param_data_for_hash = param_data
        
        # Quick check - compare hash and stats
        current_hash = hash(param_data_for_hash.numpy().tobytes())
        original_hash = weight_snapshot[full_name]['hash']
        
        if current_hash != original_hash:
            # Calculate detailed stats to see what changed
            # Convert to float32 for statistics calculation
            param_float = param_data.to(torch.float32)
            current_stats = {
                'mean': float(param_float.mean().item()),
                'std': float(param_float.std().item()),
                'min': float(param_float.min().item()),
                'max': float(param_float.max().item()),
            }
            
            # Calculate differences in stats
            stats_diff = {
                key: abs(current_stats[key] - weight_snapshot[full_name][key])
                for key in current_stats if key in weight_snapshot[full_name]
            }
            
            # If doing a full check, estimate the number of changed values
            if full_check:
                # This is memory intensive but gives a better estimate of differences
                with torch.no_grad():
                    # Calculate element-wise absolute differences
                    abs_diff = torch.abs(param_float - param_float.mean())
                    # Count values that differ significantly (using a small threshold)
                    diff_count = (abs_diff > 1e-5).sum().item()
                changed_values += diff_count
                change_desc = f"Changed values: {diff_count}/{param_data.numel()} ({diff_count/param_data.numel():.6%})"
            else:
                # Estimate change based on statistics
                change_desc = f"Stats diff: {stats_diff}"
            
            changed_params.append((full_name, change_desc))
    
    # Now check all buffers
    for name, buffer in model.named_buffers():
        full_name = "buffer:" + name
        if full_name not in weight_snapshot:
            changed_params.append((full_name, 'New buffer not in snapshot'))
            continue
            
        # Get current buffer data
        buffer_data = buffer.data.detach().cpu()
        total_values += buffer_data.numel()
        
        # Convert BFloat16 tensors to float32 before hashing
        if buffer_data.dtype == torch.bfloat16:
            buffer_data_for_hash = buffer_data.to(torch.float32)
        else:
            buffer_data_for_hash = buffer_data
        
        # Quick check - compare hash
        try:
            current_hash = hash(buffer_data_for_hash.numpy().tobytes())
            original_hash = weight_snapshot[full_name]['hash']
            
            if current_hash != original_hash:
                # Calculate detailed stats to see what changed
                # Convert to float32 for statistics calculation
                buffer_float = buffer_data.to(torch.float32)
                if buffer_float.numel() > 0:
                    current_stats = {
                        'mean': float(buffer_float.mean().item()),
                        'std': float(buffer_float.std().item()),
                        'min': float(buffer_float.min().item()),
                        'max': float(buffer_float.max().item()),
                    }
                    
                    # Calculate differences in stats
                    stats_diff = {
                        key: abs(current_stats[key] - weight_snapshot[full_name][key])
                        for key in current_stats if key in weight_snapshot[full_name]
                    }
                    change_desc = f"Stats diff: {stats_diff}"
                else:
                    change_desc = "Empty buffer changed"
                
                changed_params.append((full_name, change_desc))
        except Exception as e:
            changed_params.append((full_name, f"Error checking buffer: {str(e)}"))
    
    # Check module states
    for name, module in model.named_modules():
        # Check training mode
        training_name = "module_training:" + name
        if training_name in weight_snapshot:
            current_value = int(module.training)
            if current_value != weight_snapshot[training_name]['value']:
                changed_params.append((
                    training_name, 
                    f"Training state changed: was {bool(weight_snapshot[training_name]['value'])}, now {bool(current_value)}"
                ))
        
        # Check active adapter
        if hasattr(module, 'active_adapter'):
            adapter_name = "module_adapter:" + name
            if adapter_name in weight_snapshot:
                current_hash = hash(str(module.active_adapter))
                if current_hash != weight_snapshot[adapter_name]['hash']:
                    changed_params.append((
                        adapter_name, 
                        f"Adapter changed: was {weight_snapshot[adapter_name]['value']}, now {module.active_adapter}"
                    ))
        
        # Check LoRA-specific properties
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check rank
            if hasattr(module, 'r'):
                rank_name = f"lora_rank:{name}"
                if rank_name in weight_snapshot and module.r != weight_snapshot[rank_name]['value']:
                    changed_params.append((
                        rank_name,
                        f"LoRA rank changed: was {weight_snapshot[rank_name]['value']}, now {module.r}"
                    ))
                    
            # Check alpha
            if hasattr(module, 'lora_alpha'):
                alpha_name = f"lora_alpha:{name}"
                if alpha_name in weight_snapshot and module.lora_alpha != weight_snapshot[alpha_name]['value']:
                    changed_params.append((
                        alpha_name,
                        f"LoRA alpha changed: was {weight_snapshot[alpha_name]['value']}, now {module.lora_alpha}"
                    ))
    
    # If any parameters changed, log detailed info
    if changed_params:
        colored_print("WARNING", f"Found {len(changed_params)} changed parameters/buffers/states!", Colors.RED)
        for name, change_desc in changed_params[:10]:  # Show first 10
            colored_print("Changed", f"{name}: {change_desc}", Colors.RED)
        if len(changed_params) > 10:
            colored_print("Note", f"... and {len(changed_params)-10} more", Colors.RED)
            
        return False, changed_params, changed_values, total_values
    
    colored_print("Frozen Check", "All critic model weights, buffers, and states verified as unchanged", Colors.GREEN)
    return True, [], 0, total_values

def verify_actor_weights_changing_comprehensive(model, full_snapshot):
    """Verify that actor model weights are changing as expected.
    
    This function performs a comprehensive check of the actor model's weights,
    comparing them against a baseline snapshot to verify that:
    1. LoRA weights are changing (they should be since they're being trained)
    2. Non-LoRA weights are NOT changing (they should be frozen by PEFT)
    
    Args:
        model: The actor model to verify
        full_snapshot: Full model hash from the start of training
    """
    colored_print("Weight Verification", "Verifying actor model weights are changing properly", Colors.BLUE)
    
    # Skip if no snapshot provided
    if full_snapshot is None:
        colored_print("Weight Verification", "No snapshot provided, skipping verification", Colors.YELLOW)
        return
    
    # Get current model state
    current_hash = get_model_hash(model)
    
    if current_hash == full_snapshot:
        colored_print("Actor Weight Check", "ERROR: Actor weights did not change at all from initial snapshot!", Colors.RED)
        colored_print("Debug Info", f"Initial hash: {full_snapshot[:16]}...", Colors.RED)
        colored_print("Debug Info", f"Current hash: {current_hash[:16]}...", Colors.RED)
        raise ValueError("Actor weights did not change during training. This indicates a serious error in the training loop.")
    else:
        colored_print("Actor Weight Check", "Actor model hash has changed since initial snapshot (good)", Colors.GREEN)
    
    # Detailed weight change analysis by parameter type
    state_dict = model.state_dict()
    
    # Separate weights into LoRA and non-LoRA
    lora_params = []
    non_lora_params = []
    
    for name, param in model.named_parameters():
        if is_lora_param(name):
            lora_params.append((name, param))
        else:
            non_lora_params.append((name, param))
    
    # Check LoRA weights - SHOULD be changing
    lora_changed = []
    lora_not_changed = []
    
    # Take snapshot of LoRA params
    for name, param in lora_params:
        if param.requires_grad:
            lora_changed.append(name)
        else:
            lora_not_changed.append(name)
    
    # Report on LoRA parameters
    colored_print("LoRA Params", f"Found {len(lora_params)} LoRA parameters", Colors.BLUE)
    colored_print("LoRA Changes", f"{len(lora_changed)} LoRA params with requires_grad=True (should be > 0)", 
                 Colors.GREEN if len(lora_changed) > 0 else Colors.RED)
    
    if len(lora_not_changed) > 0:
        colored_print("LoRA Warning", f"{len(lora_not_changed)} LoRA params with requires_grad=False (should be 0)", Colors.YELLOW)
            
    # Check non-LoRA weights - should NOT be changing
    non_lora_changed = []
    non_lora_not_changed = []
    
    for name, param in non_lora_params:
        if param.requires_grad:
            non_lora_changed.append(name)
        else:
            non_lora_not_changed.append(name)
    
    # Report on non-LoRA parameters
    colored_print("Non-LoRA Params", f"Found {len(non_lora_params)} non-LoRA parameters", Colors.BLUE)
    colored_print("Non-LoRA Frozen", f"{len(non_lora_not_changed)} non-LoRA params with requires_grad=False (good)", Colors.GREEN)
    
    if len(non_lora_changed) > 0:
        colored_print("Non-LoRA ERROR", f"{len(non_lora_changed)} non-LoRA params with requires_grad=True (should be 0)", Colors.RED)
        for name in non_lora_changed[:5]:  # Show a few examples
            colored_print("Unfrozen Param", name, Colors.RED)
        if len(non_lora_changed) > 5:
            colored_print("Unfrozen Param", f"...and {len(non_lora_changed) - 5} more", Colors.RED)

def apply_model_specific_patches(model, model_type):
    """Apply compatibility patches for specific models.
    
    Args:
        model: The model to patch
        model_type: The type of model being used
        
    Returns:
        model: The patched model
    """
    if model_type == "phi":
        # Apply Phi-specific patches
        colored_print("Model Patches", "Applying Phi model compatibility patches", Colors.YELLOW)
        
        # Patch DynamicCache for Phi models
        try:
            from transformers.cache_utils import DynamicCache
            
            # Add get_max_length if it doesn't exist
            if not hasattr(DynamicCache, "get_max_length"):
                colored_print("Patching", "Adding get_max_length method to DynamicCache", Colors.YELLOW)
                DynamicCache.get_max_length = DynamicCache.get_seq_length
        except (ImportError, AttributeError):
            colored_print("Warning", "Failed to apply Phi model patch, generation may fail", Colors.RED)
    
    return model

def create_peft_model_with_adapter(base_model, peft_config):
    """Create a PEFT model with a properly initialized adapter using PEFT's standard API.
    
    This function creates a PEFT model and ensures the adapter is properly loaded and available
    for saving/loading.
    """
    # Create PEFT model with specified config
    colored_print("Creating PEFT Model", f"Using config with r={peft_config.r}, alpha={peft_config.lora_alpha}", Colors.BLUE)
    
    # Check if model is already a PeftModel
    if isinstance(base_model, PeftModel):
        colored_print("Note", "Model is already a PeftModel, adding adapter", Colors.YELLOW)
        model = base_model
        # Add a new adapter with the specified config
        adapter_name = "default"
        if adapter_name in model.peft_config:
            colored_print("Note", f"Adapter '{adapter_name}' already exists, will use it", Colors.YELLOW)
        else:
            colored_print("Adding Adapter", f"Creating adapter '{adapter_name}'", Colors.BLUE)
            model.add_adapter(adapter_name, peft_config)
    else:
        # Create a new PEFT model with the default adapter
        model = get_peft_model(base_model, peft_config)
    
    # Ensure there's an active adapter
    adapter_name = list(model.peft_config.keys())[0]
    model.active_adapter = adapter_name
    colored_print("Active Adapter", f"Set active adapter to '{adapter_name}'", Colors.GREEN)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify the adapter is properly initialized
    try:
        adapter_state = model.get_adapter_state_dict()
        colored_print("Adapter State", f"Successfully verified adapter state with {len(adapter_state)} keys", Colors.GREEN)
    except Exception as e:
        colored_print("Warning", f"Could not get adapter state: {str(e)}", Colors.YELLOW)
        colored_print("Troubleshooting", "This is expected for a newly initialized adapter and will be fixed during training", Colors.YELLOW)
    
    return model

def test_tokenization_for_model(model_name, phrases=None):
    """Test tokenization of specific phrases for a model.
    
    This is useful for debugging and verifying token IDs across model variants.
    
    Args:
        model_name: HuggingFace model name to load tokenizer for
        phrases: List of phrases to tokenize (defaults to common ones)
        
    Returns:
        Dictionary of phrase -> token IDs mappings
    """
    # Set default phrases including answer patterns for different models
    if phrases is None:
        phrases = [
            "Answer:",
            " Answer:",
            "answer:",
            " answer:",
            "Answer: ",
            "The answer is",
        ]
    
    # Load tokenizer with trust_remote_code for models that need it
    trust_remote_code = "phi" in model_name or "gemma" in model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        padding_side="left",
        trust_remote_code=trust_remote_code
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize each phrase and print results
    results = {}
    print(f"\nTokenization test for {model_name}:")
    for phrase in phrases:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        results[phrase] = tokens
        token_strings = [tokenizer.decode([t]) for t in tokens]
        print(f"{phrase!r}: {tokens} -> {token_strings}")
    
    return results

