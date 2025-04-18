import os
import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional
from constants import MISTRAL_INST_START, MISTRAL_INST_END
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
        }
    else:  # llama or gpt2 or tinystories or phi
        return {
            "inst_start": "",
            "inst_end": "",
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

    # Include the actual question or use <Redacted> placeholder
    question_placeholder = question if include_question else "<Redacted>"
    base_with_type = f"{tokens['inst_start']} {base_prompt} {question_placeholder} {tokens['inst_end']}\n{prompt_type}"

    # Add model-specific answer header to partial prompt
    return base_with_type + reasoning + f" Answer: "

def load_model(model_type="mistral"):
    """Load a frozen model for evaluation."""
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
    else:
        raise ValueError("model_type must be either 'mistral', 'llama', 'gpt2', 'tinystories', or 'phi'")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=model_type=="phi"  # Phi needs trust_remote_code=True
    )

    # Freeze the model
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device