import datetime
import torch
from torch import nn
import bitsandbytes
import random
import numpy as np
import json
from torch.nn.utils import clip_grad_norm_
import argparse
import re
import os
import subprocess
import sys
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
from evaluate_gsm8k import evaluate_model, save_results
from peft import PeftModel
from utils import (
    Colors,
    colored_print,
    construct_prompts,
    find_latest_result,
    print_batch_delimiter,
    print_parallel_overview,
    get_model_hash,
    verify_all_frozen_weights,
    verify_actor_weights_changing_comprehensive,
    calculate_threshold,
    find_latest_checkpoint,
    generate_question_answer_batches,
    load_gsm8k_dataset,
    extract_answer,
    load_mmlu_dataset,
    load_aqua_dataset,
    load_mathqa_dataset,
    load_svamp_dataset,
    load_math_dataset,
    load_arc_dataset,
    load_model,
    get_grad_norm,
)
import glob
from datasets import load_dataset



def get_default_eval_batch_size(train_batch_size: int) -> int:
    """Default evaluation batch size: floor(1.5x train batch size)."""
    return max(1, int(train_batch_size * 1.5))


def get_default_train_batch_size(task_type: str) -> int:
    """Default TRAINING batch size by task type in one place.
    - wiki_compression/wiki_continuation: 16
    - arithmetic/arithmetic-negative, mmlu, and others: 12
    - gsm8k, mathqa: 8
    """
    if task_type in ("wiki_compression", "wiki_continuation"):
        return 16
    if task_type in ("mathqa", "gsm8k"):
        return 8
    return 12


def find_answer_start_position(input_ids, model_type):
    """Find the starting position of the answer in the input_ids based on model type."""
    if model_type == "mistral":
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
    elif model_type in ["llama", "llama3.2-1b", "phi-4"]:  # phi-4 uses the same token IDs as llama for "Answer:"
        matching_indices = (
            ((input_ids[:-1] == 16533) | (input_ids[:-1] == 22559))
            & (input_ids[1:] == 25)
        ).nonzero(as_tuple=True)[0]
        pos = matching_indices[-1].item() + 2
    elif model_type in ["qwen3", "qwen3-14b"]:
        # Qwen2.5 and Qwen3 use same token IDs: " Answer" (21806) or "Answer" (16141) followed by ":" (25)
        matching_indices = (
            ((input_ids[:-1] == 21806) | (input_ids[:-1] == 16141))  # " Answer" or "Answer"
            & (input_ids[1:] == 25)  # ":"
        ).nonzero(as_tuple=True)[0]
        
        if len(matching_indices) > 0:
            pos = matching_indices[-1].item() + 2
        else:
            # Fallback in case the exact pattern isn't found
            colored_print("Warning", f"Could not find 'Answer:' in {model_type} output, using fallback position", Colors.YELLOW)
            # Try to find just the colon
            colon_indices = (input_ids == 25).nonzero(as_tuple=True)[0]
            if len(colon_indices) > 0:
                pos = colon_indices[-1].item() + 1
            else:
                # Worst case: use the last 20% of tokens
                pos = int(len(input_ids) * 0.8)
    elif model_type in ["gpt2", "tinystories"]:  # TinyStories uses same tokens as GPT2
        matching_indices = (
            (input_ids[:-1] == 23998)
            & (input_ids[1:] == 25)
        ).nonzero(as_tuple=True)[0]
        pos = matching_indices[-1].item() + 2
    elif model_type == "phi":
        # Phi-3.5-mini tokenization: "Answer:" -> [673, 29901] or " Answer:" -> [29871, 673, 29901]
        matching_indices = (
            (input_ids[:-1] == 673)  # "Answer"
            & (input_ids[1:] == 29901)  # ":"
        ).nonzero(as_tuple=True)[0]
        pos = matching_indices[-1].item() + 2
    elif model_type in ["gemma-3", "gemma-3-small"]:
        # For Gemma-3, we need to handle multiple potential tokens for "Answer"
        # followed by colon token (236787)
        matching_indices = (
            (
                (input_ids[:-1] == 25685)  # " Answer"
                | (input_ids[:-1] == 7925)  # "Answer"
                | (input_ids[:-1] == 14433)  # "answer"
                | (input_ids[:-1] == 3890)  # " answer"
            )
            & (input_ids[1:] == 236787)  # ":"
        ).nonzero(as_tuple=True)[0]
        
        if len(matching_indices) > 0:
            pos = matching_indices[-1].item() + 2
        else:
            # Fallback in case the exact pattern isn't found
            colored_print("Warning", "Could not find 'Answer:' in Gemma-3 output, using fallback position", Colors.YELLOW)
            # Try to find a plausible position - the colon might be there
            colon_indices = (input_ids == 236787).nonzero(as_tuple=True)[0]
            if len(colon_indices) > 0:
                pos = colon_indices[-1].item() + 1
            else:
                # Worst case: use the last 20% of tokens
                pos = int(len(input_ids) * 0.8)
    else:
        raise ValueError("Unsupported model type")
    return pos


def calculate_answer_log_probs(
    frozen_model,
    tokenizer,
    device,
    questions,
    reasoning,
    answers,
    hyperparameters,
    include_question=False,
):
    """Calculate the log probabilities of the answers given the reasoning.

    Args:
        frozen_model: The critic model (frozen)
        tokenizer: Tokenizer for the model
        device: The device to run on
        questions: List of question strings
        reasoning: List of reasoning strings (from either actor or critic)
        answers: List of answer strings
        hyperparameters: Dictionary of hyperparameters
        include_question: Whether to include the question in the prompt (default: False)

    Returns:
        tuple: (
            mean_answer_logprobs,  # Average log prob of each answer token
            answer_logprobs,       # Full sequence of answer token log probs
            extracted_answers      # Only for GSM8K: extracted numerical answers
        )
    """
    # Create prompts with reasoning (may have <Redacted> instead of actual question when include_question=False)
    partial_prompts = [
        construct_prompts(
            question=q,
            hyperparameters=hyperparameters,
            reasoning=r,
            include_question=include_question,
        )
        for q, r in zip(questions, reasoning)
    ]

    # Add answers to create full prompts
    full_prompts = [x + y for x, y in zip(partial_prompts, answers)]

    # Tokenize full prompts
    full_prompt_tokens = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # For GSM8K, we also generate answers to extract numerical values
    extracted_generated_answers = None
    if hyperparameters["task_type"] == "gsm8k":
        # Tokenize partial prompts (without answers) for generation
        partial_prompt_tokens = tokenizer(partial_prompts, padding=True, return_tensors="pt").to(
            device
        )

        # Generate answer tokens
        max_answer_length = 15
        with torch.no_grad():
            generated_outputs = frozen_model.generate(
                input_ids=partial_prompt_tokens.input_ids,
                attention_mask=partial_prompt_tokens.attention_mask,
                max_new_tokens=max_answer_length,
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and extract numerical answers
        generated_answers = tokenizer.batch_decode(
            generated_outputs[:, -max_answer_length - 1 :], skip_special_tokens=True
        )
        selected_answers = [x.split("\n")[-1] for x in generated_answers]
        extracted_generated_answers = [extract_answer(ans) for ans in selected_answers]

    # Find the starting positions of answers in the full prompts
    answer_start_positions = [
        find_answer_start_position(input_ids, hyperparameters["model_type"])
        for input_ids in full_prompt_tokens.input_ids
    ]

    # Verify answer positions are correct
    for i in range(len(answers)):
        decoded_answer = tokenizer.decode(
            full_prompt_tokens.input_ids[i][answer_start_positions[i] :]
        ).strip()
        expected_answer = answers[i].strip()
        if (
            decoded_answer[:3] != expected_answer[:3]
            or decoded_answer[-3:] != expected_answer[-3:]
        ):
            colored_print("Answer mismatch at index", str(i), Colors.RED)

    # Calculate log probabilities
    with torch.no_grad():
        model_logits = frozen_model(
            input_ids=full_prompt_tokens.input_ids,
            attention_mask=full_prompt_tokens.attention_mask,
        ).logits

    # Convert to log probabilities
    log_probs = torch.nn.functional.log_softmax(model_logits, dim=-1)

    # Get log probs for each answer token
    answer_logprobs = [
        log_probs[i, start - 1 : -1]
        .gather(1, full_prompt_tokens.input_ids[i, start:].unsqueeze(-1))
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
    critic_answer_logprobs: torch.Tensor
    extracted_answers: Optional[List[Any]]


@dataclass
class TrainingState:
    """Holds the state of the training process"""

    batch_index: int
    previous_normalized_rewards: List[float]
    previous_advantages: List[float]

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
    
    # Gradient accumulation tracking
    accumulation_step: int

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
        # Configure generation configs to avoid parameter conflicts
        critic_model.generation_config.temperature = None
        critic_model.generation_config.top_p = None
        critic_model.generation_config.top_k = None
        
        # Also configure actor model to avoid warnings during generation
        actor_model.generation_config.top_k = None
        actor_model.generation_config.top_p = None

        return cls(
            batch_index=start_batch,
            previous_normalized_rewards=prev_rewards,
            previous_advantages=prev_advantages,
            actor_model=actor_model,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            tokenizer=tokenizer,
            device=device,
            model_save_path=model_save_path,
            log_file=log_file,
            hyperparameters=updated_hyperparameters,  # Use the updated hyperparameters
            accumulation_step=0,
        )


def generate_reasoning_and_kl(
    state: TrainingState, questions: List[str], calculate_kl: bool = True
) -> ReasoningOutput:
    """Generate reasoning from both models and calculate KL divergence.
    
    Args:
        state: Current training state
        questions: List of input questions (dataset handles any repetition)
        calculate_kl: Whether to calculate KL divergence (if False, will return zeros)
    
    Returns:
        ReasoningOutput: Contains generated reasoning and associated metrics
    """
    # Create prompts for each question (no expansion needed - dataset handles repetition)
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
            top_k=None,
            top_p=None,
            pad_token_id=state.tokenizer.pad_token_id,
        )
        
        # Only generate critic reasoning if we're normalizing loss
        if state.hyperparameters.get("normalize_loss", True):
            parallel_mode = state.hyperparameters.get("parallel", False)
            
            if parallel_mode:
                # OPTIMIZATION: In parallel mode, all questions are identical, so only generate once
                colored_print("Critic Optimization", "Using single critic computation with replication", Colors.GREEN)
                
                # Generate critic reasoning for just the first (unique) example
                unique_tokenized = state.tokenizer(
                    [prompts[0]], padding=True, return_tensors="pt"
                ).to(state.device)
                
                q_r_tokens_unique = state.critic_model.generate(
                    unique_tokenized.input_ids,
                    attention_mask=unique_tokenized.attention_mask,
                    max_new_tokens=state.hyperparameters["cot_length"],
                    min_new_tokens=state.hyperparameters["cot_length"],
                    do_sample=False,  # Critic is deterministic
                    top_k=None,
                    top_p=None,
                    pad_token_id=state.tokenizer.pad_token_id,
                )
                
                # Replicate the result for all batch positions
                batch_size = len(questions)
                q_r_tokens = q_r_tokens_unique.repeat(batch_size, 1)
                
                # Decode once and replicate
                critic_reasoning_unique = state.tokenizer.batch_decode(
                    q_r_tokens_unique[:, -state.hyperparameters["cot_length"] :], 
                    skip_special_tokens=True
                )[0]
                critic_reasoning = [critic_reasoning_unique] * batch_size
            else:
                # Normal mode: generate for all examples
                q_r_tokens = state.critic_model.generate(
                    tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    max_new_tokens=state.hyperparameters["cot_length"],
                    min_new_tokens=state.hyperparameters["cot_length"],
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    pad_token_id=state.tokenizer.pad_token_id,
                )
                # Decode critic reasoning text
                critic_reasoning = state.tokenizer.batch_decode(
                    q_r_tokens[:, -state.hyperparameters["cot_length"] :], skip_special_tokens=True
                )
        else:
            # Skip critic reasoning generation when not normalizing
            q_r_tokens = None
            critic_reasoning = None

    # Only compute the KL if we need it (kl_penalty is not None, or if we want to track it)
    if calculate_kl:
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
    else:
        # Return zero tensors if we're not calculating KL
        device = q_R_tokens.device
        batch_size = len(q_R_tokens)
        R_mean_actor_logprobs = torch.zeros(batch_size, device=device)
        R_mean_critic_logprobs = torch.zeros(batch_size, device=device)
        kl = torch.zeros(batch_size, device=device)

    # Decode actor reasoning text
    actor_reasoning = state.tokenizer.batch_decode(
        q_R_tokens[:, -state.hyperparameters["cot_length"] :], skip_special_tokens=True
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
    """Calculate advantages for both standard and parallel sampling modes.
    
    Args:
        state: Current training state
        questions: List of questions (dataset handles any repetition)
        answers: List of answers (dataset handles any repetition)
        reasoning_output: Output from generate_reasoning functions
        
    Returns:
        AdvantageOutput: Contains advantage calculations and metrics
    """
    parallel_mode = state.hyperparameters.get("parallel", False)
    
    # Calculate log probs of answers given actor's reasoning
    # Use markovian flag to determine whether to include question context
    include_question_in_reward = not state.hyperparameters.get("markovian", True)
    
    # Check if we should use actor model for rewards
    actor_reward_weight = state.hyperparameters.get("actor_reward_weight", 0.0)
    use_actor_rewards = actor_reward_weight > 0.0
    
    # Log which reward mode is being used (only on first batch to avoid spam)
    if state.batch_index == 0:
        if include_question_in_reward:
            colored_print("Reward Mode", "Non-Markovian: P(answer | question, CoT)", Colors.CYAN)
        else:
            colored_print("Reward Mode", "Markovian: P(answer | CoT)", Colors.CYAN)
        
        if use_actor_rewards:
            colored_print("Actor Rewards", f"Using actor model for rewards with weight {actor_reward_weight}", Colors.MAGENTA)
        else:
            colored_print("Critic Rewards", "Using critic model for rewards (standard)", Colors.CYAN)
    
    # Choose reward model and reasoning based on configuration
    reward_model = state.actor_model if use_actor_rewards else state.critic_model
    reward_reasoning = reasoning_output.actor_reasoning if use_actor_rewards else reasoning_output.actor_reasoning
    
    actor_answer_logprobs, extracted_answers = calculate_answer_log_probs(
        reward_model,
        state.tokenizer,
        state.device,
        questions,
        reward_reasoning,
        answers,
        state.hyperparameters,
        include_question=include_question_in_reward,
    )

    # Calculate normalized rewards - always use critic model for baseline
    if state.hyperparameters.get("normalize_loss", True):
        if parallel_mode:
            # OPTIMIZATION: In parallel mode, calculate critic answer log prob only once
            colored_print("Critic Answer Optimization", "Computing single critic answer and replicating", Colors.GREEN)
            
            critic_answer_logprob_single, _ = calculate_answer_log_probs(
                state.critic_model,  # Always use critic for baseline
                state.tokenizer,
                state.device,
                [questions[0]],  # Just first question
                [reasoning_output.critic_reasoning[0]],  # Just first reasoning
                [answers[0]],  # Just first answer
                state.hyperparameters,
                include_question=include_question_in_reward,
            )
            
            # Replicate across batch
            batch_size = len(questions)
            critic_answer_logprobs = critic_answer_logprob_single.repeat(batch_size)
        else:
            # Normal mode: calculate for all
            critic_answer_logprobs, _ = calculate_answer_log_probs(
                state.critic_model,  # Always use critic for baseline
                state.tokenizer,
                state.device,
                questions,
                reasoning_output.critic_reasoning,
                answers,
                state.hyperparameters,
                include_question=include_question_in_reward,
            )
        
        # Normalize reward as improvement over baseline
        # If using actor rewards, don't detach actor_answer_logprobs to preserve gradients
        if use_actor_rewards:
            normalized_rewards = actor_answer_logprobs - critic_answer_logprobs.detach()
        else:
            normalized_rewards = actor_answer_logprobs - critic_answer_logprobs
    else:
        # Skip critic calculation when not normalizing
        critic_answer_logprobs = torch.zeros_like(actor_answer_logprobs)
        normalized_rewards = actor_answer_logprobs
    
    # Calculate advantages - simplified for both modes
    if parallel_mode:
        # Parallel mode: use standardized batch baseline (mean-centered, unit variance)
        if state.hyperparameters.get("r") is not None:
            colored_print("Warning", f"r parameter ({state.hyperparameters['r']}) is ignored in parallel mode", Colors.YELLOW)
            colored_print("Info", "Parallel mode uses standardized batch baseline (mean=0, std=1)", Colors.CYAN)
        
        batch_mean = normalized_rewards.mean()
        batch_std = normalized_rewards.std()
        # Add small epsilon to prevent division by zero
        advantages = (normalized_rewards - batch_mean) / (batch_std + 1e-8)
    else:
        # Standard mode: use exponential moving average baseline
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
        critic_answer_logprobs=critic_answer_logprobs,
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
    batch_index=None,
):
    """Calculate training losses using specified methods (PG/PPO/EI).

    Args:
        kl: KL divergence between actor and critic distributions
        R_mean_actor_logprobs: Mean log probs of actor's reasoning under actor
        R_mean_critic_logprobs: Mean log probs of actor's reasoning under critic
        advantages: Advantage values for actor's reasoning
        previous_advantages: History of advantages for EI threshold
        hyperparameters: Training configuration
        batch_index: Current batch index for accurate EI threshold calculation

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
    actor_reward_weight = hyperparameters.get("actor_reward_weight", 0.0)

    # Initialize metrics dictionary
    metrics = {}

    # Policy gradient loss: R_θ(τ) * ∇_θ log P_θ(τ)
    # Detach advantages to isolate this term when using actor rewards
    if actor_reward_weight > 0.0:
        pg_losses = -R_mean_actor_logprobs * advantages.detach()
        # Actor reward gradient loss: ∇_θ R_θ(τ) 
        # Don't detach advantages - let gradients flow through reward model
        reward_gradient_losses = -actor_reward_weight * advantages
        losses = pg_losses + reward_gradient_losses
        metrics["pg_losses"] = pg_losses
        metrics["reward_gradient_losses"] = reward_gradient_losses
    else:
        # Standard policy gradient loss
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
            previous_normalized_rewards, hyperparameters["use_ei"], batch_index
        )
        training_mask = (normalized_rewards > threshold).float()
        metrics["ei_threshold"] = threshold
        metrics["ei_mask"] = training_mask
        metrics["ei_enabled"] = True
    else:
        # Explicitly mark that EI is disabled
        metrics["ei_enabled"] = False

    return (
        losses,  # No longer using flatten parameter
        training_mask,
        metrics,
    )


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


def setup_training_environment(task_type, resume, hyperparameters):
    """Set up the results directory and load checkpoints if resuming."""
    if resume:
        # Get the task results directory
        results_dir = os.path.join("results", task_type)
        if not os.path.exists(results_dir):
            raise ValueError(f"No results directory found for task type: {task_type}")
        
        # Look for timestamped run directories
        run_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d)) and re.match(r"^\d{8}_\d{6}$", d)]
        
        if not run_dirs:
            raise ValueError(f"No previous runs found in {results_dir}")
        
        # Get the latest run directory
        latest_dir = max(run_dirs, key=os.path.getmtime)
        colored_print("Resume", f"Using latest run directory: {latest_dir}", Colors.BOLD)
        
        # Check if this run directory has a log file
        log_file = os.path.join(latest_dir, "log.jsonl")
        if not os.path.exists(log_file):
            # Check if there are adapter directories in this run
            adapter_dirs = sorted(
                [d for d in glob.glob(os.path.join(latest_dir, "adapter_*")) if os.path.isdir(d)],
                key=lambda x: int(x.split("_")[-1])  # Sort by batch number
            )
            
            if adapter_dirs:
                # Use the latest adapter to get batch information
                latest_adapter = adapter_dirs[-1]
                batch_number = int(latest_adapter.split("_")[-1])
                colored_print("Log File", f"Creating log file using adapter at batch {batch_number}", Colors.YELLOW)
                
                # Check if metadata exists
                metadata_path = os.path.join(latest_adapter, "training_metadata.pt")
                if os.path.exists(metadata_path):
                    # Load metadata for hyperparameters
                    metadata = torch.load(metadata_path)
                    if "hyperparameters" in metadata:
                        hyperparameters = metadata["hyperparameters"]
                        
                # Create a minimal log file with just the batch index and hyperparameters
                with open(log_file, "w") as f:
                    json.dump(hyperparameters, f)
                    f.write("\n")
                    
                    # Add an entry for the current batch
                    entry = {"Batch Index": batch_number}
                    json.dump(entry, f)
                    f.write("\n")
                
                colored_print("Log File", f"Created new log file for resuming from adapter", Colors.GREEN)
            else:
                raise ValueError(f"Missing required log file and no adapter checkpoints found in: {latest_dir}")
        
        start_batch, hyperparameters = load_training_state(log_file)
        previous_normalized_rewards, previous_advantages = (
            load_previous_rewards_and_advantages(log_file)
        )
        
        # Use the latest run directory for saving future checkpoints
        model_save_path = latest_dir
    else:
        # Create a new timestamped directory for this run
        results_dir = os.path.join(
            "results", task_type, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(results_dir, exist_ok=True)
        model_save_path = results_dir  # Directory, not specific file
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
        hyperparameters,
    )



def initialize_model_and_optimizer(model_type, hyperparameters, checkpoint_path=None):
    """Initialize the model, frozen model, tokenizer, device, and optimizer."""
    model, frozen_model, tokenizer, device = load_model(model_type, hyperparameters)
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=hyperparameters["lr"]
    )

    if checkpoint_path is not None:
        # Find the latest checkpoint in the directory if checkpoint_path points to a directory
        if os.path.isdir(checkpoint_path):
            # First check for adapter directories within this run directory
            adapter_dirs = sorted(
                [d for d in glob.glob(os.path.join(checkpoint_path, "adapter_*")) if os.path.isdir(d)],
                key=lambda x: int(x.split("_")[-1])  # Sort by batch number
            )
            
            if adapter_dirs:
                # Use the latest adapter directory
                latest_adapter = adapter_dirs[-1]
                colored_print("Resume", f"Using latest adapter: {os.path.basename(latest_adapter)}", Colors.BOLD)
                
                # Get the batch number for logging 
                batch_num = int(latest_adapter.split("_")[-1])
                
                # Load the adapter using PEFT's load_pretrained
                from peft import PeftModel
                
                colored_print("Loading Adapter", f"Loading adapter from {latest_adapter}", Colors.BLUE)
                model = PeftModel.from_pretrained(
                    model,  # Base model
                    latest_adapter,  # Adapter path
                    is_trainable=True  # Ensure it's set to training mode
                )
                
                # Load optimizer state and other metadata
                metadata_path = os.path.join(latest_adapter, "training_metadata.pt")
                if os.path.exists(metadata_path):
                    metadata = torch.load(metadata_path)
                    model_optimizer.load_state_dict(metadata["optimizer_state_dict"])
                    colored_print("Metadata", f"Loaded optimizer state from batch {batch_num}", Colors.GREEN)
                    
                    # Print key info about the loaded adapter
                    colored_print("Adapter Info", f"Loaded adapter from batch {batch_num}", Colors.GREEN)
                    colored_print("Active Adapter", f"Current active adapter: {model.active_adapter}", Colors.GREEN)
                else:
                    colored_print("Warning", f"No metadata found at {metadata_path}", Colors.YELLOW)
                
                return model, frozen_model, tokenizer, device, model_optimizer
            
            # Fall back to traditional checkpoint files if no adapters found
            colored_print("Note", "No adapter directories found in this run", Colors.YELLOW)
            latest_checkpoint = find_latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                colored_print("Resume", f"Using latest checkpoint: {os.path.basename(latest_checkpoint)}", Colors.BOLD)
            else:
                colored_print("Warning", f"No checkpoints found in {checkpoint_path}", Colors.RED)
                return model, frozen_model, tokenizer, device, model_optimizer
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Check if this is a LoRA adapter checkpoint or full model checkpoint
        if "adapter_state_dict" in checkpoint:
            colored_print("Resume", "Loading LoRA adapter weights from checkpoint file", Colors.BOLD)
            model.load_adapter_state_dict(checkpoint["adapter_state_dict"])
            size_bytes = sum(tensor.nelement() * tensor.element_size() 
                           for tensor in checkpoint["adapter_state_dict"].values())
            size_mb = size_bytes / (1024 * 1024)
            colored_print("Checkpoint Info", f"Loaded LoRA weights, size: {size_mb:.2f}MB", Colors.GREEN)
        elif "model_state_dict" in checkpoint:
            colored_print("Resume", "Loading full model weights (old format)", Colors.BOLD)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            colored_print("Warning", "Checkpoint has unknown format", Colors.RED)
        
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
    R_mean_actor_logprobs: torch.Tensor  # Reasoning logprobs
    R_mean_critic_logprobs: torch.Tensor  # Reasoning logprobs
    kl: torch.Tensor
    advantages: torch.Tensor
    normalized_rewards: torch.Tensor
    actor_answer_logprobs: torch.Tensor
    critic_answer_logprobs: torch.Tensor
    losses: torch.Tensor
    training_mask: Optional[torch.Tensor]
    metrics: Dict[str, Any]


@dataclass
class LogMetrics:
    """Holds metrics for logging"""

    # Mean metrics across batch
    loss: float
    pg_loss: float
    actor_logprobs: float
    critic_logprobs: float
    actor_answer_logprobs: float
    critic_answer_logprobs: float
    kl: float
    weighted_kl: Optional[float]
    ppo_ratio: Optional[float]
    ppo_clipped_ratio: Optional[float]
    advantage: float
    normalized_reward: float
    
    # First example metrics
    first_loss: float
    first_pg_loss: float
    first_actor_logprobs: float
    first_critic_logprobs: float
    first_actor_answer_logprobs: float
    first_critic_answer_logprobs: float
    first_kl: float
    first_weighted_kl: Optional[float]
    first_advantage: float
    first_normalized_reward: float
    
    # Other metrics
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
        previous_advantages: List[float],
        batch_size: int,
    ):
        """Create LogMetrics from batch data and training state"""
        # Calculate number of active examples
        training_mask = batch_data.training_mask
        num_active = (
            training_mask.sum().item()
            if training_mask is not None
            else len(batch_data.losses)
        )

        # Handle case where no examples are active
        if num_active == 0:
            colored_print("Warning", "No active examples in batch!", Colors.RED)
            # Use placeholder values for metrics when no examples are active
            return cls(
                # Mean metrics
                loss=float('nan'),  # NaN indicates no active examples
                pg_loss=float('nan'),
                actor_logprobs=batch_data.R_mean_actor_logprobs.mean().item(),
                critic_logprobs=batch_data.R_mean_critic_logprobs.mean().item(),
                actor_answer_logprobs=batch_data.actor_answer_logprobs.mean().item(),
                critic_answer_logprobs=batch_data.critic_answer_logprobs.mean().item(),
                kl=batch_data.kl.mean().item(),
                weighted_kl=None,
                ppo_ratio=None,
                ppo_clipped_ratio=None,
                advantage=batch_data.advantages.mean().item(),
                normalized_reward=batch_data.normalized_rewards.mean().item(),
                
                # First example metrics
                first_loss=float('nan'),
                first_pg_loss=float('nan'),
                first_actor_logprobs=batch_data.R_mean_actor_logprobs[0].item() if len(batch_data.R_mean_actor_logprobs) > 0 else float('nan'),
                first_critic_logprobs=batch_data.R_mean_critic_logprobs[0].item() if len(batch_data.R_mean_critic_logprobs) > 0 else float('nan'),
                first_actor_answer_logprobs=batch_data.actor_answer_logprobs[0].item() if len(batch_data.actor_answer_logprobs) > 0 else float('nan'),
                first_critic_answer_logprobs=batch_data.critic_answer_logprobs[0].item() if len(batch_data.critic_answer_logprobs) > 0 else float('nan'),
                first_kl=batch_data.kl[0].item() if len(batch_data.kl) > 0 else float('nan'),
                first_weighted_kl=None,
                first_advantage=batch_data.advantages[0].item() if len(batch_data.advantages) > 0 else float('nan'),
                first_normalized_reward=batch_data.normalized_rewards[0].item() if len(batch_data.normalized_rewards) > 0 else float('nan'),
                
                # Other metrics
                gradient_norm=0.0,  # No gradient if no active examples
                num_active=0,
                fraction_active=0.0,
                ei_threshold=batch_data.metrics.get("ei_threshold", None),
                mean_prev_advantage=(
                    np.mean(previous_advantages) if previous_advantages else None
                ),
                std_prev_advantage=(
                    np.std(previous_advantages) if previous_advantages else None
                ),
            )

        # Get PPO metrics
        ppo_ratio = batch_data.metrics.get("prob_ratios", [None])[0]
        ppo_clipped_ratio = batch_data.metrics.get("clipped_ratios", [None])[0]

        # Convert to float if they exist
        ppo_ratio = float(ppo_ratio.item()) if ppo_ratio is not None else None
        ppo_clipped_ratio = (
            float(ppo_clipped_ratio.item()) if ppo_clipped_ratio is not None else None
        )

        # Get KL values - average across all examples, not just first one
        raw_kl_mean = batch_data.kl.mean().item()
        raw_kl_first = batch_data.kl[0].item() if len(batch_data.kl) > 0 else float('nan')
        
        weighted_kl = batch_data.metrics.get("weighted_kl", None)
        weighted_kl_mean = float(weighted_kl.mean().item()) if weighted_kl is not None else None
        weighted_kl_first = float(weighted_kl[0].item()) if weighted_kl is not None and len(weighted_kl) > 0 else None

        # Calculate metrics that should be averaged over all examples (regardless of active status)
        mean_actor_logprobs = batch_data.R_mean_actor_logprobs.mean().item()
        mean_critic_logprobs = batch_data.R_mean_critic_logprobs.mean().item()
        mean_actor_answer_logprobs = batch_data.actor_answer_logprobs.mean().item()
        mean_critic_answer_logprobs = batch_data.critic_answer_logprobs.mean().item()
        mean_advantage = batch_data.advantages.mean().item()
        mean_normalized_reward = batch_data.normalized_rewards.mean().item()
        
        # Get metrics for the first example
        first_actor_logprobs = batch_data.R_mean_actor_logprobs[0].item()
        first_critic_logprobs = batch_data.R_mean_critic_logprobs[0].item()
        first_actor_answer_logprobs = batch_data.actor_answer_logprobs[0].item()
        first_critic_answer_logprobs = batch_data.critic_answer_logprobs[0].item()
        first_advantage = batch_data.advantages[0].item()
        first_normalized_reward = batch_data.normalized_rewards[0].item()
        
        # Calculate loss metrics across ALL examples (not just active ones)
        # This gives a more consistent view of model performance regardless of threshold
        mean_loss = batch_data.losses.mean().item()
        mean_pg_loss = batch_data.metrics["pg_losses"].mean().item()
        
        # Get loss metrics for first example
        first_loss = batch_data.losses[0].item()
        first_pg_loss = batch_data.metrics["pg_losses"][0].item()
        
        # For reference, also calculate loss metrics for active examples only
        if training_mask is not None and num_active > 0:
            # Get only the active losses for calculating means
            active_mask = training_mask.bool()
            active_losses = batch_data.losses[active_mask]
            active_pg_losses = batch_data.metrics["pg_losses"][active_mask]
            active_only_mean_loss = active_losses.mean().item()
            active_only_mean_pg_loss = active_pg_losses.mean().item()
            
            # Add these to metrics dictionary for potential logging but don't use as primary metrics
            batch_data.metrics["active_only_loss"] = active_only_mean_loss
            batch_data.metrics["active_only_pg_loss"] = active_only_mean_pg_loss

        return cls(
            # Mean metrics
            loss=mean_loss,
            pg_loss=mean_pg_loss,
            actor_logprobs=mean_actor_logprobs,
            critic_logprobs=mean_critic_logprobs,
            actor_answer_logprobs=mean_actor_answer_logprobs,
            critic_answer_logprobs=mean_critic_answer_logprobs,
            kl=raw_kl_mean,
            weighted_kl=weighted_kl_mean,
            ppo_ratio=ppo_ratio,
            ppo_clipped_ratio=ppo_clipped_ratio,
            advantage=mean_advantage,
            normalized_reward=mean_normalized_reward,
            
            # First example metrics
            first_loss=first_loss,
            first_pg_loss=first_pg_loss,
            first_actor_logprobs=first_actor_logprobs,
            first_critic_logprobs=first_critic_logprobs,
            first_actor_answer_logprobs=first_actor_answer_logprobs,
            first_critic_answer_logprobs=first_critic_answer_logprobs,
            first_kl=raw_kl_first,
            first_weighted_kl=weighted_kl_first,
            first_advantage=first_advantage,
            first_normalized_reward=first_normalized_reward,
            
            # Other metrics
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
    critic_reasoning_text = batch_data.critic_reasoning[0] if batch_data.critic_reasoning is not None else "None (normalize_loss=False)"

    # Calculate fraction of answers contained in reasoning across batch
    contains_answer_fraction = sum(
        answer in reasoning
        for answer, reasoning in zip(batch_data.answers, batch_data.actor_reasoning)
    ) / len(batch_data.answers)

    # Print the question/context and actor reasoning (always shown)
    if state.hyperparameters["task_type"] in ["wiki_compression", "wiki_continuation"]:
        colored_print("Context:", q, Colors.BLUE)
        colored_print("Actor Reasoning:", actor_reasoning_text, Colors.YELLOW)
    else:  # arithmetic or gsm8k
        colored_print("Question:", q, Colors.BLUE)
        colored_print("Actor Reasoning:", actor_reasoning_text, Colors.YELLOW)
    
    # Only show critic reasoning if normalize_loss is True
    if state.hyperparameters.get("normalize_loss", True) and batch_data.critic_reasoning is not None:
        colored_print("Critic Reasoning:", critic_reasoning_text, Colors.CYAN)

    colored_print("Answer:", a, Colors.GREEN)
    
    # Only show EI status if it's enabled
    ei_enabled = "ei_enabled" in batch_data.metrics and batch_data.metrics["ei_enabled"]
    if ei_enabled:
        ei_status = f"Enabled (std_mult={state.hyperparameters['use_ei']})"
        colored_print("Expert Iteration:", ei_status, Colors.CYAN)
    
    # Only show parallel sampling status if enabled
    parallel_mode = state.hyperparameters.get("parallel", False)
    if parallel_mode:
        batch_size = state.hyperparameters.get("batch_size", 8)
        colored_print("Parallel Sampling:", f"Enabled ({batch_size} copies per example)", Colors.BOLD)
    
    # Get raw unfiltered losses directly from the tensors
    # Always use the raw tensors instead of potentially filtered metrics
    raw_first_loss = batch_data.losses[0].item() if len(batch_data.losses) > 0 else float('nan')
    raw_mean_loss = batch_data.losses.mean().item() if len(batch_data.losses) > 0 else float('nan')
    
    raw_first_pg_loss = batch_data.metrics["pg_losses"][0].item() if "pg_losses" in batch_data.metrics and len(batch_data.metrics["pg_losses"]) > 0 else float('nan')
    raw_mean_pg_loss = batch_data.metrics["pg_losses"].mean().item() if "pg_losses" in batch_data.metrics and len(batch_data.metrics["pg_losses"]) > 0 else float('nan')
    
    # KL values
    if metrics.weighted_kl is not None and state.hyperparameters.get("kl_penalty", 0) != 0:
        # Use weighted KL only if penalty is non-zero
        first_kl_to_log = metrics.first_weighted_kl 
        kl_to_log = metrics.weighted_kl
    else:
        # Use raw KL if penalty is zero or None
        first_kl_to_log = metrics.first_kl
        kl_to_log = metrics.kl
    
    # Helper function to format values concisely
    def fmt(val):
        if isinstance(val, float) and not np.isnan(val):
            return f"{val:.4f}"
        return "NaN"
    
    # Print condensed metrics (horizontal layout)
    print("\n" + "=" * 100)
    
    # Advantage and reward metrics (line 1)
    print(f"{Colors.MAGENTA}Advantage{Colors.END} [F: {fmt(metrics.first_advantage)} | M: {fmt(metrics.advantage)}]  "
          f"{Colors.MAGENTA}Reward{Colors.END} [F: {fmt(metrics.first_normalized_reward)} | M: {fmt(metrics.normalized_reward)}]  "
          f"{Colors.GREEN}KL{Colors.END} [F: {fmt(first_kl_to_log)} | M: {fmt(kl_to_log)}]")
    
    # Log probabilities (line 2)
    print(f"{Colors.YELLOW}Actor ⟨LP⟩{Colors.END} [A: {fmt(metrics.first_actor_answer_logprobs)} | R: {fmt(metrics.first_actor_logprobs)}]  "
          f"{Colors.YELLOW}Critic ⟨LP⟩{Colors.END} [A: {fmt(metrics.first_critic_answer_logprobs)} | R: {fmt(metrics.first_critic_logprobs)}]")
    
    # Losses (line 3)
    print(f"{Colors.CYAN}Loss{Colors.END} [F: {fmt(raw_first_loss)} | M: {fmt(raw_mean_loss)}]  "
          f"{Colors.CYAN}PG Loss{Colors.END} [F: {fmt(raw_first_pg_loss)} | M: {fmt(raw_mean_pg_loss)}]  "
          f"{Colors.BOLD}Active{Colors.END} [{metrics.num_active}/{state.hyperparameters['batch_size']} ({metrics.fraction_active:.1%})]")
    
    # Legend
    print(f"{Colors.BOLD}Legend:{Colors.END} F=First example, M=Mean, A=Answer, R=Reasoning, LP=Log Probs")
    
    # If no examples were active, add a clear indicator
    if metrics.num_active == 0:
        print(f"{Colors.RED}Warning: No examples passed the EI threshold in this batch{Colors.END}")

    # Safely convert metrics to Python values, handling NaN
    def safe_float(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NaN (no active examples)"
        return float(value)

    # Add actor reward specific metrics
    actor_reward_weight = state.hyperparameters.get("actor_reward_weight", 0.0)
    actor_reward_metrics = {}
    if actor_reward_weight > 0.0:
        if "reward_gradient_losses" in batch_data.metrics:
            reward_grad_loss = batch_data.metrics["reward_gradient_losses"]
            actor_reward_metrics = {
                "Actor Reward Weight": float(actor_reward_weight),
                "Reward Gradient Loss": safe_float(reward_grad_loss.mean().item()),
                "First Reward Gradient Loss": safe_float(reward_grad_loss[0].item()),
                "PG vs Reward Ratio": safe_float(metrics.pg_loss / reward_grad_loss.mean().item()) if reward_grad_loss.mean().item() != 0 else "inf",
            }

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
            # Mean metrics
            "Loss": safe_float(metrics.loss),
            "Policy Gradient Loss": safe_float(metrics.pg_loss),
            "Actor Reasoning Log Probs": float(metrics.actor_logprobs),
            "Critic Reasoning Log Probs": float(metrics.critic_logprobs),
            "Actor Answer Log Probs": float(metrics.actor_answer_logprobs),
            "Critic Answer Log Probs": float(metrics.critic_answer_logprobs),
            "KL": float(kl_to_log),
            "KL Type": "Raw KL" if kl_to_log == metrics.kl else "Weighted KL",
            "PPO Ratio": (
                float(metrics.ppo_ratio) if metrics.ppo_ratio is not None else None
            ),
            "PPO Clipped Ratio": (
                float(metrics.ppo_clipped_ratio)
                if metrics.ppo_clipped_ratio is not None
                else None
            ),
            "Advantage": safe_float(metrics.advantage),
            "Normalized Reward": float(metrics.normalized_reward),
            
            # Raw unfiltered loss metrics
            "Raw Loss": safe_float(raw_mean_loss),
            "Raw Policy Gradient Loss": safe_float(raw_mean_pg_loss),
            "Raw First Loss": safe_float(raw_first_loss),
            "Raw First Policy Gradient Loss": safe_float(raw_first_pg_loss),
            
            # First example metrics
            "First Loss": safe_float(metrics.first_loss),
            "First Policy Gradient Loss": safe_float(metrics.first_pg_loss),
            "First Actor Reasoning Log Probs": float(metrics.first_actor_logprobs),
            "First Critic Reasoning Log Probs": float(metrics.first_critic_logprobs),
            "First Actor Answer Log Probs": float(metrics.first_actor_answer_logprobs),
            "First Critic Answer Log Probs": float(metrics.first_critic_answer_logprobs),
            "First KL": float(first_kl_to_log),
            "First KL Type": "Raw KL" if first_kl_to_log == metrics.first_kl else "Weighted KL",
            "First Advantage": safe_float(metrics.first_advantage),
            "First Normalized Reward": float(metrics.first_normalized_reward),
            
            # Other metrics
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
            "EI Enabled": ei_enabled,
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
            "Use PPO": bool(state.hyperparameters["use_ppo"]),
            "Markovian": bool(state.hyperparameters.get("markovian", True)),
        },
    }
    
    # Add actor reward metrics if using actor rewards
    if actor_reward_metrics:
        log_entry["Actor Reward Metrics"] = actor_reward_metrics
    
    # Add active-only metrics if available
    if "active_only_loss" in batch_data.metrics:
        log_entry["Training Metrics"]["Active Only Loss"] = safe_float(batch_data.metrics["active_only_loss"])
    if "active_only_pg_loss" in batch_data.metrics:
        log_entry["Training Metrics"]["Active Only PG Loss"] = safe_float(batch_data.metrics["active_only_pg_loss"])

    # Write to log file
    with open(state.log_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")


def evaluate_model_generic(
    actor_model,
    critic_model,
    tokenizer,
    device,
    test_data,
    hyperparameters,
    answer_extractor_fn,
    answer_comparator_fn=None,
    batch_size=16,
    num_samples=None,
    task_name="Task",
    max_answer_tokens=10
):
    """Generic evaluation function for all task types using critic model for answer generation.
    
    This follows the Markovian framework:
    1. Actor generates CoT at training temperature
    2. Critic generates answer deterministically (temp 0) after "Answer:"
    3. Extract and compare answers using task-specific functions
    
    Args:
        actor_model: The model to evaluate
        critic_model: The critic model (frozen)
        tokenizer: Tokenizer for the models
        device: Device to run on
        test_data: List of (question, answer) tuples
        hyperparameters: Training hyperparameters
        answer_extractor_fn: Function to extract predicted answer from generated text
                             Signature: (generated_text: str) -> extracted_answer
        answer_comparator_fn: Optional function to compare predicted vs gold answers
                             Signature: (predicted, gold) -> bool
                             If None, uses simple equality (predicted == gold)
        batch_size: Evaluation batch size
        num_samples: Optional limit on number of samples to evaluate
        task_name: Name for progress bar
        max_answer_tokens: Maximum tokens to generate for answer
        
    Returns:
        tuple: (accuracy, detailed_results)
    """
    # Limit number of samples if specified
    if num_samples and num_samples < len(test_data):
        test_data = random.sample(test_data, num_samples)
    
    # Default comparator is simple equality
    if answer_comparator_fn is None:
        answer_comparator_fn = lambda pred, gold: pred == gold
    
    actor_model.eval()
    
    correct = 0
    results = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc=f"Evaluating {task_name}"):
        batch = test_data[i:i+batch_size]
        questions, answers = zip(*batch)
        
        # Stage 1: Generate CoT with actor model
        reasoning_prompts = [
            construct_prompts(question=q, hyperparameters=hyperparameters)
            for q in questions
        ]
        
        inputs = tokenizer(
            reasoning_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=hyperparameters.get("question_length", 512)
        ).to(device)
        
        with torch.no_grad():
            reasoning_outputs = actor_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=hyperparameters.get("cot_length", 100),
                min_new_tokens=hyperparameters.get("cot_length", 100),
                do_sample=True,
                temperature=hyperparameters.get("temperature", 1.0),
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        cot_texts = tokenizer.batch_decode(
            reasoning_outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Stage 2: Generate answer with critic model (deterministic)
        include_question_in_eval = not hyperparameters.get("markovian", True)
        answer_prompts = [
            construct_prompts(
                question=q,
                hyperparameters=hyperparameters,
                reasoning=cot,
                include_question=include_question_in_eval
            )
            for q, cot in zip(questions, cot_texts)
        ]
        
        answer_inputs = tokenizer(
            answer_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=hyperparameters.get("question_length", 512) + hyperparameters.get("cot_length", 100)
        ).to(device)
        
        with torch.no_grad():
            answer_outputs = critic_model.generate(
                input_ids=answer_inputs.input_ids,
                attention_mask=answer_inputs.attention_mask,
                max_new_tokens=max_answer_tokens,
                do_sample=False,  # Deterministic
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_answers = tokenizer.batch_decode(
            answer_outputs[:, answer_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract predicted answers using task-specific function
        predicted_answers = [answer_extractor_fn(ans) for ans in generated_answers]
        
        # Calculate accuracy using task-specific comparator
        for q, cot, gen_ans, pred, gold in zip(questions, cot_texts, generated_answers, predicted_answers, answers):
            is_correct = answer_comparator_fn(pred, gold)
            if is_correct:
                correct += 1
                
            # Use both "is_correct" and "correct" for backwards compatibility
            results.append({
                "question": q,
                "reasoning": cot,
                "generated_answer": gen_ans,
                "predicted": pred,
                "answer": gold,
                "gold": gold,  # Alias for backwards compatibility
                "correct": is_correct,
                "is_correct": is_correct,  # Alias for backwards compatibility
            })
    
    accuracy = correct / len(test_data) if len(test_data) > 0 else 0.0
    return accuracy, results


def evaluate_model_on_mmlu(
    actor_model,
    critic_model,
    tokenizer,
    device,
    test_data,
    hyperparameters,
    batch_size=16,
    num_samples=500
):
    """Evaluate MMLU - extract letter A-E from generated answer."""
    def extract_letter(text):
        matches = re.findall(r"[A-E]", text.upper())
        return matches[0] if matches else "X"
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, extract_letter,
        batch_size=batch_size, num_samples=num_samples,
        task_name="MMLU", max_answer_tokens=10
    )


def save_results_mmlu(output_dir, model_path, model_type, accuracy, results, total, subject=None, batch_index_override=None):
    """Append MMLU evaluation results to a JSONL file (mirrors GSM8K pipeline)."""
    # Determine batch index from checkpoint filename if available
    batch_index = None
    if batch_index_override is not None:
        batch_index = int(batch_index_override)
    elif model_path:
        basename = os.path.basename(model_path)
        match = re.search(r"model_batch_(\d+)\.pt$", basename)
        if not match:
            match = re.search(r"model_(\d+)_", basename)
        if match:
            batch_index = int(match.group(1))

    # Prepare entry
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_index": batch_index,
        "accuracy": accuracy,
        "model_path": model_path,
        "model_type": model_type,
        "total_examples": total,
        "subject": subject,
        "results": results,
    }

    # Append to a task-level JSONL file (same as GSM8K style)
    results_file = os.path.join(output_dir, f"mmlu_results_{model_type}.jsonl")
    with open(results_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")

    return results_file


def evaluate_model_on_aqua(actor_model, critic_model, tokenizer, device, test_data, hyperparameters, batch_size=16):
    """Evaluate AQuA - extract letter A-E from generated answer."""
    def extract_letter(text):
        matches = re.findall(r"[A-E]", text.upper())
        return matches[0] if matches else "X"
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters, extract_letter,
        batch_size=batch_size,
        task_name="AQuA", max_answer_tokens=10
    )


def evaluate_model_on_numeric(actor_model, critic_model, tokenizer, device, test_data, hyperparameters, batch_size=16):
    """Evaluate numeric-answer tasks (e.g., SVAMP, MATH) - handles LaTeX formatting."""
    def normalize_numeric(text: str) -> str:
        s = text.strip()
        s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
        s = s.replace("$", "").replace("\\", "")
        s = re.sub(r"\s+", "", s)
        return s
    
    def compare_normalized(pred, gold):
        return normalize_numeric(pred) == normalize_numeric(str(gold))
    
    return evaluate_model_generic(
        actor_model, critic_model, tokenizer, device, test_data,
        hyperparameters,
        answer_extractor_fn=lambda x: x,  # Return raw text, comparison does normalization
        answer_comparator_fn=compare_normalized,
        batch_size=batch_size,
        task_name="Numeric", max_answer_tokens=16
    )


def run_periodic_evaluation(state: TrainingState, checkpoint_path: str = None):
    """Run periodic evaluation on test set for supported tasks."""
    # If GSM8K, evaluate the model
    if state.hyperparameters["task_type"] == "gsm8k":
        colored_print("Evaluation", "Running GSM8K evaluation...", Colors.BOLD)
        
        # Use test split for evaluation
        test_data = list(load_gsm8k_dataset(split="test"))
        
        # Run evaluation in eval mode
        with torch.no_grad():
            state.actor_model.eval()
            accuracy, results = evaluate_model(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"])
            )
            state.actor_model.train()
        
        # Save results into the run directory (e.g., results/gsm8k/<timestamp>)
        model_dir = state.model_save_path
        results_file = save_results(
            model_dir,
            checkpoint_path,  # Still pass for metadata, but override batch index explicitly
            state.hyperparameters["model_type"],
            accuracy,
            results,
            len(test_data),
            batch_index_override=state.batch_index,
        )
        
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)
    
    # If MMLU, evaluate the model
    elif state.hyperparameters["task_type"] == "mmlu":
        colored_print("Evaluation", "Running MMLU evaluation...", Colors.BOLD)
        
        # Get subject filter from hyperparameters
        subject = state.hyperparameters.get("mmlu_subject", None)
        
        # Use test split for evaluation
        test_data = list(load_mmlu_dataset(split="test", subject=subject))
        
        # Run evaluation in eval mode
        with torch.no_grad():
            state.actor_model.eval()
            accuracy, results = evaluate_model_on_mmlu(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"])
            )
            state.actor_model.train()
        
        # Save results in run directory and update combined plot
        model_dir = state.model_save_path
        results_file = save_results_mmlu(
            model_dir,
            checkpoint_path,
            state.hyperparameters["model_type"],
            accuracy,
            results,
            len(test_data),
            subject=subject,
            batch_index_override=state.batch_index,
        )
        
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)

    # If AQuA, evaluate using multiple choice
    elif state.hyperparameters["task_type"] == "aqua":
        colored_print("Evaluation", "Running AQuA evaluation...", Colors.BOLD)
        test_data = list(load_aqua_dataset(split="test"))
        with torch.no_grad():
            state.actor_model.eval()
            accuracy, results = evaluate_model_on_aqua(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
            )
            state.actor_model.train()
        model_dir = state.model_save_path
        results_file = os.path.join(model_dir, f"aqua_results_{state.hyperparameters['model_type']}.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_index": state.batch_index,
            "accuracy": accuracy,
            "model_path": checkpoint_path,
            "model_type": state.hyperparameters["model_type"],
            "total_examples": len(test_data),
            "results": results,
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(results_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)
    
    # If SVAMP, evaluate numeric QA
    elif state.hyperparameters["task_type"] == "svamp":
        colored_print("Evaluation", "Running SVAMP evaluation...", Colors.BOLD)
        test_data = list(load_svamp_dataset(split="test"))
        with torch.no_grad():
            state.actor_model.eval()
            accuracy, results = evaluate_model_on_numeric(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
            )
            state.actor_model.train()
        model_dir = state.model_save_path
        results_file = os.path.join(model_dir, f"svamp_results_{state.hyperparameters['model_type']}.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_index": state.batch_index,
            "accuracy": accuracy,
            "model_path": checkpoint_path,
            "model_type": state.hyperparameters["model_type"],
            "total_examples": len(test_data),
            "results": results,
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(results_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)
    
    # If MATH, evaluate numeric QA
    elif state.hyperparameters["task_type"] == "math":
        colored_print("Evaluation", "Running MATH evaluation...", Colors.BOLD)
        try:
            test_data = list(load_math_dataset(split="test"))
        except Exception:
            test_data = list(load_math_dataset(split="validation"))
        with torch.no_grad():
            state.actor_model.eval()
            accuracy, results = evaluate_model_on_numeric(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"])
            )
            state.actor_model.train()
        model_dir = state.model_save_path
        results_file = os.path.join(model_dir, f"math_results_{state.hyperparameters['model_type']}.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_index": state.batch_index,
            "accuracy": accuracy,
            "model_path": checkpoint_path,
            "model_type": state.hyperparameters["model_type"],
            "total_examples": len(test_data),
            "results": results,
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(results_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)

    # If MathQA, evaluate multiple choice using evaluate_reasoning-like flow
    elif state.hyperparameters["task_type"] == "mathqa":
        colored_print("Evaluation", "Running MathQA evaluation...", Colors.BOLD)
        test_data = list(load_mathqa_dataset(split="test"))
        with torch.no_grad():
            state.actor_model.eval()
            # Reuse evaluate_reasoning-style function already imported as evaluate_model
            accuracy, results = evaluate_model(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
            )
            state.actor_model.train()
        model_dir = state.model_save_path
        results_file = os.path.join(model_dir, f"mathqa_results_{state.hyperparameters['model_type']}.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_index": state.batch_index,
            "accuracy": accuracy,
            "model_path": checkpoint_path,
            "model_type": state.hyperparameters["model_type"],
            "total_examples": len(test_data),
            "results": results,
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(results_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)
    
    # If ARC, evaluate multiple choice using evaluate_reasoning-like flow
    elif state.hyperparameters["task_type"] == "arc":
        colored_print("Evaluation", "Running ARC evaluation...", Colors.BOLD)
        # Get subset from hyperparameters or environment variable
        subset = state.hyperparameters.get("arc_subset") or os.getenv("ARC_SUBSET", "ARC-Challenge")
        # Use validation split for ARC evaluation
        test_data = list(load_arc_dataset(split="validation", subset=subset))
        with torch.no_grad():
            state.actor_model.eval()
            # Reuse evaluate_reasoning-style function already imported as evaluate_model
            accuracy, results = evaluate_model(
                state.actor_model,
                state.critic_model,
                state.tokenizer,
                state.device,
                test_data,
                state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
            )
            state.actor_model.train()
        model_dir = state.model_save_path
        results_file = os.path.join(model_dir, f"arc_results_{state.hyperparameters['model_type']}.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_index": state.batch_index,
            "accuracy": accuracy,
            "model_path": checkpoint_path,
            "model_type": state.hyperparameters["model_type"],
            "subset": subset,
            "total_examples": len(test_data),
            "results": results,
        }
        os.makedirs(model_dir, exist_ok=True)
        with open(results_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        colored_print("Evaluation", f"Completed successfully. Accuracy: {accuracy:.2%}", Colors.GREEN)




def save_checkpoint(state: TrainingState):
    """Save model checkpoint (weights only)."""
    colored_print(
        "Checkpoint", f"Saving model at batch {state.batch_index}", Colors.BOLD
    )
    
    # Only verify critic model weights if weight verification is enabled
    critic_hash_before = None
    enable_weight_verification = state.hyperparameters.get("enable_weight_verification", False)
    
    if enable_weight_verification:
        # Take snapshot of critic model before saving to verify it doesn't change
        critic_hash_before = get_model_hash(state.critic_model)
        colored_print("Critic Verification", f"Critic hash before saving: {critic_hash_before[:16]}...", Colors.BLUE)
    
    # Create checkpoint path with batch index to avoid overwriting
    # Keep legacy checkpoint directory at task level, but store eval outputs in run dir
    checkpoint_dir = os.path.dirname(state.model_save_path)
    checkpoint_filename = f"model_batch_{state.batch_index}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Create adapter path as a subdirectory of the run folder, not at the task level
    adapter_path = os.path.join(state.model_save_path, f"adapter_{state.batch_index}")
    
    # Save only LoRA adapter weights instead of full model
    model_to_save = state.actor_model
    
    # Print diagnostics about the model before trying to save
    colored_print("Model Diagnostics", "Checking model state before saving", Colors.BLUE)
    
    # Check if model is still a PEFT model
    is_peft_model = isinstance(model_to_save, PeftModel)
    colored_print("PEFT Check", f"Is PeftModel: {is_peft_model}", Colors.BLUE if is_peft_model else Colors.RED)
    
    # Check trainable parameters
    total_params = sum(p.numel() for p in model_to_save.parameters())
    trainable_params = sum(p.numel() for p in model_to_save.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    colored_print("Params", f"Total: {total_params:,}, Trainable: {trainable_params:,} ({trainable_ratio:.4%})", 
                Colors.BLUE if trainable_ratio > 0 else Colors.RED)
    
    try:
        if is_peft_model:
            # Get the active adapter
            colored_print("Active Adapter", f"Current active adapter: {model_to_save.active_adapter}", Colors.GREEN)
            
            # Save the adapter using PEFT's built-in method
            colored_print("Saving Adapter", f"Saving adapter to {adapter_path}", Colors.BLUE)
            model_to_save.save_pretrained(adapter_path)
            
            # Also save optimizer state and batch index metadata
            metadata_path = os.path.join(adapter_path, "training_metadata.pt")
            torch.save(
                {
                    "optimizer_state_dict": state.actor_optimizer.state_dict(),
                    "batch_index": state.batch_index,
                    "hyperparameters": state.hyperparameters,
                },
                metadata_path
            )
            
            colored_print("Save Success", f"Saved adapter at batch {state.batch_index} to {adapter_path}", Colors.GREEN)
        else:
            # If not a PEFT model, raise an error - we don't want to save full model
            raise ValueError("Model is not a PEFT model with adapters. Cannot save checkpoint.")
    except Exception as e:
        colored_print("Error Saving Model", f"Error: {str(e)}", Colors.RED)
        
        # Print detailed traceback
        import traceback
        colored_print("Error Traceback", traceback.format_exc(), Colors.RED)
        
        # Don't fall back to saving full model - just report the error
        colored_print("Checkpoint Failed", "Could not save adapter weights. Check model configuration.", Colors.RED)
    
    # Only verify critic model weights if weight verification is enabled
    if enable_weight_verification and critic_hash_before is not None:
        # Verify critic model hasn't changed due to saving process
        critic_hash_after = get_model_hash(state.critic_model)
        if critic_hash_before != critic_hash_after:
            colored_print("WARNING", "Critic model changed during saving process!", Colors.RED)
            colored_print("Hash Before", critic_hash_before, Colors.RED)
            colored_print("Hash After", critic_hash_after, Colors.RED)
        else:
            colored_print("Critic Verification", "Critic model unchanged during saving", Colors.GREEN)
    


def process_batch(state: TrainingState, qa_batch: List[Tuple[str, str]]) -> BatchData:
    """Process a single batch of data.
    
    This function handles both standard and parallel sampling modes transparently.
    When --parallel is enabled, the dataset provides repeated examples and simplified
    advantage calculation is used automatically.
    """
    questions, answers = zip(*qa_batch)

    # Determine if we need to calculate KL
    kl_penalty = state.hyperparameters.get("kl_penalty", None)
    calculate_kl = kl_penalty is not None

    # Generate reasoning from both models and compute KL
    reasoning_output = generate_reasoning_and_kl(state, questions, calculate_kl=calculate_kl)

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
        state.batch_index,  # Pass batch index for accurate EI threshold calculation
    )

    batch_data = BatchData(
        questions=questions,
        answers=answers,
        actor_reasoning=reasoning_output.actor_reasoning,
        critic_reasoning=reasoning_output.critic_reasoning,
        R_mean_actor_logprobs=reasoning_output.R_mean_actor_logprobs,
        R_mean_critic_logprobs=reasoning_output.R_mean_critic_logprobs,
        kl=reasoning_output.kl,
        advantages=advantage_output.advantages,
        normalized_rewards=advantage_output.normalized_rewards,
        actor_answer_logprobs=advantage_output.actor_answer_logprobs,
        critic_answer_logprobs=advantage_output.critic_answer_logprobs,
        losses=losses,
        training_mask=training_mask,
        metrics=metrics,
    )

    return batch_data

def check_gradient_flow(model, loss_type="total"):
    """Check if gradients are flowing through the model after loss.backward()"""
    total_params = 0
    params_with_grad = 0
    max_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                grad_norm = param.grad.data.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
    
    grad_ratio = params_with_grad / total_params if total_params > 0 else 0
    
    if grad_ratio < 1.0:
        colored_print("Gradient Warning", f"{loss_type}: Only {params_with_grad}/{total_params} ({grad_ratio:.1%}) params have gradients", Colors.YELLOW)
    else:
        colored_print("Gradient Check", f"{loss_type}: All {params_with_grad}/{total_params} params have gradients (max: {max_grad_norm:.6f})", Colors.GREEN)
    
    return grad_ratio, max_grad_norm


def update_model(state: TrainingState, batch_data: BatchData) -> float:
    """Perform model update and return gradient norm"""
    num_active = (
        batch_data.training_mask.sum().item()
        if batch_data.training_mask is not None
        else len(batch_data.losses)
    )
    
    # Get gradient accumulation steps from hyperparameters
    accumulation_steps = state.hyperparameters.get("gradient_accumulation_steps", 1)
    
    # Log information about active examples
    if batch_data.training_mask is not None:
        total_examples = len(batch_data.training_mask)
        active_fraction = num_active / total_examples
        colored_print(
            "EI Active:", 
            f"{num_active}/{total_examples} examples ({active_fraction:.1%}) above threshold",
            Colors.BOLD if active_fraction < 0.1 else Colors.CYAN  # Highlight in bold if < 10%
        )
    else:
        # Explicitly log when not using EI
        colored_print(
            "EI Status:", 
            "Disabled - training on all examples without thresholding",
            Colors.GREEN
        )

    grad_norm = 0
    if num_active > 0:
        # Calculate mean loss over active examples, scaled for gradient accumulation
        # Divide by accumulation_steps to get proper gradient scaling
        loss = (
            batch_data.losses
            * (
                batch_data.training_mask
                if batch_data.training_mask is not None
                else 1.0
            )
        ).sum() / (num_active * accumulation_steps)  # Scale by accumulation steps
        loss.backward()

        # Check gradient flow if using actor rewards (only on first few batches to avoid spam)
        actor_reward_weight = state.hyperparameters.get("actor_reward_weight", 0.0)
        if actor_reward_weight > 0.0 and state.batch_index < 5:
            grad_ratio, max_grad = check_gradient_flow(state.actor_model, "Actor Reward Path")

        # Increment accumulation step
        state.accumulation_step += 1
        
        # Check if we should take an optimizer step
        if state.accumulation_step >= accumulation_steps:
            # Calculate gradient norm before optimization step
            grad_norm = get_grad_norm(state.actor_model.parameters())
            
            # Apply gradient clipping
            clip_grad_norm_(state.actor_model.parameters(), 1.0)
            
            # Take optimizer step
            state.actor_optimizer.step()
            state.actor_optimizer.zero_grad()
            
            # Reset accumulation step
            state.accumulation_step = 0
            
            if accumulation_steps > 1:
                colored_print(
                    "Gradient Accumulation:", 
                    f"Optimizer step taken after {accumulation_steps} accumulation steps",
                    Colors.CYAN
                )
        else:
            # Just accumulating gradients, don't calculate grad norm yet
            grad_norm = 0
            if accumulation_steps > 1:
                colored_print(
                    "Gradient Accumulation:", 
                    f"Step {state.accumulation_step}/{accumulation_steps} - accumulating gradients",
                    Colors.YELLOW
                )

    return grad_norm



def train(task_type: str, resume: bool, model_type: str, hyperparameters: dict):
    """Main training loop"""
    state = TrainingState.initialize(task_type, resume, model_type, hyperparameters)
    
    # Display parallel overview if parallel sampling is enabled
    print_parallel_overview(state.hyperparameters)
    
    # Baseline evaluation at timestep 0 (before any training updates)
    if not resume and state.batch_index == 0:
        try:
            if task_type == "gsm8k":
                colored_print("Baseline Eval", "Running GSM8K evaluation at timestep 0", Colors.BOLD)
                test_data = list(load_gsm8k_dataset(split="test"))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                save_results(
                    state.model_save_path,
                    None,
                    state.hyperparameters["model_type"],
                    accuracy,
                    results,
                    len(test_data),
                    batch_index_override=0,
                )
                colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
            elif task_type == "mmlu":
                colored_print("Baseline Eval", "Running MMLU evaluation at timestep 0", Colors.BOLD)
                subject = state.hyperparameters.get("mmlu_subject", None)
                test_data = list(load_mmlu_dataset(split="test", subject=subject))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model_on_mmlu(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                        batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                save_results_mmlu(
                    os.path.dirname(state.model_save_path),
                    None,
                    state.hyperparameters["model_type"],
                    accuracy,
                    results,
                    len(test_data),
                    subject=subject,
                    batch_index_override=0,
                )
                colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
            elif task_type == "aqua":
                colored_print("Baseline Eval", "Running AQuA evaluation at timestep 0", Colors.BOLD)
                test_data = list(load_aqua_dataset(split="test"))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model_on_aqua(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                        batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                # Save JSONL and combined plot
                results_file = os.path.join(state.model_save_path, f"aqua_results_{state.hyperparameters['model_type']}.jsonl")
                entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "batch_index": 0,
                    "accuracy": accuracy,
                    "model_path": None,
                    "model_type": state.hyperparameters["model_type"],
                    "total_examples": len(test_data),
                    "results": results,
                }
                os.makedirs(state.model_save_path, exist_ok=True)
                with open(results_file, "a") as f:
                    json.dump(entry, f)
                    f.write("\n")
                colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
            elif task_type == "svamp":
                colored_print("Baseline Eval", "Running SVAMP evaluation at timestep 0", Colors.BOLD)
                test_data = list(load_svamp_dataset(split="test"))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model_on_numeric(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                        batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                results_file = os.path.join(state.model_save_path, f"svamp_results_{state.hyperparameters['model_type']}.jsonl")
                entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "batch_index": 0,
                    "accuracy": accuracy,
                    "model_path": None,
                    "model_type": state.hyperparameters["model_type"],
                    "total_examples": len(test_data),
                    "results": results,
                }
                os.makedirs(state.model_save_path, exist_ok=True)
                with open(results_file, "a") as f:
                    json.dump(entry, f)
                    f.write("\n")
                colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
            elif task_type == "math":
                colored_print("Baseline Eval", "Running MATH evaluation at timestep 0", Colors.BOLD)
                # Hendrycks MATH has no official test split on HF; often uses test for eval here
                # We will use split="test" if present; otherwise default to validation
                try:
                    test_data = list(load_math_dataset(split="test"))
                except Exception:
                    test_data = list(load_math_dataset(split="validation"))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model_on_numeric(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                        batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                results_file = os.path.join(state.model_save_path, f"math_results_{state.hyperparameters['model_type']}.jsonl")
                entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "batch_index": 0,
                    "accuracy": accuracy,
                    "model_path": None,
                    "model_type": state.hyperparameters["model_type"],
                    "total_examples": len(test_data),
                    "results": results,
                }
                os.makedirs(state.model_save_path, exist_ok=True)
                with open(results_file, "a") as f:
                    json.dump(entry, f)
                    f.write("\n")
                colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
            elif task_type == "arc":
                colored_print("Baseline Eval", "Running ARC evaluation at timestep 0", Colors.BOLD)
                # Get subset from hyperparameters or environment variable
                subset = state.hyperparameters.get("arc_subset") or os.getenv("ARC_SUBSET", "ARC-Challenge")
                test_data = list(load_arc_dataset(split="validation", subset=subset))
                with torch.no_grad():
                    state.actor_model.eval()
                    accuracy, results = evaluate_model(
                        state.actor_model,
                        state.critic_model,
                        state.tokenizer,
                        state.device,
                        test_data,
                        state.hyperparameters,
                        batch_size=get_default_eval_batch_size(state.hyperparameters["batch_size"]),
                    )
                    state.actor_model.train()
                results_file = os.path.join(state.model_save_path, f"arc_results_{state.hyperparameters['model_type']}.jsonl")
                entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "batch_index": 0,
                    "accuracy": accuracy,
                    "model_path": None,
                    "model_type": state.hyperparameters["model_type"],
                    "subset": subset,
                    "total_examples": len(test_data),
                    "results": results,
                }
                os.makedirs(state.model_save_path, exist_ok=True)
                with open(results_file, "a") as f:
                    json.dump(entry, f)
                    f.write("\n")
                 colored_print("Baseline Eval", f"Completed. Accuracy: {accuracy:.2%}", Colors.GREEN)
        except Exception as e:
            colored_print("Baseline Eval", f"Failed: {str(e)}", Colors.YELLOW)
    
    # Get dataset size for tracking full passes
    if task_type == "gsm8k":
        dataset_size = len(load_dataset("openai/gsm8k", "main")["train"])
    else:
        dataset_size = float('inf')  # For generated datasets
    # Use a uniform default checkpoint frequency for all tasks
    default_checkpoint_frequency = 1000
    default_eval_frequency = 200  # Evaluate more often than checkpointing
    
    # Use configured frequency if provided, otherwise use default
    checkpoint_frequency = hyperparameters.get("checkpoint_frequency") or default_checkpoint_frequency
    # If eval_frequency not specified, use checkpoint_frequency (backwards compatible)
    eval_frequency = hyperparameters.get("eval_frequency") or hyperparameters.get("checkpoint_frequency") or default_eval_frequency
    
    batches_per_epoch = dataset_size // hyperparameters["batch_size"]
    completed_epochs = 0

    qa_generator = generate_question_answer_batches(
        num_batches=hyperparameters["num_batches"],
        batch_size=hyperparameters["batch_size"],
        task_type=task_type,
        tokenizer=state.tokenizer,
        hyperparameters=hyperparameters,
    )

    # Only create weight snapshots if verification is enabled
    enable_weight_verification = hyperparameters.get("enable_weight_verification", False)
    critic_full_snapshot = None
    actor_full_snapshot = None

    if enable_weight_verification:
        # Take full snapshots of both models' weights using the new hashing method
        colored_print("Weight Verification", "Taking complete snapshot of critic model weights", Colors.BLUE)
        critic_full_snapshot = get_model_hash(state.critic_model)
        colored_print("Weight Verification", f"Created critic model hash: {critic_full_snapshot[:16]}...", Colors.BLUE)
        
        colored_print("Weight Verification", "Taking complete snapshot of actor model weights", Colors.BLUE)
        actor_full_snapshot = get_model_hash(state.actor_model)
        colored_print("Weight Verification", f"Created actor model hash: {actor_full_snapshot[:16]}...", Colors.BLUE)
    else:
        colored_print("Weight Verification", "Weight verification disabled", Colors.YELLOW)
    
    try:
        for batch_index in range(state.batch_index, hyperparameters["num_batches"]):
            state.batch_index = batch_index
            print_batch_delimiter()
            colored_print("Batch:", str(batch_index), Colors.BOLD, inline=True)

            try:
                qa_batch = next(qa_generator)
            except StopIteration:
                # Reset generator if we run out of data
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

            # Update history and log
            state.previous_normalized_rewards.extend(
                batch_data.normalized_rewards.float().detach().cpu().numpy()
            )
            state.previous_advantages.extend(
                batch_data.advantages.float().detach().cpu().numpy()
            )

            metrics = LogMetrics.from_batch(
                batch_data,
                grad_norm,
                state.previous_advantages,
                state.hyperparameters["batch_size"],
            )
            log_batch_results(state, batch_data, metrics)

            # Periodic plotting via subprocess (non-blocking)
            try:
                plot_every = state.hyperparameters.get("plot_every", 15)
                if plot_every and plot_every > 0 and batch_index > 0 and (batch_index % plot_every == 0):
                    plotter_path = os.path.join(os.path.dirname(__file__), "plot_training_metrics.py")
                    window = str(state.hyperparameters.get("plot_window_size", 10))
                    cmd = [
                        sys.executable,
                        plotter_path,
                        "--window_size", window,
                        "--files", state.log_file,
                    ]
                    subprocess.Popen(cmd)
            except Exception as e:
                colored_print("Plotting Error", f"Failed to spawn plotter: {str(e)}", Colors.YELLOW)

            # Save checkpoint periodically
            if batch_index % checkpoint_frequency == 0 and batch_index > 0:
                save_checkpoint(state)
            
            # Run evaluation periodically (independent of checkpointing)
            if batch_index % eval_frequency == 0 and batch_index > 0:
                checkpoint_path = os.path.join(os.path.dirname(state.model_save_path), f"model_batch_{batch_index}.pt")
                run_periodic_evaluation(state, checkpoint_path)

            # Every X batches, verify model weights are behaving as expected if verification is enabled
            if enable_weight_verification and batch_index % state.hyperparameters.get("weight_verification_freq", 10) == 0 and batch_index > 0:
                colored_print("Weight Verification", f"Performing verification at batch {batch_index}", Colors.BLUE)
                
                # Verify critic model weights aren't changing (frozen correctly)
                verify_all_frozen_weights(state.critic_model, critic_full_snapshot)
                    
                # Verify actor model weights are changing properly
                verify_actor_weights_changing_comprehensive(state.actor_model, actor_full_snapshot)
            


    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if completed_epochs > 0:
            print("Saving checkpoint and running final evaluation")
            save_checkpoint(state)
        else:
            print("No checkpoint saved (no full epochs completed)")
        return
    
    # Handle any remaining accumulated gradients at the end of training
    if state.accumulation_step > 0:
        colored_print(
            "Gradient Accumulation:", 
            f"Applying final accumulated gradients ({state.accumulation_step} steps)",
            Colors.CYAN
        )
        grad_norm = get_grad_norm(state.actor_model.parameters())
        clip_grad_norm_(state.actor_model.parameters(), 1.0)
        state.actor_optimizer.step()
        state.actor_optimizer.zero_grad()
        state.accumulation_step = 0



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
    kl_penalty: Optional[float]
    gradient_accumulation_steps: int
    batch_size: int
    normalize_loss: bool
    lr: float
    num_batches: int
    ppo_epsilon: float
    checkpoint_frequency: Optional[int]
    eval_frequency: Optional[int]
    weight_verification_freq: int
    enable_weight_verification: bool
    # LoRA parameters
    lora_rank: int
    lora_alpha: float
    # Debug options
    debug_repeat_datapoint: bool
    # Parallel sampling (whole-batch repetition)
    parallel: bool = False
    # Markovian vs Non-Markovian rewards
    markovian: bool = True
    # Actor reward gradients
    actor_reward_weight: float = 0.0
    # Plotting controls
    plot_every: int = 15
    plot_window_size: int = 10

    @classmethod
    def from_args(cls, args):
        """Create config from parsed command line arguments"""
        # Handle markovian flag logic: default True unless --no-markovian is specified
        markovian_mode = not args.no_markovian
        
        # Determine task-specific default CoT length if not explicitly provided
        cot_defaults = {
            "wiki_continuation": 50,
            "wiki_compression": 50,
            "gsm8k": 100,
            "arithmetic": 150,
            "arithmetic-negative": 150,
            "mmlu": 150,
            "mathqa": 150,
        }
        final_cot_length = (
            args.cot_length if args.cot_length is not None else cot_defaults.get(args.task_type, 50)
        )
        
        # Create config with all arguments
        return cls(
            task_type=args.task_type,
            model_type=args.model_type,
            resume=args.resume,
            use_ei=args.use_ei,
            use_ppo=args.use_ppo,
            cot_length=final_cot_length,
            r=args.r,
            temperature=args.temperature,
            question_length=args.question_length,
            target_length=args.target_length,
            kl_penalty=args.kl_penalty,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_size=args.batch_size,
            normalize_loss=args.normalize_loss,
            lr=args.lr,
            num_batches=args.num_batches,
            ppo_epsilon=args.ppo_epsilon,
            checkpoint_frequency=args.checkpoint_frequency,
            eval_frequency=args.eval_frequency,
            weight_verification_freq=args.weight_verification_freq,
            enable_weight_verification=args.enable_weight_verification,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            debug_repeat_datapoint=args.debug_repeat_datapoint,
            parallel=args.parallel,
            markovian=markovian_mode,
            actor_reward_weight=args.actor_reward_weight,
            plot_every=args.plot_every,
            plot_window_size=args.plot_window_size,
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
            "arithmetic-negative",
            "gsm8k",
            "mmlu",
            "math",
            "mathqa",
            "svamp",
            "aqua",
            "arc",
            "wiki_compression",
            "wiki_continuation",
        ],
        help="Task type (default: wiki_continuation)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=["llama", "llama3.2-1b", "mistral", "gpt2", "tinystories", "phi", "phi-4", "qwen3", "qwen3-14b", "gemma-3", "gemma-3-small"],
        help="Model type (default: llama)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use_ei",
        type=float,
        default=None,
        help="Use Expert Iteration with specified number of standard deviations (default: None, which disables thresholding)",
    )
    parser.add_argument("--use_ppo", action="store_true")
    parser.add_argument("--cot_length", type=int, default=None)
    parser.add_argument("--r", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--question_length", type=int, default=50)
    parser.add_argument("--target_length", type=int, default=50)
    parser.add_argument("--kl_penalty", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                       help="Number of batches to accumulate gradients before updating (default: 1)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--normalize_loss", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_batches", type=int, default=100000)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        help="How often to save model checkpoints in batches (default: 1000)",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        help="How often to run evaluation on test set in batches (default: same as checkpoint_frequency)",
    )
    # Add weight verification frequency parameter
    parser.add_argument(
        "--weight_verification_freq",
        type=int,
        default=10,
        help="Frequency (in batches) to run comprehensive weight verification (default: 10)",
    )
    # Add new flag to enable/disable weight verification
    parser.add_argument(
        "--enable_weight_verification",
        action="store_true",
        help="Enable weight verification (disabled by default)",
    )

    # MMLU-specific arguments
    parser.add_argument(
        "--mmlu_subject",
        type=str,
        default=None,
        help="Specific MMLU subject to train on (default: all subjects)",
    )
    parser.add_argument(
        "--mmlu_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="MMLU dataset split to use (default: validation)",
    )
    # LoRA configuration arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Rank for LoRA adapter (default: 8). Higher values use more parameters but can capture more complex patterns.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="Alpha scaling for LoRA adapter (default: 16). Usually set to 2x the rank.",
    )
    # Debug options
    parser.add_argument(
        "--debug_repeat_datapoint",
        action="store_true",
        help="Debug mode: train on the same datapoint repeatedly to test optimization",
    )
    # Parallel sampling (whole-batch repetition)
    # Parallel mode is on by default; use --no-parallel to disable
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel sampling (process different examples in each batch)",
    )
    parser.set_defaults(parallel=True)
    # Markovian vs Non-Markovian reward calculation
    parser.add_argument(
        "--no-markovian",
        action="store_true",
        help="Use Non-Markovian rewards P(answer|question,CoT) instead of default Markovian P(answer|CoT)",
    )
    # Actor reward gradients
    parser.add_argument(
        "--actor_reward_weight",
        type=float,
        default=1.0,
        help="Weight for actor reward gradients. If > 0, use actor model for rewards with this weight (default: 1.0)",
    )
    # Plotting controls
    parser.add_argument(
        "--plot_every",
        type=int,
        default=15,
        help="How often (in batches) to spawn plotting process (0 disables). Default: 15",
    )
    parser.add_argument(
        "--plot_window_size",
        type=int,
        default=10,
        help="Smoothing window size passed to plot_training_metrics.py (default: 10)",
    )

    args = parser.parse_args()
    # Apply task-specific default training batch size if not provided
    if args.batch_size is None:
        args.batch_size = get_default_train_batch_size(args.task_type)
    config = TrainingConfig.from_args(args)
    main(config)



