#!/usr/bin/env python
"""
Minimal Vector Quantization (VQ) training implementation for language models.

This standalone script implements VQ training with:
1. Codebook loss - hidden states should be close to token embeddings
2. Answer log probability - reasoning should lead to correct answers
3. Prevention of empty reasoning by penalizing common tokens
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import random
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union, Any

# -----------------------------------------
# Configuration and Data Structures
# -----------------------------------------

@dataclass
class VQConfig:
    """Configuration for VQ training"""
    # Model parameters
    model_name: str = "google/gemma-3-1b-it"
    lora_rank: int = 8
    lora_alpha: int = 16
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_batches: int = 1000
    debug_repeat_datapoint: bool = False  # Whether to use the same datapoint repeatedly
    use_8bit_adam: bool = False       # New: Use 8-bit Adam optimizer
    
    # VQ specific parameters
    vq_reason_length: int = 50
    vq_sampling_temp: float = 0.5  # Temperature for softmax sampling (lower=more peaked, higher=more diverse)
    vq_use_argmax: bool = False # Use argmax instead of sampling for VQ tokens
    actor_hidden_layer_index: int = -2 # Which hidden layer from actor to use (-1 for last, -2 for second to last etc)
    normalize_reasoning_states: bool = True # Normalize the actor hidden states before VQ process
    vq_sequential_generation: bool = False # Generate VQ reasoning tokens sequentially (autoregressively)
    use_gumbel_softmax_vq: bool = False # New: Use Gumbel-Softmax for VQ
    gumbel_tau: float = 1.0           # New: Temperature for Gumbel-Softmax
    placeholder_token_id: Optional[int] = None  # Token ID to use as placeholder, defaults in setup_model
    debug_answer_is_question: bool = False # New: If true, answer is a copy of the question
    
    # Logging parameters
    print_frequency: int = 10
    plot_frequency: int = 50 # Changed from plot_interval, new default for direct batch frequency
    checkpoint_frequency: int = 50
    save_final_model: bool = True
    debug_gradients: bool = False  # Add debug flag for gradient logging
    weights_check_frequency: int = 100 # Frequency to check model weights
    codebook_loss_weight: float = 0.25 # Weight for codebook loss term
    
    # Task parameters
    task_type: str = "arithmetic"
    context_length: int = 100  # Length of input context (for wiki)
    max_target_length: int = 100  # Maximum length of target (for wiki)
    cot_length: int = 50
    
    # Resume parameters
    resume_from_checkpoint: Optional[str] = None # Path to checkpoint directory to resume from

    # Testing parameters
    copy_test_frequency: int = 0  # Run copy test every n batches, 0 to disable
    copy_test_samples: int = 3  # Number of samples to use for copy test


@dataclass
class VQTrainingState:
    """State for VQ training"""
    # Models and tokenizer
    actor_model: torch.nn.Module  # Model that generates reasoning (trainable)
    critic_model: torch.nn.Module  # Model that evaluates reasoning (frozen)
    tokenizer: Any
    optimizer: torch.optim.Optimizer
    device: torch.device
    
    # Token embeddings
    token_embeddings: torch.Tensor
    
    # Training state
    batch_index: int = 0
    
    # Tracking metrics
    metrics_history: Dict[str, List[float]] = None
    
    # Output paths
    output_dir: str = None
    log_file: str = None
    
    # Config
    config: VQConfig = None
    answer_suffix_len: Optional[int] = None # New: To store tokenized length of " Answer: "


# -----------------------------------------
# Model Setup Functions
# -----------------------------------------

def setup_model(config: VQConfig) -> VQTrainingState:
    """Initialize the actor and critic models, tokenizer, and training state, handling resume logic."""
    print(f"Loading model: {config.model_name}")
    
    start_batch_index = 0
    loaded_metrics_history = None
    output_dir = None
    log_file = None
    original_config_loaded = None

    if config.resume_from_checkpoint:
        print(f"Attempting to resume from checkpoint: {config.resume_from_checkpoint}")
        if not os.path.isdir(config.resume_from_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint directory not found: {config.resume_from_checkpoint}")

        output_dir = os.path.dirname(config.resume_from_checkpoint)
        log_file = os.path.join(output_dir, "log.jsonl")
        print(f"Resuming in directory: {output_dir}")

        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Original log file not found for resuming: {log_file}")

        original_config_data = None
        with open(log_file, "r") as f_orig_config:
            try:
                original_config_data_str = f_orig_config.readline()
                original_config_data = json.loads(original_config_data_str)
                original_config_loaded = VQConfig(**original_config_data)
                print("Loaded original config for resume.")
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse original config from {log_file}")
        
        current_config_dict = asdict(config)
        original_config_dict = asdict(original_config_loaded)
        sticky_params = [
            'model_name', 'lora_rank', 'lora_alpha', 'vq_reason_length', 
            'actor_hidden_layer_index', 'normalize_reasoning_states', 
            'vq_sequential_generation', 'use_gumbel_softmax_vq', 'gumbel_tau',
            'placeholder_token_id', 'task_type', 'context_length', 
            'max_target_length', 'cot_length', 'codebook_loss_weight',
            # 'weights_check_frequency' could also be sticky
        ]
        default_vq_config = VQConfig()
        for param_name in sticky_params:
            if getattr(config, param_name) == getattr(default_vq_config, param_name) and \
               hasattr(original_config_loaded, param_name) and \
               getattr(original_config_loaded, param_name) is not None:
                print(f"Resuming with original config value for '{param_name}': {getattr(original_config_loaded, param_name)}")
                setattr(config, param_name, getattr(original_config_loaded, param_name))

        # Determine truncation_point_batch_idx and current_start_batch_index (for the training loop)
        checkpoint_name = os.path.basename(config.resume_from_checkpoint)
        parsed_batch_from_cp_name = -1
        is_regular_checkpoint_format = False
        current_start_batch_index = 0 # Default to 0, will be updated
        truncation_point_batch_idx = -1 # Batch index to truncate logs/metrics AT (inclusive)

        if checkpoint_name.startswith("checkpoint_"):
            try:
                parsed_batch_from_cp_name = int(checkpoint_name.split("_")[-1])
                is_regular_checkpoint_format = True
            except ValueError:
                print(f"Warning: Could not parse batch number from checkpoint name '{checkpoint_name}'.")
                pass # Fall through to other methods or log-based determination

        # Read all metric log entries to determine max batch in log and for filtering
        all_metric_log_entries_tuples = [] # List of (dict_entry, raw_line_string)
        max_batch_in_log = -1
        raw_log_lines_for_rewrite = []

        if os.path.exists(log_file):
            with open(log_file, "r") as f_log_read:
                temp_lines = f_log_read.readlines()
                if len(temp_lines) > 0:
                    raw_log_lines_for_rewrite.append(temp_lines[0]) # Keep config line for rewrite
                    if len(temp_lines) > 1:
                        for line_str in temp_lines[1:]:
                            if line_str.strip():
                                try:
                                    entry_dict = json.loads(line_str)
                                    if 'batch' in entry_dict:
                                        all_metric_log_entries_tuples.append((entry_dict, line_str))
                                        max_batch_in_log = max(max_batch_in_log, entry_dict['batch'])
                                except json.JSONDecodeError:
                                    print(f"Warning: Skipping unparseable log line during scan: {line_str.strip()}")
        
        if is_regular_checkpoint_format:
            truncation_point_batch_idx = parsed_batch_from_cp_name
            current_start_batch_index = parsed_batch_from_cp_name + 1
            print(f"Resuming from checkpoint_{parsed_batch_from_cp_name}. Log/metrics up to batch {truncation_point_batch_idx}. Next batch: {current_start_batch_index}.")
        elif checkpoint_name == "interrupted_checkpoint":
            truncation_point_batch_idx = max_batch_in_log
            current_start_batch_index = max_batch_in_log + 1
            print(f"Resuming from interrupted_checkpoint. Log/metrics up to batch {truncation_point_batch_idx} (last logged). Next batch: {current_start_batch_index}.")
        else:
            # Fallback for unusually named checkpoint dir, or if path is to run dir itself.
            print(f"Warning: Checkpoint path {config.resume_from_checkpoint} is not 'checkpoint_N' or 'interrupted_checkpoint'. Using log to determine resume point.")
            truncation_point_batch_idx = max_batch_in_log
            current_start_batch_index = max_batch_in_log + 1 # If max_batch_in_log is -1, this becomes 0.
            print(f"Using log: Log/metrics up to batch {truncation_point_batch_idx}. Next batch: {current_start_batch_index}.")

        if truncation_point_batch_idx == -1 and current_start_batch_index == 0:
            print("Warning: No valid checkpoint batch num or log entries found. Starting as if from scratch in this directory, or there might be issues.")

        # Load metrics history, TRUNCATING based on truncation_point_batch_idx
        loaded_metrics_history = {key: [] for key in ['total_loss', 'answer_logprobs', 'grad_norm', 'quantization_error_norm', 'codebook_loss']} # Ensure all keys exist
        for entry_dict, _ in all_metric_log_entries_tuples:
            if entry_dict['batch'] <= truncation_point_batch_idx:
                for key_metric in loaded_metrics_history.keys():
                    if key_metric in entry_dict:
                        loaded_metrics_history[key_metric].append(entry_dict[key_metric])
        print(f"Loaded metrics history with {len(loaded_metrics_history['total_loss'])} entries (up to batch {truncation_point_batch_idx}).")

        # Rewrite the log file to be truncated
        with open(log_file, "w") as f_rewrite:
            f_rewrite.write(json.dumps(original_config_data) + "\n") # Write original config (already loaded)
            
            lines_to_write_for_metrics = []
            for entry_dict, raw_line_str in all_metric_log_entries_tuples:
                if entry_dict['batch'] <= truncation_point_batch_idx:
                    # Ensure raw_line_str ends with a newline if it contains data
                    current_line_to_write = raw_line_str.strip() # Remove existing newlines/whitespace
                    if current_line_to_write: # Only add newline if there's content
                        lines_to_write_for_metrics.append(current_line_to_write + '\n')
            
            f_rewrite.writelines(lines_to_write_for_metrics)
        print(f"Log file {log_file} has been rewritten, truncated to batch {truncation_point_batch_idx}.")
        
        # Load actor model from checkpoint path (config.resume_from_checkpoint)
        actor_model = AutoModelForCausalLM.from_pretrained(
            config.resume_from_checkpoint, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        # Critic is reloaded based on the (potentially original) config.model_name
        critic_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.resume_from_checkpoint) # Load tokenizer from checkpoint path
        start_batch_index = current_start_batch_index # Use the determined start index for the VQState

    else: # Not resuming, standard setup
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", config.task_type, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "log.jsonl")
        
        # Initialize log file with config
        with open(log_file, "w") as f:
            json.dump(asdict(config), f)
            f.write("\\n")
        
        actor_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        critic_model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Common setup for both resume and new run
    for param in actor_model.parameters():
        param.requires_grad = True
    for param in critic_model.parameters():
        param.requires_grad = False
        
    print("Actor model parameters enabled for gradients")
    print("Critic model parameters frozen - no gradients will be computed")
    
    actor_trainable = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
    critic_trainable = sum(p.numel() for p in critic_model.parameters() if p.requires_grad)
    print(f"Actor trainable parameters: {actor_trainable}")
    print(f"Critic trainable parameters: {critic_trainable} (should be 0)")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if config.placeholder_token_id is None:
        if tokenizer.unk_token_id is not None:
            config.placeholder_token_id = tokenizer.unk_token_id
            print(f"Using UNK token (ID: {config.placeholder_token_id}) as default placeholder for VQ reasoning.")
        elif tokenizer.eos_token_id is not None:
            config.placeholder_token_id = tokenizer.eos_token_id
            print(f"Warning: UNK token not found. Using EOS token (ID: {config.placeholder_token_id}) as placeholder for VQ reasoning.")
        else:
            raise ValueError("Tokenizer does not have an UNK or EOS token, and no placeholder_token_id was specified.")
    else:
        try:
            _ = tokenizer.decode([config.placeholder_token_id])
            print(f"Using pre-configured placeholder_token_id: {config.placeholder_token_id}")
        except Exception as e:
            raise ValueError(f"Provided placeholder_token_id {config.placeholder_token_id} is not valid for the tokenizer: {e}")
    
    if config.lora_rank > 0: # Apply LoRA if resuming and original had LoRA, or if new run with LoRA
        try:
            from peft import LoraConfig, get_peft_model
            if "gpt2" in config.model_name.lower():
                lora_target_modules = ["c_attn", "c_proj"]
            else:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            lora_config = LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha,
                target_modules=lora_target_modules, bias="none", task_type="CAUSAL_LM"
            )
            # If model is already a PeftModel (e.g. from resume_from_checkpoint),
            # this call might error or behave unexpectedly.
            # However, from_pretrained on a PEFT checkpoint should load it as a PeftModel directly.
            # We only call get_peft_model if it's NOT already a PeftModel.
            if not hasattr(actor_model, 'print_trainable_parameters'): # Heuristic: check if it's already PEFT
                 print(f"Applying LoRA with rank={config.lora_rank}, alpha={config.lora_alpha} to ACTOR model.")
                 actor_model = get_peft_model(actor_model, lora_config)
            else:
                print("Actor model appears to be already PEFT-enabled (likely from checkpoint). Skipping get_peft_model.")
            actor_model.print_trainable_parameters()
        except ImportError:
            print("PEFT library not found. Running without LoRA.")
            if config.lora_rank > 0 : # if lora was specified but peft not found
                print("WARNING: LoRA rank > 0 but PEFT not found. LoRA will not be applied.")
    
    device = next(actor_model.parameters()).device
    print(f"Using device: {device}")
    
    token_embeddings = None
    if hasattr(actor_model, 'get_input_embeddings'):
        token_embeddings = actor_model.get_input_embeddings().weight
    else: # Common for PEFT models, need to get from base_model
        if hasattr(actor_model, 'base_model') and hasattr(actor_model.base_model, 'get_input_embeddings'):
             token_embeddings = actor_model.base_model.get_input_embeddings().weight
        elif hasattr(actor_model, 'transformer') and hasattr(actor_model.transformer, 'wte'): # GPT-2 specific
            token_embeddings = actor_model.transformer.wte.weight
        elif hasattr(actor_model, 'model') and hasattr(actor_model.model, 'embed_tokens'): # Other models
            token_embeddings = actor_model.model.embed_tokens.weight
        else:
            raise ValueError("Could not locate token embeddings in model or its base_model")

    print(f"Token embeddings shape: {token_embeddings.shape}")
    
    trainable_params = [p for p in actor_model.parameters() if p.requires_grad]
    optimizer_class = torch.optim.AdamW
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("Using 8-bit AdamW optimizer from bitsandbytes.")
        except ImportError:
            print("Warning: bitsandbytes not found. Falling back to standard AdamW optimizer.")
    
    # Ensure learning_rate from potentially updated config is used for new/resumed optimizer
    current_lr = config.learning_rate 
    optimizer = optimizer_class(trainable_params, lr=current_lr)
    print(f"Using optimizer: {optimizer.__class__.__name__} with LR: {current_lr}")

    if config.resume_from_checkpoint:
        optimizer_path = os.path.join(config.resume_from_checkpoint, "optimizer.pt")
        if os.path.exists(optimizer_path):
            try:
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                print(f"Loaded optimizer state from {optimizer_path}")
            except Exception as e:
                print(f"Warning: Could not load optimizer state from {optimizer_path}: {e}. Initializing new optimizer state.")
        else:
            print(f"Warning: Optimizer state file not found at {optimizer_path}. Initializing new optimizer state.")
    
    # Initialize metrics_history for both new and resumed runs
    # Ensure all_metric_keys are present.
    metrics_history_to_use = {key: [] for key in ['total_loss', 'answer_logprobs', 'grad_norm', 'quantization_error_norm', 'codebook_loss']}

    if loaded_metrics_history is not None: # Resuming run
        # Reconstruct metrics history, including batch_indices and padding with NaN
        # First, determine the maximum length based on any of the loaded metric lists (they should be same length)
        max_len_loaded = 0
        if loaded_metrics_history and loaded_metrics_history.get('total_loss'): # Check if any metrics were loaded
            max_len_loaded = len(loaded_metrics_history['total_loss'])
            
        # Find all batch numbers that were part of the loaded history up to truncation_point_batch_idx
        # This assumes 'batch' was always present in the logged entries that contributed to loaded_metrics_history
        logged_batches_up_to_truncation = []
        for entry_dict_scan, _ in all_metric_log_entries_tuples:
            if entry_dict_scan['batch'] <= truncation_point_batch_idx:
                if 'batch' in entry_dict_scan: # Ensure batch key exists
                    logged_batches_up_to_truncation.append(entry_dict_scan['batch'])
                else: # This case should ideally not happen if logs are consistent
                    print(f"Warning: Log entry for batch around {truncation_point_batch_idx} missing 'batch' key. Cannot accurately get its batch index.")
        
        # Sort and unique batch numbers, in case of any disorder or duplicates in raw log before processing
        # However, all_metric_log_entries_tuples should already be ordered by file read.
        # The critical part is associating the correct batch index to each *position* in the loaded_metrics_history lists.
        # This assumes that loaded_metrics_history['some_metric'][i] corresponds to the i-th valid log entry
        # up to truncation_point_batch_idx.
        
        # We need to populate 'batch_indices' based on the 'batch' field from the *original log entries*
        # that made it into loaded_metrics_history.
        
        temp_metrics_from_log = {key: [] for key in ['total_loss', 'answer_logprobs', 'grad_norm', 'quantization_error_norm', 'codebook_loss']}
        processed_log_batches = [] # Store batch numbers from log to populate batch_indices

        for entry_dict, _ in all_metric_log_entries_tuples:
            if entry_dict['batch'] <= truncation_point_batch_idx:
                if 'batch' not in entry_dict:
                    # This should not happen if logs are well-formed, but good to be defensive
                    print(f"Warning: Log entry around batch {truncation_point_batch_idx} is missing 'batch' key. Skipping this entry for historical data.")
                    continue
                
                processed_log_batches.append(entry_dict['batch'])
                
                for key in ['total_loss', 'answer_logprobs', 'grad_norm', 'quantization_error_norm', 'codebook_loss']:
                    if key == 'batch_indices': # Handled by processed_log_batches
                        continue
                    temp_metrics_from_log[key].append(entry_dict.get(key, float('nan')))
        
        metrics_history_to_use['batch_indices'] = processed_log_batches
        for key in ['total_loss', 'answer_logprobs', 'grad_norm', 'quantization_error_norm', 'codebook_loss']:
            if key != 'batch_indices':
                metrics_history_to_use[key] = temp_metrics_from_log[key]
        
        print(f"Reconstructed metrics_history for resume. Found {len(metrics_history_to_use['batch_indices'])} historical batch entries.")
        if metrics_history_to_use['batch_indices']:
            print(f"  Historical batch range: {min(metrics_history_to_use['batch_indices'])} to {max(metrics_history_to_use['batch_indices'])}")


    state = VQTrainingState(
        actor_model=actor_model, critic_model=critic_model, tokenizer=tokenizer,
        optimizer=optimizer, device=device, token_embeddings=token_embeddings,
        batch_index=start_batch_index, # This is the key for resuming batch count
        metrics_history=metrics_history_to_use,
        output_dir=output_dir, log_file=log_file, config=config,
    )
    
    answer_suffix_str = " Answer: "
    temp_tokenized_suffix = tokenizer(answer_suffix_str, add_special_tokens=False, return_tensors="pt")
    state.answer_suffix_len = temp_tokenized_suffix.input_ids.shape[1]
    
    return state


# -----------------------------------------
# Data Generation Functions
# -----------------------------------------

def generate_arithmetic_qa(num_numbers: int = 15, max_num: int = 100) -> Tuple[str, str]:
    """Generate a simple arithmetic addition question and answer."""
    numbers = [random.randint(1, max_num) for _ in range(num_numbers)]
    question = " + ".join([str(n) for n in numbers])
    answer = str(sum(numbers))
    return question, answer


def load_wiki_dataset(max_samples: int = 1000) -> List[Tuple[str, str]]:
    """Load Wikipedia dataset from disk or download if needed.
    
    Returns pairs of (context, continuation) where the task is to generate the continuation.
    """
    try:
        from datasets import load_dataset
        
        # Try to load local dataset first
        wiki_path = os.path.join("data", "wiki")
        if os.path.exists(wiki_path):
            print(f"Loading wiki dataset from {wiki_path}")
            try:
                dataset = load_dataset(wiki_path)
                if "train" in dataset:
                    wiki_data = dataset["train"]
                else:
                    wiki_data = dataset
            except Exception as e:
                print(f"Error loading local dataset: {e}")
                print("Falling back to downloading dataset...")
                wiki_data = None
        else:
            wiki_data = None
            
        # If local loading failed, download from HuggingFace
        if wiki_data is None:
            print("Downloading Wikipedia dataset from HuggingFace...")
            dataset = load_dataset("wikipedia", "20220301.en", split="train[:10000]")
            wiki_data = dataset
        
        # Extract context/continuation pairs
        pairs = []
        for i, item in enumerate(wiki_data):
            if i >= max_samples:
                break
                
            # Extract text
            if "text" in item:
                text = item["text"]
            elif "article" in item:
                text = item["article"]
            else:
                continue
                
            # Skip short texts
            if len(text) < 200:
                continue
                
            # Split into context and continuation
            split_point = len(text) // 2
            context = text[:split_point]
            continuation = text[split_point:split_point + 500]  # Limit continuation length
            
            pairs.append((context, continuation))
            
        print(f"Loaded {len(pairs)} context/continuation pairs from Wikipedia")
        return pairs
        
    except ImportError:
        print("Error: datasets library not found. Please install with 'pip install datasets'")
        print("Generating synthetic wikipedia-like data instead...")
        
        # Generate synthetic Wikipedia-like data
        pairs = []
        topics = ["science", "history", "technology", "art", "literature", "sports", "music"]
        
        for _ in range(max_samples):
            topic = random.choice(topics)
            context = f"This article is about {topic}. It explores various aspects of {topic} throughout history."
            context += f" There are many important events in the development of {topic}."
            
            continuation = f"One of the key developments in {topic} was the discovery of new methods."
            continuation += f" As {topic} evolved, it influenced many areas of human knowledge."
            
            pairs.append((context, continuation))
            
        print(f"Generated {len(pairs)} synthetic context/continuation pairs")
        return pairs


def get_batch_data(batch_size: int, task_type: str, config: VQConfig) -> List[Tuple[str, str]]:
    """Generate a batch of question-answer pairs based on task type."""
    qa_batch_raw: List[Tuple[str, str]]
    
    # --- Step 1: Determine initial qa_batch_raw --- 
    if config.debug_repeat_datapoint:
        if not hasattr(get_batch_data, "debug_datapoint"):
            # Create a single datapoint to reuse
            if task_type == "arithmetic":
                debug_question, debug_answer = generate_arithmetic_qa()
                print(f"DEBUG MODE: Using single arithmetic datapoint repeatedly")
                print(f"Question: {debug_question}")
                print(f"Answer: {debug_answer}")
            elif task_type == "wiki_continuation":
                # For wiki, load_wiki_dataset might be too slow for repeated calls if not cached.
                # We rely on its internal caching or one-time load.
                wiki_data_for_debug = load_wiki_dataset(max_samples=10) # Get a small sample
                if wiki_data_for_debug:
                    # Original character-based truncation for debug display
                    # The actual token length handling is done later in process_batch or by AIQ logic
                    debug_context_str = wiki_data_for_debug[0][0]
                    debug_continuation_str = wiki_data_for_debug[0][1]
                    print(f"DEBUG MODE: Using single wiki continuation datapoint repeatedly")
                    print(f"Original Context (truncated for display): {debug_context_str[:100]}...")
                    print(f"Original Continuation (truncated for display): {debug_continuation_str[:100]}...")
                    get_batch_data.debug_datapoint = (debug_context_str, debug_continuation_str)
                else:
                    # Fallback if wiki loading fails for debug
                    get_batch_data.debug_datapoint = ("Debug Wiki Context", "Debug Wiki Continuation")
            else:
                get_batch_data.debug_datapoint = ("Sample question", "Sample answer")
        
        qa_batch_raw = [get_batch_data.debug_datapoint] * batch_size
    
    else: # Normal data generation (not debug_repeat_datapoint)
        if task_type == "arithmetic":
            qa_batch_raw = [generate_arithmetic_qa() for _ in range(batch_size)]
        elif task_type == "wiki_continuation":
            if not hasattr(get_batch_data, "wiki_data"):
                get_batch_data.wiki_data = load_wiki_dataset(max_samples=1000)
            
            selected_pairs = random.sample(
                get_batch_data.wiki_data, 
                min(batch_size, len(get_batch_data.wiki_data))
            )
            
            # Character-based truncation (original logic for non-AIQ path)
            # Token-based truncation happens later in process_batch
            truncated_pairs = []
            for context, continuation in selected_pairs:
                truncated_context = context[:config.context_length * 4]  
                truncated_continuation = continuation[:config.max_target_length * 4]
                truncated_pairs.append((truncated_context, truncated_continuation))
            qa_batch_raw = truncated_pairs
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    # --- Step 2: Apply debug_answer_is_question if active --- 
    # This print will now always execute before the AIQ check
    # print(f"DEBUG get_batch_data: Value of config.debug_answer_is_question before check: {config.debug_answer_is_question}") # Commented out

    if config.debug_answer_is_question:
        # print(f"DEBUG get_batch_data: debug_answer_is_question is True. Overriding answers to be copies of questions.") # Commented out
        processed_qa_batch = []
        for q_text, _ in qa_batch_raw: # Original answer from qa_batch_raw is ignored
            processed_qa_batch.append((q_text, q_text)) # Answer is now a copy of question
        return processed_qa_batch # Unindented to align with processed_qa_batch=[]
    else:
        # If not AIQ, return the qa_batch_raw (which could be from debug_repeat or normal path)
        return qa_batch_raw


def get_model_specific_tokens_for_vq(model_name_str: str) -> Dict[str, str]:
    """
    Returns a dictionary of model-specific tokens for prompt construction.
    Simplified version for vq_training_minimal.py.
    Actual token strings should align with the tokenizer's special tokens.
    """
    lower_model_name = model_name_str.lower()
    
    # For Gemma, using placeholders. Actual tokens from tokenizer.bos_token, etc. or chat template are better.
    if "gemma" in lower_model_name:
        # These are illustrative based on src/constants.py but might need tokenizer-specific values
        # For robust Gemma templating, tokenizer.apply_chat_template is preferred if not manually aligning.
        return {
            "bos": "<bos>", # Placeholder, tokenizer.bos_token if available and correct
            "start_of_turn_user": "<start_of_turn>user\n",
            "start_of_turn_model": "<start_of_turn>model\n",
            "end_of_turn": "<end_of_turn>\n",
            "format_type": "gemma"
        }
    elif "mistral" in lower_model_name:
        return {
            "bos": "<s>", # Common BOS for Mistral
            "inst_start": "[INST]",
            "inst_end": "[/INST]",
            "format_type": "mistral"
        }
    elif "phi" in lower_model_name: # For models like Phi-2, Phi-3, etc.
        return {
            "im_start": "<|im_start|>",
            "im_sep": "<|im_sep|>",
            "im_end": "<|im_end|>",
            "format_type": "phi"
        }
    # Example for Llama, if added later (ensure tokens are correct for specific Llama versions)
    # elif "llama" in lower_model_name:
    #     return {
    #         "bos": "<s>",
    #         "inst_start": "[INST]",
    #         "inst_end": "[/INST]",
    #         "sys_start": "<<SYS>>\n", 
    #         "sys_end": "\n<</SYS>>\n\n",
    #         "format_type": "llama"
    #     }
    else:  # Default for GPT2 and others not explicitly listed (e.g., "standard" type)
        return {
            "bos": "", # GPT-2 usually doesn't prepend BOS via manual prompt construction
            "inst_start": "",
            "inst_end": "",
            "start_of_turn_user": "",
            "start_of_turn_model": "",
            "end_of_turn": "",
            "format_type": "standard"
        }


def construct_prompt(question: str, task_type: str, config: VQConfig) -> str:
    """Construct a prompt for the given question or context, aligned with src/utils.py style and model-specific tokens."""
    
    model_tokens = get_model_specific_tokens_for_vq(config.model_name)
    format_type = model_tokens["format_type"]

    # Define base prompts and suffix based on task_type (as before)
    if task_type == "arithmetic":
        base_prompt_text = f"You will be given an arithmetic problem, which you have {config.vq_reason_length} tokens to work through step-by-step. Question:"
        prompt_type_suffix = "Reasoning:"
    elif task_type == "wiki_continuation":
        base_prompt_text = (
            f"You will need to predict the next {config.max_target_length} tokens which follow the provided passage."
            f"You can write {config.vq_reason_length} thinking tokens which will be your sole context for prediction."
            f"Feel free to be creative in your thinking strategy!\n\nOpening text:"
        )
        prompt_type_suffix = "Helpful Text:"
    else:
        print(f"Warning: construct_prompt received an unexpected task_type: '{task_type}'. Using a generic prompt.")
        return f"Question: {question}\nAnswer:" # Basic fallback

    # Apply model-specific formatting (reasoning is None case from src/utils.py)
    if format_type == "gemma":
        return (
            f"{model_tokens['bos']}{model_tokens['start_of_turn_user']}"
            f"{base_prompt_text} {question}{model_tokens['end_of_turn']}"
            f"{model_tokens['start_of_turn_model']}"
            f"{prompt_type_suffix}"
        )
    elif format_type == "mistral":
        # Mistral format from src/utils.py (reasoning=None case) was: f"{tokens['inst_start']} {base_prompt} {question} {tokens['inst_end']}\n{prompt_type}"
        # BOS is often added by tokenizer automatically for Mistral, but if manual, it's usually `<s>`
        return f"{model_tokens.get('bos', '')}{model_tokens['inst_start']} {base_prompt_text} {question} {model_tokens['inst_end']}\n{prompt_type_suffix}"
    elif format_type == "phi":
        system_message = "You are a helpful AI assistant."
        # Structure based on src/utils.py for phi-4, (reasoning is None case)
        return (
            f"{model_tokens['im_start']}system{model_tokens['im_sep']}\n{system_message}{model_tokens['im_end']}\n"
            f"{model_tokens['im_start']}user{model_tokens['im_sep']}\n{base_prompt_text} {question}{model_tokens['im_end']}\n"
            f"{model_tokens['im_start']}assistant{model_tokens['im_sep']}\n{prompt_type_suffix}"
        )
    elif format_type == "standard": # For GPT-2 etc.
        return f"{base_prompt_text} {question}\n{prompt_type_suffix}"
    else: # Should not happen if format_type is always one of the above
        print(f"Warning: Unknown format_type '{format_type}' in construct_prompt. Using standard format.")
        return f"{base_prompt_text} {question}\n{prompt_type_suffix}"


# -----------------------------------------
# Vector Quantization Core Functions
# -----------------------------------------

def _vq_process_single_hidden_state(
    raw_hidden_state_for_token: torch.Tensor, # Shape [B, 1, H] or [B, H]
    state: VQTrainingState,
    config: VQConfig # Pass config for normalization, argmax, temp options
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Helper to VQ-process a single hidden state for one token position."""
    model = state.actor_model # For device info, not used for forward pass here
    token_embeddings = state.token_embeddings
    # Ensure raw_hidden_state_for_token is [B, H] if it's [B, 1, H]
    if raw_hidden_state_for_token.ndim == 3:
        raw_hidden_state_for_token = raw_hidden_state_for_token.squeeze(1)

    current_raw_hidden_state = raw_hidden_state_for_token
    if config.normalize_reasoning_states:
        current_raw_hidden_state = torch.nn.functional.normalize(current_raw_hidden_state, p=2, dim=-1)

    hidden_dim = current_raw_hidden_state.shape[-1]
    # Similarities: current_raw_hidden_state [B, H] x token_embeddings.T [H, V] -> [B, V]
    normalized_target_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)
    similarities_for_token = torch.matmul(current_raw_hidden_state, normalized_target_embeddings.T)
    token_probs_for_token = torch.nn.functional.softmax(similarities_for_token / config.vq_sampling_temp, dim=-1)

    if config.vq_use_argmax:
        sampled_token_id_for_token = torch.argmax(token_probs_for_token, dim=-1) # Shape [B]
    else:
        with torch.no_grad():
            distribution = torch.distributions.Categorical(probs=token_probs_for_token)
            sampled_token_id_for_token = distribution.sample() # Shape [B]
    
    sampled_token_embedding_for_token = token_embeddings[sampled_token_id_for_token] # Shape [B, H]
    
    if config.codebook_loss_weight > 0:
        codebook_loss_for_token = torch.nn.functional.mse_loss(current_raw_hidden_state, sampled_token_embedding_for_token.detach())
    else:
        codebook_loss_for_token = torch.tensor(0.0, device=current_raw_hidden_state.device, dtype=torch.float32)
    
    ste_hidden_state_for_token = current_raw_hidden_state + (sampled_token_embedding_for_token - current_raw_hidden_state).detach()
    
    # Calculate quantization error for this token (normalized space)
    norm_ste_for_metric = torch.nn.functional.normalize(ste_hidden_state_for_token, p=2, dim=-1)
    norm_sampled_for_metric = torch.nn.functional.normalize(sampled_token_embedding_for_token, p=2, dim=-1)
    quant_error_for_token_val = torch.norm(norm_ste_for_metric - norm_sampled_for_metric, p=2, dim=-1).mean().item()

    return sampled_token_id_for_token, ste_hidden_state_for_token, codebook_loss_for_token, similarities_for_token, quant_error_for_token_val


def encode_reasoning(
    state: VQTrainingState,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reasoning_length: int = 50,
    # sampling_temperature and placeholder_token_id are now taken from config inside
) -> Dict[str, Any]: # Return type changed to Any to accommodate new structure temporarily
    """Generate reasoning tokens using the VQ approach with the ACTOR model."""
    model = state.actor_model
    tokenizer = state.tokenizer
    device = state.device
    config = state.config # Get config from state
    token_embeddings = state.token_embeddings # For the helper
    
    batch_size, seq_len = input_ids.shape
    hidden_dim = model.config.hidden_size # Assuming this is available
    vocab_size = model.config.vocab_size # Needed for one-hot encoding in Gumbel

    if config.use_gumbel_softmax_vq:
        if config.vq_sequential_generation:
            raise NotImplementedError("Gumbel-Softmax VQ for sequential generation is not yet implemented.")

        # --- Gumbel-Softmax VQ Path (Parallel Generation) ---
        placeholder_token_id_val = config.placeholder_token_id
        if placeholder_token_id_val is None:
            placeholder_token_id_val = tokenizer.eos_token_id
        
        placeholder_ids = torch.full((batch_size, reasoning_length), placeholder_token_id_val, dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([input_ids, placeholder_ids], dim=1)
        extended_attention_mask = attention_mask # Will be extended next
        if attention_mask is not None:
            extended_attention_mask = torch.cat([attention_mask, torch.ones((batch_size, reasoning_length), dtype=attention_mask.dtype, device=device)], dim=1)
        else:
            extended_attention_mask = torch.ones_like(extended_input_ids)

        outputs = model(input_ids=extended_input_ids, attention_mask=extended_attention_mask, output_hidden_states=False, return_dict=True)
        all_logits = outputs.logits # [B, S_extended, V]
        # Select logits for the reasoning part
        reasoning_logits = all_logits[:, -reasoning_length:, :] # [B, L_reason, V]
        reasoning_logits_flat = reasoning_logits.reshape(-1, vocab_size) # [B*L_reason, V]

        # Gumbel-Softmax STE
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(reasoning_logits_flat))) # [B*L_reason, V]
        y_soft_flat = torch.softmax((reasoning_logits_flat + gumbel_noise) / config.gumbel_tau, dim=-1) # [B*L_reason, V]
        
        sampled_token_ids_flat = torch.argmax(y_soft_flat, dim=-1) # [B*L_reason]
        y_hard_flat = torch.nn.functional.one_hot(sampled_token_ids_flat, num_classes=vocab_size).to(y_soft_flat.dtype) # [B*L_reason, V]
        
        # STE output in vocab space (one-hot like with soft grads)
        ste_vocab_output_flat = y_hard_flat + (y_soft_flat - y_soft_flat.detach()) # [B*L_reason, V]

        # Convert STE output from vocab space to Embeddings
        straight_through_hidden_flat = torch.matmul(ste_vocab_output_flat, token_embeddings) # [B*L_reason, H]
        straight_through_hidden = straight_through_hidden_flat.reshape(batch_size, reasoning_length, hidden_dim) # [B, L_reason, H]

        quantized_token_ids = sampled_token_ids_flat.reshape(batch_size, reasoning_length) # [B, L_reason]

        # Codebook Loss (y_soft vs y_hard.detach() in vocab space)
        if config.codebook_loss_weight > 0:
            codebook_loss_term = torch.nn.functional.mse_loss(y_soft_flat, y_hard_flat.detach())
        else:
            codebook_loss_term = torch.tensor(0.0, device=model.device, dtype=torch.float32)

        # Quantization Error Norm (STE embeddings vs. hard-sampled embeddings)
        sampled_token_embeds_flat = token_embeddings[sampled_token_ids_flat] # [B*L_reason, H]
        sampled_token_embeds_reshaped = sampled_token_embeds_flat.reshape(batch_size, reasoning_length, -1) # [B, L_reason, H]

        norm_ste_for_metric = torch.nn.functional.normalize(straight_through_hidden, p=2, dim=-1)
        norm_sampled_for_metric = torch.nn.functional.normalize(sampled_token_embeds_reshaped, p=2, dim=-1)
        quantization_error_norm_val = torch.norm(norm_ste_for_metric - norm_sampled_for_metric, p=2, dim=-1).mean().item()
        
        similarities_to_return = reasoning_logits # Logits are pre-softmax, akin to original similarities
        token_probs_to_return = y_soft_flat.reshape(batch_size, reasoning_length, -1)

        if config.debug_gradients:
            print(f"GRAD DEBUG (Gumbel-Softmax VQ) - Tau: {config.gumbel_tau}")
            print(f"  Quantization Error Norm: {quantization_error_norm_val:.4f}, Codebook Loss (vocab_mse): {codebook_loss_term.item():.4f}")
            print(f"  y_soft_flat norm: {torch.norm(y_soft_flat).item():.4f}, y_hard_flat norm: {torch.norm(y_hard_flat).item():.4f}")
            print(f"  straight_through_hidden norm: {torch.norm(straight_through_hidden).item():.4f}")

        return {
            'quantized_token_ids': quantized_token_ids,
            'similarities': similarities_to_return,
            'token_probs': token_probs_to_return,
            'hidden_states': straight_through_hidden,
            'quantization_error_norm': quantization_error_norm_val,
            'codebook_loss_term': codebook_loss_term
        }

    # --- Existing VQ Logic (Similarity-based) ---
    if not config.vq_sequential_generation:
        # --- Parallel VQ Generation (Existing Logic Slightly Adapted) ---
        placeholder_token_id_val = config.placeholder_token_id
        if placeholder_token_id_val is None:
            placeholder_token_id_val = tokenizer.eos_token_id
        
        placeholder_ids = torch.full((batch_size, reasoning_length), placeholder_token_id_val, dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([input_ids, placeholder_ids], dim=1)
        extended_attention_mask = attention_mask # Will be extended next
        if attention_mask is not None:
            extended_attention_mask = torch.cat([attention_mask, torch.ones((batch_size, reasoning_length), dtype=attention_mask.dtype, device=device)], dim=1)
        else:
            extended_attention_mask = torch.ones_like(extended_input_ids) # Create if not present

        outputs = model(input_ids=extended_input_ids, attention_mask=extended_attention_mask, output_hidden_states=True, return_dict=True)
        all_actor_hidden_states = outputs.hidden_states
        selected_layer_hs = all_actor_hidden_states[config.actor_hidden_layer_index]
        raw_reasoning_hs_parallel = selected_layer_hs[:, -reasoning_length:, :] # [B, L_reason, H]

        current_raw_hidden_states_for_vq = raw_reasoning_hs_parallel
        if config.normalize_reasoning_states:
            current_raw_hidden_states_for_vq = torch.nn.functional.normalize(raw_reasoning_hs_parallel, p=2, dim=-1)
        
        # Reshape for batch VQ processing: [B*L_reason, H]
        flat_current_raw_hs = current_raw_hidden_states_for_vq.reshape(-1, hidden_dim)
        normalized_token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)
        
        similarities_parallel = torch.matmul(flat_current_raw_hs, normalized_token_embeddings.T) # [B*L, V]
        token_probs_parallel = torch.nn.functional.softmax(similarities_parallel / config.vq_sampling_temp, dim=-1)

        if config.vq_use_argmax:
            sampled_token_ids_flat = torch.argmax(token_probs_parallel, dim=-1) # [B*L]
        else:
            with torch.no_grad():
                distribution = torch.distributions.Categorical(probs=token_probs_parallel)
                sampled_token_ids_flat = distribution.sample() # [B*L]
        
        quantized_token_ids = sampled_token_ids_flat.reshape(batch_size, reasoning_length)
        sampled_token_embeds_flat = token_embeddings[sampled_token_ids_flat] # [B*L, H]
        
        if config.codebook_loss_weight > 0:
            codebook_loss_term = torch.nn.functional.mse_loss(flat_current_raw_hs, sampled_token_embeds_flat.detach())
        else:
            codebook_loss_term = torch.tensor(0.0, device=model.device, dtype=torch.float32) # Ensure consistent device and dtype
        
        ste_hidden_states_flat = flat_current_raw_hs + (sampled_token_embeds_flat - flat_current_raw_hs).detach()
        straight_through_hidden = ste_hidden_states_flat.reshape(batch_size, reasoning_length, hidden_dim)
        
        norm_ste_for_metric = torch.nn.functional.normalize(straight_through_hidden, p=2, dim=-1)
        norm_sampled_for_metric = torch.nn.functional.normalize(sampled_token_embeds_flat.reshape(batch_size, reasoning_length, -1), p=2, dim=-1)
        quantization_error_norm_val = torch.norm(norm_ste_for_metric - norm_sampled_for_metric, p=2, dim=-1).mean().item()

        if config.debug_gradients:
            # Simplified debug prints for parallel mode
            print(f"GRAD DEBUG (Parallel VQ) - Selected actor layer: {config.actor_hidden_layer_index}, Normalized: {config.normalize_reasoning_states}")
            print(f"  Quantization Error Norm: {quantization_error_norm_val:.4f}, Codebook Loss: {codebook_loss_term.item():.4f}")

        return {
            'quantized_token_ids': quantized_token_ids,
            'similarities': similarities_parallel.reshape(batch_size, reasoning_length, -1), # Reshape for consistency if needed
            'token_probs': token_probs_parallel.reshape(batch_size, reasoning_length, -1),
            'hidden_states': straight_through_hidden,
            'quantization_error_norm': quantization_error_norm_val,
            'codebook_loss_term': codebook_loss_term
        }

    else:
        # --- Sequential VQ Generation --- 
        #print(f"DEBUG encode_reasoning (seq): actor_model.config.vocab_size: {model.config.vocab_size}")
        #print(f"DEBUG encode_reasoning (seq): token_embeddings.shape[0]: {token_embeddings.shape[0]}")

        collected_quantized_token_ids = []
        collected_ste_hidden_states = []
        collected_codebook_losses = []
        collected_similarities = [] # For debugging/analysis if needed
        total_quantization_error_norm = 0.0

        current_dynamic_input_ids = input_ids
        current_dynamic_attention_mask = attention_mask

        for _ in range(reasoning_length):
            outputs_step = model(input_ids=current_dynamic_input_ids, 
                                 attention_mask=current_dynamic_attention_mask, 
                                 output_hidden_states=True, 
                                 return_dict=True)
            
            all_step_hidden_states = outputs_step.hidden_states
            # Get hidden state for the *last* token position from the selected layer
            hidden_state_for_next_token = all_step_hidden_states[config.actor_hidden_layer_index][:, -1, :] # Shape [B, H]
            
            # VQ process for this single token step
            sampled_id, ste_hs, cb_loss, sims, q_err = _vq_process_single_hidden_state(
                hidden_state_for_next_token, state, config
            )
            
            collected_quantized_token_ids.append(sampled_id.unsqueeze(1)) # [B, 1]
            collected_ste_hidden_states.append(ste_hs.unsqueeze(1))       # [B, 1, H]
            collected_codebook_losses.append(cb_loss)
            collected_similarities.append(sims.unsqueeze(1)) # [B, 1, V] for consistency
            total_quantization_error_norm += q_err

            # Update inputs for the next step
            current_dynamic_input_ids = torch.cat([current_dynamic_input_ids, sampled_id.unsqueeze(1)], dim=1)
            if current_dynamic_attention_mask is not None:
                current_dynamic_attention_mask = torch.cat([current_dynamic_attention_mask, torch.ones((batch_size, 1), dtype=current_dynamic_attention_mask.dtype, device=device)], dim=1)
            else: # Should not happen if original input_ids had no mask, but defensive
                current_dynamic_attention_mask = torch.ones_like(current_dynamic_input_ids)

        quantized_token_ids = torch.cat(collected_quantized_token_ids, dim=1) # [B, L_reason]
        straight_through_hidden = torch.cat(collected_ste_hidden_states, dim=1) # [B, L_reason, H]
        codebook_loss_term = torch.stack(collected_codebook_losses).mean() # Average codebook loss
        avg_quantization_error_norm = total_quantization_error_norm / reasoning_length
        # For similarities and token_probs, we might just log the first/last step or not return them to simplify
        # Here, concatenating all similarities: [B, L_reason, V]
        all_similarities = torch.cat(collected_similarities, dim=1) 

    if config.debug_gradients:
        print(f"GRAD DEBUG (Sequential VQ) - Selected actor layer: {config.actor_hidden_layer_index}, Normalized: {config.normalize_reasoning_states}")
        print(f"  Avg Quantization Error Norm: {avg_quantization_error_norm:.4f}, Avg Codebook Loss: {codebook_loss_term.item():.4f}")
    
    return {
        'quantized_token_ids': quantized_token_ids,
            'similarities': all_similarities, # Or handle differently
            'token_probs': torch.nn.functional.softmax(all_similarities / config.vq_sampling_temp, dim=-1), # Recompute for all if needed
            'hidden_states': straight_through_hidden,
            'quantization_error_norm': avg_quantization_error_norm,
            'codebook_loss_term': codebook_loss_term
        }


def calculate_answer_logprobs(
    state: VQTrainingState,
    actor_reasoning_embeddings: torch.Tensor, # Shape: [B, L_reason, H]
    answer_input_ids: torch.Tensor,           # Shape: [B, L_answer_padded]
    answer_attention_mask: torch.Tensor,    # Shape: [B, L_answer_padded]
    task_type: str = "arithmetic" # task_type might influence label creation slightly if needed
) -> torch.Tensor:
    """Calculate the log probability of answers using the CRITIC model, 
    given actor's reasoning embeddings and tokenized answers.
    
    This function:
    1. Uses the CRITIC model (frozen weights).
    2. Concatenates actor's reasoning embeddings with critic's answer embeddings.
    3. Prepares labels for the answer part, ignoring reasoning and padding.
    4. Passes inputs_embeds, attention_mask, and labels to the critic.
    5. Returns -outputs.loss (mean log probability of target answer tokens).
    """
    critic_model = state.critic_model
    tokenizer = state.tokenizer
    device = state.device
    config = state.config # For debug_gradients

    batch_size_from_actor_embeds = actor_reasoning_embeddings.shape[0]
    reasoning_seq_len = actor_reasoning_embeddings.shape[1]

    # 1. Define and tokenize the " Answer: " suffix anew within this function for its embeddings
    answer_suffix_str = " Answer: " 
    tokenized_suffix = tokenizer(
        [answer_suffix_str] * batch_size_from_actor_embeds, 
        return_tensors="pt", 
        padding=False, # Suffix is short and constant
        add_special_tokens=False 
    ).to(device)
    answer_suffix_ids = tokenized_suffix.input_ids
    answer_suffix_attention_mask = tokenized_suffix.attention_mask # Will be all 1s if no padding
    answer_suffix_seq_len = answer_suffix_ids.shape[1] # This should match state.answer_suffix_len if tokenizer is consistent
    
    answer_suffix_embeddings = critic_model.get_input_embeddings()(answer_suffix_ids)

    # 2. Get embeddings for the original answer_input_ids from the CRITIC model
    answer_embeddings = critic_model.get_input_embeddings()(answer_input_ids)

    # 3. Concatenate actor_reasoning_embeddings, answer_suffix_embeddings, and critic's answer_embeddings
    combined_inputs_embeds = torch.cat([actor_reasoning_embeddings, answer_suffix_embeddings, answer_embeddings], dim=1)

    # 4. Create labels for the CausalLM loss calculation
    reasoning_labels = torch.full((batch_size_from_actor_embeds, reasoning_seq_len), -100, dtype=torch.long, device=device)
    suffix_labels = torch.full((batch_size_from_actor_embeds, answer_suffix_seq_len), -100, dtype=torch.long, device=device)
    
    answer_labels_masked = answer_input_ids.clone()
    if tokenizer.pad_token_id is not None:
        answer_labels_masked[answer_attention_mask == 0] = -100 
    else: 
        answer_labels_masked[answer_attention_mask == 0] = -100 
        
    combined_labels = torch.cat([reasoning_labels, suffix_labels, answer_labels_masked], dim=1)

    # 5. Create combined attention mask
    reasoning_attention_mask_part = torch.ones((batch_size_from_actor_embeds, reasoning_seq_len), dtype=torch.long, device=device)
    combined_attention_mask = torch.cat([reasoning_attention_mask_part, answer_suffix_attention_mask, answer_attention_mask], dim=1)

    # 6. Forward pass through CRITIC model
    outputs = critic_model(
        inputs_embeds=combined_inputs_embeds,
        attention_mask=combined_attention_mask,
        labels=combined_labels,
        return_dict=True
    )

    # 7. The loss returned is the mean cross-entropy over valid (non -100) labels.
    # We want to maximize log probabilities, so we return -loss.
    mean_answer_log_prob = -outputs.loss

    if config.debug_gradients:
        print(f"GRAD DEBUG - calculate_answer_logprobs (new):")
        print(f"  actor_reasoning_embeddings norm: {torch.norm(actor_reasoning_embeddings, p=2, dim=-1).mean().item():.4f}") # Keep this print for original norm
        print(f"  actor_reasoning_embeddings shape: {actor_reasoning_embeddings.shape}")
        print(f"  answer_input_ids shape: {answer_input_ids.shape}")
        print(f"  combined_inputs_embeds shape: {combined_inputs_embeds.shape}")
        print(f"  combined_labels shape: {combined_labels.shape}")
        # print(f"  Sample labels (first item): {combined_labels[0, reasoning_seq_len:]}") # Potentially long
        print(f"  critic_outputs.loss: {outputs.loss.item():.4f}")
        print(f"  mean_answer_log_prob (=-loss): {mean_answer_log_prob.item():.4f}")
        print(f"  mean_answer_log_prob requires_grad: {mean_answer_log_prob.requires_grad}")
        if mean_answer_log_prob.requires_grad:
            print(f"  mean_answer_log_prob grad_fn: {mean_answer_log_prob.grad_fn}")
    
    return mean_answer_log_prob


def process_batch(state: VQTrainingState, qa_batch: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Process a batch of question-answer pairs.
    
    This function:
    1. Tokenizes the questions
    2. Generates reasoning using actor model (gradients enabled)
    3. Calculates answer log probabilities using critic model
    4. Returns metrics and losses
    
    Args:
        state: VQ training state
        qa_batch: List of (question, answer) pairs
        
    Returns:
        Dictionary with processed data, losses, and metrics
    """
    actor_model = state.actor_model
    critic_model = state.critic_model
    tokenizer = state.tokenizer
    device = state.device
    config = state.config
    task_type = config.task_type
    
    # Unpack batch
    questions, answers = zip(*qa_batch)
    batch_size = len(questions)
    
    # Create prompts and tokenize
    # OLD: prompts = [construct_prompt(q, task_type, config) for q in questions]
    
    # Determine actor and critic model max lengths
    actor_max_len = state.actor_model.config.max_position_embeddings
    critic_max_len = state.critic_model.config.max_position_embeddings 
    safety_margin = 10 # Safety margin for special tokens etc.

    # --- Calculate max_prompt_tokens_for_actor ---
    # Max length the actor can take for (prompt + reasoning), leaving a safety margin
    max_len_for_prompt_plus_reasoning = actor_max_len - safety_margin
    # Max length the prompt itself can be, leaving space for reasoning
    max_prompt_tokens_for_actor = max_len_for_prompt_plus_reasoning - config.vq_reason_length

    if max_prompt_tokens_for_actor <= 0:
        raise ValueError(
            f"vq_reason_length ({config.vq_reason_length}) is too large for actor_max_len ({actor_max_len}) with safety_margin ({safety_margin}). "
            f"Max prompt tokens becomes <= 0. Reduce vq_reason_length or check model."
        )

    # --- Prompt Construction with content length control --- 
    final_prompts_for_tokenizer = []
    for q_content_str in questions: # 'questions' are the raw content strings from qa_batch
        if config.task_type == "wiki_continuation":
            # Tokenize just the content string to enforce config.context_length
            content_input_ids = tokenizer.encode(
                q_content_str,
                max_length=config.context_length, # Apply user-defined token limit for content
                truncation=True,
                add_special_tokens=False # Content part shouldn't get BOS/EOS typically when inserted
            )
            # Decode the (potentially truncated) content back to string
            truncated_content_str = tokenizer.decode(content_input_ids, skip_special_tokens=True)
            
            # Construct the full prompt using this token-truncated content
            prompt_for_item = construct_prompt(truncated_content_str, task_type, config)
        else: # For arithmetic or other tasks where content is typically short
            prompt_for_item = construct_prompt(q_content_str, task_type, config)
        final_prompts_for_tokenizer.append(prompt_for_item)
    
    # The old effective_prompt_max_len logic is no longer needed here,
    # as prompts are pre-constructed to desired conceptual length.
    # max_prompt_tokens_for_actor will serve as the final safeguard in tokenizer.

    tokenized_inputs = tokenizer(
        final_prompts_for_tokenizer, # Use the processed prompts
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens_for_actor, # Overall safeguard
        return_tensors="pt",
    ).to(device)
    
    # Check verbose setting
    verbose = (state.batch_index % config.print_frequency) == 0
    
    # Gradient debugging
    if config.debug_gradients:
        print(f"\nGRAD DEBUG - before encode_reasoning:")
        sample_param = next(iter(actor_model.parameters()))
        print(f"  Actor model requires_grad for first parameter: {sample_param.requires_grad}")
        print(f"  Actor model sample parameter requires_grad: {sample_param.requires_grad}")
        print(f"  Critic model requires_grad for first parameter: {next(critic_model.parameters()).requires_grad}")
    
    # Get reasoning tokens using VQ approach from actor model
    reasoning_outputs = encode_reasoning(
        state=state,
        input_ids=tokenized_inputs.input_ids,
        attention_mask=tokenized_inputs.attention_mask,
        reasoning_length=config.vq_reason_length
        # sampling_temperature and placeholder_token_id are now accessed from state.config
    )
    
    # Extract outputs from reasoning_outputs
    quantized_token_ids = reasoning_outputs['quantized_token_ids']
    # Get hidden states from the actor model (these are the straight_through_hidden states)
    actor_reasoning_embeddings = reasoning_outputs['hidden_states']
    
    # Gradient debugging
    if config.debug_gradients:
        print(f"GRAD DEBUG - after encode_reasoning:")
        print(f"  actor_reasoning_embeddings shape: {actor_reasoning_embeddings.shape}")
        print(f"  actor_reasoning_embeddings requires_grad: {actor_reasoning_embeddings.requires_grad}")
        print(f"  actor_reasoning_embeddings grad_fn: {actor_reasoning_embeddings.grad_fn}")
    
    # Decode reasoning token IDs to text FOR LOGGING ONLY if verbose
    if verbose:
        # Decode actual input tokens seen by the model
        decoded_input_for_log = tokenizer.decode(tokenized_inputs.input_ids[0], skip_special_tokens=True)
        input_token_count = len(tokenized_inputs.input_ids[0])
        
        reasoning_text_for_logging = tokenizer.decode(quantized_token_ids[0], skip_special_tokens=True)
        reasoning_token_count = len(quantized_token_ids[0])

        print(f"\nInput (decoded, {input_token_count} tokens): {decoded_input_for_log}")
        # Target will be printed after answer_input_ids is available and decoded
        print(f"Generated VQ reasoning (decoded, {reasoning_token_count} tokens): {reasoning_text_for_logging}")
        # Print first 10 reasoning tokens as individual tokens for more detail
        reasoning_tokens_for_logging = [tokenizer.decode([token_id.item()]) for token_id in quantized_token_ids[0]]
        print(f"First 10 VQ reasoning tokens: {reasoning_tokens_for_logging}")
    
    # Tokenize answers for the critic model
    # Ensure max_length is appropriate for your task and model

    answer_suffix_len = state.answer_suffix_len # Retrieve from state
    # answer_suffix_str = " Answer: " # Define the suffix - not needed here if len is from state
    # Calculate suffix length (ensure tokenizer is available - it is, via state)
    # We tokenize it once here to get its length for truncation purposes.
    # add_special_tokens=False is important as this suffix is mid-sequence.
    # temp_tokenized_suffix = state.tokenizer(answer_suffix_str, add_special_tokens=False, return_tensors="pt")
    # answer_suffix_len = temp_tokenized_suffix.input_ids.shape[1] # Get sequence length from [batch, seq_len]
    # If doing for a single string: answer_suffix_len = len(state.tokenizer.encode(answer_suffix_str, add_special_tokens=False))
    # print(f"DEBUG process_batch: Determined answer_suffix_len: {answer_suffix_len} for suffix '{answer_suffix_str}'") # Removed from here

    # --- Answer Tokenization ---
    # Max length the critic can take for (reasoning + suffix + answer), leaving a safety margin
    max_len_for_reasoning_plus_suffix_plus_answer = critic_max_len - safety_margin
    # Max length the answer itself can be, leaving space for reasoning and the suffix
    max_answer_tokens_for_critic = max_len_for_reasoning_plus_suffix_plus_answer - config.vq_reason_length - answer_suffix_len

    if max_answer_tokens_for_critic <= 0:
        raise ValueError(
            f"vq_reason_length ({config.vq_reason_length}) is too large for critic_max_len ({critic_max_len}) with safety_margin ({safety_margin}). "
            f"Max answer tokens becomes <= 0. Reduce vq_reason_length, check suffix, or check model."
        )

    # Determine the user-defined conceptual length for the answer part
    if config.debug_answer_is_question:
        user_desired_answer_len = config.context_length # If answer is question, its conceptual length matches context_length
    else:
        user_desired_answer_len = config.max_target_length

    if config.task_type == "wiki_continuation" or config.debug_answer_is_question:
        # For wiki, or if answer is question (applies to any task_type then for length constraint)
        effective_answer_max_len = min(user_desired_answer_len, max_answer_tokens_for_critic)
        if verbose:
            print(f"DEBUG process_batch ('{config.task_type}', AIQ={config.debug_answer_is_question}): Effective answer max_length for tokenizer: {effective_answer_max_len} "
                  f"(User desired len: {user_desired_answer_len}, Critic-constrained max: {max_answer_tokens_for_critic})")
    else:
        # For arithmetic etc., when not debug_answer_is_question
        effective_answer_max_len = max_answer_tokens_for_critic 
        if verbose:
            print(f"DEBUG process_batch ('{config.task_type}', AIQ={config.debug_answer_is_question}): Effective answer max_length for tokenizer: {effective_answer_max_len} "
                  f"(Critic-constrained max: {max_answer_tokens_for_critic})")

    # Unpack batch now that answers might have been modified by debug_answer_is_question
    # And questions are needed for prompts if answers were copied from them.
    # qa_batch itself is already processed by get_batch_data if the flag is on.
    questions, answers = zip(*qa_batch)

    # Create prompts (can be done here as questions are now final for the batch)
    prompts = [construct_prompt(q, task_type, config) for q in questions]

    answer_tokenizer_output = tokenizer(
        list(answers), # Ensure answers is a list of strings
        padding=True,
        truncation=True,
        max_length=effective_answer_max_len, 
        return_tensors="pt"
    ).to(device)
    answer_input_ids = answer_tokenizer_output.input_ids
    answer_attention_mask = answer_tokenizer_output.attention_mask
    
    if verbose: # Now print DECODED target with its token count
        decoded_target_for_log = tokenizer.decode(answer_input_ids[0], skip_special_tokens=True)
        target_token_count = len(answer_input_ids[0])
        print(f"Target (decoded, {target_token_count} tokens): {decoded_target_for_log}")
    
    # Calculate answer log probabilities using the CRITIC model
    # This now takes actor's reasoning embeddings and tokenized answers
    answer_logprobs = calculate_answer_logprobs(
        state=state,
        actor_reasoning_embeddings=actor_reasoning_embeddings,
        answer_input_ids=answer_input_ids,
        answer_attention_mask=answer_attention_mask,
        task_type=task_type
    )
    
    # Gradient debugging for answer log probs
    if config.debug_gradients:
        print(f"GRAD DEBUG - after calculate_answer_logprobs:")
        print(f"  answer_logprobs shape: {answer_logprobs.shape}")
        print(f"  answer_logprobs requires_grad: {answer_logprobs.requires_grad}")
        print(f"  answer_logprobs grad_fn: {answer_logprobs.grad_fn}")
    
    # Calculate total loss
    # Total loss = -answer_logprobs (maximize) + weighted codebook_loss (minimize)
    codebook_loss_term = reasoning_outputs['codebook_loss_term']
    total_loss = -answer_logprobs + config.codebook_loss_weight * codebook_loss_term
    
    # Add small epsilon to make sure loss doesn't immediately go to zero (if answer_logprobs is high and codebook low)
    epsilon = 1e-6
    total_loss = total_loss + epsilon
    
    # Gradient debugging for total loss
    if config.debug_gradients:
        print(f"GRAD DEBUG - total loss:")
        print(f"  total_loss shape: {total_loss.shape}")
        print(f"  total_loss requires_grad: {total_loss.requires_grad}")
        print(f"  total_loss grad_fn: {total_loss.grad_fn}")
    
    # Print loss components if verbose
    if verbose:
        print(f"Loss components [batch {state.batch_index}]:")
        print(f"  Answer log prob: {answer_logprobs.item():.4f}")
        print(f"  Total loss: {total_loss.item():.4f}")
        
        # Debug gradient flow - check if ANY actor parameter requires gradients
        actor_has_grads = any(p.requires_grad for p in actor_model.parameters())
        # Debug gradient flow
        print(f"Gradient flow check:")
        print(f"  Actor has grads enabled: {actor_has_grads}")
        print(f"  Critic has grads enabled: {any(p.requires_grad for p in critic_model.parameters())}")
        print(f"  Answer logprobs requires_grad: {answer_logprobs.requires_grad}")
        print(f"  Total loss requires_grad: {total_loss.requires_grad}")
    
    # Return all necessary data
    return {
        'questions': questions,
        'answers': answers,
        'quantized_token_ids': quantized_token_ids,
        'answer_logprobs': answer_logprobs,
        'total_loss': total_loss,
        'quantization_error_norm': reasoning_outputs['quantization_error_norm'],
        'codebook_loss_term': codebook_loss_term # Return the unweighted term for logging
    }


# -----------------------------------------
# Training and Visualization Functions
# -----------------------------------------

def plot_metrics(metrics_history: Dict[str, List[float]], save_path: str, config: VQConfig):
    """Plot training metrics over time."""
    # Determine the number of metrics to plot that have data
    available_metrics = {k: v for k, v in metrics_history.items() if v and len(v) > 0}
    
    # Extract batch_indices for x-axis, and remove it from metrics to be plotted as y-axes
    # .pop returns None if 'batch_indices' isn't in available_metrics or if its list is empty
    x_coords = available_metrics.pop('batch_indices', None) 

    num_metrics_to_plot = len(available_metrics)

    if num_metrics_to_plot == 0:
        print("No metrics with data to plot.")
        return

    # Define the order and appearance of known metrics
    plot_config = {
        'total_loss': {'title': 'Total Loss', 'color': 'r-', 'ylabel': 'Loss'},
        'answer_logprobs': {'title': 'Answer Log Probabilities', 'color': 'b-', 'ylabel': 'Log Prob'},
        'grad_norm': {'title': 'Actor Gradient Norm (Pre-clipping)', 'color': 'g-', 'ylabel': 'Norm'},
        'quantization_error_norm': {'title': 'Quantization L2 Error (Normalized)', 'color': 'm-', 'ylabel': 'Error Norm'},
        'codebook_loss': {'title': 'Codebook Loss (Unweighted)', 'color': 'c-', 'ylabel': 'Loss'} # Added codebook loss
    }

    # Filter and order metrics based on plot_config and availability
    metrics_to_display = []
    for key in plot_config.keys():
        if key in available_metrics:
            metrics_to_display.append(key)
    
    if not metrics_to_display:
        print("None of the configured metrics have data to plot.")
        return

    # Adjust grid size for up to 6 plots
    num_display_metrics = len(metrics_to_display)
    if num_display_metrics <= 2:
        rows, cols = 1, num_display_metrics
        figsize = (6 * num_display_metrics, 5)
    elif num_display_metrics <= 4:
        rows, cols = 2, 2
        figsize = (12, 10)
    elif num_display_metrics <= 6:
        rows, cols = 3, 2 # Changed to 3x2 grid
        figsize = (12, 15) # Adjusted figsize
    else: # More than 6, just use a tall 2-column layout
        rows = (num_display_metrics + 1) // 2
        cols = 2
        figsize = (12, 5 * rows)
        
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Construct a detailed title
    title_parts = [
        f"Model: {config.model_name.split('/')[-1]}",
        f"LR: {config.learning_rate:.0e}", # Scientific notation for LR
        f"BS: {config.batch_size}",
        f"VQ_L: {config.vq_reason_length}",
        f"VQ_T: {config.vq_sampling_temp}",
        f"CBW: {config.codebook_loss_weight}",
        f"L{config.actor_hidden_layer_index}",
        f"NormRS: {'T' if config.normalize_reasoning_states else 'F'}", # T/F for boolean
        f"SeqGen: {'T' if config.vq_sequential_generation else 'F'}",
        f"Argmax: {'T' if config.vq_use_argmax else 'F'}"
    ]
    if config.lora_rank > 0:
        title_parts.append(f"LoRA: {config.lora_rank}")
    if config.use_gumbel_softmax_vq:
        title_parts.append(f"GumbelVQ: T")
        title_parts.append(f"GTau: {config.gumbel_tau:.2f}")
    else:
        title_parts.append(f"GumbelVQ: F")
    title_parts.append(f"8bitAdam: {'T' if config.use_8bit_adam else 'F'}")
    
    param_summary = ", ".join(title_parts)
    
    # Split title if too long, adjust layout
    max_title_len_per_line = 100 # Approximate chars per line
    if len(param_summary) > max_title_len_per_line:
        # Find a good split point (e.g., after a comma)
        split_idx = param_summary.rfind(',', 0, max_title_len_per_line)
        if split_idx == -1 or len(param_summary) - split_idx < 20: # Avoid tiny second line
            split_idx = param_summary.find(',', max_title_len_per_line // 2) # Try middle
        if split_idx != -1:
            param_summary_line1 = param_summary[:split_idx+1]
            param_summary_line2 = param_summary[split_idx+1:].strip()
            suptitle_text = f'VQ Training Metrics\n{param_summary_line1}\n{param_summary_line2}'
            fig.suptitle(suptitle_text, fontsize=10) # Smaller font for multi-line
            plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust for 3 lines total
        else:
            fig.suptitle(f'VQ Training Metrics\n({param_summary})', fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.94]) # Adjust for 2 lines total
    else:
        fig.suptitle(f'VQ Training Metrics ({param_summary})', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Standard adjustment

    axs_flat = axs.flatten() if num_display_metrics > 1 else [axs] # Handle single plot case
    
    plot_idx = 0
    for metric_key in metrics_to_display:
        if plot_idx >= len(axs_flat):
            print(f"Warning: More metrics ({len(metrics_to_display)}) than available subplots ({len(axs_flat)}). Some metrics will not be plotted.")
            break
        
        metric_plot_config = plot_config[metric_key] # Renamed to avoid conflict
        ax = axs_flat[plot_idx]
        y_values = available_metrics[metric_key]

        # Use actual batch numbers for x-axis if available and lengths match
        if x_coords and len(x_coords) == len(y_values):
            ax.plot(x_coords, y_values, metric_plot_config['color'])
            ax.set_xlabel('Batch Number (from log)')
        else:
            if x_coords and len(x_coords) != len(y_values):
                 print(f"Warning: Mismatch length between batch_indices ({len(x_coords)}) and metric {metric_key} ({len(y_values)}). Plotting against index.")
            elif not x_coords:
                 print(f"Warning: 'batch_indices' not found or empty. Plotting metric {metric_key} against index.")
            ax.plot(y_values, metric_plot_config['color']) # Fallback to plotting against index
            ax.set_xlabel('Batch Index (sequential)')

        ax.set_title(metric_plot_config['title'])
        # ax.set_xlabel('Batch') # X-label is set conditionally above
        ax.set_ylabel(metric_plot_config['ylabel'])
        ax.grid(True)
        plot_idx += 1
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axs_flat)):
        axs_flat[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved metrics plot to {save_path}")


def run_copy_test(state: VQTrainingState, qa_batch: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Run a test to see if prefilling reasoning with answer tokens improves log probabilities.
    
    NOTE: This test is currently NOT FULLY COMPATIBLE with the new embedding-based 
    calculate_answer_logprobs function. It needs significant refactoring to generate
    actor_reasoning_embeddings for its test cases. Returning default values.

    This test originally:
    1. Computes normal log probs with empty reasoning
    2. Computes log probs with reasoning containing answer prefix
    3. Compares the improvement to the historical average
    
    Returns:
        Dictionary with test results (currently empty or default)
    """
    print("\n" + "="*80)
    print("WARNING: run_copy_test is not fully compatible with the current calculate_answer_logprobs.")
    print("It requires refactoring to use actor_reasoning_embeddings instead of reasoning_texts.")
    print("Skipping effective test and returning empty results.")
    print("="*80 + "\n")
    return { # Return empty or default dictionary
        "empty": 0.0,
        "step_by_step": 0.0,
        "hint_first_token": 0.0,
        "answer_prefix": 0.0,
        "improvement": 0.0,
        "significant_threshold": 0.0,
        "is_significant": False
    }


def train_vq(config: VQConfig):
    """Main training loop for Vector Quantization.
    
    This function:
    1. Initializes the model and training state
    2. Generates batches of data
    3. Trains the model using VQ approach
    4. Tracks and visualizes metrics
    5. Saves checkpoints
    """
    # Setup the model and state
    # print("DEBUG: Entered train_vq() function.") # Commented out
    state = setup_model(config)
    
    # print(f"DEBUG train_vq: state.batch_index immediately after setup_model: {state.batch_index}") # DEBUG PRINT REMOVED

    print(f"\nStarting VQ training with:")
    print(f"  Model: {config.model_name}")
    print(f"  Task: {config.task_type}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of batches: {config.num_batches}")
    print(f"  VQ reasoning length: {config.vq_reason_length}")
    print(f"  VQ sampling temperature: {config.vq_sampling_temp} (lower=more peaked, higher=more diverse)")
    if config.copy_test_frequency > 0:
        print(f"  Copy test frequency: Every {config.copy_test_frequency} batches")
    print(f"  Debug repeat datapoint: {config.debug_repeat_datapoint}")
    print(f"  Debug gradients: {config.debug_gradients}")
    print(f"  Output directory: {state.output_dir}")
    print(f"  Weights check frequency: {config.weights_check_frequency} batches")

    # Store initial sums of weights for checking
    initial_actor_weights_sum = sum(p.sum().item() for p in state.actor_model.parameters() if p.requires_grad)
    initial_critic_weights_sum = sum(p.sum().item() for p in state.critic_model.parameters())
    if config.debug_gradients: # Also log initial sums if debugging
        print(f"Initial actor weights sum: {initial_actor_weights_sum}")
        print(f"Initial critic weights sum: {initial_critic_weights_sum}")
    
    # Main training loop
    try:
        # first_loop_batch_idx = -1 # DEBUG PRINT REMOVED
        for batch_idx in range(state.batch_index, config.num_batches): # Start from state.batch_index
            # if first_loop_batch_idx == -1: # DEBUG PRINT REMOVED
            #     first_loop_batch_idx = batch_idx # DEBUG PRINT REMOVED
            #     print(f"DEBUG train_vq: First value of batch_idx in loop: {first_loop_batch_idx}") # DEBUG PRINT REMOVED

            state.batch_index = batch_idx # Keep state.batch_index updated for internal use (e.g. process_batch verbose)
            
            # Whether to print verbose output for this batch
            verbose = (batch_idx % config.print_frequency) == 0
            
            if verbose:
                print(f"\n{'='*80}\n\nBatch {batch_idx}")
            else:
                # Print simple progress
                print(f"Batch {batch_idx}", end="\r", flush=True)
            
            # Generate batch data
            # print(f"DEBUG train_vq: About to call get_batch_data for batch {batch_idx}.") # Commented out
            qa_batch = get_batch_data(config.batch_size, config.task_type, config)
            
            # Run copy test if configured and it's time
            if config.copy_test_frequency > 0 and batch_idx > 0 and batch_idx % config.copy_test_frequency == 0:
                copy_test_results = run_copy_test(state, qa_batch)
                
                # Log copy test results
                with open(os.path.join(state.output_dir, "copy_test_results.jsonl"), "a") as f:
                    result_entry = {
                        "batch": batch_idx,
                        **copy_test_results
                    }
                    json.dump(result_entry, f)
                    f.write("\n")
            
            # Process batch
            batch_data = process_batch(state, qa_batch)
            
            # Optimization step
            state.optimizer.zero_grad()
            batch_loss = batch_data['total_loss'].mean()
            
            # Debug gradient information before backward
            if config.debug_gradients:
                print(f"\nGRAD DEBUG - before backward:")
                print(f"  batch_loss: {batch_loss.item()}")
                print(f"  batch_loss requires_grad: {batch_loss.requires_grad}")
                print(f"  batch_loss grad_fn: {batch_loss.grad_fn}")
                
                # Check if actor has gradients
                for name, param in state.actor_model.named_parameters():
                    if param.requires_grad:
                        print(f"  Actor param {name}: requires_grad={param.requires_grad}, has_grad={param.grad is not None}")
                        # Print only a few parameters to avoid cluttering
                        if name.endswith(".weight"):
                            break
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient Clipping (returns total norm of params gradients BEFORE clipping)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(state.actor_model.parameters(), max_norm=1.0)
            
            # Debug gradient information after backward
            if config.debug_gradients:
                print(f"GRAD DEBUG - after backward:")
                print(f"  batch_loss: {batch_loss.item()}")
                print(f"  Actor grad norm (pre-clipping): {actor_grad_norm.item():.4f}") # Print pre-clipping norm
                
                # Check which parts of the actor have gradients (simplified logging)
                modules_with_grad = set()
                for name, param in state.actor_model.named_parameters():
                    if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
                        module_name = name.split('.')[0]
                        modules_with_grad.add(module_name)
                print(f"  Modules with non-zero gradients: {modules_with_grad}")
                
                # Also check critic model to confirm it doesn't have gradients
                critic_has_grad = False
                for name, param in state.critic_model.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        critic_has_grad = True
                        print(f"  WARNING: Critic has gradients in {name}")
                        break
                if not critic_has_grad:
                    print("  Confirmed: Critic model has no gradients")
            
            # Verify that only actor parameters get gradients (only on first verbose batch)
            if batch_idx == 0 and verbose:
                # Check if ANY actor parameter has gradients after backward
                actor_has_grad = any(p.grad is not None for p in state.actor_model.parameters() if p.requires_grad)
                critic_has_grad = any(p.grad is not None for p in state.critic_model.parameters() if p.requires_grad)
                print(f"Gradient check after backward:")
                print(f"  Actor has gradients: {actor_has_grad} (should be True)")
                print(f"  Critic has gradients: {critic_has_grad} (should be False)")
            
            state.optimizer.step()
            
            # Debug gradient after optimizer step
            if config.debug_gradients:
                print(f"GRAD DEBUG - after optimizer step:")
                # Check if gradients were actually applied
                for name, param in state.actor_model.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            print(f"  {name} still has gradient after step: {param.grad is not None}")
                            break
            
            # Update metrics history
            state.metrics_history['total_loss'].append(batch_loss.item())
            state.metrics_history['answer_logprobs'].append(batch_data['answer_logprobs'].mean().item())
            state.metrics_history['grad_norm'].append(actor_grad_norm.item())
            state.metrics_history['quantization_error_norm'].append(batch_data['quantization_error_norm'])
            state.metrics_history['codebook_loss'].append(batch_data['codebook_loss_term'].item()) # Add codebook loss
            # Also append the current batch_idx to batch_indices
            state.metrics_history.setdefault('batch_indices', []).append(batch_idx)

            
            # Log metrics to file
            if verbose:
                metrics_entry = {
                    'batch': batch_idx,
                    'total_loss': batch_loss.item(),
                    'answer_logprobs': batch_data['answer_logprobs'].mean().item(),
                    'grad_norm': actor_grad_norm.item(),
                    'quantization_error_norm': batch_data['quantization_error_norm'],
                    'codebook_loss': batch_data['codebook_loss_term'].item() # Add to JSON log
                }
                
                with open(state.log_file, "a") as f:
                    json.dump(metrics_entry, f)
                    f.write("\n")
            
            # Plot metrics periodically
            if config.plot_frequency > 0 and batch_idx > 0 and (batch_idx % config.plot_frequency == 0):
                plot_path = os.path.join(state.output_dir, "metrics.png")
                plot_metrics(state.metrics_history, plot_path, state.config)
            
            # Save checkpoint periodically
            checkpoint_freq = config.checkpoint_frequency
            if checkpoint_freq > 0 and batch_idx > 0 and batch_idx % checkpoint_freq == 0:
                checkpoint_path = os.path.join(state.output_dir, f"checkpoint_{batch_idx}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save only the actor model (what we're training)
                state.actor_model.save_pretrained(checkpoint_path)
                state.tokenizer.save_pretrained(checkpoint_path)
                # Save optimizer state
                optimizer_save_path = os.path.join(checkpoint_path, "optimizer.pt")
                torch.save(state.optimizer.state_dict(), optimizer_save_path)
                print(f"\nSaved checkpoint at batch {batch_idx} to {checkpoint_path} (incl. optimizer)")
        
        # Save final metrics plot
        final_plot_path = os.path.join(state.output_dir, "metrics.png")
        plot_metrics(state.metrics_history, final_plot_path, state.config)
        
        # Save final model if requested
        if config.save_final_model:
            final_model_path = os.path.join(state.output_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            
            # Save only the actor model (what we're training)
            state.actor_model.save_pretrained(final_model_path)
            state.tokenizer.save_pretrained(final_model_path)
            print(f"\nSaved final model to {final_model_path}")
        
        print("\nTraining complete!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        # Save current state
        interrupt_path = os.path.join(state.output_dir, "interrupted_checkpoint")
        os.makedirs(interrupt_path, exist_ok=True)
        
        # Save only the actor model (what we're training)
        state.actor_model.save_pretrained(interrupt_path)
        state.tokenizer.save_pretrained(interrupt_path)
        # Save optimizer state on interrupt
        optimizer_interrupt_save_path = os.path.join(interrupt_path, "optimizer.pt")
        torch.save(state.optimizer.state_dict(), optimizer_interrupt_save_path)
        print(f"Saved interrupted state to {interrupt_path} (incl. optimizer)")


# -----------------------------------------
# Main Program
# -----------------------------------------

def main():
    """Parse command line arguments and start training."""
    # print("DEBUG: Entered main() function.") # Commented out
    parser = argparse.ArgumentParser(description="Train with Vector Quantization")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it",
                       help="Hugging Face model name")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="Rank for LoRA (0 to disable)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="Alpha for LoRA")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_batches", type=int, default=1000,
                       help="Number of batches to train")
    parser.add_argument("--debug_repeat_datapoint", action="store_true",
                       help="Debug by training on the same datapoint repeatedly")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer (requires bitsandbytes).")
    
    # VQ specific parameters
    parser.add_argument("--vq_reason_length", type=int, default=50,
                       help="Length of reasoning to generate")
    parser.add_argument("--vq_sampling_temp", type=float, default=0.5,
                       help="Temperature for softmax sampling (lower=more peaked, higher=more random)")
    parser.add_argument("--vq_use_argmax", action="store_true",
                       help="Use argmax instead of sampling for VQ reasoning tokens")
    parser.add_argument("--actor_hidden_layer_index", type=int, default=-2,
                       help="Index of actor hidden layer to use for reasoning (e.g., -1 for last, -2 for 2nd-to-last)")
    parser.add_argument("--normalize-reasoning-states", 
                        dest="normalize_reasoning_states",
                        action=argparse.BooleanOptionalAction, default=False,
                       help="Enable normalization of actor hidden states for reasoning (controls --normalize-reasoning-states and --no-normalize-reasoning-states)")
    parser.add_argument("--vq_sequential_generation", action="store_true",
                       help="Enable sequential (autoregressive) generation for VQ reasoning tokens")
    parser.add_argument("--use_gumbel_softmax_vq", action="store_true",
                        help="Use Gumbel-Softmax for VQ instead of similarity-based sampling.")
    parser.add_argument("--gumbel_tau", type=float, default=1.0,
                        help="Temperature tau for Gumbel-Softmax.")
    parser.add_argument("--filler_token", type=str, default="<REASONING>",
                       help="Filler token for reasoning placeholders")
    
    # Task parameters
    parser.add_argument("--task_type", type=str, default="arithmetic",
                       choices=["arithmetic", "wiki_continuation"],
                       help="Task type to train on")
    parser.add_argument("--context_length", type=int, default=100,
                       help="Context length for wiki continuation task")
    parser.add_argument("--max_target_length", type=int, default=100,
                       help="Maximum target length for wiki continuation task")
    
    # Logging parameters
    parser.add_argument("--print_frequency", type=int, default=10,
                       help="How often to print detailed output")
    parser.add_argument("--plot_frequency", type=int, default=50, # Changed from plot_interval
                       help="How often to plot metrics (e.g., every N batches, 0 to disable)")
    parser.add_argument("--checkpoint_frequency", type=int, default=50,
                       help="How often to save checkpoints (0 to disable)")
    parser.add_argument("--save_final_model", action="store_true",
                       help="Whether to save the final model")
    
    # Add debug gradients flag
    parser.add_argument("--debug_gradients", action="store_true",
                       help="Enable extensive gradient debugging prints")
    
    # Add copy test parameters
    parser.add_argument("--copy_test", type=int, default=0,
                       help="Run the copy test every n batches (0 to disable)")
    parser.add_argument("--copy_test_samples", type=int, default=3,
                       help="Number of samples to use for the copy test")
    
    # Add codebook loss weight argument
    parser.add_argument("--codebook_loss_weight", type=float, default=0.25,
                       help="Weight for the codebook loss term")
    parser.add_argument("--debug_answer_is_question", action="store_true", 
                        help="Set answer to be identical to the question for debugging.")
    
    # Add resume argument
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to a checkpoint directory (e.g., results/task/timestamp/checkpoint_N) to resume training from.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # print(f"DEBUG main: Parsed args.debug_answer_is_question: {args.debug_answer_is_question}") # Commented out
    
    # Create config from arguments
    config = VQConfig(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        debug_repeat_datapoint=args.debug_repeat_datapoint,
        use_8bit_adam=args.use_8bit_adam, # New
        vq_reason_length=args.vq_reason_length,
        vq_sampling_temp=args.vq_sampling_temp,
        vq_use_argmax=args.vq_use_argmax,
        actor_hidden_layer_index=args.actor_hidden_layer_index,
        normalize_reasoning_states=args.normalize_reasoning_states,
        vq_sequential_generation=args.vq_sequential_generation,
        use_gumbel_softmax_vq=args.use_gumbel_softmax_vq, # New
        gumbel_tau=args.gumbel_tau,                     # New
        codebook_loss_weight=args.codebook_loss_weight,
        print_frequency=args.print_frequency,
        plot_frequency=args.plot_frequency, # Changed from plot_interval
        checkpoint_frequency=args.checkpoint_frequency,
        save_final_model=args.save_final_model,
        task_type=args.task_type,
        context_length=args.context_length,
        max_target_length=args.max_target_length,
        debug_gradients=args.debug_gradients,
        copy_test_frequency=args.copy_test,
        copy_test_samples=args.copy_test_samples,
        debug_answer_is_question=args.debug_answer_is_question # New arg
    )
    
    # Update config with resume path if provided
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint

    # Start training
    train_vq(config)


if __name__ == "__main__":
    main() 