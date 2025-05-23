#!/usr/bin/env python
"""
Example script showing how to use Parallel Vector Quantization for token generation and training.

This demonstrates:
1. How to process an entire sequence in parallel
2. How to quantize hidden states to the nearest token embeddings
3. How to train a model as an autoencoder with Vector Quantization loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Dict, List, Optional, Tuple, Any

from src.vq import ParallelVQEncoder
from src.utils import colored_print, Colors

def test_vq_parallel(model_name="google/gemma-3-1b-it", text="The quick brown fox jumps over the lazy dog."):
    """Demonstrate the parallel VQ encoder by encoding and decoding a text sequence."""
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create parallel VQ encoder
    vq_encoder = ParallelVQEncoder(model, tokenizer)
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Print the tokenized input
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    colored_print("Tokenized Input", f"{token_strings}", Colors.BLUE)
    colored_print("Token IDs", f"{tokens}", Colors.BLUE)
    
    # Test with different temperatures
    for temp in [0.5, 1.0, 2.0]:
        colored_print("Processing", f"Using temperature: {temp}", Colors.CYAN)
        
        # Encode-decode with VQ
        quantized_data = vq_encoder.encode_decode(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            temperature=temp,
            return_all_stages=True
        )
        
        # Get quantized token IDs
        quantized_ids = quantized_data['quantized_token_ids']
        
        # Decode to text
        reconstructed_text = vq_encoder.decode_to_text(quantized_ids)[0]
        
        # Print tokens
        token_ids = quantized_ids[0]
        individual_tokens = [tokenizer.decode([t]) for t in token_ids]
        colored_print("Quantized Tokens", f"{individual_tokens}", Colors.MAGENTA)
        
        # Print reconstruction
        colored_print("Reconstructed", f"{reconstructed_text}", Colors.GREEN)
        print("\n---\n")
        
    return vq_encoder

def train_vq_parallel(
    model_name="google/gemma-3-1b-it", 
    text="The quick brown fox jumps over the lazy dog.",
    num_iterations=30,
    learning_rate=5e-5,
    temperature=1.0
):
    """Demonstrate training with parallel VQ approach."""
    # Load model and tokenizer
    colored_print("Model", f"Loading {model_name}", Colors.BOLD)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create the VQ autoencoder
    vq_encoder = ParallelVQEncoder(model, tokenizer)
    
    # Train to memorize the text
    colored_print("Training", f"Training for {num_iterations} iterations with lr={learning_rate}", Colors.BOLD)
    history, metrics = vq_encoder.memorize_string(
        text=text,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        temperature=temperature
    )
    
    # Print final reconstruction
    final_iter, final_text = history[-1]
    
    if final_text == text:
        colored_print("Success", f"Perfectly memorized text at iteration {final_iter}!", Colors.GREEN)
    else:
        colored_print("Final Result", f"After {final_iter} iterations:", Colors.YELLOW)
        colored_print("Original", text, Colors.BLUE)
        colored_print("Reconstructed", final_text, Colors.GREEN)
    
    # Test on variations
    colored_print("Testing Variations", "Testing reconstruction on variations of the original", Colors.BOLD)
    
    variations = [
        text,  # Original
        text.replace("quick", "swift") if "quick" in text else text.replace("the", "a"),  # Word substitution
        text + " The end.",  # Extended
        text[:len(text)//2],  # Truncated
    ]
    
    vq_encoder.test_memorization(variations, temperature=temperature)
    
    return vq_encoder, history, metrics

def test_fresh_context_decoding(
    vq_encoder,
    text="The quick brown fox jumps over the lazy dog.",
    model_name="google/gemma-3-1b-it"
):
    """Demonstrate using fresh context for decoding quantized tokens."""
    colored_print("Fresh Context Test", "Testing decoding with fresh context", Colors.BOLD)
    
    # Load a fresh critic model
    colored_print("Critic Model", "Loading fresh critic model", Colors.BOLD)
    critic_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    # Freeze the critic model (set requires_grad=False) but DON'T use torch.no_grad()
    # This allows gradient flow through the model without updating weights
    for param in critic_model.parameters():
        param.requires_grad_(False)
    
    # Tokenize the input
    inputs = vq_encoder.tokenizer(text, return_tensors="pt").to(vq_encoder.device)
    
    # Get quantized token IDs
    quantized_data = vq_encoder.encode_decode(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        temperature=1.0,
        return_all_stages=True
    )
    
    token_ids = quantized_data['quantized_token_ids']
    
    # Process through different segments of the quantized tokens
    # to demonstrate that the decoder only uses the provided tokens
    segments = [
        ("Full sequence", token_ids),
        ("First half", token_ids[:, :token_ids.size(1)//2]),
        ("Last half", token_ids[:, token_ids.size(1)//2:])
    ]
    
    for name, segment_ids in segments:
        colored_print(f"Testing {name}", f"Token count: {segment_ids.size(1)}", Colors.YELLOW)
        
        # Get hidden states with fresh context
        hidden_states = vq_encoder.decode_with_fresh_context(
            quantized_token_ids=segment_ids,
            critic_model=critic_model
        )
        
        # Convert hidden states back to tokens
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)
        normalized_hidden = F.normalize(flat_hidden, p=2, dim=1)
        normalized_embeddings = F.normalize(vq_encoder.token_embeddings, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(normalized_hidden, normalized_embeddings.T)
        
        # Get closest tokens
        closest_token_ids = torch.argmax(similarities, dim=-1).reshape(batch_size, seq_len)
        
        # Decode to text
        decoded_text = vq_encoder.tokenizer.decode(closest_token_ids[0], skip_special_tokens=True)
        
        # Print individual tokens for comparison
        used_tokens = [vq_encoder.tokenizer.decode([t]) for t in segment_ids[0]]
        colored_print("Input Tokens", f"{used_tokens}", Colors.CYAN)
        
        # Print output tokens
        output_tokens = [vq_encoder.tokenizer.decode([t]) for t in closest_token_ids[0]]
        colored_print("Output Tokens", f"{output_tokens}", Colors.MAGENTA)
        
        # Print final decoded text
        colored_print("Decoded Result", decoded_text, Colors.GREEN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Vector Quantization example")
    parser.add_argument("--test", action="store_true", help="Test parallel VQ encoding/decoding")
    parser.add_argument("--train", action="store_true", help="Test parallel VQ training")
    parser.add_argument("--fresh_context", action="store_true", help="Test fresh context decoding")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="Model to use")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.", help="Text to process/memorize")
    parser.add_argument("--iterations", type=int, default=30, help="Number of training iterations")
    
    args = parser.parse_args()
    
    # Default behavior if no options specified
    if not (args.test or args.train or args.fresh_context):
        args.test = True
    
    vq_encoder = None
    
    if args.test:
        vq_encoder = test_vq_parallel(model_name=args.model, text=args.text)
        
    if args.train:
        vq_encoder, history, metrics = train_vq_parallel(
            model_name=args.model, 
            text=args.text,
            num_iterations=args.iterations
        )
        
    if args.fresh_context:
        if vq_encoder is None:
            # If we haven't already created a VQ encoder in test or train
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            vq_encoder = ParallelVQEncoder(model, tokenizer)
            
        test_fresh_context_decoding(vq_encoder, text=args.text, model_name=args.model) 