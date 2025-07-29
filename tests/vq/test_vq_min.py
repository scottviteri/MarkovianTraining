#!/usr/bin/env python
"""
Minimal test script for VQ Token Reduction

This script tests how few tokens are needed to reconstruct strings
using the Vector Quantization approach.
"""

import torch
import torch.nn.functional as F
import argparse
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import colored_print, Colors
from src.vq import ParallelVQEncoder

def test_token_reduction(model_name="google/gemma-3-1b-it", text="The quick brown fox jumps over the lazy dog.", num_iterations=30):
    """Test how few tokens are needed to reconstruct a string using VQ."""
    colored_print("Model", f"Loading {model_name}", Colors.BOLD)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    # Load a separate critic/decoder model
    colored_print("Critic Model", "Loading frozen critic model", Colors.BOLD)
    critic_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda()
    
    # Freeze the critic model
    for param in critic_model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create the VQ autoencoder
    vq_encoder = ParallelVQEncoder(model, tokenizer)
    
    # Print the tokenized input
    tokens = tokenizer.encode(text)
    token_strings = [tokenizer.decode([t]) for t in tokens]
    colored_print("Tokenized Input", f"{token_strings}", Colors.BLUE)
    colored_print("Token IDs", f"{tokens}", Colors.BLUE)
    
    # Train to memorize the text
    colored_print("Training", f"Training for {num_iterations} iterations", Colors.BOLD)
    history, metrics = vq_encoder.memorize_string(
        text=text,
        num_iterations=num_iterations,
        learning_rate=5e-5,
        temperature=1.0
    )
    
    # Print final reconstruction
    final_iter, final_text = history[-1]
    
    if final_text == text:
        colored_print("Success", f"Perfectly memorized text at iteration {final_iter}!", Colors.GREEN)
    else:
        colored_print("Final Result", f"After {final_iter} iterations:", Colors.YELLOW)
        colored_print("Original", text, Colors.BLUE)
        colored_print("Reconstructed", final_text, Colors.GREEN)
        # If not perfectly memorized, skip token reduction testing
        return history, metrics
    
    # Test with fresh context
    colored_print("Testing Fresh Context", "Ensuring decoder only uses quantized tokens with a fresh context", Colors.BOLD)
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
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
        decoded_text = tokenizer.decode(closest_token_ids[0], skip_special_tokens=True)
        
        # Print individual tokens for comparison
        used_tokens = [tokenizer.decode([t]) for t in segment_ids[0]]
        colored_print("Input Tokens", f"{used_tokens}", Colors.CYAN)
        
        # Print output tokens
        output_tokens = [tokenizer.decode([t]) for t in closest_token_ids[0]]
        colored_print("Output Tokens", f"{output_tokens}", Colors.MAGENTA)
        
        # Print final decoded text
        colored_print("Decoded Result", decoded_text, Colors.GREEN)
    
    # Now test precise token reduction with fine-grained steps
    colored_print("Token Reduction Test", "Finding minimum tokens needed for reconstruction:", Colors.BOLD)
    
    # Define token counts to test
    token_count = token_ids.size(1)
    
    if token_count <= 10:
        token_counts_to_test = range(1, token_count + 1)
    else:
        # For longer sequences, use percentages
        percentages = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        token_counts_to_test = sorted(set([max(1, int(p * token_count / 100)) for p in percentages]))
        # Add single token increments for the first few tokens
        for i in range(1, min(6, token_count)):
            if i not in token_counts_to_test:
                token_counts_to_test.append(i)
        token_counts_to_test.sort()
    
    # Test each token count
    min_perfect_tokens = token_count  # Initialize to max
    
    for count in token_counts_to_test:
        # Get subset of tokens
        reduced_tokens = token_ids[0, :count]
        reduced_text = tokenizer.decode(reduced_tokens, skip_special_tokens=True)
        
        # Calculate similarity
        similarity = difflib.SequenceMatcher(None, text, reduced_text).ratio() * 100
        perfect_match = reduced_text == text
        
        # Update minimum perfect token count
        if perfect_match and count < min_perfect_tokens:
            min_perfect_tokens = count
        
        # Print result
        token_percent = f"{count}/{token_count} tokens ({count/token_count*100:.1f}%)"
        similarity_str = f"{similarity:.1f}% similar"
        match_indicator = "✓" if perfect_match else "✗"
        
        # Choose color based on match quality
        color = Colors.GREEN if perfect_match else (
            Colors.YELLOW if similarity > 80 else Colors.RED
        )
        
        colored_print(
            f"{token_percent}",
            f"{match_indicator} {similarity_str}: '{reduced_text}'",
            color
        )
        
        # Print used tokens
        used_tokens = [tokenizer.decode([t]) for t in reduced_tokens]
        colored_print("Used tokens", f"{used_tokens}", Colors.CYAN)
    
    # Print summary
    if min_perfect_tokens < token_count:
        colored_print(
            "Result", 
            f"Perfect reconstruction with only {min_perfect_tokens}/{token_count} tokens ({min_perfect_tokens/token_count*100:.1f}%)!",
            Colors.GREEN
        )
    else:
        colored_print(
            "Result",
            "Required all tokens for perfect reconstruction",
            Colors.YELLOW
        )
    
    return history, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VQ Token Reduction")
    
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", 
                      help="Model to use")
    parser.add_argument("--text", type=str, 
                      default="The quick brown fox jumps over the lazy dog.",
                      help="Text to memorize")
    parser.add_argument("--iterations", type=int, default=30,
                      help="Number of training iterations")
    
    args = parser.parse_args()
    
    test_token_reduction(
        model_name=args.model,
        text=args.text,
        num_iterations=args.iterations
    ) 