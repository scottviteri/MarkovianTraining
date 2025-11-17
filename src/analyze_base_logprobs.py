import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import load_model_for_evaluation
from train import generate_question_answer_batches
import argparse
import numpy as np
from tqdm import tqdm
import json
import os

def calculate_token_logprobs_with_context(model, tokenizer, device, text, context_start, q_end=200, a_end=300):
    """Calculate log probabilities for answer tokens given varying context."""
    
    # Tokenize the text
    tokens = tokenizer(
        text, 
        return_tensors="pt",
        truncation=False
    )
    
    # Return None if text isn't long enough for full answer
    if tokens.input_ids.size(1) < a_end:
        return None
    
    # Truncate to include only context + answer
    tokens = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=a_end - context_start,
        padding=False
    ).to(device)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=tokens.input_ids)
        logits = outputs.logits
    
    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get log prob of each actual token
    token_log_probs = log_probs[:, :-1].gather(
        2, tokens.input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    
    # Calculate average log prob for answer section only
    answer_start_idx = q_end - context_start - 1  # -1 because we lost one position in the gather
    answer_end_idx = a_end - context_start - 1
    answer_logprobs = token_log_probs[0, answer_start_idx:answer_end_idx]
    
    return torch.mean(answer_logprobs).item()

def process_samples(args):
    """Process samples and save raw data to file."""
    print("Loading model...")
    model, _, tokenizer, device = load_model_for_evaluation(
        model_path=None,
        use_base_model=True,
        model_type="llama"
    )

    # Initialize data generator
    print("Initializing data generator...")
    data_gen = generate_question_answer_batches(
        num_batches=args.num_samples,
        batch_size=1,
        task_type="wiki_continuation",
        tokenizer=tokenizer,
        hyperparameters={
            "target_length": args.max_length,
            "question_length": args.max_length
        },
        chunk_size=args.num_samples
    )

    # Define context lengths to try
    context_lengths = range(10, 201, 10)  # Try contexts from 10 to 200 tokens
    
    # Initialize arrays to store results
    total_logprobs = {length: 0.0 for length in context_lengths}
    counts = {length: 0 for length in context_lengths}

    # Process samples
    print("Processing samples...")
    for batch in tqdm(data_gen, desc="Processing articles"):
        text = batch[0][0]
        
        # Try each context length
        for context_length in context_lengths:
            context_start = 200 - context_length
            avg_logprob = calculate_token_logprobs_with_context(
                model, tokenizer, device, text, 
                context_start=context_start
            )
            
            if avg_logprob is not None:
                total_logprobs[context_length] += avg_logprob
                counts[context_length] += 1

    # Calculate averages
    avg_logprobs = {
        length: total_logprobs[length] / counts[length] 
        for length in context_lengths 
        if counts[length] > 0
    }

    # Save data
    data = {
        "context_lengths": list(context_lengths),
        "avg_logprobs": list(avg_logprobs.values()),
        "counts": list(counts.values())
    }
    
    with open(args.intermediate_file, 'w') as f:
        json.dump(data, f)
    
    print(f"Raw data saved to {args.intermediate_file}")

def smooth_curve(data, window_size):
    """Apply moving average smoothing to the data."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')

def find_crossing_point(x, y, threshold=-1):
    """Find where smoothed curve crosses threshold."""
    crossing_idx = np.where(y < threshold)[0]
    if len(crossing_idx) > 0:
        return x[crossing_idx[0]]
    return None

def plot_results(args):
    """Create plot from saved data with smoothing."""
    print(f"Loading data from {args.intermediate_file}")
    with open(args.intermediate_file, 'r') as f:
        data = json.load(f)
    
    context_lengths = np.array(data["context_lengths"])
    avg_logprobs = np.array(data["avg_logprobs"])

    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(context_lengths, avg_logprobs, 'b-', alpha=0.3, label='Raw')
    
    # Apply smoothing if enough data points
    if len(context_lengths) > args.window_size:
        smoothed = smooth_curve(avg_logprobs, args.window_size)
        valid_x = context_lengths[args.window_size-1:]
        plt.plot(valid_x, smoothed, 'r-', label=f'Smoothed (window={args.window_size})')
        
        # Find crossing point
        crossing_point = find_crossing_point(valid_x, smoothed)
        if crossing_point is not None:
            plt.annotate(f'Crosses -1 at context length {int(crossing_point)}',
                        xy=(crossing_point, -1),
                        xytext=(crossing_point + 20, -0.8),
                        arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.axhline(y=-1, color='k', linestyle='--', label='y = -1')
    plt.xlabel('Context Length (tokens)')
    plt.ylabel('Average Log Probability of Target Section')
    plt.title('Target Section (200-300) Log Probability vs Context Length')
    plt.grid(True)
    plt.legend()

    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum sequence length to analyze")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of Wikipedia samples to average over")
    parser.add_argument("--window_size", type=int, default=50,
                       help="Window size for smoothing")
    parser.add_argument("--output", type=str, default="token_logprobs.png",
                       help="Output plot filename")
    parser.add_argument("--intermediate_file", type=str, default="logprobs_data.json",
                       help="File to store/load raw data")
    parser.add_argument("--process_only", action="store_true",
                       help="Only process samples and save data")
    parser.add_argument("--plot_only", action="store_true",
                       help="Only create plot from saved data")
    args = parser.parse_args()

    if args.plot_only and args.process_only:
        raise ValueError("Cannot specify both --plot_only and --process_only")

    if args.plot_only:
        plot_results(args)
    elif args.process_only:
        process_samples(args)
    else:
        process_samples(args)
        plot_results(args)

if __name__ == "__main__":
    main() 