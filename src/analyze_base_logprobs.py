import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from evaluate_gsm8k import load_model
from train import generate_question_answer_batches
import argparse
import numpy as np
from tqdm import tqdm
import json
import os

def calculate_token_logprobs(model, tokenizer, device, text, max_length):
    """Calculate log probabilities for each token position."""
    
    # Tokenize the text
    tokens = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=max_length
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
    
    return token_log_probs[0]  # Remove batch dimension

def process_samples(args):
    """Process samples and save raw data to file."""
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(
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

    # Initialize arrays to store results
    total_logprobs = torch.zeros(args.max_length - 1).to(device)
    counts = torch.zeros(args.max_length - 1).to(device)

    # Process samples
    print("Processing samples...")
    all_logprobs = []  # Store individual sample logprobs
    for i, batch in enumerate(tqdm(data_gen)):
        if i >= args.num_samples:
            break
            
        text = batch[0][0]
        token_logprobs = calculate_token_logprobs(model, tokenizer, device, text, args.max_length)
        
        # Store individual sample data
        all_logprobs.append(token_logprobs.cpu().tolist())
        
        # Add to running totals
        seq_len = min(len(token_logprobs), args.max_length - 1)
        total_logprobs[:seq_len] += token_logprobs[:seq_len]
        counts[:seq_len] += 1

    # Save raw data
    data = {
        "individual_logprobs": all_logprobs,
        "counts": counts.cpu().tolist(),
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
    
    # Calculate average logprobs
    total_logprobs = np.zeros(args.max_length - 1)
    counts = np.array(data["counts"])
    
    for sample_logprobs in data["individual_logprobs"]:
        seq_len = min(len(sample_logprobs), args.max_length - 1)
        total_logprobs[:seq_len] += sample_logprobs[:seq_len]
    
    avg_logprobs = total_logprobs / counts

    # Apply smoothing
    valid_x = np.arange(1, args.max_length - args.window_size + 1)
    smoothed_logprobs = smooth_curve(avg_logprobs, args.window_size)

    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot both raw and smoothed data
    plt.plot(range(1, args.max_length), avg_logprobs, 'b-', alpha=0.3, label='Raw')
    plt.plot(valid_x, smoothed_logprobs, 'r-', label=f'Smoothed (window={args.window_size})')
    
    plt.axhline(y=-1, color='k', linestyle='--', label='y = -1')
    plt.xlabel('Token Position')
    plt.ylabel('Average Log Probability')
    plt.title('Average Token Log Probabilities in LLaMA Base Model')
    plt.grid(True)
    plt.legend()
    
    # Find crossing point on smoothed curve
    crossing_point = find_crossing_point(valid_x, smoothed_logprobs)
    if crossing_point is not None:
        plt.annotate(f'Smoothed curve crosses -1 at position {int(crossing_point)}',
                    xy=(crossing_point, -1),
                    xytext=(crossing_point + 50, -0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

    # Print statistics
    print("\nStatistics:")
    print(f"Average log probability (raw): {np.mean(avg_logprobs):.4f}")
    print(f"Average log probability (smoothed): {np.mean(smoothed_logprobs):.4f}")
    if crossing_point is not None:
        print(f"Smoothed curve crosses -1 at position: {int(crossing_point)}")
    print(f"Min log probability (smoothed): {np.min(smoothed_logprobs):.4f}")
    print(f"Max log probability (smoothed): {np.max(smoothed_logprobs):.4f}")

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