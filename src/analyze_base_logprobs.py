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
    
    # Tokenize the text first to check length
    tokens = tokenizer(
        text, 
        return_tensors="pt",
        truncation=False  # Don't truncate yet, we want to check full length
    )
    
    # Return None if text is too short
    if tokens.input_ids.size(1) < max_length:
        return None
    
    # Now tokenize with truncation for processing
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
        num_batches=args.num_samples * 2,  # Request more samples to account for filtering
        batch_size=1,
        task_type="wiki_continuation",
        tokenizer=tokenizer,
        hyperparameters={
            "target_length": args.max_length,
            "question_length": args.max_length
        },
        chunk_size=args.num_samples * 2
    )

    # Initialize arrays to store results
    total_logprobs = torch.zeros(args.max_length - 1).to(device)
    counts = torch.zeros(args.max_length - 1).to(device)

    # Process samples
    print("Processing samples...")
    all_logprobs = []  # Store individual sample logprobs
    processed_samples = 0
    skipped_samples = 0
    total_samples_seen = 0
    
    print("Starting sample processing loop...")
    for batch in tqdm(data_gen, desc="Processing articles"):
        if processed_samples >= args.num_samples:
            break
            
        total_samples_seen += 1
        if total_samples_seen % 10 == 0:  # Print status every 10 samples
            print(f"\rSeen: {total_samples_seen}, Processed: {processed_samples}, "
                  f"Skipped: {skipped_samples}, Target: {args.num_samples}", end="")
            
        text = batch[0][0]
        
        # Print length info for debugging
        initial_tokens = tokenizer(text, return_tensors="pt", truncation=False)
        print(f"\nArticle length: {initial_tokens.input_ids.size(1)} tokens "
              f"(need {args.max_length})")
        
        token_logprobs = calculate_token_logprobs(model, tokenizer, device, text, args.max_length)
        
        if token_logprobs is None:
            skipped_samples += 1
            continue
        
        # Store individual sample data
        all_logprobs.append(token_logprobs.cpu().tolist())
        
        # Add to running totals
        seq_len = min(len(token_logprobs), args.max_length - 1)
        total_logprobs[:seq_len] += token_logprobs[:seq_len]
        counts[:seq_len] += 1
        
        processed_samples += 1

    print(f"\nFinal stats:")
    print(f"Total samples seen: {total_samples_seen}")
    print(f"Processed {processed_samples} samples")
    print(f"Skipped {skipped_samples} samples that were too short")

    # Save raw data
    data = {
        "individual_logprobs": all_logprobs,
        "counts": counts.cpu().tolist(),
        "metadata": {
            "processed_samples": processed_samples,
            "skipped_samples": skipped_samples,
            "total_samples_seen": total_samples_seen,
            "max_length": args.max_length
        }
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
    
    # Get actual sequence length from the data
    seq_length = len(data["individual_logprobs"][0])  # Length of first sample
    print(f"Actual sequence length in data: {seq_length}")
    
    # Adjust window size if it's too large
    window_size = min(args.window_size, seq_length // 2)
    if window_size != args.window_size:
        print(f"Adjusting window size from {args.window_size} to {window_size} due to short sequence length")
    
    # Initialize arrays with correct size
    total_logprobs = np.zeros(seq_length)
    counts = np.array(data["counts"][:seq_length])  # Trim to actual length
    
    # Sum up logprobs
    for sample_logprobs in data["individual_logprobs"]:
        total_logprobs += sample_logprobs[:seq_length]
    
    avg_logprobs = total_logprobs / counts

    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(range(1, seq_length + 1), avg_logprobs, 'b-', alpha=0.3, label='Raw')
    
    # Only apply smoothing if we have enough data points
    if seq_length > window_size:
        # Apply smoothing
        valid_x = np.arange(1, seq_length - window_size + 2)
        smoothed_logprobs = smooth_curve(avg_logprobs, window_size)
        
        # Plot smoothed data
        plt.plot(valid_x, smoothed_logprobs, 'r-', 
                label=f'Smoothed (window={window_size})')
        
        # Find crossing point on smoothed curve
        crossing_point = find_crossing_point(valid_x, smoothed_logprobs)
        if crossing_point is not None:
            plt.annotate(f'Smoothed curve crosses -1 at position {int(crossing_point)}',
                        xy=(crossing_point, -1),
                        xytext=(crossing_point + 50, -0.8),
                        arrowprops=dict(facecolor='black', shrink=0.05))
    else:
        print(f"Warning: Sequence length ({seq_length}) too short for smoothing with window size {window_size}")
        smoothed_logprobs = avg_logprobs  # Use raw data for statistics
    
    plt.axhline(y=-1, color='k', linestyle='--', label='y = -1')
    plt.xlabel('Token Position')
    plt.ylabel('Average Log Probability')
    plt.title('Average Token Log Probabilities in LLaMA Base Model')
    plt.grid(True)
    plt.legend()

    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

    # Print statistics
    print("\nStatistics:")
    print(f"Sequence length: {seq_length} tokens")
    print(f"Average log probability (raw): {np.mean(avg_logprobs):.4f}")
    if seq_length > window_size:
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