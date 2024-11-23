import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from evaluate_gsm8k import load_model
from train import generate_question_answer_batches
import argparse
import numpy as np
from tqdm import tqdm

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum sequence length to analyze")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of Wikipedia samples to average over")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--output", type=str, default="token_logprobs.png",
                       help="Output plot filename")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model(
        model_path=None,  # Not loading trained weights
        use_base_model=True,
        model_type="llama"
    )

    # Initialize data generator with correct hyperparameters
    print("Initializing data generator...")
    data_gen = generate_question_answer_batches(
        num_batches=args.num_samples,
        batch_size=1,  # Process one at a time for simplicity
        task_type="wiki_continuation",
        tokenizer=tokenizer,
        hyperparameters={
            "target_length": args.max_length,  # Changed from question_length to target_length
            "question_length": args.max_length  # Keep this if needed
        }
    )

    # Initialize arrays to store results
    total_logprobs = torch.zeros(args.max_length - 1).to(device)
    counts = torch.zeros(args.max_length - 1).to(device)

    # Process samples
    print("Processing samples...")
    for i, batch in enumerate(tqdm(data_gen)):
        if i >= args.num_samples:
            break
            
        text = batch[0][0]  # Get text from first (and only) item in batch
        
        # Calculate log probs for this sample
        token_logprobs = calculate_token_logprobs(model, tokenizer, device, text, args.max_length)
        
        # Add to running totals
        seq_len = min(len(token_logprobs), args.max_length - 1)
        total_logprobs[:seq_len] += token_logprobs[:seq_len]
        counts[:seq_len] += 1

    # Calculate averages
    avg_logprobs = (total_logprobs / counts).cpu().numpy()

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, args.max_length), avg_logprobs)
    plt.axhline(y=-1, color='r', linestyle='--', label='y = -1')
    plt.xlabel('Token Position')
    plt.ylabel('Average Log Probability')
    plt.title('Average Token Log Probabilities in LLaMA Base Model')
    plt.grid(True)
    plt.legend()
    
    # Add text annotation for crossing point
    crossing_point = np.where(avg_logprobs < -1)[0][0] if any(avg_logprobs < -1) else None
    if crossing_point is not None:
        plt.annotate(f'Crosses -1 at position {crossing_point}',
                    xy=(crossing_point, -1),
                    xytext=(crossing_point + 50, -0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

    # Print some statistics
    print("\nStatistics:")
    print(f"Average log probability across all positions: {np.mean(avg_logprobs):.4f}")
    if crossing_point is not None:
        print(f"Log probability crosses -1 at position: {crossing_point}")
    print(f"Min log probability: {np.min(avg_logprobs):.4f}")
    print(f"Max log probability: {np.max(avg_logprobs):.4f}")

if __name__ == "__main__":
    main() 