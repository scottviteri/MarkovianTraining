#!/usr/bin/env python
"""
Test script that demonstrates the effect of prefilling reasoning with answer prefixes.

This test:
1. Creates reasoning with random tokens (baseline)
2. Creates reasoning with the prefix of the answer (cheating)
3. Compares answer log probabilities between both cases
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from vq_training_minimal import calculate_answer_logprobs, CustomGradientFunction, VQTrainingState, VQConfig

def create_state(model_name):
    """Create a minimal VQ training state for testing."""
    print(f"Loading model: {model_name}")
    
    # Load actor and critic models - for testing we use the same model for both
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # Freeze critic model
    for param in model.parameters():
        param.requires_grad = False
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get device
    device = next(model.parameters()).device
    
    # Create minimal config
    config = VQConfig(
        model_name=model_name,
        debug_gradients=False  # Disable verbose gradient debugging output
    )
    
    # Create minimal state
    state = VQTrainingState(
        actor_model=model,
        critic_model=model,
        tokenizer=tokenizer,
        optimizer=None,
        device=device,
        token_embeddings=model.get_input_embeddings().weight,
        config=config
    )
    
    return state

def get_next_token_probs(model, tokenizer, context, device):
    """Get probabilities for the next token after a given context."""
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
    return probs

def get_top_tokens(probs, tokenizer, top_k=5):
    """Get the top-k most likely next tokens and their probabilities."""
    topk_probs, topk_indices = torch.topk(probs, top_k)
    tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
    probs = [p.item() for p in topk_probs]
    return list(zip(tokens, probs))

def test_next_token_prediction(state, questions_answers, n_tests=3, task_type="arithmetic"):
    """Test how different reasoning contexts affect next token prediction."""
    model = state.actor_model
    tokenizer = state.tokenizer
    device = state.device
    
    print("\n" + "="*80)
    print(f"TESTING NEXT TOKEN PREDICTION WITH DIFFERENT CONTEXTS (task_type: {task_type})")
    print("="*80)
    
    # Test different prompting strategies
    for i, (question, answer) in enumerate(questions_answers[:n_tests]):
        print(f"\nExample #{i+1}: {question}")
        print(f"Correct answer: {answer}")
        
        # For arithmetic, we typically want to predict the entire answer
        # For QA, often just the first token is critical
        answer_tokens = tokenizer.tokenize(answer)
        first_answer_token = answer_tokens[0] if answer_tokens else ""
        partial_answer = answer[:len(answer)//2]
        
        # Test different reasoning contexts
        contexts = {
            "No context": f"Question: {question}\nAnswer:",
            "Zero-shot": f"Question: {question}\nLet's solve this problem. The answer is:",
            "Step-by-step": f"Question: {question}\nLet me solve this step by step. First I'll analyze the problem, then determine the approach. The answer is:",
            "Hint (first token)": f"Question: {question}\nI know the answer starts with '{first_answer_token}'. The answer is:",
            "Hint (partial)": f"Question: {question}\nI know the answer begins with '{partial_answer}'. The answer is:",
            "Full answer": f"Question: {question}\nThe answer is clearly {answer}. So the answer is:",
        }
        
        # Get the top predicted tokens for each context
        print("\nTop 5 predicted next tokens for each context:")
        print("-" * 60)
        
        for context_name, context in contexts.items():
            probs = get_next_token_probs(model, tokenizer, context, device)
            top_tokens = get_top_tokens(probs, tokenizer)
            
            # Check if the correct first token is in top predictions
            correct_first_token_id = tokenizer.encode(answer, add_special_tokens=False)[0]
            correct_first_token = tokenizer.decode([correct_first_token_id])
            correct_token_prob = probs[correct_first_token_id].item()
            correct_token_rank = (probs > probs[correct_first_token_id]).sum().item() + 1
            
            print(f"{context_name}:")
            for token, prob in top_tokens:
                print(f"  {token}: {prob:.4f}" + (" ✓" if token.strip() == correct_first_token.strip() else ""))
            
            print(f"  Correct token '{correct_first_token}': rank {correct_token_rank}, prob {correct_token_prob:.4f}")
            print("-" * 60)

    print("\nTest completed!")

def main():
    parser = argparse.ArgumentParser(description="Test how reasoning context affects answer prediction")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-it",
                       help="Model name or path")
    parser.add_argument("--task_type", type=str, default="arithmetic",
                       help="Task type (arithmetic or qa)")
    parser.add_argument("--n_tests", type=int, default=3,
                       help="Number of test examples to run")
    args = parser.parse_args()
    
    # Create test state
    state = create_state(args.model_name)
    
    # Create test questions and answers
    if args.task_type == "arithmetic":
        questions_answers = [
            ("42 + 31", "73"),
            ("13 * 7", "91"),
            ("256 / 8", "32"),
            ("100 - 35", "65"),
            ("25^2", "625"),
            ("17 + 38", "55"),
            ("11 * 11", "121"),
            ("144 / 12", "12"),
            ("1024 - 512", "512"),
            ("10^3", "1000")
        ]
    else:  # QA tasks
        questions_answers = [
            ("What is the capital of France?", "Paris"),
            ("What color is the sky?", "Blue"),
            ("Who wrote Hamlet?", "William Shakespeare"),
            ("What is the square root of 144?", "12"),
            ("What planet is known as the Red Planet?", "Mars"),
            ("What is the chemical symbol for gold?", "Au"),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
            ("What is the largest ocean on Earth?", "Pacific Ocean"),
            ("What year did World War II end?", "1945"),
            ("What is the tallest mountain in the world?", "Mount Everest")
        ]
    
    # Run test
    test_next_token_prediction(
        state, 
        questions_answers, 
        n_tests=args.n_tests, 
        task_type=args.task_type,
    )

if __name__ == "__main__":
    main() 