from transformers import AutoTokenizer

def load_tokenizer(model_name="tinystories"):
    """
    Load a tokenizer for testing.
    
    Args:
        model_name: One of ["tinystories", "phi", "gpt2", "mistral", "llama"]
    """
    if model_name == "tinystories":
        model_id = "roneneldan/TinyStories-33M"
    elif model_name == "phi":
        model_id = "microsoft/Phi-3.5-mini-instruct"
    elif model_name == "gpt2":
        model_id = "openai-community/gpt2"
    elif model_name == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_name == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from ['tinystories', 'phi', 'gpt2', 'mistral', 'llama']")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

if __name__ == "__main__":
    # Test loading with command line argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="tinystories",
        choices=["tinystories", "phi", "gpt2", "mistral", "llama"],
        help="Model tokenizer to test"
    )
    args = parser.parse_args()
    
    # Load the specified tokenizer
    tokenizer = load_tokenizer(args.model)
    print(f"\nTesting {args.model} tokenizer:")

    # Test strings
    test_strings = [
        "Answer:",
        "Answer: 42",
        "The Answer: is",
        "Here is the Answer: to your question",
        "answer:",
        "ANSWER:"
    ]

    # Test each string
    for text in test_strings:
        print(f"\nAnalyzing: '{text}'")
        tokens = tokenizer.encode(text)
        
        # Print token details
        print("Tokens:")
        for token in tokens:
            token_text = tokenizer.decode([token])
            print(f"  Token {token}: '{token_text}'")