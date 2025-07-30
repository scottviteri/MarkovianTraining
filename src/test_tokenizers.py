from transformers import AutoTokenizer

def load_tokenizer(model_name="tinystories"):
    if model_name == "tinystories":
        tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories", padding_side="left")
    elif model_name == "phi":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", padding_side="left", trust_remote_code=True)
    elif model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left")
    elif model_name == "mistral":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side="left")
    elif model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")
    elif model_name == "qwen25":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side="left")
    elif model_name == "qwen3":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", padding_side="left")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Set pad token to eos token for consistency
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Print tokenizer info
    print(f"Tokenizer: {model_name}")
    print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    return tokenizer

if __name__ == "__main__":
    # Test loading with command line argument
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="tinystories",
        choices=["tinystories", "phi", "gpt2", "mistral", "llama", "qwen25", "qwen3"],
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