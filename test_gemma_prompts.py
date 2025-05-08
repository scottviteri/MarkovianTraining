from transformers import AutoTokenizer
from src.constants import GEMMA3_BOS, GEMMA3_START_OF_TURN, GEMMA3_END_OF_TURN

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

# Sample prompt with reasoning
question = "Calculate 5 + 7 + 3"
reasoning = "I'll add 5 + 7 first to get 12. Then I'll add 12 + 3."
prompt_type = "Reasoning:"

# Create full prompt with Gemma-3 formatting
full_prompt = (
    f"{GEMMA3_BOS}{GEMMA3_START_OF_TURN}user\n"
    f"You will be given an arithmetic problem, which you have 40 tokens to work through step-by-step. Question: {question}{GEMMA3_END_OF_TURN}\n"
    f"{GEMMA3_START_OF_TURN}model\n"
    f"{prompt_type}{reasoning} Answer: "
)

print("Full prompt:")
print(full_prompt)
print("\n" + "-"*80 + "\n")

# Tokenize the prompt
tokens = tokenizer.encode(full_prompt)
token_strings = [tokenizer.decode([token]) for token in tokens]

print("Token IDs:")
print(tokens)
print("\nToken strings:")
for i, (token, token_str) in enumerate(zip(tokens, token_strings)):
    print(f"{i}: {token} -> {token_str}")

# Now verify the 'Answer:' detection
# Find where "Answer:" appears in the tokens
answer_tokens = []
answer_positions = []

for i in range(len(tokens) - 1):
    # Check for variants of "Answer" token followed by colon
    if (((tokens[i] == 25685) or  # " Answer"
        (tokens[i] == 7925) or   # "Answer"
        (tokens[i] == 14433) or  # "answer"
        (tokens[i] == 3890)) and  # " answer" 
        (tokens[i+1] == 236787)):  # ":"
        answer_tokens.append((tokens[i], tokens[i+1]))
        answer_positions.append(i)

print("\n" + "-"*80 + "\n")
print("Found 'Answer:' at positions:", answer_positions)
print("Token pairs:", answer_tokens)

# Test the position that would be returned by our function
if answer_positions:
    pos = answer_positions[-1] + 2
    print(f"\nAnswer start position would be: {pos}")
    print(f"Tokens from this position: {tokens[pos:]}")
    print(f"Decoded: {tokenizer.decode(tokens[pos:])}") 