from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

# Try different variations
answer_variations = [
    "Answer:",
    " Answer:",
    "Answer: ",
    " Answer: ",
    "answer:",
    " answer:",
    "answer: ",
    " answer: "
]

for text in answer_variations:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = [tokenizer.decode([token]) for token in tokens]
    print(f"{text!r} -> {tokens} -> {decoded}")

# Also check for model specific tokens
print("\nSpecial tokens:")
for token_name, token_value in tokenizer.special_tokens_map.items():
    print(f"{token_name}: {token_value} -> {tokenizer.encode(token_value, add_special_tokens=False)}") 