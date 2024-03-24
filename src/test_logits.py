import torch
import os
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from einops import repeat
from training_types import *
from utilities import extend_initial_config, test_string
from matplotlib import pyplot as plt


p1 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(
    "cpu"
)
p2 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cpu")
t1 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
t2 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# s1 = "Argosy was an American magazine, founded by Frank Munsey in 1882 as a children's weekly."
# s2 = "In 1896 it became the first pulp magazine, printing only fiction and using cheap pulp paper. Circulation rose and remained strong for decades, but fell to no more than 50,000 by 1942."

p1.to("cuda")
p2.to("cuda")


def plot_test_strings(predictor, tokenizer, sentences, model_name, device="cuda"):
    # predictor.to(device)
    for i, sentence in enumerate(sentences, start=1):
        plt.figure(i)
        plt.clf()
        encoding_true, neg_log_probs_true = test_string(
            predictor, tokenizer, sentence, add_bos=True
        )
        encoding_false, neg_log_probs_false = test_string(
            predictor, tokenizer, sentence, add_bos=False
        )
        plt.plot(neg_log_probs_true, label="add_bos=True")
        plt.plot(neg_log_probs_false, label="add_bos=False")
        plt.title(f"{model_name} {i}")
        plt.legend()
        plt.savefig(f"{model_name.lower()}_{i}.png")
    # predictor.to("cpu")


# regular_sentences = [
#    "The sun sets in the west and rises in the east.",
#    "A group of flamingos is called a flamboyance.",
#    "Mount Everest is the highest mountain in the world.",
#    "The Pacific Ocean is the largest ocean on Earth.",
# ]
# plot_test_strings(p1, t1, regular_sentences, "Instruct-Regular")
# plot_test_strings(p2, t2, regular_sentences, "Base-Regular")

# sentences = [
#    "Argosy was an American magazine, founded by Frank Munsey in 1882 as a children's weekly.",
#    "The quick brown fox jumps over the lazy dog.",
#    "[INST] Summarize the following text. [/INST] Argosy was an American magazine, founded by Frank Munsey in 1882 as a children's weekly. In 1896 it became the first pulp magazine, printing only fiction and using cheap pulp paper. Circulation rose and remained strong for decades, but fell to no more than 50,000 by 1942.",
#    "[INST] Explain the significance of 'The quick brown fox jumps over the lazy dog.' [/INST] This sentence is known for containing every letter of the English alphabet, making it a pangram. It is commonly used for typing practice and in font displays.",
# ]
arithmetic_sentences = [
    "What is the product of 123 and 456? The answer is fifty-six thousand and eighty-eight.",
    "If you have 1000 apples and give away 250, then buy another 150, how many do you have in total? You would have nine hundred apples in total.",
    "[INST] Calculate the following expression. What is the product of 123 and 456? [/INST] The answer is fifty-six thousand and eighty-eight.",
    "[INST] Solve the following problem. If you have 1000 apples and give away 250, then buy another 150, how many do you have in total? [/INST] You would have nine hundred apples in total.",
]
plot_test_strings(p1, t1, arithmetic_sentences, "Instruct-Arithmetic")
plot_test_strings(p2, t2, arithmetic_sentences, "Base-Arithmetic")

# plot_test_strings(p1, t1, sentences, "Instruct")
# plot_test_strings(p2, t2, sentences, "Base")
