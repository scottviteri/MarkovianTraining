"""
Mock classes for testing.
"""

import time
from typing import Any

import numpy as np
import torch
from transformers import (
    GenerationMixin,
    PretrainedConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def make_mock_llama_model() -> Any:
    """
    Creates a mock LlamaForCausalLM model.

    The llama model has one hidden layer, hidden size 2, and 2 attention heads.
    We increase vocab size to accomodate a special pad token.
    """
    config = LlamaConfig(
        vocab_size=32001,
        hidden_size=1,
        intermediate_size=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=32000,  # '[PAD]'
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )

    model = LlamaForCausalLM(config)
    return model


def make_mock_llama_tokenizer() -> Any:
    """
    Creates a mock LlamaTokenizer.
    """
    return LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


def save_normal_mock_llama():
    """
    Saves a small llama model to the hub
    """
    print("Testing making a llama model")
    llama_model = make_mock_llama_model()
    print(llama_model)
    # print the number of parameters this model has
    print("Number of parameters:", llama_model.num_parameters())
    # save mock model to the hub
    print("Saving to hub")
    # llama_model.push_to_hub("mock_llama")
    tokenizer = make_mock_llama_tokenizer()
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(tokenizer)
    # pylint: disable=not-callable
    tokenizer.push_to_hub("mock_llama")


def save_multi_adapter_model():
    """
    Todo: save  model wtih multiple adapters on it randomly initalized
    """
    pass


if __name__ == "__main__":
    print("Testing mocking.py")
    # save_normal_mock_llama()
