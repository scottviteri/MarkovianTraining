"""
A class for providing mock implementations of things
"""
import torch

from collaborative_experiments.utils import get_device
from collaborative_experiments.constants import DEFAULT_MSG_CONTEXT_LENGTH


class MockConfig:
    def __init__(self):
        self.vocab_size = None


class MockOutput:
    def __init__(self, logits):
        self.logits = logits


class mockCausalGPT2(torch.nn.Module):
    def __init__(self, causal_lm_tokenizer):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = MockConfig()
        self.config.vocab_size = causal_lm_tokenizer.vocab_size
        self.device = get_device("mock")

    def forward(self, input_ids):
        """
        Sets the first token to be 1.0 and the rest to be 0.0
        Args:
            input_ids (torch.tensor): shape (batch_size, seq_len)
        Returns:
            output.logits (torch.tensor): shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        logits = torch.zeros(
            (batch_size, seq_len, vocab_size), device=input_ids.device
        ).to(torch.float32)
        # Set the first token of every sequence to have its first value as 1.0
        logits[:, 0, 0] = 1.0

        return MockOutput(logits)

    def generate(
        self,
        input_ids,
        max_length=DEFAULT_MSG_CONTEXT_LENGTH,
        num_return_sequences=1,
        **kwargs
    ):
        """
        Returns a tensor of shape (num_return_sequences, max_length) with random integers.
        """
        return torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(num_return_sequences, max_length),
            device=input_ids.device,
        )
