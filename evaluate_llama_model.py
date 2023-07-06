"""
This is a script to evaluate a llama model on some data with respect to a reward model.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from tqdm import tqdm

from typing import Any

# TODO: move this into source code so that imports are cleaner and others can 
# pip install CollaborativeTraining
from data import get_superhf_prompts

class MockRewardModel(torch.nn.Module):
    """Mocks a HuggingFace AutoModelForSequenceClassification class."""

    def __init__(self) -> None:
        """Mocks initialization."""
        super().__init__()
        self.device = torch.device("cpu")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        **_: Any,
    ) -> Any:
        """Mocks the __call__ method for sequence classification."""
        output = type("", (), {})()  # TODO use an actual mocking library
        # Return a random float for each input in the batch
        output.logits = torch.randn(input_ids.shape[0])
        return output

    def forward(
        self,
        **kwargs: Any,
    ) -> Any:
        """Mocks the forward method for sequence classification."""
        return self(**kwargs)


# This is a class that evaluates a llama model.
# It stores things like the reward model, the dataset, the language model,
# and the tokenizer.
class LlamaEvaluator:
    def __init__(self, language_model_name, reward_model_name, dataset_name):
        self.tokenizer, self.language_model = self.load_language_model(language_model_name)
        self.reward_model = self.load_reward_model(reward_model_name)
        self.dataset = self.load_dataset(dataset_name)

    def load_language_model(self, language_model_name):
        """
        Loads a llama model
        """
        tokenizer = LlamaTokenizer.from_pretrained(language_model_name)
        model = LlamaForCausalLM.from_pretrained(language_model_name)
        return tokenizer, model


    def load_reward_model(self, reward_model_name):
        """
        Loads a reward model
        """
        if "mock" in reward_model_name:
            return MockRewardModel()
        
    def load_dataset(self, dataset_name):
        """
        Loads a dataset
        """
        return get_superhf_prompts(dataset_name)
    
    def generate_one_completion(self, prompt):
        """
        Generates a completion for a prompt
        """
        self.language_model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if len(input_ids) > 200:
            input_ids = input_ids[:100]
            print("Warning: Truncating prompt to 100 tokens.")
        outputs = self.language_model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        return [self.tokenizer.decode(output) for output in outputs]

    def evaluate(self, cap_num_prompts=-1):
        """
        Evaluates the model on the dataset
        """

        # (1) generate completions for each prompt in the dataset
        # (2) feed the completions into the reward model
        # (3) compute the average reward for each completion

        # (1)
        self.language_model.eval()
        completions = []
        if cap_num_prompts > 0:
            dataset = self.dataset[:cap_num_prompts]
            print("Warning: Capping number of prompts to {}.".format(cap_num_prompts))
        else:
            dataset = self.datasets
        for prompt in tqdm(dataset, desc="Generating completions"):
            completions.extend(self.generate_one_completion(prompt))
        
        # (2)
        rewards = []
        for completion in tqdm(completions, desc="Computing rewards"):
            input_ids = self.tokenizer.encode(completion, return_tensors="pt")
            rewards.append(self.reward_model(input_ids).logits.item())
        
        # (3)
        return sum(rewards) / len(rewards)


def evaluate_test():
    """
    Uses mockLM and mock RM to evaluate anthropic red team data
    """
    evaluator = LlamaEvaluator("peterchatain/mock_llama", "mockRM", "anthropic-red-team")
    print(evaluator.evaluate(cap_num_prompts=20))


def main():
    evaluate_test()

if __name__ == "__main__":
    main()