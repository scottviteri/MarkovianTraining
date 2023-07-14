"""
A file demonstrating how to use multiple adapters for generating text

TODO:
    1. Fix tokenizer special tokens
"""

import timeit
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
from peft import LoraConfig, peft_model, PeftConfig, PeftModel
from tqdm import tqdm

from typing import Any, List


class MultipleLlamas:
    def __init__(self, language_model_name, agent_names: List[str], verbose=False):
        self.verbose = verbose
        self.tokenizer, self.language_model = self.load_language_model(language_model_name, agent_names)

    def load_language_model(self, language_model_name, agent_names: List[str]):
        """
        Loads a llama model

        Args:
            language_model_name (str): the name of the language model to load
            agents (List[str]): the names of the agents to add to the model. Each as LORA adapter.
        """
        tokenizer = LlamaTokenizer.from_pretrained(language_model_name)
        model = LlamaForCausalLM.from_pretrained(language_model_name)
        if self.verbose: print("We loaded the model, and it has the following parameters:")
        if self.verbose: print(model.num_parameters())
        # add peft adapter
        for agent_name in agent_names:
            lora_r = 4
            if agent_name == "large_lora_r":
                lora_r = 32 * (10 ** 6)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                fan_in_fan_out=False,
            )
            # check if model is a peft model already
            if isinstance(model, PeftModel):
                model.add_adapter(agent_name, lora_config)
            else:
                model = PeftModel(model, lora_config, agent_name)

            if self.verbose:
                print(f"Loaded adapter for agent {agent_name}")
                print(model.num_parameters())
                model.print_trainable_parameters()
                print("---")
        return tokenizer, model
    
    def generate_one_completion(self, prompt, agent_name):
        """
        Generates a completion for a prompt
        """
        self.language_model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if len(input_ids) > 200:
            input_ids = input_ids[:100]
            print("Warning: Truncating prompt to 100 tokens.")

        # set adapter for this particular agent
        self.language_model.set_adapter(agent_name)
        outputs = self.language_model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        return [self.tokenizer.decode(output) for output in outputs]

    def simulate_one_turn(self, agent_nmes: List[str] = None):
        """
        Simulates one turn of each agent in the list.

        Args:
            agent_names (List[str]): the names of the agents to simulate. If None, simulates all agents.
        
        Returns:
            agent_completions (Dict[str, List[str]]): a dictionary mapping agent names to their completions
        """
        prompt = "Hi there. This is an example prompt."
        agent_completions = {}

        for agent_name in agent_nmes:
            if self.verbose: print(f"Generating completion for agent {agent_name}")
            completion = self.generate_one_completion(prompt, agent_name)
            if self.verbose: print(completion)
            agent_completions[agent_name] = completion
        return agent_completions


def run_mock_llama_test(agent_names, agent_exp_list: List[List[str]]):
    """
    For a given set of experiments, evaluates the agents to simulate multiple times

    Args:
        agent_names (List[str]): the names of the agents to add to the model. Each as LORA adapter.
        agents_to_simulate (List[List[str]]): a list of agents to simulate. Each element is a list of agents to simulate.
    """
    # load the model
    trainer_1 = MultipleLlamas(
        "peterchatain/mock_llama",
        agent_names,
    )
    times = {}
    for agents_to_simulate in agent_exp_list:
        # simulate one turn, and time it using time
        start = timeit.default_timer()
        output = trainer_1.simulate_one_turn(agents_to_simulate)
        stop = timeit.default_timer()
        print("Time taken:", stop - start)
        times[tuple(agents_to_simulate)] = stop - start
    return times

def test_one_adapter_creation(num_tests=1):
    """
    This test ensure that despite multiple adapters being created, only the active one is used.
    """
    for _ in range(num_tests):
        time_1 = run_mock_llama_test(["agent1", "agent2"], [["agent1"]])[0]
        time_2 = run_mock_llama_test(["agent1", "large_lora_r"], [["agent1"]])[0]
        time_3 = run_mock_llama_test(["agent1", "large_lora_r"], [["large_lora_r"]])[0]
        time_4 = run_mock_llama_test(["agent1", "agent2"], [["agent1", "agent2"]])[0]

        print(f"expected that {time_3} is longest by far, followed by {time_4}, {time_2} which is closly folowed by {time_1}")
        assert time_3 > time_1 + time_2 + time_4


def test_set_adapter(num_tests=1):
    """
    This test ensures that multiple calls to set_adapter do not
    mean that adapters are stacked on top of each other.
    e.g. model.generate() should only utilize the active adapter, and not all the adapters that have been set.
    """
    for _ in range(num_tests):
        times_dict = run_mock_llama_test(["agent1", "large_lora_r", "agent2"], [["agent1"], ["agent2"], ["large_lora_r"], ["agent2", "agent1"]])
        print(times_dict)


if __name__ == "__main__":
    print("Testing peft_multi_adapters.py")
    # test_one_adapter_creation(3)
    test_set_adapter(1)