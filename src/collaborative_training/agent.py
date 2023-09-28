"""
Implements the agent class.

TODO:
    1. Decide how to store the context history. (Dictionary? list of messages with role and content?)
    2. Implement the agent class
    3. Edit the dataset to be correct
"""

from typing import List, Tuple, Dict, Any, Optional
from peft import LoraConfig, peft_model, PeftConfig, PeftModel

from .data import get_superhf_prompts
from .constants import SYSTEM_PROMPT


class Agent:
    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        agent_dataset: str,
        agent_lora_config: Dict,
    ):
        """
        Agent class consists of:
            - a language model (either an adapter on top of a base model, or a separate LM itself)
            - a context history of accumulated experiences
            - a training corpus unique to this agent

        Args:
            agent_name (str): the name of the agent
            agent_type (str): the type of the agent (e.g. llama, gpt2, llama_lora, gpt2_lora).
                If lora is in the name, use lora adapter.
            agent_dataset (str): the name of the dataset to use for training
            agent_lora_config (Dict): the config for the lora adapter
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.agent_dataset = agent_dataset
        self.agent_lora_config = agent_lora_config

        if "lora" in self.agent_type:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
                fan_in_fan_out=False,
            )
            for key, value in agent_lora_config.items():
                setattr(lora_config, key, value)
        else:
            raise NotImplementedError("Only lora agents are supported at this time.")

        self.dataset = get_superhf_prompts(self.agent_dataset)  # TODO
        self.context_history = []
        self.context_prompt = ""

    def set_context_history(history: List[Dict]):
        """
        Sets the history of the agent, including fine tune observations, messages, and prompts
        """
        self.context_history = history

    def _flatten_context(self, provided_context=None):
        """
        Uses the current context window and flattens is into a prompt which then updates
        self.context_prompt to be this prompt.

        Args:
            provided_context (List[Dict]): the context to use, only provided for debugging purposes
        """
        context = self.context_history
        if provided_context is not None:
            context = provided_context

        prompt = ""
        for message in context:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += "<System> " + content + " </System>"
            elif role == "user":
                prompt += "<User> " + content + " </User>"
            elif role == "assistant":
                prompt += "<assistant> " + content + " </assistant>"
        if provided_context is None:
            self.context_prompt = prompt
        return prompt

    def generate_one_completion(**generation_kwargs):
        """
        Flattens the context history into one prompt. Generates a copmletion and returns the completion only.
        Stores the completion in it's own context history
        """
