"""
A file that defines the training object for collaborative training.
"""

from typing import List, Tuple, Dict, Any, Optional
from transformers import Trainer

from tqdm import tqdm

from .agent import Agent
from .constants import SYSTEM_PROMPT


class CollaborativeTrainer(Trainer):
    def __init__(
        self,
        num_agents: Optional[int] = 2,
        num_rounds: Optional[int] = 2,
        live_print: Optional[bool] = True,
        *args,
        **kwargs
    ):
        """
        A trainer that orchestrates the collaborative training process.

        This consists of:
            - a base language model
            - a collection of agents (either separate LMs or strings that are adapter names for the base LM)
            - a training pipeline and standardized interface for having agents communicate
        """

        super().__init__(*args, **kwargs)

        agent_names = ["agent"] * num_agents
        for i, agent_name in enumerate(agent_names):
            agent_names[i] = agent_name + str(i)

        for agent_name in agent_names:
            agent = Agent(
                agent_name,
                agent_type="lora",
                agent_dataset="mock",
                agent_lora_config={},
            )
            context_history = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        agent_name=agent.agent_name, other_agents=agent_names
                    ),
                }
            ]
            agent.set_context_history(context_history)

        self.num_rounds = num_rounds
        self.live_print = live_print

    def complete_one_round(self):
        """
        Completes one round of collaborative training.
        """

        # generate the state for each agent

        # generate the action for each agent

        # distribute the actions among the agents

        # collect a training corpus observation from each agent

        # compute a loss for each agent on the training corpus observation
        # update the model parameters for each agent using the loss on training corpus

        # distribute the feedback reward (loss of that agent) to each of the corresponding agents that sent messages
        # update the model on it's message it sent based on the reward received from other agents

    def train(self):
        """
        Executes the total training pipeline
        """
        for _ in tqdm(range(self.num_rounds)):
            self.complete_one_round()
