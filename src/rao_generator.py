"""
Main Definitions:

1. RaoGenerator class
2. __init__ method
3. gen_rao_tensor method
4. trunc_documents method

Short Description:

We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.

Longer Desciption:

The rao_generator.py script is responsible for generating reward-action-observation (RAO) sequences from Wikipedia articles for training a transformer model. Here's a high-level overview:

1. Import necessary libraries and modules: This includes libraries like torch, torchtyping, datasets, einops, and numpy, as well as the RaoConfig and log_and_print_info from rao_tools.py.

2. Define RaoGenerator class: This class is responsible for generating RAO sequences. It includes several methods:

- __init__: Initializes the RaoGenerator with a RaoConfig and the number of data points. It also sets up the dataloader and various tensor attributes.

- gen_rao_tensor: Generates an RAO tensor for a given batch of data. It iterates over each observation in the batch, generates an action using the transformer model, calculates the loss for the actual and filler actions, and updates the RAO tensor. It also logs and prints information for each observation.

- trunc_documents: Truncates the documents in the dataset to a fixed token count and discards the rest. The truncated documents are then used to generate the RAO sequences.

3. Use of RaoConfig: The RaoConfig class from rao_tools.py is used to configure the RAO generation process. It encapsulates various configuration parameters, such as the model, tokenizer, batch size, and token counts for the reward, action, and observation.

4. Use of log_and_print_info: The log_and_print_info function from rao_tools.py is used to log and print information during the RAO generation process. It logs and prints information such as the batch number, loss values, previous observation, actual loss, action, predicted observation, true observation, size of RAO triple, size of the context window, and the current learning rate.

In summary, rao_generator.py is responsible for generating RAO sequences from Wikipedia articles using a transformer model, which are then used for training. It uses the RaoConfig class to configure the RAO generation process and the log_and_print_info function to log and print information during the process.
"""
import einops
import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value, Array3D
from einops import repeat, rearrange
from itertools import islice
from functools import reduce
import numpy as np
from src.rao_tools import (
    RaoConfig,
    log_and_print_info,
    condense_triples,
    create_loss_tokens_tensor,
)


class RaoGenerator:
    """RaoRaoRaoRaoRaoRaoRaoRaoRaoRao"""

    def __init__(self, cfg: RaoConfig):
        self._cfg = cfg

        # sets self._dataloader, self._tokens_per_pure_reward
        # self._reward_prefix_tensor, self._tokens_per_pure_action,
        # and self._action_prefix_tensor
        self.set_prefixes()
        self._dataset = self.prepare_dataset()

    def gen_rao_tensor(
        self, input_ids, loss_fn, aggregate_losses, optimistic_loss: float, prev_obs, batch_index
    ):
        causal_lm = self._cfg.model
        causal_lm_tokenizer = self._cfg.tokenizer

        rao_tensor_triples = []
        default_tensor = torch.tensor(
            [[] for _ in range(self._cfg.batch_size)],
            dtype=torch.int64,
            device=self._cfg.device,
        )
        losses = torch.tensor(
            [[] for _ in range(self._cfg.batch_size)],
            dtype=torch.float32,
            device=self._cfg.device,
        )
        optimistic_loss_tokens_tensor = create_loss_tokens_tensor(
            torch.tensor(
                [[optimistic_loss] for _ in range(self._cfg.batch_size)],
                dtype=torch.float32,
                device=self._cfg.device,
            ),
            self._cfg.tokenizer,
            self._cfg.device,
            self._tokens_per_pure_reward,
        )

        for observation_index in range(self._cfg.obs_between_weight_updates):
            incentive_rao = torch.cat(
                (
                    condense_triples(
                        rao_tensor_triples[-self._cfg.num_rao :], default_tensor
                    ),
                    optimistic_loss_tokens_tensor,
                    self._action_prefix_tensor,
                ),
                dim=-1,
            )

            # RAOR_A
            # argmax in Action (P_theta_t (helpful_msg_{t+1} | lhes ++ optimistic_loss_{t+1})))
            if self._cfg.use_multirao_for_action_gen:
                action_input = incentive_rao[
                    :,
                    -(self._cfg.ctxt_size - self._tokens_per_pure_action) :,
                ]
            else:
                action_input = incentive_rao[
                    :,
                    -(
                        self._cfg.tok_p_rao * (self._cfg.num_rao + 1)
                        + self._cfg.tok_p_loss
                        + self._action_prefix_tensor.shape[-1]
                    ) :,
                ]
            with torch.no_grad():
                full_action = causal_lm.generate(
                    inputs=action_input,
                    output_scores=True,
                    do_sample=True,
                    num_beams=self._cfg.num_beams,
                    min_new_tokens=self._tokens_per_pure_action,
                    max_new_tokens=self._tokens_per_pure_action,
                    pad_token_id=causal_lm_tokenizer.eos_token_id,
                )
            action: TensorType["batch", "seq_length"] = full_action[
                :, -self._cfg.tok_p_action :
            ]

            true_obs: TensorType["batch", "seq_length"] = input_ids[
                :, observation_index, :
            ].to(self._cfg.device)

            # Calculate loss for the actual observation, using only the loss and action as context
            # actual_loss_t = log P_theta (external_text_t | lhes ++ optimistic_loss_t ++ helpful_msg_t)
            # target num_rao = 0
            context = torch.cat(
                (
                    condense_triples(
                        rao_tensor_triples[-self._cfg.num_rao :], default_tensor
                    )
                    if self._cfg.num_rao > 0
                    else torch.tensor(
                        [[] for _ in range(self._cfg.batch_size)],
                        dtype=torch.int32,
                        device=self._cfg.device,
                    ),
                    optimistic_loss_tokens_tensor,
                    action,
                    true_obs,
                ),
                dim=-1,
            )
            prediction = causal_lm(context)
            predicted_logits = prediction.logits[:, -self._cfg.tok_p_obs - 1 : -1, :]
            predicted_obs = predicted_logits.argmax(dim=-1)
            out = loss_fn(
                input=rearrange(
                    predicted_logits,
                    "batch seq_length vocab_size -> batch vocab_size seq_length",
                ),
                target=true_obs,
            )
            batch_loss = out.mean(dim=-1, keepdim=True)

            # Calculate loss for the filler action
            if self._cfg.use_loss_difference:
                with torch.no_grad():
                    prediction = causal_lm(true_obs)
                    predicted_logits = prediction.logits[:, :-1, :]
                    predicted_obs = predicted_logits.argmax(dim=-1)
                    out = loss_fn(
                        input=rearrange(
                            predicted_logits,
                            "batch seq_length vocab_size -> batch vocab_size seq_length",
                        ),
                        target=true_obs[:, 1:],
                    )
                    batch_loss = batch_loss - out.mean(dim=-1)

            optimistic_loss_tokens_tensor = causal_lm_tokenizer.batch_encode_plus(
                [
                    str(round(self.inject_noise(optimistic_loss).item(), 3))
                    for _ in range(self._cfg.batch_size)
                ],
                return_tensors="pt",
                truncation="longest_first",
                padding="max_length",
                max_length=self._tokens_per_pure_reward,
            ).input_ids.to(self._cfg.device)
            assert (
                optimistic_loss_tokens_tensor.shape[-1] == self._tokens_per_pure_reward
            )
            optimistic_loss_tokens_tensor = torch.cat(
                (self._reward_prefix_tensor, optimistic_loss_tokens_tensor), dim=-1
            )

            losses = torch.cat((losses, batch_loss), dim=-1)
            rao_tensor_triples.append((optimistic_loss_tokens_tensor, action, true_obs))

            log_and_print_info(
                self._cfg,
                batch_index,
                observation_index,
                batch_loss,
                aggregate_losses,
                prev_obs,
                action,
                predicted_obs,
                true_obs,
            )
            prev_obs = true_obs

        return rao_tensor_triples, losses, true_obs 

    @staticmethod
    def inject_noise(loss: torch.Tensor) -> torch.Tensor:
        return loss * (1.0 + torch.randn(1) * 0.05)

    def prepare_dataset(self):
        itr_ds = load_dataset(self._cfg.dataset_name, self._cfg.task_name, split="train", streaming=True)
        ds_tokenized = map(
            lambda x: self._cfg.tokenizer(x["text"], return_tensors="pt")["input_ids"].to(self._cfg.device), 
            itr_ds)
        pure_obs = get_pure_obs(self._cfg.batch_size, self._tokens_per_pure_observation, self._cfg.device, ds_tokenized)
        obs_ds = prepend_obs_tensor(self._observation_prefix_tensor, pure_obs)
        return take(self._cfg.num_batches, 
            group_obs_tensors(self._cfg.obs_between_weight_updates, obs_ds))    

    def set_prefixes(self):
        observation_prefix = "\nObservation: "
        observation_prefix_tokens = self._cfg.tokenizer.encode(
            observation_prefix, add_special_tokens=False
        )
        observation_prefix_tensor = repeat(
            torch.tensor(observation_prefix_tokens),
            "tokens -> batch tokens",
            batch=self._cfg.batch_size,
        ).to(self._cfg.device)
        tokens_per_pure_observation = self._cfg.tok_p_obs - len(
            observation_prefix_tokens
        )

        action_prefix = "\nAction: "
        action_prefix_tokens = self._cfg.tokenizer.encode(
            action_prefix, add_special_tokens=False
        )
        action_prefix_tensor = repeat(
            torch.tensor(action_prefix_tokens),
            "tokens -> batch tokens",
            batch=self._cfg.batch_size,
        ).to(self._cfg.device)
        tokens_per_pure_action = self._cfg.tok_p_action - len(action_prefix_tokens)

        reward_prefix = "\nLoss: "
        reward_prefix_tokens = self._cfg.tokenizer.encode(
            reward_prefix, add_special_tokens=False
        )
        reward_prefix_tensor = repeat(
            torch.tensor(reward_prefix_tokens),
            "tokens -> batch tokens",
            batch=self._cfg.batch_size,
        ).to(self._cfg.device)
        tokens_per_pure_reward = self._cfg.tok_p_loss - len(reward_prefix_tokens)

        self._tokens_per_pure_reward = tokens_per_pure_reward
        self._reward_prefix_tensor = reward_prefix_tensor
        self._tokens_per_pure_action = tokens_per_pure_action
        self._action_prefix_tensor = action_prefix_tensor
        self._tokens_per_pure_observation = tokens_per_pure_observation
        self._observation_prefix_tensor = observation_prefix_tensor

    @property
    def dataset(self):
        return self._dataset

    @property
    def tokens_per_pure_reward(self):
        return self._tokens_per_pure_reward


def get_pure_obs(batch_size, tok_per_pure_obs, device, itr_ds):
    batches = [torch.empty((1,0), dtype=torch.int64, device=device) 
        for _ in range(batch_size)]
    while 1:
        for i in range(len(batches)):
            while batches[i].shape[1] < tok_per_pure_obs:
                batches[i] = torch.cat((batches[i], next(itr_ds)),dim=1)
        for batch in batches: assert  batch.shape[-1] >= tok_per_pure_obs
        out_tensor = torch.cat([batch[:,:tok_per_pure_obs] for batch in batches], dim=0)
        for i in range(len(batches)):
            batches[i] = batches[i][:, tok_per_pure_obs:]
        yield out_tensor
    return batches 

def prepend_obs_tensor(obs_prefix_tensor, itr_ds):
    return map(lambda x: torch.cat((obs_prefix_tensor, x), dim=1), itr_ds)

def group_obs_tensors(obs_per_weight_update, itr_ds):
    while 1:
        grouped = torch.stack([next(itr_ds) for _ in range(obs_per_weight_update)])
        yield einops.rearrange(grouped, 'buffer batch tokens -> batch buffer tokens')

def take(num_batches, itr_ds): 
    for _ in range(num_batches): yield next(itr_ds)