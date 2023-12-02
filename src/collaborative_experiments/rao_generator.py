"""
We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""

import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value, Array2D
from einops import repeat, rearrange
import numpy as np
from collaborative_experiments.rao_tools import RaoConfig, log_and_print_info

# POINTS_FROM_DATASET = NUM_DATAPOINTS


class RaoGenerator:
    """RaoRaoRaoRaoRaoRaoRaoRaoRaoRao"""

    def __init__(
        self,
        cfg: RaoConfig,
        num_data_points: int,
    ):
        self._cfg = cfg
        self._points_from_data = num_data_points 
        self._num_data_points = num_data_points

        # sets self._dataloader, self._tokens_per_pure_reward
        # self._reward_prefix_tensor, self._tokens_per_pure_action,
        # and self._action_prefix_tensor
        self.trunc_documents()

    def gen_rao_tensor(
        self,
        data,
        optimizer,
        loss_fn,
        aggregate_losses,
        batch_index=None,
        wandb_table=None,
    ):
        rao_tensor = torch.tensor(
            [[] for _ in range(self._cfg.batch_size)],
            device=self._cfg.device,
            dtype=torch.int32,
        )
        causal_lm = self._cfg.model
        causal_lm_tokenizer = self._cfg.tokenizer

        for observation_index in range(self._cfg.obs_p_doc):
            optimizer.zero_grad()
            high_reward_value = (
                round(np.mean(aggregate_losses) - np.std(aggregate_losses), 3)
                if aggregate_losses
                else 6.0
            )
            high_reward = causal_lm_tokenizer(
                [str(high_reward_value) for _ in range(self._cfg.batch_size)],
                return_tensors="pt",
                padding="max_length",
                # Fixme: one context uses cfg.tok_p_reward here
                max_length=self._tokens_per_pure_reward,
            ).input_ids.to(self._cfg.device)
            # _reward_prefix_tensor already on device
            high_reward = torch.cat(
                (self._reward_prefix_tensor, high_reward), dim=-1
            )
            incentive_rao = torch.cat(
                (rao_tensor, high_reward, self._action_prefix_tensor), dim=-1
            )
            full_action = causal_lm.generate(
                inputs=incentive_rao,
                output_scores=True,
                do_sample=True,
                return_dict_in_generate=True,
                max_new_tokens=self._tokens_per_pure_action,
                pad_token_id=causal_lm_tokenizer.pad_token_id,
                eos_token_id=None,
            )
            action: TensorType["batch", "seq_length"] = full_action.sequences[
                :, -self._cfg.tok_p_action :
            ]
            if observation_index > 1:
                prev_obs: TensorType["batch", "seq_length"] = data["input_ids"][
                    :, observation_index - 1, :
                ]
            else:
                prev_obs: TensorType["batch", "seq_length"] = torch.full_like(
                    data["input_ids"][:, 0, :], causal_lm_tokenizer.pad_token_id
                )
            true_obs: TensorType["batch", "seq_length"] = data["input_ids"][
                :, observation_index, :
            ]
            true_obs = true_obs.to(self._cfg.device)
            with torch.no_grad():
                prediction = causal_lm(
                    torch.cat((rao_tensor, high_reward, action, true_obs), dim=-1)
                )
                predicted_logits = prediction.logits[
                    :, -self._cfg.tok_p_obs - 1 : -1, :
                ]
                predicted_obs = predicted_logits.argmax(dim=-1)
                out = loss_fn(
                    input=rearrange(
                        predicted_logits,
                        "batch seq_length vocab_size -> batch vocab_size seq_length",
                    ),
                    target=true_obs,
                )
                batch_loss = out.mean(dim=-1)
            string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
            losses_tensor: TensorType["batch", "seq_length"] = causal_lm_tokenizer(
                string_losses, return_tensors="pt", padding=True
            ).input_ids.to(self._cfg.device)
            actual_reward = torch.cat(
                (self._reward_prefix_tensor, losses_tensor), dim=-1
            )
            rao_tensor = torch.cat(
                (rao_tensor, actual_reward, action, true_obs), dim=-1
            )[:, -(self._cfg.ctxt_size - self._cfg.tok_p_rao) :]

            # print("rao tensor: ", repr(causal_lm_tokenizer.decode(rao_tensor[0])))
            # print()
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
                optimizer,
                wandb_table,
            )

            return rao_tensor

    def trunc_documents(self):
        truncated_documents = {"text": [], "input_ids": []}

        while len(truncated_documents["input_ids"]) < self._num_data_points:
            # This while loop is used to load the dataset. It will keep running until a valid dataset is loaded.
            # The termination condition is when a valid dataset is loaded without any exceptions.
            # not creating enough datapoints
            dataset = load_dataset(
                "wikipedia",
                "20220301.simple",
                split=f"train[:{self._points_from_data}]"
                if self._points_from_data
                else "train",
            )
            dataset = dataset.map(
                lambda example: self._cfg.tokenizer(example["text"]), batched=True
            )
            dataset.set_format(type="torch", columns=["input_ids", "text"])

            # Define your features
            truncated_document_features = Features(
                {
                    "text": Value(dtype="string", id=None),
                    #'input_ids': Array2D(shape=(TOKENS_PER_OBSERVATION, OBSERVATIONS_PER_DOCUMENT), dtype='int32')
                    "input_ids": Array2D(
                        shape=(self._cfg.obs_p_doc, self._cfg.tok_p_obs), dtype="int32"
                    ),
                }
            )

            truncated_documents = {"text": [], "input_ids": []}

            observation_prefix = "\nObservation: "
            observation_prefix_tokens = self._cfg.tokenizer.encode(
                observation_prefix, add_special_tokens=False
            )
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

            reward_prefix = "\nReward: "
            reward_prefix_tokens = self._cfg.tokenizer.encode(
                reward_prefix, add_special_tokens=False
            )
            reward_prefix_tensor = repeat(
                torch.tensor(reward_prefix_tokens),
                "tokens -> batch tokens",
                batch=self._cfg.batch_size,
            ).to(self._cfg.device)
            tokens_per_pure_reward = self._cfg.tok_p_reward - len(reward_prefix_tokens)

            # currently need to use tensors in input_ids as per the truncated_document_features
            for document in dataset:
                if (
                    self._num_data_points
                    and len(truncated_documents["input_ids"]) == self._num_data_points
                ):
                    break
                # Only accept long enough samples
                if document["input_ids"].shape[-1] <= self._cfg.tok_p_doc:
                    continue
                # truncate documents
                tokens = document["input_ids"].tolist()
                # Inject "Observation: " prefix every TOKENS_PER_OBSERVATION - observation_prefix_tokens
                new_tokens = []
                for j in range(0, len(tokens), tokens_per_pure_observation):
                    new_tokens.extend(observation_prefix_tokens)
                    # new_tokens.extend(torch.randint(0, causal_lm_tokenizer.vocab_size, (tokens_per_pure_observation,)).tolist())
                    new_tokens.extend(tokens[j : j + tokens_per_pure_observation])
                new_document = torch.tensor(new_tokens[: self._cfg.tok_p_doc])
                reshaped_document = new_document.reshape(
                    (self._cfg.obs_p_doc, self._cfg.tok_p_obs)
                )
                truncated_documents["input_ids"].append(reshaped_document)
                # random_tokens = torch.randint(0, causal_lm_tokenizer.vocab_size, reshaped_document.shape)
                # truncated_documents["input_ids"].append(random_tokens)
                assert len(new_tokens[: self._cfg.tok_p_doc]) == self._cfg.tok_p_doc
                # leave the text alone (not truncated)
                truncated_documents["text"].append(document["text"])
            if not self._num_data_points:
                num_data_points = len(truncated_documents["text"])
                NUM_BATCHES = num_data_points // self._cfg.batch_size
                break
            self._points_from_data *= 2

        assert len(truncated_documents["input_ids"]) == self._num_data_points

        # Convert the list of dictionaries into a Dataset in one go
        truncated_dataset = Dataset.from_dict(
            truncated_documents, features=truncated_document_features
        )
        truncated_dataset.set_format(type="torch", columns=["input_ids", "text"])

        dataloader = DataLoader(
            truncated_dataset,
            batch_size=self._cfg.batch_size,
            drop_last=True,
            shuffle=True,
        )

        self._dataloader = dataloader
        self._tokens_per_pure_reward = tokens_per_pure_reward
        self._reward_prefix_tensor = reward_prefix_tensor
        self._tokens_per_pure_action = tokens_per_pure_action
        self._action_prefix_tensor = action_prefix_tensor

    @property
    def points_from_data(self):
        return self._points_from_data

    @property
    def num_data_points(self):
        return self._num_data_points

    @property
    def dataloader(self):
        return self._dataloader

    @property
    def tokens_per_pure_reward(self):
        return self._tokens_per_pure_reward

    @property
    def reward_prefix_tensor(self):
        return self._reward_prefix_tensor

    @property
    def tokens_per_pure_action(self):
        return self._tokens_per_pure_action

    @property
    def action_prefix_tensor(self):
        return self._action_prefix_tensor
