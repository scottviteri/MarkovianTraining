"""
We will pull in passages from wikipedia articles.
For each article that is long enough, we will break it into chunks of fixed token count, and discard the rest.
The dataloader will feed the ith segment of BATCH_SIZE different articles to the transformer simultaneously to generate rewards.
We reassemble the article subsequences to include (reward, prediction, article snippet) triples.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value, Array2D
from einops import repeat
from collaborative_experiments.rao_tools import RaoConfig

# POINTS_FROM_DATASET = NUM_DATAPOINTS


class RaoGenerator:
    """RaoRaoRaoRaoRaoRaoRaoRaoRaoRao"""

    def __init__(
        self,
        cfg: RaoConfig,
        points_from_data: int,
        num_data_points: int,
    ):
        self._cfg = cfg
        self._points_from_data = points_from_data
        self._num_data_points = num_data_points

        # sets self._dataloader, self._tokens_per_pure_reward
        # self._reward_prefix_tensor, self._tokens_per_pure_action,
        # and self._action_prefix_tensor
        self.trunc_documents()

    def trunc_documents(self):
        truncated_documents = {"text": [], "input_ids": []}

        while len(truncated_documents["input_ids"]) < self._num_data_points:
            # This while loop is used to load the dataset. It will keep running until a valid dataset is loaded.
            # The termination condition is when a valid dataset is loaded without any exceptions.
            # not creating enough datapoints
            dataset = load_dataset(
                "wikipedia",
                "20220301.en",
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
