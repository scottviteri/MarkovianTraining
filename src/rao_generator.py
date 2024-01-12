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

import torch
from torchtyping import TensorType
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, Features, Value, Array2D
from einops import repeat, rearrange
import numpy as np
from rao_tools import RaoConfig, log_and_print_info

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
        average_losses,
        aggregate_losses,
        batch_index=None,
    ):
        rao_tensor = torch.zeros(
            (self._cfg.batch_size, self._cfg.tok_p_rao * self._cfg.num_rao),
            device=self._cfg.device,
            dtype=torch.int32,
        )
        causal_lm = self._cfg.model
        causal_lm_tokenizer = self._cfg.tokenizer

        new_losses = []
        for observation_index in range(self._cfg.obs_p_doc):
            optimizer.zero_grad()
            low_loss_value = (
                round(
                    np.mean(average_losses)
                    - np.std(average_losses),
                    3,
                )
                if average_losses
                else 0.5 
            )
            low_loss = causal_lm_tokenizer.batch_encode_plus(
                [str(low_loss_value) for _ in range(self._cfg.batch_size)],
                return_tensors="pt",
                truncation="longest_first",
                padding="max_length",
                max_length=self._tokens_per_pure_reward,
            ).input_ids.to(self._cfg.device)
            assert low_loss.shape[-1] == self._tokens_per_pure_reward
            low_loss = torch.cat((self._reward_prefix_tensor, low_loss), dim=-1)
            incentive_rao = torch.cat(
                (rao_tensor, low_loss, self._action_prefix_tensor), dim=-1
            )

            # RAOR_A
            # argmax in Action (P_theta_t (helpful_msg_{t+1} | lhes ++ optimistic_loss_{t+1})))
            full_action = causal_lm.generate(
                inputs=incentive_rao[
                    :, -(self._cfg.tok_p_rao*self._cfg.num_rao + self._cfg.tok_p_loss + 
                        self._action_prefix_tensor.shape[-1]) :
                ],
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

            # Generate a filler action of the same length as the actual action
            filler_action: TensorType["batch", "seq_length"] = torch.cat(
                (
                    self._action_prefix_tensor,
                    torch.full(
                        (self._cfg.batch_size, self._tokens_per_pure_action),
                        causal_lm_tokenizer.pad_token_id,
                        device=self._cfg.device,
                    ),
                ),
                dim=-1,
            )

            if observation_index > 0:
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

            # Calculate loss for the actual observation, using only the loss and action as context
            with torch.no_grad():
                 # actual_loss_t = log P_theta (external_text_t | lhes ++ optimistic_loss_t ++ helpful_msg_t) 
                prediction = causal_lm(torch.cat((rao_tensor, low_loss, action, true_obs), dim=-1))
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
                batch_loss_action = out.mean(dim=-1)

            # Calculate loss for the filler action
            if self._cfg.use_loss_difference:
                with torch.no_grad():
                    prediction = causal_lm(torch.cat((filler_action, true_obs), dim=-1))
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
                    batch_loss_filler = out.mean(dim=-1)
                    # Calculate the difference in loss
                    loss_difference = batch_loss_action - batch_loss_filler
            else:
                batch_loss_filler = None
                loss_difference = batch_loss_action
            new_losses.append(loss_difference.mean().item())

            string_losses: str = [str(round(r.item(), 3)) for r in loss_difference]
            losses_tensor: TensorType[
                "batch", "seq_length"
            ] = causal_lm_tokenizer.batch_encode_plus(
                string_losses,
                return_tensors="pt",
                padding="max_length",
                truncation="longest_first",
                max_length=self._tokens_per_pure_reward,
            ).input_ids.to(
                self._cfg.device
            )
            assert causal_lm_tokenizer.eos_token_id not in losses_tensor
            assert losses_tensor.shape[-1] == self._tokens_per_pure_reward
            actual_loss = torch.cat((self._reward_prefix_tensor, losses_tensor), dim=-1)
            # so we are adding to the end and removing from the front
            rao_tensor = torch.cat((rao_tensor, actual_loss, action, true_obs), dim=-1)[:,self._cfg.tok_p_rao:]

            log_and_print_info(
                self._cfg,
                batch_index,
                observation_index,
                batch_loss_action,
                batch_loss_filler,
                loss_difference,
                aggregate_losses,
                prev_obs,
                actual_loss,
                action,
                predicted_obs,
                true_obs,
                optimizer,
                rao_tensor,
            )

        return rao_tensor, new_losses

    def trunc_documents(self):
        truncated_documents = {"text": [], "input_ids": []}

        # Check the total memory of the GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(
                self._cfg.device
            ).total_memory / (
                1024**3
            )  # in GB
        else:
            # Force simple for non_cuda
            # Todo: need exception for mac mps
            gpu_memory = 1

        # Select the dataset based on the GPU memory
        if gpu_memory > 50:  # adjust this value based on your requirements
            dataset_name = "20220301.en"
        else:
            dataset_name = "20220301.simple"

        while len(truncated_documents["input_ids"]) < self._num_data_points:
            dataset = load_dataset(
                "wikipedia",
                dataset_name,
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
    def dataloader(self):
        return self._dataloader

