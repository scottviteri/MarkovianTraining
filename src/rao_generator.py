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
from src.rao_tools import RaoConfig, log_and_print_info

class RaoGenerator:
    """RaoRaoRaoRaoRaoRaoRaoRaoRaoRao"""

    def __init__(
        self,
        cfg: RaoConfig
    ):
        self._cfg = cfg

        # sets self._dataloader, self._tokens_per_pure_reward
        # self._reward_prefix_tensor, self._tokens_per_pure_action,
        # and self._action_prefix_tensor
        self.set_prefixes()
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
        causal_lm = self._cfg.model
        causal_lm_tokenizer = self._cfg.tokenizer

        input_ids = data["input_ids"][0]

        rao_tensor = torch.tensor(
            [[] for _ in range(self._cfg.batch_size)],
            device=self._cfg.device,
            dtype=torch.int32,
        )

        new_losses = []
        for observation_index in range(self._cfg.obs_p_doc):
            optimizer.zero_grad()
            low_loss_value = (
                round(
                    np.mean(average_losses) - np.std(average_losses),
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
                (
                    rao_tensor[:, -self._cfg.tok_p_rao * self._cfg.num_rao :],
                    low_loss,
                    self._action_prefix_tensor,
                ),
                dim=-1,
            )

            # RAOR_A
            # argmax in Action (P_theta_t (helpful_msg_{t+1} | lhes ++ optimistic_loss_{t+1})))
            full_action = causal_lm.generate(
                inputs=incentive_rao[
                    :,
                    -(
                        self._cfg.tok_p_rao * (self._cfg.num_rao+1)
                        + self._cfg.tok_p_loss
                        + self._action_prefix_tensor.shape[-1]
                    ) :,
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
                prev_obs: TensorType["batch", "seq_length"] = input_ids[
                    :, observation_index - 1, :
                ]
            else:
                prev_obs: TensorType["batch", "seq_length"] = torch.full_like(
                    input_ids[:, 0, :], causal_lm_tokenizer.pad_token_id
                )
            true_obs: TensorType["batch", "seq_length"] = input_ids[
                :, observation_index, :
            ]
            true_obs = true_obs.to(self._cfg.device)

            # Calculate loss for the actual observation, using only the loss and action as context
            with torch.no_grad():
                # actual_loss_t = log P_theta (external_text_t | lhes ++ optimistic_loss_t ++ helpful_msg_t)
                # target num_rao = 0
                prediction = causal_lm(
                    torch.cat(
                        (
                            rao_tensor[:, -self._cfg.tok_p_rao * self._cfg.num_rao :],
                            low_loss,
                            action,
                            true_obs,
                        ),
                        dim=-1,
                    )
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
            rao_tensor = torch.cat((rao_tensor, actual_loss, action, true_obs), dim=-1)

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

    def batch(self, iterable):
        iterator = iter(iterable)
        while True:
            chunk = list(islice(iterator, self._cfg.batch_size))
            if not chunk:
                return
            yield chunk

    @staticmethod
    def merge_dictionaries(d0, d1):
        # return {k: d0[k] + d1[k] for k in d0.keys()}
        return {
            "input_ids": d0["input_ids"] + d1["input_ids"],
            "text": d0["text"] + [d1["text"]],
        }

    @staticmethod
    def intersperse_1D_tensors(tensor, interspersed_tensor, interval):
        # Chunk the tensor into segments of size interval
        chunks = tensor.chunk(len(tensor) // interval)
        # Intersperse the interspersed_tensor with the chunks
        interspersed = torch.cat([torch.cat([interspersed_tensor, chunk]) for chunk in chunks])
        return interspersed

    @staticmethod
    def intersperse_lists(list1, list2, interval):
        # Split list1 into chunks of size interval
        chunks = [list1[i:i + interval] for i in range(0, len(list1), interval)]
        # Intersperse list2 with the chunks
        interspersed = [item for sublist in zip([list2]*len(chunks), chunks) for item in sublist]
        # Flatten the list
        interspersed = [item for sublist in interspersed for item in sublist]
        return interspersed

    def cat_raw_data(self):
        ds = iter(load_dataset(self._cfg.dataset_name, self._cfg.task_name, split="train", streaming=True))
        ds_text = map(lambda x: {"text": x["inputs"]+x['targets'][0]}, ds) if self._cfg.dataset_name == "bigbench" else ds
        ds_tokenized = map(lambda x: {"text": x["text"], "input_ids": self._cfg.tokenizer(x["text"])["input_ids"]}, ds_text)
        while 1:
            obs_multiple_batches = {"text": [], "input_ids": []}
            while (
                len(obs_multiple_batches["input_ids"])
                < self._cfg.batch_size * self._cfg.tok_p_doc
            ):
                data = next(ds_tokenized)
                obs_multiple_batches = self.merge_dictionaries(obs_multiple_batches, data)
            # intersperse  self._observation_prefix_tensor (batch_size,  x) in obs_multiple_batches["input_ids"]  (batch_size, y) every self._tokens_per_pure_observation
            prefixed_input_ids = self.intersperse_lists(
                list1 = obs_multiple_batches["input_ids"],  
                list2 = self._observation_prefix_tensor[0].tolist(), 
                interval = self._tokens_per_pure_observation
            )    
            yield {"input_ids": prefixed_input_ids[:self._cfg.batch_size*self._cfg.tok_p_doc], "text": obs_multiple_batches["text"]}

    def to_batched_tensor(self, data_iterator):
        return map(lambda datapt: 
            {"input_ids": einops.rearrange(
                torch.tensor(datapt["input_ids"]),
                f"(batch_size obs_p_doc tok_p_obs) -> batch_size obs_p_doc tok_p_obs",
                batch_size=self._cfg.batch_size,
                tok_p_obs=self._cfg.tok_p_obs
            ),
                "text": datapt["text"],
            },
            data_iterator
        )

    def trunc_documents(self):
        # Select the dataset based on the GPU memory
        def concat_tensor(batch):
            return {
                "input_ids": torch.concat(
                    [torch.tensor(b["input_ids"]) for b in batch], dim=-1
                )
            }

        data_1D = self.cat_raw_data()
        data_batched = self.to_batched_tensor(data_1D)
        dataset = {"input_ids":[], "text":[]}
        for _ in range(self._cfg.num_batches):
            data = next(data_batched)
            dataset["input_ids"].append(data["input_ids"])
            dataset["text"].append(data["text"])

        data_set_features = Features(
            {
                "text": Value(dtype="string", id=None),
                "input_ids": Array3D(
                    shape=(self._cfg.batch_size, self._cfg.obs_p_doc, self._cfg.tok_p_obs), dtype="int32"
                ),
            }
        )

        truncated_dataset = Dataset.from_dict(dataset, features=data_set_features)
        #truncated_dataset = Dataset.from_dict(data_set, features=data_set_features)
        truncated_dataset.set_format(type="torch", columns=["input_ids", "text"])

        dataloader = DataLoader(
            truncated_dataset,
            #batch_size=self._cfg.batch_size,
            drop_last=True,
            shuffle=True,
        )

        self._dataloader = dataloader


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
    def dataloader(self):
        return self._dataloader
