"""
Main Definitions:

1. RaoConfig class
2. log_and_print_info

High level structure:

The rao_tools.py script provides tools for reward-action-observation (RAO) training and sequence generation. Here's the high-level structure with an emphasis on the RaoConfig class and the log_and_print_info function:

1. Import necessary libraries and modules: This includes libraries like torch, torchtyping, transformers, peft, and wandb.

2. Define RaoConfig class: This class is used to set up RAO-like training and data generation. It includes several attributes that represent various configuration parameters, such as device, wandb, load_model, do_lora, model_name, lr, tok_p_loss, tok_p_action, tok_p_obs, num_beams, batch_size, num_batches,  interval_save_weights, and interval_print. The class also includes methods to set the model and create an attention mask. The RaoConfig class is particularly important as it encapsulates the configuration parameters for RAO training and data generation. It also includes methods to set the model and create an attention mask, which are crucial steps in the setup process.

3. Define log_and_print_info function: This function is used to log and print information during the training process. It takes in various parameters such as the configuration, batch index, loss values, optimizer, and the RAO tensor. It then logs and prints information such as the batch number, loss values, previous observation, actual loss, action, predicted observation, true observation, size of RAO triple, size of the context window, and the current learning rate. This function is crucial for monitoring the training process and debugging.

4. Define main function: This function is used to test the RaoConfig class. It creates an instance of this class and prints it.

5. Run the main function: If the script is run as a standalone program (i.e., not imported as a module), the main function is called.
"""

from dataclasses import dataclass
import torch
import torchtyping
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import wandb


# from collections import namedtuple
# Config = namedtuple("Config", ["setting1", "setting2"])


class RaoConfig:
    """Immutable config class to set up rao-like training and data generation."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        dataset_name: str = "wikipedia",
        task_name: str = None,
        lr: float = 1e-4,
        num_rao: int = 1,
        batch_size: int = 10,
        num_batches: int = 100,
        tok_p_loss: int = 12,
        obs_between_weight_updates: int = 10,
        obs_to_action_ratio: float = 1.0,
        num_beams: int = 1,
        interval_save_weights: int = 100,
        interval_print: int = 10,
        training_ctxt_size=None,
        wandb: bool = False,
        load_model: bool = False,
        do_lora: bool = True,
        use_loss_difference: bool = True,
        use_multirao_for_action_gen: bool = False,
        use_rewards_to_go: bool = False,
        alternate_training: bool = False,
        regular_training: bool = False,
        gpt_eval: bool = False 
    ):
        self._model_name = model_name
        self._lr = lr
        self._num_rao = num_rao
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._tok_p_loss = tok_p_loss
        self._obs_between_weight_updates = obs_between_weight_updates
        self._obs_to_action_ratio = obs_to_action_ratio
        self._num_beams = num_beams
        self._interval_save_weights = interval_save_weights
        self._interval_print = interval_print
        self._wandb = wandb
        self._load_model = load_model
        self._do_lora = do_lora
        self._use_loss_difference = use_loss_difference
        self._use_multirao_for_action_gen = use_multirao_for_action_gen
        self._use_rewards_to_go = use_rewards_to_go
        self._training_ctxt_size = training_ctxt_size
        self._dataset_name = dataset_name
        self._alternate_training = alternate_training 
        self._regular_training = regular_training
        self._gpt_eval = gpt_eval 

        self._device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self._task_name = self._set_task_name(task_name)
        self._path_2_model = f"saved_weights_and_losses/{self.model_name}_weights"
        self._path_2_tokenizer = (
            f"saved_weights_and_losses/{self._model_name}_tokenizer"
        )

        # sets model, tokenizer and ctxt_size
        self._set_model()
        if self._training_ctxt_size is None:
            self._training_ctxt_size = self._ctxt_size

        self._tok_p_action = int(
            self._training_ctxt_size
            / ((self._obs_to_action_ratio + 1.0) * (self._num_rao + 1.0))
            - self._tok_p_loss / (self._obs_to_action_ratio + 1)
        )
        if self.regular_training:
            self._tok_p_obs = self._ctxt_size
        else:
            self._tok_p_obs = int(self._tok_p_action * self._obs_to_action_ratio)
            self._tok_p_doc = self._tok_p_obs * self._obs_between_weight_updates
            self._tok_p_rao = self._tok_p_loss + self._tok_p_action + self._tok_p_obs
            assert (self._num_rao + 1) * self._tok_p_rao <= self._training_ctxt_size

    def _set_task_name(self, task_name):
        if task_name:
            return task_name
        if self._dataset_name == "wikipedia":
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(
                    self.device
                ).total_memory / (1023**3)
                if gpu_memory > 49:
                    task_name = "20220301.en"
                else:
                    task_name = "20220301.simple"
            else:
                task_name = "20220301.simple"
        else:  # self._dataset_name == "bigbench":
            task_name = "arithmetic"
        return task_name

    def _set_model(self):
        """Load model"""
        model_dict = {
            "tinystories": "roneneldan/TinyStories-1M",
            "llama": "meta-llama/Llama-2-7b-hf",
            "distilgpt2": "distilgpt2",
            "gptj": "EleutherAI/gpt-j-6b",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "gpt2-large": "gpt2-large",
            "gpt2-xl": "gpt2-xl",
            "gpt2": "gpt2",
            # Add other models here
        }

        with self._device:
            if self._load_model:
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    self._path_2_model,
                    torch_dtype=torch.float16,
                    use_flash_attention_2=self._model_name == "mistral"
                    or self._model_name == "llama",
                )
                causal_lm.bfloat16()
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    self._path_2_tokenizer,
                    padding_side="left",
                )
                if self._model_name == "mistral":
                    self._ctxt_size = causal_lm.config.sliding_window
                elif self._model_name == "tinystories":
                    self._ctxt_size = causal_lm.config.window_size
                elif self._model_name == "llama":
                    self._ctxt_size = causal_lm.config.max_position_embeddings
                elif self._model_name == "distilgpt2":
                    self._ctxt_size = causal_lm.config.n_ctx
                elif self._model_name == "gptj":
                    self._ctxt_size = causal_lm.config.n_positions
                elif self._model_name == "gpt2-large":
                    self._ctxt_size = causal_lm.config.n_positions
                else:
                    self._ctxt_size = causal_lm.config.n_positions

            elif self._model_name == "tinystories":
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_size="left"
                )
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                self._ctxt_size = causal_lm.config.window_size
            elif self._model_name == "llama":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name],
                    torch_dtype=torch.float16,
                    use_flash_attention_2=True,
                )
                causal_lm.bfloat16()
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.max_position_embeddings
            elif self._model_name == "mistral":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name],
                    torch_dtype=torch.float16,
                    use_flash_attention_2=True,
                )
                causal_lm.bfloat16()
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.max_position_embeddings

            elif self._model_name == "distilgpt2":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.n_ctx

            elif self._model_name == "gptj":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.n_positions

            elif self._model_name == "gpt2-large":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.n_ctx

            elif self._model_name == "gpt2-xl":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.n_ctx

            elif self._model_name == "gpt2":
                causal_lm = AutoModelForCausalLM.from_pretrained(
                    model_dict[self._model_name]
                )
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    model_dict[self._model_name], padding_side="left"
                )
                self._ctxt_size = causal_lm.config.n_ctx

        if self._do_lora:
            peft_config = LoraConfig(
                # basemodel_name_or_path=MODEL,
                r=64,
                lora_alpha=128,
                lora_dropout=0.1,
                target_modules=self._get_linear_layers(causal_lm),
            )

            causal_lm = get_peft_model(causal_lm, peft_config)

        # causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        causal_lm_tokenizer.padding_side = "left"
        causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.encode(" ")[0]

        self._model = causal_lm
        self._tokenizer = causal_lm_tokenizer

    @staticmethod
    def _get_linear_layers(model):
        return list(
            set(
                map(
                    lambda x: x[0].split(".")[-1],
                    filter(
                        lambda x: isinstance(x[1], torch.nn.Linear),
                        model.named_modules(),
                    ),
                )
            )
        )

    @property
    def model_name(self):
        return self._model_name

    @property
    def lr(self):
        return self._lr

    @property
    def num_rao(self):
        return self._num_rao

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def tok_p_loss(self):
        return self._tok_p_loss

    @property
    def obs_to_action_ratio(self):
        return self._obs_to_action_ratio

    @property
    def num_beams(self):
        return self._num_beams

    @property
    def interval_save_weights(self):
        return self._interval_save_weights

    @property
    def interval_print(self):
        return self._interval_print

    @property
    def wandb(self):
        return self._wandb

    @property
    def load_model(self):
        return self._load_model

    @property
    def do_lora(self):
        return self._do_lora

    @property
    def use_loss_difference(self):
        return self._use_loss_difference

    @property
    def training_ctxt_size(self):
        return self._training_ctxt_size

    @property
    def use_multirao_for_action_gen(self):
        return self._use_multirao_for_action_gen

    @property
    def use_rewards_to_go(self):
        return self._use_rewards_to_go

    @property
    def device(self):
        return self._device

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def task_name(self):
        return self._task_name

    @property
    def path_2_model(self):
        return self._path_2_model

    @property
    def path_2_tokenizer(self):
        return self._path_2_tokenizer

    @property
    def tok_p_action(self):
        return self._tok_p_action

    @property
    def tok_p_obs(self):
        return self._tok_p_obs

    @property
    def tok_p_doc(self):
        return self._tok_p_doc

    @property
    def obs_between_weight_updates(self):
        return self._obs_between_weight_updates

    @property
    def tok_p_rao(self):
        return self._tok_p_rao

    @property
    def ctxt_size(self):
        return self._ctxt_size

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def alternate_training(self):
        return self._alternate_training

    @property
    def regular_training(self):
        return self._regular_training

    @property
    def gpt_eval(self):
        return self._gpt_eval

    def __repr__(self):
        return (
            f"RaoConfig({self._model_name}, do_lora {self._do_lora}, "
            + f"batch_size {self._batch_size}, tok_p_action {self._tok_p_action}, "
            + f"ctxt_size {self._ctxt_size})"
        )


def multi_print(string, f):
    print(string); print(string, file=f)

def log_and_print_info(
    cfg,
    batch_index,
    observation_index,
    batch_loss,
    aggregate_losses,
    prev_obs,
    action,
    predicted_obs,
    true_obs,
):
    tokenizer = cfg.tokenizer
    if batch_index % cfg.interval_print == 0:
        with open(f"saved_weights_and_losses/{cfg.model_name}_log.txt", "a") as f:
            multi_print(f"\nBatch Number {batch_index}", f)
            multi_print(f"Loss: {batch_loss[0][0]:.3f}", f)
            if aggregate_losses:
                multi_print(f"Aggregate Loss: {aggregate_losses[-1]}", f)
            multi_print(f"Previous Obs: {repr(tokenizer.batch_decode(prev_obs)[0])}", f)
            multi_print(f"Action: {repr(tokenizer.batch_decode(action)[0])}", f)
            multi_print(f"Predicted Obs: {repr(tokenizer.batch_decode(predicted_obs)[0])}", f)
            multi_print(f"True Obs: {repr(tokenizer.batch_decode(true_obs)[0])}", f)
            multi_print("___________________________________________", f)
        if cfg.wandb:
            wandb.log(
                {
                    "Batch Index": batch_index,
                    "Batch Loss": batch_loss[0],
                }
            )


def condense_triples(rao_tensor_triples, default_tensor):
    if not rao_tensor_triples:
        return default_tensor
    return torch.cat(
        [torch.cat(triple, dim=-1) for triple in rao_tensor_triples], dim=-1
    )


def compute_cumulative_averages(losses: torch.Tensor) -> torch.Tensor:
    # Flip the tensor to compute right-sided cumulative averages
    losses = torch.flip(losses, dims=[1])
    cumulative_averages = torch.cumsum(losses, dim=1) / torch.arange(
        1, losses.shape[1] + 1, device=losses.device
    )
    # Flip the tensor back
    cumulative_averages = torch.flip(cumulative_averages, dims=[1])
    return cumulative_averages


def create_loss_tokens_tensor(
    batch_loss: torch.Tensor, tokenizer, device: torch.device, tokens_per_pure_reward
):
    string_losses: str = [str(round(r.item(), 3)) for r in batch_loss]
    loss_tokens_tensor: TensorType["batch", "seq_length"] = tokenizer.batch_encode_plus(
        string_losses,
        return_tensors="pt",
        padding="max_length",
        truncation="longest_first",
        max_length=tokens_per_pure_reward,
    ).input_ids.to(device)
    assert tokenizer.eos_token_id not in loss_tokens_tensor
    assert loss_tokens_tensor.shape[-1] == tokens_per_pure_reward
    return loss_tokens_tensor


def main():
    # define each class in default to check if they work
    test0 = RaoConfig()
    print(test0)


if __name__ == "__main__":
    main()
