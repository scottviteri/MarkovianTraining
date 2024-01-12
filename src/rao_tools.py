"""
Main Definitions:

1. RaoConfig class
2. log_and_print_info

High level structure:

The rao_tools.py script provides tools for reward-action-observation (RAO) training and sequence generation. Here's the high-level structure with an emphasis on the RaoConfig class and the log_and_print_info function:

1. Import necessary libraries and modules: This includes libraries like torch, torchtyping, transformers, peft, and wandb.

2. Define RaoConfig class: This class is used to set up RAO-like training and data generation. It includes several attributes that represent various configuration parameters, such as device, wandb, load_model, do_lora, model_name, lr, tok_p_loss, tok_p_action, tok_p_obs, normalize_to_ctxt_size, num_beams, batch_size, num_batches,  interval_save_weights, and interval_print. The class also includes methods to set the model and create an attention mask. The RaoConfig class is particularly important as it encapsulates the configuration parameters for RAO training and data generation. It also includes methods to set the model and create an attention mask, which are crucial steps in the setup process.

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
        device=None,
        wandb: bool = False,
        load_model: bool = False,
        do_lora: bool = True,
        use_loss_difference: bool = True,
        model_name: str = "distilgpt2",
        lr: float = 1e-4,
        tok_p_loss: int = 10,
        tok_p_action: int = 100,
        tok_p_obs: int = 50,
        obs_p_doc: int = 10,
        normalize_to_ctxt_size: int = False,
        impose_ctxt_size=None,
        num_beams: int = 1,
        batch_size: int = 2,
        num_batches: int = 4,
        interval_save_weights: int = 30,
        interval_print: int = None,
    ):
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        else:
            self._device = device

        # bools
        self._load_model = load_model
        self._wandb = wandb
        self._do_lora = do_lora
        self._use_loss_difference = use_loss_difference

        # strs
        self._model_name = model_name

        # ints
        self._tok_p_loss = tok_p_loss
        self._tok_p_action = tok_p_action
        self._normalize_to_ctxt_size = normalize_to_ctxt_size
        self._impose_ctxt_size = impose_ctxt_size
        if tok_p_obs is None:  # what is the point of this?
            self._tok_p_obs = self._ctxt_size - self._tok_p_action - self._tok_p_loss
        else:
            self._tok_p_obs = tok_p_obs
        self._obs_p_doc = obs_p_doc
        self._num_beams = num_beams
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._interval_save_weights = interval_save_weights
        self._path_2_model = f"saved_weights_and_losses/{self._model_name}_weights"
        self._path_2_tokenizer = (
            f"saved_weights_and_losses/{self._model_name}_tokenizer"
        )

        if interval_print is None:
            self._interval_print = 10
        else:
            self._interval_print = interval_print

        # float
        self._lr = lr

        # sets model, tokenizer and ctxt_size
        self._set_model()
        if self._impose_ctxt_size:
            self._ctxt_size = self._impose_ctxt_size
        assert self._ctxt_size > self._tok_p_loss
        if self._normalize_to_ctxt_size:
            ctxt_left = self._ctxt_size - self._tok_p_loss
            combined_len = self._tok_p_action + self._tok_p_obs
            self._tok_p_action = int(ctxt_left * (self._tok_p_action / combined_len))
            self._tok_p_obs = int(ctxt_left * (self._tok_p_obs / combined_len))
        self._tok_p_doc = self._tok_p_obs * self._obs_p_doc

        self._tok_p_rao = self._tok_p_loss + self._tok_p_action + self._tok_p_obs

    def __repr__(self):
        return (
            f"RaoConfig({self._model_name}, do_lora {self._do_lora}, "
            + f"batch_size {self._batch_size}, tok_p_action {self._tok_p_action}, "
            + f"ctxt_size {self._ctxt_size})"
        )

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
                    # model_dict[self._model_name],
                    torch_dtype=torch.float16,
                    use_flash_attention_2=self._model_name == "mistral"
                    or self._model_name == "llama",
                )
                causal_lm.bfloat16()
                causal_lm_tokenizer = AutoTokenizer.from_pretrained(
                    self._path_2_tokenizer,
                    # model_dict[self._model_name],
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
                )  # .to(self._device)
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
                )  # .to(self._device)
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
                # base_model_name_or_path=MODEL,
                r=64,
                lora_alpha=128,
                lora_dropout=0.1,
                target_modules=self.get_linear_layers(causal_lm),
            )

            causal_lm = get_peft_model(causal_lm, peft_config)

        # causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        causal_lm_tokenizer.padding_side = "left"
        causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.encode(" ")[0]

        self._model = causal_lm
        self._tokenizer = causal_lm_tokenizer

    @property
    def device(self):
        return self._device

    @property
    def load_model(self):
        return self._load_model

    @property
    def wandb(self):
        return self._wandb

    @property
    def do_lora(self):
        return self._do_lora

    @property
    def use_loss_difference(self):
        return self._use_loss_difference

    @property
    def model_name(self):
        return self._model_name

    # I do not know if immutability is an issue for a trainable model
    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def ctxt_size(self):
        return self._ctxt_size

    @property
    def tok_p_loss(self):
        return self._tok_p_loss

    @property
    def tok_p_action(self):
        return self._tok_p_action

    @property
    def normalize_to_ctxt_size(self):
        return self._normalize_to_ctxt_size

    @property
    def tok_p_obs(self):
        return self._tok_p_obs

    @property
    def tok_p_rao(self):
        return self._tok_p_rao

    @property
    def tok_p_doc(self):
        return self._tok_p_doc

    @property
    def obs_p_doc(self):
        return self._obs_p_doc

    @property
    def num_beams(self):
        return self._num_beams

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def path_2_model(self):
        return self._path_2_model

    @property
    def path_2_tokenizer(self):
        return self._path_2_tokenizer

    @property
    def interval_save_weights(self):
        return self._interval_save_weights

    @property
    def interval_print(self):
        return self._interval_print

    @property
    def lr(self):
        return self._lr

    @staticmethod
    def get_linear_layers(model):
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


def log_and_print_info(
    cfg,
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
):
    tokenizer = cfg.tokenizer
    if batch_index % cfg.interval_print == 0:
        print(f"\nBatch Number {batch_index}")
        if cfg.normalize_to_ctxt_size or not cfg.use_loss_difference:
            print("Loss: ", f"{batch_loss_action[0]:.3f}")
        else:
            print(
                "Loss (Action - Filler = Difference): ",
                f"{batch_loss_action[0]:.3f}/{batch_loss_filler[0]:.3f}/{loss_difference[0]:.3f}",
            )
        if aggregate_losses:
            print("Aggregate Loss: ", aggregate_losses[-1])
        print("Previous Obs:", repr(tokenizer.batch_decode(prev_obs)[0]))
        print("Actual Loss:", repr(tokenizer.batch_decode(actual_loss)[0]))
        print("Action: ", repr(tokenizer.batch_decode(action)[0]))
        print(
            "Predicted Obs: ",
            repr(tokenizer.batch_decode(predicted_obs)[0].encode("utf-8")),
        )
        print("True Obs:", repr(tokenizer.batch_decode(true_obs)[0]))
        if cfg.normalize_to_ctxt_size:
            print("Size of RAO tensor: ", rao_tensor.size())
            print("Size of the context window: ", cfg.ctxt_size)
        for param_group in optimizer.param_groups:
            print("Current Learning Rate: ", param_group["lr"])
        print("____________________________________________")
    with open(f"saved_weights_and_losses/{cfg.model_name}_log.txt", "a") as f:
        print(f"\nBatch Number {batch_index}", file=f)
        if cfg.normalize_to_ctxt_size or not cfg.use_loss_difference:
            print("Loss: ", f"{batch_loss_action[0]:.3f}", file=f)
        else:
            print(
                "Loss (Action - Filler = Difference): ",
                f"{batch_loss_action[0]:.3f}/{batch_loss_filler[0]:.3f}/{loss_difference[0]:.3f}",
                file=f
            )
        if aggregate_losses:
            print("Aggregate Loss: ", aggregate_losses[-1], file=f)
        print("Previous Obs:", repr(tokenizer.batch_decode(prev_obs)[0]), file=f)
        print("Actual Loss:", repr(tokenizer.batch_decode(actual_loss)[0]), file=f)
        print("Action: ", repr(tokenizer.batch_decode(action)[0]), file=f)
        print(
            "Predicted Obs: ",
            repr(tokenizer.batch_decode(predicted_obs)[0].encode("utf-8")),
            file=f,
        )
        print("True Obs:", repr(tokenizer.batch_decode(true_obs)[0]), file=f)
        if cfg.normalize_to_ctxt_size:
            print("Size of RAO tensor: ", rao_tensor.size(), file=f)
            print("Size of the context window: ", cfg.ctxt_size, file=f)
        for param_group in optimizer.param_groups:
            print("Current Learning Rate: ", param_group["lr"], file=f)
        print("", file=f)
        if cfg.wandb:
            wandb.log(
                {
                    "Batch Index": batch_index,
                    "Batch Loss": batch_loss_action[0],
                    "Filler Loss": batch_loss_filler[0],
                    "Loss Difference": loss_difference[0],
                }
            )


def main():
    # define each class in default to check if they work
    test0 = RaoConfig()
    print(test0)


if __name__ == "__main__":
    main()
