import torch
import torchtyping
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from dataclasses import dataclass
import einops
from datasets import load_dataset
import json
import copy
from contextlib import nullcontext

import torch.nn as nn

from src.training_types import *
from src.prepare_dataset import prepare_dataset

from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationMixin,
    GPT2LMHeadModel,
)
import torch
import torch.nn as nn


class ModelWithQHead(PreTrainedModel, GenerationMixin):
    def __init__(self, model_name_or_path, config):
        super().__init__(config)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=config
        )
        self.qhead = copy.deepcopy(self.transformer)

        mlp_modules = get_mlp_modules(self.qhead)
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=mlp_modules,
        )
        ## print("Num Linear Layers: ", len(linear_layers))
        self.qhead = get_peft_model(self.qhead, peft_config)
        self.qhead.print_trainable_parameters()
        ## Grouping q_head and q_head_block together for easier parameter management
        last_layer = (
            self.transformer.transformer.h[-1]
            if "gpt2" in model_name_or_path
            else self.transformer.model.layers[-1]
        )
        self.v_head_group = nn.ModuleDict(
            {
                "v_head_block": copy.deepcopy(last_layer),
                "v_head": nn.Linear(
                    self.transformer.lm_head.weight.shape[1], 1, bias=True
                ),
            }
        )

        ## Zero-initialize weights in q_head_block and q_head
        # for name, param in self.q_head_group["q_head_block"].named_parameters():
        #    if "weight" in name:
        #        torch.nn.init.zeros_(param)
        # torch.nn.init.zeros_(self.q_head_group["q_head"].weight)

        # To set q_head and q_head_block parameters to require_grad=True, use:
        # for param in self.q_head_group.parameters():
        #     param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        add_q_head=True,
        get_v_head=False,
        **kwargs,
    ):
        # if add_q_head:
        #    self.transformer.enable_adapter_layers()
        # else:
        #    self.transformer.disable_adapter_layers()
        outputs = (self.qhead if add_q_head else self.transformer)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=get_v_head,
            # **kwargs,
            **{k: v for k, v in kwargs.items() if k != "output_hidden_states"},
        )
        if get_v_head:
            pre_values = self.v_head_group["v_head_block"](outputs.hidden_states[-1])[0]
            values = self.v_head_group["v_head"](pre_values).squeeze(-1)
            return outputs, values
        # if add_q_head:
        #    hidden_states = outputs.hidden_states[-1]
        #    pre_q_values = self.q_head_group["q_head_block"](hidden_states)[0]
        #    q_values = self.q_head_group["q_head"](pre_q_values)
        #    outputs.logits += q_values
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.transformer.prepare_inputs_for_generation(input_ids, **kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.transformer._reorder_cache(past_key_values, beam_idx)


def load_cfg_from_file(file_location: str) -> InitialConfig:
    with open(file_location) as f:
        cfg_dict = json.load(f)["parameters"]
    cfg_dict["training_type"] = eval(cfg_dict["training_type"][0])(
        **cfg_dict["training_type"][1]
    )
    cfg_dict["dataset"] = eval(cfg_dict["dataset"][0])(**cfg_dict["dataset"][1])
    return InitialConfig(**cfg_dict)


def extend_initial_config(init_cfg: InitialConfig) -> Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_2_log = f"saved_weights_and_losses/{init_cfg.model_name}_log.txt"
    traj_path = f"saved_weights_and_losses/{init_cfg.model_name}_traj"
    path_2_model = f"saved_weights_and_losses/{init_cfg.model_name}_weights"
    path_2_tokenizer = f"saved_weights_and_losses/{init_cfg.model_name}_tokenizer"

    causal_lm, causal_lm_tokenizer, ctxt_size = get_model(
        device,
        init_cfg.load_model,
        init_cfg.model_name,
        path_2_tokenizer,
        path_2_model,
        init_cfg.do_lora,
    )

    # causal_lm.q_head = torch.nn.Linear(
    #    causal_lm.lm_head.weight.shape[-1],
    #    causal_lm.lm_head.weight.shape[0],
    #    bias=False,
    #    device=device,
    # )
    # torch.nn.init.zeros_(causal_lm.q_head.weight)

    # predictor_lm = copy.deepcopy(inference_lm)
    # for param in predictor_lm.parameters():
    #    param.requires_grad = False

    training_ctxt_size = (
        ctxt_size
        if init_cfg.training_ctxt_size is None
        else init_cfg.training_ctxt_size
    )
    tok_p_action = int(training_ctxt_size / (init_cfg.obs_to_action_ratio + 2))
    tok_p_obs = int(tok_p_action * init_cfg.obs_to_action_ratio)
    tok_p_loss = None

    tok_per_pure, prefix_tensors = get_prefixes(
        causal_lm_tokenizer,
        init_cfg.batch_size,
        device,
        tok_p_loss,
        tok_p_action,
        tok_p_obs,
    )
    tok_p_pure_loss, tok_p_pure_action, tok_p_pure_obs = tok_per_pure
    loss_prefix, action_prefix, obs_prefix = prefix_tensors

    dataset = prepare_dataset(
        init_cfg,
        causal_lm_tokenizer,
        device,
        action_prefix,
        obs_prefix,
        tok_p_pure_action,
        tok_p_pure_obs,
        action_prefix,
        obs_prefix,
    )

    return Config(
        model_name=init_cfg.model_name,
        lr=init_cfg.lr,
        optimizer=init_cfg.optimizer,
        qhead_optimizer=init_cfg.optimizer,
        batch_size=init_cfg.batch_size,
        num_batches=init_cfg.num_batches,
        obs_to_action_ratio=init_cfg.obs_to_action_ratio,
        interval_save_weights=init_cfg.interval_save_weights,
        interval_print=init_cfg.interval_print,
        wandb=init_cfg.wandb,
        load_model=init_cfg.load_model,
        do_lora=init_cfg.do_lora,
        num_beams=init_cfg.num_beams,
        training_ctxt_size=init_cfg.training_ctxt_size,
        device=device,
        dataset=DatasetType(
            task=init_cfg.dataset.task,
            peek_every=init_cfg.dataset.peek_every,
            dataloader=dataset,
        ),
        path_2_log=path_2_log,
        traj_path=traj_path,
        path_2_model=path_2_model,
        path_2_tokenizer=path_2_tokenizer,
        tok_p_action=tok_p_action,
        tok_p_obs=tok_p_obs,
        tok_p_pure_action=tok_p_pure_action,
        tok_p_pure_obs=tok_p_pure_obs,
        action_prefix_tensor=action_prefix,
        obs_prefix_tensor=obs_prefix,
        ctxt_size=ctxt_size,
        causal_lm=causal_lm,  # predictor_lm,
        causal_lm_tokenizer=causal_lm_tokenizer,
        inference_cfg=init_cfg.inference_cfg,
        prediction_cfg=init_cfg.prediction_cfg,
        trainer_cfg=init_cfg.trainer_cfg,
        training_predictor_mode=True,
        perturbation_cfg=init_cfg.perturbation_cfg,
        debug=init_cfg.debug,
    )


def multi_print(string, f):
    print(string)
    print(string, file=f)


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
    if batch_index % cfg.interval_print == 0:
        with open(cfg.path_2_log, "a") as f:
            multi_print(f"\nBatch Number {batch_index}", f)
            multi_print(f"Loss: {batch_loss[0][0]:.3f}", f)
            if aggregate_losses:
                multi_print(f"Aggregate Loss: {aggregate_losses[-1]}", f)
            multi_print(
                f"Previous Obs: {repr(cfg.causal_lm_tokenizer.batch_decode(prev_obs)[0])}",
                f,
            )
            multi_print(
                f"StepByStep: {repr(cfg.causal_lm_tokenizer.batch_decode(action)[0])}",
                f,
            )
            multi_print(
                f"Predicted Obs: {repr(cfg.causal_lm_tokenizer.batch_decode(predicted_obs)[0])}",
                f,
            )
            multi_print(
                f"True Obs: {repr(cfg.causal_lm_tokenizer.batch_decode(true_obs)[0])}",
                f,
            )
            multi_print("___________________________________________", f)


def get_prefixes(
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    device: torch.device,
    tok_p_loss: Optional[int],
    tok_p_action: Optional[int],
    tok_p_obs: Optional[int],
):
    if tok_p_obs:
        observation_prefix = ""
        observation_prefix_tokens = tokenizer.encode(
            observation_prefix, add_special_tokens=False
        )
        observation_prefix_tensor = einops.repeat(
            torch.tensor(observation_prefix_tokens, dtype=torch.int64),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        tokens_per_pure_observation = tok_p_obs - len(observation_prefix_tokens)
    else:
        observation_prefix_tensor = None
        tokens_per_pure_observation = None

    if tok_p_action:
        # action_prefix = "\nStepByStep:"
        action_prefix = "Reasoning:"
        action_prefix_tokens = tokenizer.encode(
            action_prefix, add_special_tokens=False)
        action_prefix_tensor = einops.repeat(
            torch.tensor(action_prefix_tokens, dtype=torch.int64),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        tokens_per_pure_action = tok_p_action - len(action_prefix_tokens)
    else:
        action_prefix_tensor = None
        tokens_per_pure_action = None

    if tok_p_loss:
        reward_prefix = "\nLoss:"
        reward_prefix_tokens = tokenizer.encode(reward_prefix, add_special_tokens=False)
        reward_prefix_tensor = einops.repeat(
            torch.tensor(reward_prefix_tokens),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        tokens_per_pure_reward = tok_p_loss - len(reward_prefix_tokens)
    else:
        reward_prefix_tensor = None
        tokens_per_pure_reward = None

    return (
        (tokens_per_pure_reward, tokens_per_pure_action, tokens_per_pure_observation),
        (reward_prefix_tensor, action_prefix_tensor, observation_prefix_tensor),
    )


def get_ctxt_size(model_name, causal_lm):
    if model_name == "mistral":
        ctxt_size = causal_lm.config.sliding_window
    elif model_name == "tinystories":
        ctxt_size = causal_lm.config.window_size
    elif model_name == "llama":
        ctxt_size = causal_lm.config.max_position_embeddings
    elif model_name == "distilgpt2":
        ctxt_size = causal_lm.config.n_ctx
    elif model_name == "gptj":
        ctxt_size = causal_lm.config.n_positions
    elif model_name == "gpt2-large":
        ctxt_size = causal_lm.config.n_positions
    elif model_name == "phi2":
        ctxt_size = causal_lm.config.max_position_embeddings
    else:
        ctxt_size = causal_lm.config.n_positions


def get_padding_side(model_name):
    if model_name in ["llama"]:
        return "right"
    return "left"


def get_model(
    device, load_model, model_name, path_2_tokenizer, path_2_model, do_lora=None
):
    """Load model"""
    model_dict = {
        "tinystories": "roneneldan/TinyStories-1M",
        "llama": "meta-llama/Llama-2-13b-hf",
        "distilgpt2": "distilgpt2",
        "gptj": "EleutherAI/gpt-j-6b",
        "mistral": "mistralai/Mistral-7B-v0.1",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl",
        "phi2": "microsoft/phi-2",
        # Add other models here
    }
    with device:
        padding_side = get_padding_side(model_name)
        if load_model:
            model_location = "./saved_weights_and_losses/" + model_name + "_weights"
            config = AutoConfig.from_pretrained(model_location)
            causal_lm = ModelWithQHead(model_location, config)
        else:
            config = AutoConfig.from_pretrained(model_dict[model_name])
            causal_lm = ModelWithQHead(model_dict[model_name], config)

        causal_lm.bfloat16()
        causal_lm_tokenizer = AutoTokenizer.from_pretrained(
            model_dict[model_name], padding_side=padding_side
        )
        causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        ctxt_size = get_ctxt_size(model_name, causal_lm)
        for name, param in causal_lm.transformer.named_parameters():
            param.requires_grad = False
        for name, param in causal_lm.qhead.named_parameters():
            param.requires_grad = True
        for name, param in causal_lm.v_head_group.named_parameters():
            param.requires_grad = True

    causal_lm.tokenizer = causal_lm_tokenizer
    return causal_lm, causal_lm_tokenizer, ctxt_size


def get_mlp_modules(model):
    modules = []
    for x in model.named_parameters():
        parts = x[0].split(".")
        if "mlp" in parts:
            modules.append(".".join(parts[: parts.index("mlp") + 2]))
    # return [x[0].split(".") for x in model.named_parameters() if "mlp" in x[0]]
    return list(set(modules))


def create_run_name(cfg: Config) -> str:
    prediction_cfg = cfg.prediction_cfg
    inference_cfg = cfg.inference_cfg
    run_name = ""
    run_name += f"{cfg.model_name[:4]}_"
    run_name += f"{cfg.optimizer[:3]}_"
    run_name += f"pu{cfg.trainer_cfg.prediction_training_length}_gu{cfg.trainer_cfg.inference_training_length}_"
    if isinstance(cfg.dataset.task, ArithmeticTask):
        run_name += (
            f"ari_nt={cfg.dataset.task.num_terms}_nd={cfg.dataset.task.num_digits}_"
        )
    else:
        run_name += "wiki_"
    if cfg.lr != 1e-4:
        run_name += f"lr{cfg.lr}_"
    if prediction_cfg.train_O_given_prev_O:
        run_name += f"AR_"
    if prediction_cfg.train_O_given_A:
        run_name += f"M_"
    if cfg.dataset.peek_every is not None:
        run_name += f"pe{cfg.dataset.peek_every}_"

    if isinstance(cfg.debug, RepeatNPoints):
        run_name += f"r{cfg.debug.num_points}_"
    elif isinstance(cfg.debug, RepeatPointNTimes):
        run_name += f"re{cfg.debug.num_times}_"
    elif isinstance(cfg.debug, ReplaceWithRandomTokens):
        run_name += "rd_"
    elif isinstance(cfg.debug, NoWeightUpdates):
        run_name += "nwu_"

    if cfg.batch_size != 1:
        run_name += f"bs{cfg.batch_size}_"
    run_name += f"nb{cfg.num_batches}_"
    if cfg.obs_to_action_ratio != 1:
        run_name += f"o:a={cfg.obs_to_action_ratio}:1_"
    if cfg.load_model:
        run_name += f"load_"
    if cfg.do_lora:
        run_name += "lra_"
    if cfg.num_beams:
        run_name += f"nbe{cfg.num_beams}_"
    if cfg.training_ctxt_size:
        run_name += f"ics{cfg.training_ctxt_size}_"
    return run_name


def entropy_from_logits(logits):
    # Apply softmax to get the probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Compute the entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def predict_action(cfg, prev_action, prev_obs, action, add_q_head, per_batch=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = torch.cat([prev_action, prev_obs, action], dim=1)
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    prediction, values = cfg.causal_lm(
        input_sequence,
        attention_mask=attention_mask,
        add_q_head=add_q_head,
        get_v_head=True,
    )
    action_logits = prediction.logits[:, :-1, :].log_softmax(dim=-1)
    action_loss_tensor = loss_fn(
        input=einops.rearrange(
            action_logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )[:, -cfg.tok_p_pure_action :]
    negentropy = -entropy_from_logits(action_logits).mean()
    pure_action_attention_mask = attention_mask[:, -cfg.tok_p_pure_action :]
    if per_batch:
        action_loss = (action_loss_tensor * pure_action_attention_mask).sum(
            dim=1
        ) / pure_action_attention_mask.sum(dim=1)
    else:
        action_loss = (
            action_loss_tensor * pure_action_attention_mask
        ).sum() / pure_action_attention_mask.sum()
    return action_loss, values, negentropy


def predict_observation(cfg, action, obs, add_q_head, per_batch=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    mkv_input_sequence = torch.cat([action, obs], dim=1)
    mkv_attention_mask = (
        mkv_input_sequence != cfg.causal_lm_tokenizer.pad_token_id
    ).long()
    prediction = cfg.causal_lm(
        mkv_input_sequence,
        attention_mask=mkv_attention_mask,
        add_q_head=add_q_head,
        get_v_head=False,
    )
    mkv_logits = prediction.logits[:, :-1, :].log_softmax(dim=-1)
    mkv_loss_tensor = loss_fn(
        input=einops.rearrange(
            mkv_logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=mkv_input_sequence[:, 1:],
    )[:, -cfg.tok_p_pure_obs :]
    pure_obs_attention_mask = mkv_attention_mask[:, -cfg.tok_p_pure_obs :]
    if per_batch:
        obs_loss = (mkv_loss_tensor * pure_obs_attention_mask).sum(
            dim=1
        ) / pure_obs_attention_mask.sum(dim=1)
    else:
        obs_loss = (
            mkv_loss_tensor * pure_obs_attention_mask
        ).sum() / pure_obs_attention_mask.sum()
    return obs_loss

    # obs_tensor = (mkv_loss_tensor * mkv_attention_mask[:, 1:])[:, -cfg.tok_p_pure_obs :]
    # obs_losses = obs_tensor.sum(dim=-1) / mkv_attention_mask[:, 1:][
    #    :, -cfg.tok_p_pure_obs :
    # ].sum(dim=-1)
    # return obs_losses, obs_tensor


def get_neg_log_probs(cfg, input_sequence):
    """
    Computes the loss tensor for a given input sequence.

    Args:
        cfg: Configuration object containing model and tokenizer information.
        input_sequence: The input sequence tensor for which the loss is to be computed.

    Returns:
        The computed loss tensor.
    """
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    logits = cfg.predictor_lm(input_sequence, attention_mask=attention_mask).logits[
        :, :-1, :
    ]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_tensor = loss_fn(
        input=einops.rearrange(
            logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )
    return loss_tensor


def get_masked_mean(arr, mask):
    return (arr * mask).sum() / mask.sum()
