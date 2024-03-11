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
        # self.q_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        self.q_head = nn.Linear(
            self.transformer.lm_head.weight.shape[1],
            self.transformer.lm_head.weight.shape[0],
            bias=True,
        )
        torch.nn.init.zeros_(self.q_head.weight)
        self.transformer.bfloat16()
        self.q_head.bfloat16()
        # self.post_init()

    def forward(self, input_ids=None, attention_mask=None, add_q_head=True, **kwargs):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=add_q_head,
            **{k: v for k, v in kwargs.items() if k != "output_hidden_states"},
        )
        if add_q_head:
            hidden_states = outputs.hidden_states[-1]
            q_values = self.q_head(hidden_states)
            outputs.logits += q_values
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
        observation_prefix = "\nObservation: "
        observation_prefix_tokens = tokenizer.encode(
            observation_prefix, add_special_tokens=False
        )
        observation_prefix_tensor = einops.repeat(
            torch.tensor(observation_prefix_tokens),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        tokens_per_pure_observation = tok_p_obs - len(observation_prefix_tokens)
    else:
        observation_prefix_tensor = None
        tokens_per_pure_observation = None

    if tok_p_action:
        action_prefix = "\nStepByStep: "
        action_prefix_tokens = tokenizer.encode(action_prefix, add_special_tokens=False)
        action_prefix_tensor = einops.repeat(
            torch.tensor(action_prefix_tokens),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        tokens_per_pure_action = tok_p_action - len(action_prefix_tokens)
    else:
        action_prefix_tensor = None
        tokens_per_pure_action = None

    if tok_p_loss:
        reward_prefix = "\nLoss: "
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
        config = AutoConfig.from_pretrained(model_dict[model_name])
        if load_model:
            causal_lm = ModelWithQHead(model_name, config)
            # AutoModelForCausalLM.from_pretrained(
            #    path_2_model,
            #    torch_dtype=torch.float16,
            # )
        else:
            causal_lm = ModelWithQHead(model_dict[model_name], config)
            # AutoModelForCausalLM.from_pretrained(
            #    model_dict[model_name],
            #    use_flash_attention_2=model_name in ["llama"],
            # )
        if not do_lora:
            for name, param in causal_lm.named_parameters():
                param.requires_grad = "q_head" in name

        causal_lm.bfloat16()
        causal_lm_tokenizer = AutoTokenizer.from_pretrained(
            model_dict[model_name], padding_side=padding_side
        )
        #  causal_lm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        ## causal_lm.config.pad_token_id = causal_lm_tokenizer.pad_token_id
        # causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.vocab_size
        causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        # causal_lm.resize_token_embeddings(len(causal_lm_tokenizer))
        # causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        ctxt_size = get_ctxt_size(model_name, causal_lm)

        # causal_lm.v_head = torch.nn.Linear(
        #    causal_lm.lm_head.weight.shape[-1], 1, bias=False
        # )
        ## Initialize weights of v_head
        # torch.nn.init.xavier_uniform_(causal_lm.v_head.weight)
        # causal_lm.q_head = causal_lm.lm_head

    if do_lora:
        # linear_layers = get_linear_layers(causal_lm)
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            # target_modules=linear_layers
        )
        # print("Num Linear Layers: ", len(linear_layers))
        causal_lm.transformer = get_peft_model(causal_lm.transformer, peft_config)
        causal_lm.transformer.print_trainable_parameters()
        for name, param in causal_lm.named_parameters():
            if "q_head" in name:
                param.requires_grad = True

    causal_lm.tokenizer = causal_lm_tokenizer
    return causal_lm, causal_lm_tokenizer, ctxt_size


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
        if cfg.dataset.task.cumulative:
            run_name += "cm_"
    else:
        run_name += "wiki_"
    if cfg.lr != 1e-4:
        run_name += f"lr{cfg.lr}_"
    if prediction_cfg.train_O_given_prev_O:
        run_name += f"AR_"
    if prediction_cfg.train_O_given_A:
        run_name += f"M_"
    if prediction_cfg.train_A_given_AO:
        run_name += f"EI_"
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


def predict_action(cfg, prev_action, prev_obs, action, per_batch=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = torch.cat([prev_action, prev_obs, action], dim=1)
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    # with torch.no_grad() if cfg.training_predictor_mode else nullcontext():
    prediction = cfg.causal_lm(
        input_sequence,
        attention_mask=attention_mask,
        add_q_head=True,
    )
    # action_bias = cfg.causal_lm.q_head(prediction.hidden_states[-1])[
    #    :, -cfg.tok_p_action :, 0
    # ]
    # assert action_bias.shape == prediction.logits.shape
    # action_logits = (prediction.logits + action_bias)[:, :-1, :]
    action_logits = prediction.logits[:, :-1, :].log_softmax(dim=-1)
    # batch_indices, sequence_indices = torch.meshgrid(
    #    torch.arange(action_logits.size(0), device=action_logits.device),
    #    torch.arange(action_logits.size(1), device=action_logits.device),
    #    indexing="ij",
    # )
    # q_values = action_logits[batch_indices, sequence_indices, input_sequence[:, 1:]]
    q_values = torch.gather(
        action_logits, 2, input_sequence[:, 1:].unsqueeze(-1)
    ).squeeze(-1)[:, : -cfg.tok_p_pure_action]
    action_loss_tensor = loss_fn(
        input=einops.rearrange(
            action_logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )
    if per_batch:
        action_loss = (action_loss_tensor * attention_mask[:, 1:]).sum(
            dim=1
        ) / attention_mask[:, 1:].sum(dim=1)
    else:
        action_loss = (
            action_loss_tensor * attention_mask[:, 1:]
        ).sum() / attention_mask[:, 1:].sum()
    return action_loss, q_values


def predict_observation(cfg, action, obs, per_batch=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    mkv_input_sequence = torch.cat([action, obs], dim=1)
    mkv_attention_mask = (
        mkv_input_sequence != cfg.causal_lm_tokenizer.pad_token_id
    ).long()
    # with nullcontext() if cfg.training_predictor_mode else torch.no_grad():
    prediction = cfg.causal_lm(
        mkv_input_sequence, attention_mask=mkv_attention_mask, add_q_head=False
    )
    mkv_logits = prediction.logits[:, :-1, :]
    mkv_loss_tensor = loss_fn(
        input=einops.rearrange(
            mkv_logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=mkv_input_sequence[:, 1:],
    )
    if per_batch:
        obs_loss = (mkv_loss_tensor * mkv_attention_mask[:, 1:]).sum(
            dim=1
        ) / mkv_attention_mask[:, 1:].sum(dim=1)
    else:
        obs_loss = (
            mkv_loss_tensor * mkv_attention_mask[:, 1:]
        ).sum() / mkv_attention_mask[:, 1:].sum()
    return obs_loss

    # obs_tensor = (mkv_loss_tensor * mkv_attention_mask[:, 1:])[:, -cfg.tok_p_pure_obs :]
    # obs_losses = obs_tensor.sum(dim=-1) / mkv_attention_mask[:, 1:][
    #    :, -cfg.tok_p_pure_obs :
    # ].sum(dim=-1)
    # return obs_losses, obs_tensor
