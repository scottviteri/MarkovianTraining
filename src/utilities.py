import torch
import torchtyping
import torch.distributed as dist
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

from training_types import *
from prepare_dataset import prepare_dataset

import transformers
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationMixin,
    GPT2LMHeadModel,
)
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


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
        assert (
            self.qhead.base_model.model.model.layers[-3].mlp.up_proj.base_layer.weight
            == self.transformer.model.layers[-3].mlp.up_proj.weight
        ).all(), "Should be same weights"

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
        model = self.qhead if add_q_head else self.transformer
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=get_v_head,
            **{k: v for k, v in kwargs.items() if k != "output_hidden_states"},
        )
        if get_v_head:
            # pre_values = self.v_head_group["v_head_block"](
            #    outputs.hidden_states[-1].detach()
            # )[0]
            # values = self.v_head_group["v_head"](pre_values).squeeze(-1)
            hidden_states = outputs.hidden_states[-1].detach()
            values = self.v_head_group["v_head"](hidden_states).squeeze(-1)
            return outputs, values
        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        generation_inputs = self.transformer.prepare_inputs_for_generation(
            input_ids, **kwargs
        )
        generation_inputs.update(
            {"add_q_head": kwargs["add_q_head"], "get_v_head": kwargs["get_v_head"]}
        )
        return generation_inputs

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
    if init_cfg.use_mac:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_2_log = f"saved_weights_and_losses/{init_cfg.model_name}_log.txt"
    traj_path = f"saved_weights_and_losses/{init_cfg.model_name}_traj"
    path_2_model = f"saved_weights_and_losses/{init_cfg.model_name}_weights"
    path_2_tokenizer = f"saved_weights_and_losses/{init_cfg.model_name}_tokenizer"

    assert init_cfg.num_beams == 1, "Only supporting num_beams = 1 currently"

    causal_lm, causal_lm_tokenizer = get_model(
        device,
        init_cfg.load_model,
        init_cfg.model_name,
        path_2_tokenizer,
        path_2_model,
        init_cfg.inference_cfg.num_return_sequences,
        init_cfg.do_lora,
        use_mac=init_cfg.use_mac,
    )

    pure_ctxt_sizes, prefix_tensors = get_prefixes(
        causal_lm_tokenizer, init_cfg.batch_size, device, init_cfg.ctxt_sizes
    )
    causal_lm.generation_config.min_new_tokens = pure_ctxt_sizes.action_size
    causal_lm.generation_config.max_new_tokens = pure_ctxt_sizes.action_size

    dataset = prepare_dataset(
        init_cfg,
        causal_lm_tokenizer,
        device,
        pure_ctxt_sizes,
        prefix_tensors,
    )

    return Config(
        model_name=init_cfg.model_name,
        lr=init_cfg.lr,
        optimizer=init_cfg.optimizer,
        qhead_optimizer=init_cfg.optimizer,
        batch_size=init_cfg.batch_size,
        num_batches=init_cfg.num_batches,
        replay_buffer_size=init_cfg.replay_buffer_size,
        obs_to_action_ratio=init_cfg.obs_to_action_ratio,
        interval_save_weights=init_cfg.interval_save_weights,
        interval_print=init_cfg.interval_print,
        use_mac=init_cfg.use_mac,
        wandb=init_cfg.wandb,
        load_model=init_cfg.load_model,
        do_lora=init_cfg.do_lora,
        num_beams=init_cfg.num_beams,
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
        pure_ctxt_sizes=pure_ctxt_sizes,
        ctxt_sizes=init_cfg.ctxt_sizes,
        prefix_tensors=prefix_tensors,
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
    ctxt_sizes: ContextSizes,
):
    def tokenize_and_repeat(prefix_string):
        prefix_tokens = tokenizer.encode(prefix_string, add_special_tokens=False)
        repeated_prefix_tokens = einops.repeat(
            torch.tensor(prefix_tokens, dtype=torch.int64),
            "tokens -> batch tokens",
            batch=batch_size,
        ).to(device)
        return repeated_prefix_tokens

    first_action_prefix_tensor = tokenize_and_repeat("")
    tokens_per_pure_first_action = (
        ctxt_sizes.first_action_size - first_action_prefix_tensor.shape[1]
    )

    first_obs_prefix_tensor = tokenize_and_repeat("Question:")
    tokens_per_pure_first_obs = (
        ctxt_sizes.first_obs_size - first_obs_prefix_tensor.shape[1]
    )

    action_prefix_tensor = tokenize_and_repeat("Reasoning:")
    tokens_per_pure_action = ctxt_sizes.action_size - action_prefix_tensor.shape[1]

    obs_prefix_tensor = tokenize_and_repeat("Answer:")
    tokens_per_pure_obs = ctxt_sizes.obs_size - obs_prefix_tensor.shape[1]

    return (
        ContextSizes(
            first_action_size=tokens_per_pure_first_action,
            first_obs_size=tokens_per_pure_first_obs,
            action_size=tokens_per_pure_action,
            obs_size=tokens_per_pure_obs,
        ),
        PrefixTensors(
            first_action_prefix_tensor=first_action_prefix_tensor,
            first_obs_prefix_tensor=first_obs_prefix_tensor,
            action_prefix_tensor=action_prefix_tensor,
            obs_prefix_tensor=obs_prefix_tensor,
        ),
    )


def get_ctxt_window_size(model_name, causal_lm):
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
    device,
    load_model,
    model_name,
    path_2_tokenizer,
    path_2_model,
    num_return_sequences,
    do_lora=None,
    use_mac=False,
):
    """Load model"""
    model_dict = {
        "tinystories": "roneneldan/TinyStories-1M",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "distilgpt2": "distilgpt2",
        "gptj": "EleutherAI/gpt-j-6b",
        # "mistral": "mistralai/Mistral-7B-v0.1",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
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

        if not use_mac:
            causal_lm.bfloat16()
        causal_lm_tokenizer = AutoTokenizer.from_pretrained(
            model_dict[model_name], padding_side=padding_side
        )
        causal_lm_tokenizer.pad_token_id = causal_lm_tokenizer.eos_token_id
        for name, param in causal_lm.transformer.named_parameters():
            param.requires_grad = False
        for name, param in causal_lm.qhead.named_parameters():
            param.requires_grad = ".lora" in name
        for name, param in causal_lm.v_head_group.named_parameters():
            param.requires_grad = True

    causal_lm.tokenizer = causal_lm_tokenizer
    bad_words_ids = [
        [
            causal_lm_tokenizer.bos_token_id,
            causal_lm_tokenizer.eos_token_id,
        ]
    ]
    logits_warper = transformers.generation.LogitsProcessorList(
        [
            transformers.generation.NoBadWordsLogitsProcessor(
                bad_words_ids,
                eos_token_id=causal_lm_tokenizer.eos_token_id,
            ),
            # transformers.generation.MinNewTokensLengthLogitsProcessor(
            #    prompt_length_to_skip=input_ids.shape[-1],
            #    min_new_tokens=cfg.pure_ctxt_sizes.action_size,
            #    eos_token_id=causal_lm_tokenizer.eos_token_id,
            # ),
            transformers.generation.TemperatureLogitsWarper(1.0),
            transformers.generation.InfNanRemoveLogitsProcessor(),
            transformers.LogitNormalization(),
        ]
    )
    # stopping_criteria = transformers.generation.StoppingCriteriaList(
    #    [
    #        transformers.generation.MaxLengthCriteria(
    #            max_length=input_ids.shape[-1] + cfg.pure_ctxt_sizes.action_size
    #        )
    #    ]
    # )
    generation_config = transformers.GenerationConfig(
        do_sample=True,
        logits_warper=logits_warper,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        pad_token_id=causal_lm_tokenizer.pad_token_id,
        eos_token_id=causal_lm_tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )
    causal_lm.generation_config = generation_config
    return causal_lm, causal_lm_tokenizer


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
    run_name += f"cs{cfg.ctxt_sizes}"
    return run_name


def entropy_from_logits(logits):
    # Apply softmax to get the probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Compute the entropy
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def wrap_input_tokens(
    cfg,
    instruct,
    rest,
    use_start_token=False,
    use_instruct_tokens=False,
    is_prediction=False,
):
    start_tokens = torch.full(
        (cfg.batch_size, 1),
        cfg.causal_lm_tokenizer.bos_token_id,
        dtype=torch.int64,
        device=cfg.device,
    )
    # needs_start_token = cfg.model_name in ["mistral", "llama"]
    # needs_instruct_token = "Instruct" in cfg.causal_lm.name_or_path
    begin_instruct_tokens = (
        cfg.causal_lm_tokenizer.encode(
            (
                "[INST] Use the following reasoning to help predict the answer."
                if is_prediction
                else "[INST]"
            ),
            return_tensors="pt",
            add_special_tokens=False,
        )
        .to(cfg.device)
        .repeat((cfg.batch_size, 1))
    )
    end_instruct_tokens = (
        cfg.causal_lm_tokenizer.encode(
            "[/INST]", return_tensors="pt", add_special_tokens=False
        )
        .to(cfg.device)
        .repeat((cfg.batch_size, 1))
    )
    token_parts = []
    if use_start_token:
        token_parts.append(start_tokens)
    if use_instruct_tokens:
        token_parts.append(begin_instruct_tokens)
    token_parts.extend(instruct)
    if use_instruct_tokens:
        token_parts.append(end_instruct_tokens)
    token_parts.extend(rest)
    final_input_sequence = torch.cat(token_parts, dim=1)
    return final_input_sequence


def predict_action(cfg, prev_action, prev_obs, action, add_q_head, per_batch=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = wrap_input_tokens(
        cfg,
        [prev_action, prev_obs],
        [action],
        use_start_token=True,
        use_instruct_tokens=True,
        is_prediction=False,
    )
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
    )[:, -cfg.pure_ctxt_sizes.action_size :]
    negentropy = -entropy_from_logits(action_logits).mean()
    pure_action_attention_mask = attention_mask[:, -cfg.pure_ctxt_sizes.action_size :]
    masked_losses = action_loss_tensor * pure_action_attention_mask
    # if True:
    #    plt.figure()
    #    plt.plot(masked_losses[0].tolist())
    #    plt.savefig("action.png")
    #    # print(
    #    #    [
    #    #        cfg.causal_lm_tokenizer.decode([x])
    #    #        for x in input_sequence[0, -cfg.pure_ctxt_sizes.action_size :].tolist()
    #    #    ]
    #    # )
    #    plt.clf()
    if per_batch:
        action_loss = masked_losses.sum(dim=1) / pure_action_attention_mask.sum(dim=1)
    else:
        action_loss = masked_losses.sum() / pure_action_attention_mask.sum()
    return action_loss, values, negentropy


def test_sequence(predictor, input_sequence):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    prediction = predictor(
        input_sequence,
        # add_q_head=False,
        # get_v_head=False,
    )
    logits = prediction.logits[:, :-1, :].log_softmax(dim=-1)
    loss_tensor = loss_fn(
        input=einops.rearrange(
            logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )
    return loss_tensor


def test_string(predictor, tokenizer, s, add_bos=False):
    encoding = tokenizer.encode(s, add_special_tokens=add_bos)
    neg_log_probs = test_sequence(
        predictor,
        torch.tensor(
            [encoding],
            device=predictor.device,
        ),
    )[0].tolist()
    rounded_nll = list(map(lambda x: round(x, 3), neg_log_probs))
    # print(encoding)
    # print(rounded_nll)
    return encoding, rounded_nll


def inspect_string(predictor, tokenizer, s, add_bos=False):
    encoding, losses = test_string(predictor, tokenizer, s, add_bos=add_bos)
    c = list(zip(encoding[1:], losses))
    return [(tokenizer.decode([x]), y) for x, y in c]


def translate_single_tokens(tokenizer, string):
    encoded = tokenizer.encode(string, add_special_tokens=False)
    return [tokenizer.decode([x]) for x in encoded]


# test_string(cfg, "Answer: 123")
# tensor([[12.2443,  0.0759,  3.6301,  0.9889,  2.1633,  3.1531]],
#       device='cuda:0')
# translate_single_tokens(cfg, "Answer: 123")
# ['Answer', ':', '', '1', '2', '3']


def predict_observation(
    cfg, action, obs, add_q_head, per_batch=False, is_default_action=False
):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = wrap_input_tokens(
        cfg,
        [action],
        [obs],
        use_start_token=True,
        use_instruct_tokens=True,
        is_prediction=True,
    )
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    prediction = cfg.causal_lm(
        input_sequence,
        attention_mask=attention_mask,
        add_q_head=add_q_head,
        get_v_head=False,
    )
    logits = prediction.logits[:, :-1, :].log_softmax(dim=-1)
    loss_tensor = loss_fn(
        input=einops.rearrange(
            logits,
            "batch seq_length vocab_size -> batch vocab_size seq_length",
        ),
        target=input_sequence[:, 1:],
    )[:, -cfg.pure_ctxt_sizes.obs_size :]
    pure_obs_attention_mask = attention_mask[:, -cfg.pure_ctxt_sizes.obs_size :]
    masked_losses = loss_tensor * pure_obs_attention_mask
    # plt.figure()
    # plt.plot(masked_losses[0, 1:].tolist())  # also sliced to remove space
    # if is_default_action:
    #    plt.savefig("obs_default.png")
    # else:
    #    plt.savefig("obs.png")
    token_loss_pairs = inspect_string(
        cfg.causal_lm,
        cfg.causal_lm_tokenizer,
        cfg.causal_lm_tokenizer.decode(input_sequence[0].tolist()),
    )
    targeted_pairs = token_loss_pairs[
        -cfg.pure_ctxt_sizes.obs_size - 3 : -cfg.pure_ctxt_sizes.obs_size + 5
    ]
    plt.clf()

    # slicing [:,1:] is a hack because of mistral number tokenization creating a leading space token!
    batch_obs_losses = masked_losses[:, 1:].sum(dim=1) / pure_obs_attention_mask[
        :, 1:
    ].sum(dim=1)
    if dist.get_rank() == 0:
        print(
            "Default:" if is_default_action else "Updated:",
            batch_obs_losses[0].item(),
            targeted_pairs,
        )
    return batch_obs_losses if per_batch else batch_obs_losses.mean()

    # obs_tensor = (loss_tensor * attention_mask[:, 1:])[:, -cfg.pure_ctxt_sizes.obs_size :]
    # obs_losses = obs_tensor.sum(dim=-1) / attention_mask[:, 1:][
    #    :, -cfg.pure_ctxt_sizes.obs_size :
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
