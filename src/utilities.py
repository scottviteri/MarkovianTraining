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
import bitsandbytes
import wandb
import dataclasses
import einops
from datasets import load_dataset
import json
import copy
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import os
import glob

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from training_types import *
from prepare_dataset import prepare_dataset

import transformers
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationMixin,
    GPT2Model,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


class VHead(nn.Module):
    def __init__(self, input_dim):
        super(VHead, self).__init__()
        # Assuming the last layer's output dimension is used as input_dim
        gpt2_config = AutoConfig.from_pretrained(
            "gpt2", n_head=1, n_embd=1, hidden_size=input_dim
        )
        self.gpt2_block = GPT2Block(gpt2_config)
        self.v_head = nn.Linear(input_dim, 1, bias=True)

    def forward(self, hidden_states, attention_mask=None):
        pre_values = self.gpt2_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )[0]
        values = self.v_head(pre_values).squeeze(-1)
        return values


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

    def forward(
        self,
        input_ids,
        add_q_head,
        get_v_head,
        attention_mask=None,
        **kwargs,
    ):
        model = self.qhead if add_q_head else self.transformer
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=get_v_head,
            **{k: v for k, v in kwargs.items() if k != "output_hidden_states"},
        )
        # if get_v_head:
        #    # pre_values = self.v_head_group["v_head_block"](
        #    #    outputs.hidden_states[-1].detach()
        #    # )[0]
        #    # values = self.v_head_group["v_head"](pre_values).squeeze(-1)
        #    hidden_states = outputs.hidden_states[-1].detach()
        #    values = self.v_head_group["v_head"](hidden_states).squeeze(-1)
        #    return outputs, values
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
    path_2_log = f"saved_weights_and_losses/{init_cfg.model_name}_log.txt"
    current_time = datetime.now(timezone(timedelta(hours=-7)))
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    traj_path = f"saved_weights_and_losses/{init_cfg.model_name}_traj_{timestamp}.json"
    path_2_model = f"saved_weights_and_losses/{init_cfg.model_name}_weights"
    path_2_tokenizer = f"saved_weights_and_losses/{init_cfg.model_name}_tokenizer"

    assert init_cfg.num_beams == 1, "Only supporting num_beams = 1 currently"

    if not init_cfg.use_mac:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        device = torch.device(dist.get_rank())
    rank = dist.get_rank()
    print("rank", rank)

    causal_lm, v_head, tokenizer = get_model(
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
        tokenizer, init_cfg.batch_size, device, init_cfg.ctxt_sizes
    )
    causal_lm.generation_config.min_new_tokens = pure_ctxt_sizes.action_size
    causal_lm.generation_config.max_new_tokens = pure_ctxt_sizes.action_size

    # vhead = DDP(vhead, device_ids=[rank])
    causal_lm = DDP(
        causal_lm,
        device_ids=[rank],
        # find_unused_parameters=True,
    )

    if not init_cfg.load_model:
        with open(path_2_log, "w") as f:
            print("")
    with open(path_2_log, "a") as f:
        f.write("")

    if init_cfg.wandb and (init_cfg.use_mac or rank == 0):
        wandb.init(
            project="collaborative-training-many-per-context-window",
            name=create_run_name(init_cfg),
        )

    with open(traj_path, "w") as file:
        json.dump(dataclasses.asdict(init_cfg), file, indent=4)

    if rank == 0:
        print("Causal LM: ", causal_lm)

    parameters = list(
        param
        for name, param in causal_lm.module.qhead.named_parameters()
        if ".lora" in name
    ) + list(v_head.parameters())

    if init_cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD if init_cfg.use_mac else bitsandbytes.optim.SGD8bit
    elif init_cfg.optimizer == "adam":
        optimizer = (
            torch.optim.AdamW if init_cfg.use_mac else bitsandbytes.optim.AdamW8bit
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. Please choose 'sgd' or 'adam'."
        )
    optimizer = optimizer(parameters, lr=init_cfg.lr)

    dataset = prepare_dataset(
        init_cfg,
        tokenizer,
        device,
        pure_ctxt_sizes,
        prefix_tensors,
    )

    return Config(
        model_name=init_cfg.model_name,
        rank=rank,
        lr=init_cfg.lr,
        optimizer=optimizer,
        batch_size=init_cfg.batch_size,
        num_batches=init_cfg.num_batches,
        replay_buffer_size=init_cfg.replay_buffer_size,
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
        v_head=v_head,
        tokenizer=tokenizer,
        inference_cfg=init_cfg.inference_cfg,
        prediction_cfg=init_cfg.prediction_cfg,
        trainer_cfg=init_cfg.trainer_cfg,
        perturbation_cfg=init_cfg.perturbation_cfg,
        debug=init_cfg.debug,
    )


def multi_print(string, f):
    print(string)
    print(string, file=f)


def log_print_oa(
    cfg,
    batch_index,
    prev_action,
    prev_obs,
    action,
    default_action,
    obs,
    aggregate_loss,
):
    if batch_index % cfg.interval_print == 0 and (cfg.use_mac or cfg.rank == 0):
        with open(cfg.path_2_log, "a", encoding="utf-8") as f:
            multi_print(f"Batch Index: {batch_index}", f)
            if aggregate_loss:
                multi_print(f"Aggregate Loss: {aggregate_loss}", f)
            multi_print(
                f"Prev Action: {repr(cfg.tokenizer.decode(prev_action[0])) if prev_action is not None else None}",
                f,
            )
            multi_print(
                f"Prev Observation: {repr(cfg.tokenizer.decode(prev_obs[0]))}",
                f,
            )
            multi_print(
                f"Action: {repr(cfg.tokenizer.decode(action[0]))}",
                f,
            )
            multi_print(
                f"Default Action: {repr(cfg.tokenizer.decode(default_action[0]))}",
                f,
            )
            multi_print(f"Observation: {repr(cfg.tokenizer.decode(obs[0]))}", f)


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


def load_latest_model_head(model_name: str, head_type: str) -> torch.nn.Module:
    assert head_type in [
        "qhead",
        "vhead",
    ], "head_type must be either 'qhead' or 'vhead'"
    pth_files = glob.glob(
        os.path.join("saved_weights_and_losses", f"*{head_type}*.pth")
    )
    model_pth_files = list(filter(lambda x: model_name in x, pth_files))
    assert (
        len(model_pth_files) > 0
    ), f"No {head_type} files found for model {model_name}"
    model_pth_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_pth_file = model_pth_files[0]
    return torch.load(latest_pth_file)


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
            causal_lm = load_latest_model_head(model_name, "qhead").module
            v_head = load_latest_model_head(model_name, "vhead")
        else:
            config = AutoConfig.from_pretrained(model_dict[model_name])
            causal_lm = ModelWithQHead(model_dict[model_name], config)
            v_head = VHead(input_dim=4096)

        if not use_mac:
            causal_lm.bfloat16()
            v_head.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(
            model_dict[model_name], padding_side=padding_side
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        for name, param in causal_lm.transformer.named_parameters():
            param.requires_grad = False
        for name, param in causal_lm.qhead.named_parameters():
            param.requires_grad = ".lora" in name
        # for name, param in causal_lm.v_head_group.named_parameters():
        #    param.requires_grad = True
        for name, param in v_head.named_parameters():
            param.requires_grad = True

    causal_lm.tokenizer = tokenizer
    bad_words_ids = [
        [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
    ]
    logits_warper = transformers.generation.LogitsProcessorList(
        [
            transformers.generation.NoBadWordsLogitsProcessor(
                bad_words_ids,
                eos_token_id=tokenizer.eos_token_id,
            ),
            # transformers.generation.MinNewTokensLengthLogitsProcessor(
            #    prompt_length_to_skip=input_ids.shape[-1],
            #    min_new_tokens=cfg.pure_ctxt_sizes.action_size,
            #    eos_token_id=tokenizer.eos_token_id,
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )
    causal_lm.generation_config = generation_config
    return causal_lm, v_head, tokenizer


def get_mlp_modules(model):
    modules = []
    for x in model.named_parameters():
        parts = x[0].split(".")
        if "mlp" in parts:
            modules.append(".".join(parts[: parts.index("mlp") + 2]))
    # return [x[0].split(".") for x in model.named_parameters() if "mlp" in x[0]]
    return list(set(modules))


def create_run_name(init_cfg: InitialConfig) -> str:
    prediction_cfg = init_cfg.prediction_cfg
    run_name = ""
    run_name += f"{init_cfg.model_name[:4]}_"
    run_name += f"{init_cfg.optimizer}_"
    run_name += f"pu{init_cfg.trainer_cfg.prediction_training_length}_gu{init_cfg.trainer_cfg.inference_training_length}_"
    if isinstance(init_cfg.dataset.task, ArithmeticTask):
        run_name += f"ari_nt={init_cfg.dataset.task.num_terms}_nd={init_cfg.dataset.task.num_digits}_"
    else:
        run_name += "wiki_"
    if init_cfg.lr != 1e-4:
        run_name += f"lr{init_cfg.lr}_"
    if prediction_cfg.train_O_given_prev_O:
        run_name += f"AR_"
    if prediction_cfg.train_O_given_A:
        run_name += f"M_"
    if init_cfg.dataset.peek_every is not None:
        run_name += f"pe{init_cfg.dataset.peek_every}_"

    if isinstance(init_cfg.debug, RepeatNPoints):
        run_name += f"r{init_cfg.debug.num_points}_"
    elif isinstance(init_cfg.debug, RepeatPointNTimes):
        run_name += f"re{init_cfg.debug.num_times}_"
    elif isinstance(init_cfg.debug, ReplaceWithRandomTokens):
        run_name += "rd_"
    elif isinstance(init_cfg.debug, NoWeightUpdates):
        run_name += "nwu_"

    if init_cfg.batch_size != 1:
        run_name += f"bs{init_cfg.batch_size}_"
    run_name += f"nb{init_cfg.num_batches}_"
    if init_cfg.load_model:
        run_name += f"load_"
    if init_cfg.do_lora:
        run_name += "lra_"
    if init_cfg.num_beams:
        run_name += f"nbe{init_cfg.num_beams}_"
    run_name += f"cs{init_cfg.ctxt_sizes}"
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
    assert instruct[0].shape[0] == rest[0].shape[0]
    start_tokens = torch.full(
        (instruct[0].shape[0], 1),
        cfg.tokenizer.bos_token_id,
        dtype=torch.int64,
        device=cfg.device,
    )
    # needs_start_token = cfg.model_name in ["mistral", "llama"]
    # needs_instruct_token = "Instruct" in cfg.causal_lm.name_or_path
    begin_instruct_tokens = (
        cfg.tokenizer.encode(
            (
                "[INST] Use the following possibly mistaken reasoning to help predict the true numerical answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction."
                if is_prediction
                else "[INST]"
            ),
            return_tensors="pt",
            add_special_tokens=False,
        )
        .to(cfg.device)
        .repeat((instruct[0].shape[0], 1))
    )
    end_instruct_tokens = (
        cfg.tokenizer.encode("[/INST]", return_tensors="pt", add_special_tokens=False)
        .to(cfg.device)
        .repeat((instruct[0].shape[0], 1))
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


def predict_action(cfg, prev_action, prev_obs, action, add_q_head, add_v_head):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = wrap_input_tokens(
        cfg,
        [prev_action, prev_obs],
        [action],
        use_start_token=True,
        use_instruct_tokens=True,
        is_prediction=False,
    )
    attention_mask = (input_sequence != cfg.tokenizer.pad_token_id).long()
    prediction = cfg.causal_lm(
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
    negentropies = -entropy_from_logits(action_logits).mean(dim=1)
    pure_action_attention_mask = attention_mask[:, -cfg.pure_ctxt_sizes.action_size :]
    masked_losses = action_loss_tensor * pure_action_attention_mask
    # plt.figure(); plt.plot(masked_losses[0].tolist()); plt.savefig("action.png"); plt.clf()
    action_losses = masked_losses.sum(dim=1) / pure_action_attention_mask.sum(dim=1)
    if add_v_head:
        hidden_states = prediction.hidden_states[-1].detach()
        values = cfg.v_head(hidden_states)
        return (
            action_losses,
            values[:, : -cfg.pure_ctxt_sizes.action_size],
            negentropies,
        )
    return action_losses, negentropies


def test_sequence(predictor, input_sequence):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    prediction = predictor(
        input_sequence, add_q_head=False, get_v_head=False, attention_mask=None
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


def predict_observation(cfg, action, obs, add_q_head, is_default_action=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    input_sequence = wrap_input_tokens(
        cfg,
        [action],
        [obs],
        use_start_token=True,
        use_instruct_tokens=True,
        is_prediction=True,
    )
    attention_mask = (input_sequence != cfg.tokenizer.pad_token_id).long()
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
    # plt.figure(); plt.plot(masked_losses[0, 1:].tolist()); plt.savefig("obs_default.png"); plt.clf()
    token_loss_pairs = inspect_string(
        cfg.causal_lm,
        cfg.tokenizer,
        cfg.tokenizer.decode(input_sequence[0].tolist()),
    )
    targeted_pairs = [
        p
        for p in token_loss_pairs[-cfg.pure_ctxt_sizes.obs_size - 3 :]
        if str(cfg.tokenizer.pad_token) != p[0]
    ]
    # slicing [:,1:] is a hack because of mistral and llama number tokenization create a leading space token!
    obs_losses = masked_losses[:, 1:].sum(dim=1) / pure_obs_attention_mask[:, 1:].sum(
        dim=1
    )
    if cfg.rank == 0:
        print(
            "Default:" if is_default_action else "Updated:",
            obs_losses[0].item(),
            targeted_pairs,
        )
    return obs_losses


def get_neg_log_probs(cfg, input_sequence):
    """
    Computes the loss tensor for a given input sequence.

    Args:
        cfg: Configuration object containing model and tokenizer information.
        input_sequence: The input sequence tensor for which the loss is to be computed.

    Returns:
        The computed loss tensor.
    """
    attention_mask = (input_sequence != cfg.tokenizer.pad_token_id).long()
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
