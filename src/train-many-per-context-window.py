# pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation
# huggingface-cli login

import torch
from tqdm import tqdm
from einops import rearrange
import wandb
from rao_tools import RaoConfig
from rao_generator import RaoGenerator


sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'load_model': {'values': [True]},
        'use_wandb': {'values': [False]},  
        'model_name': {'values': ["tinystories"]},
        'lr': {'values': [1e-4]},
        'do_lora': {'values': [False]},
        'tok_p_loss': {'values': [9]},
        'tok_p_action': {'values': [30]},
        'tok_p_obs': {'values': [30]},
        #'obs_p_doc': {'values': [10]},
        'num_beams': {'values': [1]},
        'batch_size': {'values': [200]},
        'num_batches': {'values': [30]},
        'interval_save_weights': {'values': [10]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="collaborative-training-many-per-context-window")

def train():
    run = None
    if sweep_config['parameters']['use_wandb']['values'][0]:
        run = wandb.init(resume=True)
        wb_cfg = run.config
        config_params = {param: getattr(wb_cfg, param) for param in sweep_config['parameters']}
    else:
        wb_cfg = None
        config_params = {param: sweep_config['parameters'][param]['values'][0] for param in sweep_config['parameters']}

    obs_p_doc = 1024 // (config_params['tok_p_loss'] + config_params['tok_p_action'] + config_params['tok_p_obs']) 
    cfg = RaoConfig(
        load_model=config_params['load_model'],
        wandb=sweep_config['parameters']['use_wandb']['values'][0],
        model_name=config_params['model_name'],
        lr = config_params['lr'],
        do_lora = config_params['do_lora'], 
        tok_p_loss=config_params['tok_p_loss'],
        tok_p_action=config_params['tok_p_action'],
        tok_p_obs=config_params['tok_p_obs'],
        obs_p_doc=obs_p_doc,
        num_beams =config_params['num_beams'],
        batch_size=config_params['batch_size'],
        num_batches=config_params['num_batches'],
        interval_save_weights=config_params['interval_save_weights'],
        interval_print = 5
    )
    # todo add flag for ld
    lora_string = "L" if cfg.do_lora else "nL"
    if run is not None:
        run.name = f"ld_b{cfg.num_beams}_{lora_string}{cfg.model_name[:4]}_lr{cfg.lr}_rao{cfg.tok_p_loss}/{cfg.tok_p_action}/{cfg.tok_p_obs}_bs{cfg.batch_size}_nb{cfg.num_batches}"

    if not cfg.load_model:
        with open(f"saved_weights_and_losses/{cfg.model_name}", "w") as f:
            print("")

    NUM_DATAPOINTS = cfg.batch_size * cfg.num_batches if cfg.num_batches else None
    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.tokenizer

    raogen = RaoGenerator(
        cfg=cfg,
        num_data_points=NUM_DATAPOINTS,
    )
    dataloader = raogen.dataloader

    average_loss_differences = []
    aggregate_losses = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(causal_lm.parameters(), lr=cfg.lr)

    for batch_index, data in (
        tqdm(enumerate(dataloader), total=cfg.num_batches) if cfg.num_batches else tqdm(dataloader)
    ):
        if cfg.num_batches and batch_index > cfg.num_batches:
            break
        if batch_index > 0 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name}")
            causal_lm_tokenizer.save_pretrained(cfg.path_2_tokenizer)
            causal_lm.save_pretrained(cfg.path_2_model)

        rao_tensor, new_loss_differences = raogen.gen_rao_tensor(
            data=data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            average_loss_differences = average_loss_differences,
            aggregate_losses=aggregate_losses,
            batch_index=batch_index,
        )

        average_loss_differences.extend(new_loss_differences)

        rao_tensor_logits = causal_lm(rao_tensor).logits[:, :-1, :]
        rao_tensor_loss = loss_fn(
            input=rearrange(
                rao_tensor_logits,
                "batch seq_length vocab_size -> batch vocab_size seq_length",
            ),
            target=rao_tensor[:, 1:],
        )
        aggregate_loss = rao_tensor_loss.mean()
        aggregate_losses.append(aggregate_loss.item())
        aggregate_loss.backward()
        print("Aggregate loss: ", aggregate_loss)
        if wb_cfg:
            wandb.log({"Aggregate loss": aggregate_loss})
        optimizer.step()

    if wb_cfg:
        run.finish()


if sweep_config['parameters']['use_wandb']['values'][0]:
    wandb.agent(sweep_id, function=train)
else:
    train()
