# pip install transformers datasets==2.14.6 torchtyping==0.1.4
# pip install peft einops apache_beam==2.51.0 matplotlib wandb
# pip install -U flash-attn --no-build-isolation


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
        #'load_model': {'values': [False]},
        #'wandb': {'values': [True]},
        'model_name': {'values': ["distilgpt2"]},
        #'save_dir': {'values': ["."]},
        'lr': {'values': [1e-3, 5e-4, 1e-4]},
        'tok_p_reward': {'values': [10,30,50]},
        'tok_p_action': {'values': [10,30,50]},
        'tok_p_obs': {'values': [10,30,50]},
        #'obs_p_doc': {'values': [10]},
        'batch_size': {'values': [8]},
        'num_batches': {'values': [10]},
        #'interval_save_weights': {'values': [30]},
    }
}

sweep_id = wandb.sweep(sweep_config)

def train():
    run = wandb.init(project="collaborative-training-many-per-context-window")
    wb_cfg = run.config
    obs_p_doc = 1024 // (wb_cfg.tok_p_reward + wb_cfg.tok_p_action + wb_cfg.tok_p_obs) 
    cfg = RaoConfig(
        load_model=False,
        wandb=True,
        model_name=wb_cfg.model_name,
        lr = wb_cfg.lr,
        save_dir=".",
        tok_p_reward=wb_cfg.tok_p_reward,
        tok_p_action=wb_cfg.tok_p_action,
        tok_p_obs=wb_cfg.tok_p_obs,
        obs_p_doc=obs_p_doc,
        batch_size=wb_cfg.batch_size,
        num_batches=wb_cfg.num_batches,
        interval_save_weights=30
    )
    run.name = f"{wb_cfg.model_name}_lr{wb_cfg.lr:e}_rao{wb_cfg.tok_p_reward}/{wb_cfg.tok_p_action}/{wb_cfg.tok_p_obs}_bs{wb_cfg.batch_size}"
    run.save()

    wandb_table = wandb.Table(
        data=[],
        columns=[
            "Previous Observation",
            "Action",
            "Predicted Observation",
            "Actual Observation",
        ],
    )

    NUM_DATAPOINTS = cfg.batch_size * cfg.num_batches if cfg.num_batches else None
    causal_lm = cfg.model
    causal_lm_tokenizer = cfg.tokenizer

    raogen = RaoGenerator(
        cfg=cfg,
        num_data_points=NUM_DATAPOINTS,
    )
    dataloader = raogen.dataloader

    aggregate_losses = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(causal_lm.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.LinearLR(
    #    optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.num_batches
    #)

    for batch_index, data in (
        tqdm(enumerate(dataloader), total=cfg.num_batches) if cfg.num_batches else tqdm(dataloader)
    ):
        if cfg.num_batches and batch_index > cfg.num_batches:
            break
        if batch_index > 1 and batch_index % cfg.interval_save_weights == 0:
            print(f"Saving trained_{cfg.model_name}")
            causal_lm_tokenizer.save_pretrained(
                f"./saved_weights_and_losses/tokenizer_{cfg.model_name}"
            )
            causal_lm.save_pretrained(
                f"./saved_weights_and_losses/trained_{cfg.model_name}"
            )

        rao_tensor = raogen.gen_rao_tensor(
            data=data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            aggregate_losses=aggregate_losses,
            batch_index=batch_index,
            wandb_table=wandb_table,
        )

        # Compute the loss on the whole rao_tensor sequence and perform backpropagation
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
        if cfg.wandb:
            wandb.log({"Aggregate loss": aggregate_loss})
        optimizer.step()
        #scheduler.step()

    run.log({"Prediction Accuracy Table": wandb_table})
    run.finish()


wandb.agent(sweep_id, function=train)