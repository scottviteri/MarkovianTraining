# pip install transformers datasets==2.14.6 torchtyping==0.1.4
# pip install peft einops apache_beam==2.51.0 matplotlib wandb
# pip install -U flash-attn --no-build-isolation

import torch
from tqdm import tqdm
from einops import rearrange
import wandb
from rao_tools import RaoConfig
from rao_generator import RaoGenerator

cfg = RaoConfig(
    load_model=False,
    wandb=False,  # True,
    model_name="distilgpt2",  # "mistral",  # "gpt2" #"gpt2-xl" #"distilgpt2" #gpt2-large" # distilgpt2  ;  EleutherAI/gpt-j-6b
    save_dir=".",  # "/home/scottviteri/Projects/CollaborativeTraining/CollaborativeTraining/saved_weights_and_losses"
    tok_p_reward=10,
    tok_p_action=100,
    # if None, calculates: CTXT_WINDOW_SIZE - TOKENS_PER_ACTION - TOKENS_PER_REWARD
    tok_p_obs=None,
    obs_p_doc=5,  # 20,
    batch_size=2,
    num_batches=4,
    interval_save_weights=30,
)

if cfg.wandb:
    run = wandb.init(
        project="collaborative-training-many-per-context-window", entity="scottviteri"
    )
    wandb_table = wandb.Table(
        data=[],
        columns=[
            "Previous Observation",
            "Action",
            "Predicted Observation",
            "Actual Observation",
        ],
    )
else:
    wandb_table = None


NUM_DATAPOINTS = cfg.batch_size * cfg.num_batches if cfg.num_batches else None
causal_lm = cfg.model
causal_lm_tokenizer = cfg.tokenizer

raogen = RaoGenerator(
    cfg=cfg,
    num_data_points=NUM_DATAPOINTS,
)
dataloader = raogen.dataloader

batch_index = 0
aggregate_losses = []
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(causal_lm.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.num_batches
)

for batch_index, data in (
    tqdm(enumerate(dataloader), total=cfg.num_batches) if cfg.num_batches else tqdm(dataloader)
):
    if cfg.num_batches and batch_index > cfg.num_batches:
        break
    batch_index += 1
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

    # Fixme: different than many per context
    # rao_tensor = torch.cat((losses, action, true_obs), dim=-1)
    # rao_tensor_logits = causal_lm(rao_tensor).logits
    # rao_tensor_loss = loss_fn(
    #     input=rearrange(
    #         rao_tensor_logits,
    #         "batch seq_length vocab_size -> batch vocab_size seq_length",
    #     ),
    #     target=rao_tensor,
    # )

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
    scheduler.step()

if cfg.wandb:
    run.log({"Prediction Accuracy Table": wandb_table})
    wandb.finish()
