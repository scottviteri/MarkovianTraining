import collections

import pytest
import torch

from train import TrainingState, process_batch, update_model


@pytest.mark.slow
def test_training_step_updates_actor_not_critic(
    gpt2_components, tiny_training_hparams, arithmetic_batch, tmp_path
):
    actor_model = gpt2_components["actor"]
    critic_model = gpt2_components["critic"]
    tokenizer = gpt2_components["tokenizer"]
    device = gpt2_components["device"]

    actor_model.train()
    critic_model.eval()

    optimizer = torch.optim.Adam(actor_model.parameters(), lr=5e-5)

    state = TrainingState(
        batch_index=0,
        previous_normalized_rewards=[],
        previous_advantages=[],
        actor_model=actor_model,
        critic_model=critic_model,
        actor_optimizer=optimizer,
        tokenizer=tokenizer,
        device=device,
        model_save_path=str(tmp_path / "model"),
        log_file=str(tmp_path / "train_log.jsonl"),
        hyperparameters=tiny_training_hparams,
        accumulation_step=0,
        skip_history=collections.deque(maxlen=10),
    )

    # Capture small snapshots of actor/critic parameters for comparison
    actor_samples = []
    for param in state.actor_model.parameters():
        if param.requires_grad:
            actor_samples.append((param, param.detach().cpu().clone()))
            if len(actor_samples) == 2:
                break
    critic_param = next(state.critic_model.parameters())
    critic_snapshot = critic_param.detach().cpu().clone()

    batch_data = process_batch(state, arithmetic_batch)
    grad_norm = update_model(state, batch_data)

    assert grad_norm >= 0

    actor_changed = any(
        not torch.allclose(snapshot, param.detach().cpu())
        for param, snapshot in actor_samples
    )
    assert actor_changed, "Actor parameters did not change after optimizer step"

    assert torch.allclose(
        critic_snapshot, critic_param.detach().cpu()
    ), "Critic weights should remain frozen"

