import torch

from train import calculate_losses


def test_calculate_losses_preserves_actor_reward_gradients():
    # Inputs that require gradients to simulate actor reward training
    R_mean_actor_logprobs = torch.tensor([0.2, -0.1], requires_grad=True)
    R_mean_critic_logprobs = torch.tensor([0.05, 0.03], requires_grad=True)
    advantages = torch.tensor([0.4, -0.6], requires_grad=True)
    normalized_rewards = torch.tensor([0.1, -0.2])
    kl = torch.zeros_like(R_mean_actor_logprobs, requires_grad=True)

    hyperparameters = {
        "use_ppo": False,
        "ppo_epsilon": 0.2,
        "use_ei": None,
        "actor_reward_weight": 0.5,
        "kl_penalty": 0.0,
        "entropy_bonus": 0.0,
    }

    losses, training_mask, metrics = calculate_losses(
        kl=kl,
        R_mean_actor_logprobs=R_mean_actor_logprobs,
        R_mean_critic_logprobs=R_mean_critic_logprobs,
        advantages=advantages,
        normalized_rewards=normalized_rewards,
        previous_advantages=[],
        previous_normalized_rewards=[],
        hyperparameters=hyperparameters,
    )

    # No EI mask expected
    assert training_mask is None

    total_loss = losses.sum()
    total_loss.backward()

    # Gradients should flow through both PG and reward components
    assert R_mean_actor_logprobs.grad is not None
    assert torch.any(R_mean_actor_logprobs.grad != 0)
    assert advantages.grad is not None
    assert torch.any(advantages.grad != 0)

    # Metrics are stored for logging only, so they should be detached
    assert metrics["pg_losses"].requires_grad is False
    assert metrics["reward_gradient_losses"].requires_grad is False

