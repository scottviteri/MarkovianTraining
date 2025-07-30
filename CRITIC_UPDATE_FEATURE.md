# Critic Model Update Feature

This feature allows the critic model to be periodically updated with the actor model's learned weights during training.

## Overview

By default, the critic model remains frozen throughout training. With this feature, you can configure the critic to periodically sync with the actor model, incorporating its learned improvements. This is particularly useful in reinforcement learning scenarios where you want the baseline (critic) to gradually improve along with the policy (actor).

## Usage

Add the `--critic_update_freq` parameter to your training command:

```bash
python src/train.py \
    --task_type arithmetic \
    --model_type qwen25 \
    --use_ppo \
    --cot_length 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_batches 1000 \
    --temperature 1.0 \
    --ppo_epsilon 0.2 \
    --critic_update_freq 100  # Update critic every 100 batches
```

## Parameters

- `--critic_update_freq N`: Update the critic model with actor weights every N batches
  - Default: 0 (disabled)
  - Set to any positive integer to enable periodic updates
  - Recommended values: 50-200 depending on your task

## How It Works

1. During training, the actor model learns via LoRA adapters while the base model stays frozen
2. At specified intervals, the actor's LoRA weights are merged with the base model
3. The merged weights are copied to the critic model
4. The critic remains frozen between updates
5. The actor model continues training with separate LoRA adapters

## Example Output

When critic updates occur, you'll see messages like:
```
Critic Update: Updating critic model with actor weights at batch 100
Critic Update: Merging LoRA weights into base model
Critic Update: Critic model successfully updated with actor weights
```

## Best Practices

- Start with a moderate update frequency (e.g., every 100 batches)
- Monitor the advantage values to ensure they remain stable
- Use with `--enable_weight_verification` to verify updates are working correctly
- Adjust frequency based on training stability and performance

## Technical Details

The implementation handles the complexity of PEFT/LoRA models by:
- Temporarily merging LoRA adapters into the base model
- Copying the merged state to the critic
- Restoring the actor's original state to continue LoRA training
- Re-freezing all critic parameters after each update 