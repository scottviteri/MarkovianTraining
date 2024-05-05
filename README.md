# MarkovianTraining 

## Installation
Run the following line:
```
apt update && apt install -y vim nano ncurses-term tmux && pip install transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb scipy && pip install -U flash-attn --no-build-isolation && pip install openai
```
Once that is done, you can run `torchrun src/train.py`.
If you are not running on a machine with >= 80GB VRAM, then you will want to change the last line of `src/config_examples.py` from `config_examples=[mst]` to `config_examples=[g2]`.
