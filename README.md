# MarkovianTraining 

## Installation
```
pip install scipy transformers datasets==2.14.6 torchtyping==0.1.4 && pip install peft einops apache_beam==2.51.0 matplotlib wandb && pip install -U flash-attn --no-build-isolation && pip install openai bitsandbytes scipy scikit-learn
```

## Evaluation
Once that is done, you can run `torchrun src/train.py`.
If you are not running on a machine with >= 80GB VRAM, then you will want to change the last line of `src/config_examples.py` from `config_examples=[mst]` to `config_examples=[g2]`.

## Files
* `src/train.py` sets up the main training loop and logging
* `src/config_examples.py` is user facing -- it contains the main parameters behind a particular training run and stores the configurations in a variable called configs, which is used by `src/train.py`. 
* `src/utilities.py` extends the user specified configuration and defines the main components of the reward function. 
* `src/training_types.py` contains several typed, frozen dataclasses that are used throughout the codebase, and especially in `src/config_examples.py`.
* `src/test_generation.py` is a debugging tool which is not used in the main `src/train.py` loop -- it checks that a handful of handwritten chain-of-thought (CoT) prompts have expected relative usefulnesses toward predicting the answer in a QA pair.actions
* `src/prepare_dataset.py` generates a dataset in the form of an iter of dicts with keys "Action" and "Observation". Conceptually, it outputs just observations, but having an optional "Action" allows us to supply fake CoTs to the model, which is useful for debugging.
* `src/evaluate_via_gpt.py` is disconnected from the main loop, and can generate `arithmetic_explanations.json`, a gold standard (gpt4-generated) table of (question, CoT, answer) triples. 
* `src/evaluate_actions.py` generates plots (the ones in the paper) from the full output log of a training run in the form of a json file such as `mistral_traj_20240329_051521.json`.
* * `src/plot_lss.py` generates the loss plot (the one in the paper) from the full output log of a training run in the form of a csv file.
