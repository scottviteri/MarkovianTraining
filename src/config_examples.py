import os

from training_types import *

g2 = InitialConfig(
    model_name="distilgpt2",
    lr=1e-6,
    optimizer="adam",
    batch_size=3,
    num_batches=300,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    interval_print=1,
    use_mac=False,
    wandb=False,
    load_model=False,
    do_lora=True,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=6,
            num_digits=2,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
        ),
        peek_every=None,
    ),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    trainer_cfg=TrainerConfig(
        prediction_training_length=1, inference_training_length=30
    ),
    perturbation_cfg=None,
    # perturbation_cfg=PerturbationConfig(
    #    eval_every=10,
    #    frac_of_tokens_to_randomize=0.5,
    #    frac_of_tokens_to_pad=0.5,
    # ),
    debug=None,
)

g2_ar = InitialConfig(
    model_name="distilgpt2",
    lr=1e-4,
    optimizer="adam",
    batch_size=2,
    num_batches=10,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    use_mac=False,
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2,
            num_digits=3,
            operations=None,
            probs=None,
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=5, inference_training_length=5
    ),
    perturbation_cfg=None,
    debug=None,
)

g2_ei = InitialConfig(
    model_name="distilgpt2",
    lr=1e-4,
    optimizer="sgd",
    batch_size=2,
    num_batches=10,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    use_mac=False,
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2,
            num_digits=3,
            operations=None,
            probs=None,
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=5, inference_training_length=5
    ),
    perturbation_cfg=None,
    debug=None,
)


g2_p2 = InitialConfig(
    model_name="distilgpt2",
    lr=1e-4,
    optimizer="sgd",
    batch_size=4,
    num_batches=1000,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    use_mac=False,
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2,
            num_digits=3,
            operations=None,
            probs=None,
        ),
        peek_every=2,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=5, inference_training_length=5
    ),
    perturbation_cfg=None,
    debug=None,
)

gj = InitialConfig(
    model_name="gptj",
    lr=1e-5,
    optimizer="sgd",
    batch_size=4,
    num_batches=1000,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    use_mac=False,
    interval_print=5,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2,
            num_digits=3,
            operations=None,
            probs=None,
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=5, inference_training_length=5
    ),
    perturbation_cfg=None,
    debug=None,
)

lma = InitialConfig(
    model_name="llama",
    lr=1e-6,
    optimizer="adam",
    batch_size=1,
    num_batches=1000,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    use_mac=False,
    interval_print=11,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2,
            num_digits=3,
            operations=None,
            probs=None,
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=5, inference_training_length=5
    ),
    perturbation_cfg=None,
    debug=None,
)

phi2 = InitialConfig(
    model_name="phi2",
    lr=1e-6,
    optimizer="adam",
    batch_size=1,
    num_batches=1000,
    obs_to_action_ratio=0.5,
    interval_save_weights=900,
    use_mac=False,
    interval_print=1,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=3,
            num_digits=3,
            operations=["+", "-", "*"],
            probs=[0.0, 0.0, 1.0],
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=1, inference_training_length=1000
    ),
    perturbation_cfg=None,
    debug=None,
)

mst = InitialConfig(
    model_name="mistral",
    lr=1e-5,
    optimizer="adam",
    batch_size=4,
    num_batches=5000,
    obs_to_action_ratio=0.5,
    interval_save_weights=4900,
    use_mac=False,
    interval_print=1,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(40, 70, 300, 20),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=15,
            num_digits=2,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
        ),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=1),
    prediction_cfg=PredictionConfig(
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    trainer_cfg=TrainerConfig(
        prediction_training_length=1, inference_training_length=1000
    ),
    perturbation_cfg=None,
    debug=None,
)

configs = [mst]
