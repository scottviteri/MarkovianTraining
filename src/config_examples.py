import os

from training_types import *

g2 = InitialConfig(
    model_name="distilgpt2",
    lr=1e-6,
    optimizer="adam",
    batch_size=3,
    num_batches=300,
    replay_buffer_size=50,
    interval_save_weights=1000,
    interval_print=1,
    use_mac=False,
    wandb=False,
    load_model=False,
    do_lora=True,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 20),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=6,
            num_digits=2,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
            cumulative=True,
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

lma = InitialConfig(
    model_name="llama",
    lr=1e-6,
    optimizer="adam",
    batch_size=1,
    num_batches=1000,
    replay_buffer_size=50,
    interval_save_weights=1000,
    use_mac=False,
    interval_print=11,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(50, 50, 100, 50),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=2, num_digits=3, operations=None, probs=None, cumulative=True
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
    replay_buffer_size=50,
    interval_save_weights=1000,
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
            cumulative=True,
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
    lr=5e-5,
    optimizer="adam",
    batch_size=4,
    num_batches=501,
    replay_buffer_size=None,
    interval_save_weights=500,
    use_mac=False,
    interval_print=1,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(40, 70, 400, 20),
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=15,
            num_digits=2,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
            cumulative=True,
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

q_mst = InitialConfig(
    model_name="mistral",
    lr=1e-4,
    optimizer="adam",
    batch_size=4,
    num_batches=501,
    replay_buffer_size=None,
    interval_save_weights=500,
    use_mac=False,
    interval_print=1,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    ctxt_sizes=ContextSizes(40, 70, 400, 100),
    dataset=InitDatasetType(
        task=QuestionTask(
            num_terms=15,
            num_digits=2,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
            cumulative=True,
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

configs = [q_mst]
