from src.training_types import *
import os

g2 = InitialConfig(
    model_name="gpt2",
    lr=1e-7,
    optimizer="adam",
    batch_size=2,
    num_batches=1000,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    interval_print=21,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=200,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False), peek_every=None
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=100, fraction_to_update=1.0
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions = 2, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
    ),
     perturbation_cfg=None,
    #perturbation_cfg=PerturbationConfig(
    #    eval_every=10,
    #    frac_of_tokens_to_randomize=0.5,
    #    frac_of_tokens_to_pad=0.5,
    #),
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
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=150,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
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
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=150,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
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
    interval_print=21,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=150,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False), peek_every=2
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
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
    interval_print=5,
    wandb=False,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=150,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
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
    interval_print=11,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=200,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
         filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
    ),
    perturbation_cfg=None,
    debug=None,
)

mst = InitialConfig(
    model_name="mistral",
    lr=1e-7,
    optimizer="adam",
    batch_size=1,
    num_batches=10,
    obs_to_action_ratio=0.5,
    interval_save_weights=99,
    interval_print=1,
    wandb=True,
    load_model=False,
    do_lora=False,
    num_beams=1,
    training_ctxt_size=200,
    dataset=InitDatasetType(
        task=ArithmeticTask(num_terms=2, num_digits=3, cumulative=False),
        peek_every=None,
    ),
    inference_cfg=InferenceConfig(
        num_return_sequences=1, update_every=None, fraction_to_update=None
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None, train_A_given_AO=False, train_O_given_A=True, train_O_given_prev_O=False
    ),
    perturbation_cfg=None,
    debug=None,
)

# configs = [g2, g2_ar, g2_ei]
configs = [g2]
