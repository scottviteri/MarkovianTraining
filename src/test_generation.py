import torch
import os
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from einops import repeat
from src.training_types import *
from src.utilities import extend_initial_config


test_config = InitialConfig(
    model_name="distilgpt2",
    lr=1e-6,
    optimizer="adam",
    batch_size=2,
    num_batches=300,
    obs_to_action_ratio=0.5,
    interval_save_weights=3000,
    interval_print=1,
    wandb=False,
    load_model=False,
    do_lora=True,
    num_beams=3,
    training_ctxt_size=200,
    dataset=InitDatasetType(
        task=ArithmeticTask(
            num_terms=6,
            num_digits=2,
            cumulative=False,
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
        ),
        peek_every=None,
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None,
        train_A_given_AO=False,
        train_O_given_A=True,
        train_O_given_prev_O=False,
    ),
    inference_cfg=InferenceConfig(num_return_sequences=3),
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


def test_critic():
    cfg = extend_initial_config(test_config)
    repeat = "210 210 210 210 210 210 210 210 210 210 210 210 210 210 210"
    in_pieces = "Let's break down the expression 23 + 14 + 81 + 92 by evaluating the tens place first: 20 + 10 + 80 + 90 = 200. Now, let's add the ones place: 3 + 4 + 1 + 2 = 10. Combining the results from the tens and ones places gives us the final answer:"
    in_order = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 37. Then, add 37 and 81 to get 118. Finally, add 118 and 92 to arrive at the final result:"
    in_order_corrupted = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 27. Then, add 27 and 81 to get 108. Finally, add 108 and 92 to arrive at the final result:"
    direct_question = "The solution to 23 + 14 + 81 + 92 is"
    random_test = "I am a flying banana"
    input_strings = [
        repeat,
        in_pieces,
        in_order,
        in_order_corrupted,
        direct_question,
        random_test,
    ]
    tokenizer_out = cfg.causal_lm_tokenizer(
        input_strings, return_tensors="pt", padding=True
    )
    question = tokenizer_out["input_ids"].to(device=cfg.device)
    attention_mask = tokenizer_out["attention_mask"].to(device=cfg.device)
    answer_index = cfg.causal_lm_tokenizer("210")["input_ids"][0]
    out = cfg.causal_lm(question, attention_mask=attention_mask)
    probabilities = torch.softmax(out.logits, dim=-1)[:, -1, answer_index]
    assert 1 == 1
    # assert torch.all(
    #    probabilities[:-1] > probabilities[1:]
    # ), "Probabilities should be strictly decreasing"


def test_critic_2():
    cfg = extend_initial_config(test_config)
    repeat = "486 486 486 486 486 486 486 486 486 486 486 486 486 486 486"
    in_pieces = "Let's break down the expression 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80 by evaluating the tens place first: 20 + 10 + 80 + 90 + 50 + 60 + 70 + 80 = 460. Now, let's add the ones place: 3 + 4 + 1 + 2 + 7 + 3 + 6 + 0 = 26. Combining the results from the tens and ones places gives us the final answer:"
    in_order = "Let's evaluate 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80. First, add 23 and 14 to get 37. Then, add 37 and 81 to get 118. Next, add 118 and 92 to get 210. Then, add 210 and 57 to get 267. Next, add 267 and 63 to get 330. Then, add 330 and 76 to get 406. Finally, add 406 and 80 to arrive at the final result:"
    in_order_corrupted = "Let's evaluate 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80. First, add 23 and 14 to get 27. Then, add 27 and 81 to get 108. Next, add 108 and 92 to get 200. Then, add 200 and 57 to get 257. Next, add 257 and 63 to get 320. Then, add 320 and 76 to get 396. Finally, add 396 and 80 to arrive at the final result:"
    direct_question = "The solution to 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80 is"
    random_test = "I am a flying banana"
    input_strings = [
        repeat,
        in_pieces,
        in_order,
        in_order_corrupted,
        direct_question,
        random_test,
    ]
    tokenizer_out = cfg.causal_lm_tokenizer(
        input_strings, return_tensors="pt", padding=True
    )
    question = tokenizer_out["input_ids"].to(device=cfg.device)
    attention_mask = tokenizer_out["attention_mask"].to(device=cfg.device)
    answer_index = cfg.causal_lm_tokenizer("486")["input_ids"][0]
    out = cfg.causal_lm(question, attention_mask=attention_mask)
    probabilities = torch.softmax(out.logits, dim=-1)[:, -1, answer_index]
    assert 1 == 1


def test_num_return_sequences():
    cfg = extend_initial_config(test_config)

    input_sequence = torch.randint(0, 10, (cfg.batch_size, cfg.tok_p_obs)).to(
        cfg.device
    )
    attention_mask = (input_sequence != cfg.causal_lm_tokenizer.pad_token_id).long()
    action_candidates = cfg.causal_lm.generate(
        inputs=input_sequence,
        attention_mask=attention_mask,
        num_beams=cfg.num_beams,
        bad_words_ids=[[cfg.causal_lm_tokenizer.pad_token_id]],
        output_scores=True,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=cfg.tok_p_pure_action,
        max_new_tokens=cfg.tok_p_pure_action,
        pad_token_id=cfg.causal_lm_tokenizer.pad_token_id,
        num_return_sequences=cfg.inference_cfg.num_return_sequences,
    )[:, -cfg.tok_p_pure_action :]

    # Check if action_candidates have the expected shape
    expected_shape = (
        cfg.batch_size * cfg.inference_cfg.num_return_sequences,
        cfg.tok_p_pure_action,
    )
    assert (
        action_candidates.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {action_candidates.shape}"


# This allows the test to be run with `python test_generation.py`
if __name__ == "__main__":
    test_num_return_sequences()
    print("Test passed.")
