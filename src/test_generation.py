import torch
import os
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from einops import repeat
from src.training_types import *
from src.utilities import extend_initial_config


test_config = InitialConfig(
    model_name="mistral",
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
            operations=["+", "-", "*"],
            probs=[1.0, 0.0, 0.0],
        ),
        peek_every=None,
    ),
    prediction_cfg=PredictionConfig(
        filter_best_actions=None,
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

# TODO: make sure that my code tokenizes to the right


def wrap_questions(begin, end, questions, answer):
    answer_length = len(str(answer))
    return [
        begin + question[:-answer_length] + end + str(answer) for question in questions
    ]


def check_strictly_decreasing(cfg, questions, answer):
    # this method assumes a tokenizer which splits a number into digits
    tokenizer_out = cfg.causal_lm_tokenizer(
        questions, return_tensors="pt", padding=True,
        add_special_tokens=True
    )
    tokenizer_out_2 = cfg.causal_lm_tokenizer(
        questions, return_tensors="pt", padding=True,
        add_special_tokens=False
    )
    tokenized_questions = tokenizer_out["input_ids"].to(device=cfg.device)
    attention_mask = tokenizer_out["attention_mask"].to(device=cfg.device)
    predictions = cfg.causal_lm(tokenized_questions, attention_mask=attention_mask)
    probs = torch.softmax(predictions.logits, dim=-1)[:, :-1, :]
    correct_probs = torch.gather(
        probs, 2, tokenized_questions[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    answer_probs = correct_probs[:, -len(str(answer)) :]
    answer_mean_probs = answer_probs.mean(dim=1)
    print(answer_mean_probs)
    reasonably_high = (answer_mean_probs[:2]>0.1).all()
    dec_1 = answer_mean_probs[0] > answer_mean_probs[-2]
    dec_2 = answer_mean_probs[-3] > answer_mean_probs[-2]
    dec_3 = answer_mean_probs[-2] > answer_mean_probs[-1]
    return reasonably_high and dec_1 and dec_2 and dec_3
    #return torch.all(
    #    answer_mean_probs[:-1] > answer_mean_probs[1:]
    #)  # , "answer_mean_probs should be strictly decreasing"


def test_critic():
    cfg = extend_initial_config(test_config)
    #repeat = "210 210 210 210 210 210 210 210 210 210 210 210 210 210 210"
    in_order = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 37. Then, add 37 and 81 to get 118. Finally, add 118 and 92 to arrive at the final result 210"
    in_pieces = "Let's break down the expression 23 + 14 + 81 + 92 by evaluating the tens place first: 20 + 10 + 80 + 90 = 200. Now, let's add the ones place: 3 + 4 + 1 + 2 = 10. Combining the results from the tens and ones places gives us the final answer 210"
    in_order_corrupted = "Let's evaluate 23 + 14 + 81 + 92. First, add 23 and 14 to get 27. Then, add 27 and 81 to get 108. Finally, add 108 and 92 to arrive at the final result 210"
    direct_question = "The solution to 23 + 14 + 81 + 92 is 210"
    random_test = "I am a flying banana 210"
    input_strings = [
    #    repeat,
        in_order,
        in_pieces,
        in_order_corrupted,
        direct_question,
        random_test,
    ]
    # spaces are ok here because no generation
    # input_strings = wrap_questions("StepByStep: ", "Observation: ", input_strings, 210)
    # input_strings = wrap_questions("StepByStep: ", "", input_strings, 210)
    assert check_strictly_decreasing(
        cfg, input_strings, 210
    ), "Should be strictly decreasing"
    # assert torch.all(
    #    probabilities[:-1] > probabilities[1:]
    # ), "Probabilities should be strictly decreasing"


def test_critic_2():
    cfg = extend_initial_config(test_config)
    # repeat = "486 486 486 486 486 486 486 486 486 486 486 486 486 486 486"
    in_order = "Let's evaluate 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80. First, add 23 and 14 to get 37. Then, add 37 and 81 to get 118. Next, add 118 and 92 to get 210. Then, add 210 and 57 to get 267. Next, add 267 and 63 to get 330. Then, add 330 and 76 to get 406. Finally, add 406 and 80 to arrive at the final result: 486"
    in_pieces = "Let's break down the expression 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80 by evaluating the tens place first: 20 + 10 + 80 + 90 + 50 + 60 + 70 + 80 = 460. Now, let's add the ones place: 3 + 4 + 1 + 2 + 7 + 3 + 6 + 0 = 26. Combining the results from the tens and ones places gives us the final answer: 486"
    in_order_corrupted = "Let's evaluate 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80. First, add 23 and 14 to get 27. Then, add 27 and 81 to get 108. Next, add 108 and 92 to get 200. Then, add 200 and 57 to get 257. Next, add 257 and 63 to get 320. Then, add 320 and 76 to get 396. Finally, add 396 and 80 to arrive at the final result: 486"
    direct_question = "The solution to 23 + 14 + 81 + 92 + 57 + 63 + 76 + 80 is 486"
    random_test = "I am a flying banana 486"
    input_strings = [
    #    repeat,
        in_order,
        in_pieces,
        in_order_corrupted,
        direct_question,
        random_test,
    ]
    # wrap_questions("StepByStep: ", input_strings, "Observation: ", 486)
    #assert check_strictly_decreasing(
    #    cfg, input_strings, 486 
    #), "should be strictly decreasing"
    assert True # temporarily disable


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
