from src.training_types import *
import os

g2_p2 = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=1,
        num_batches=100,
        obs_to_action_ratio=0.5,
        interval_save_weights=3000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(
              name="arithmetic_explanations.jsonl", 
              task=None, peek_every=2),
        training_type=EI(
                prev_action=False, prev_observation=False, action=False, num_samples=3, 
                reinforce=False, rf_baseline=False, autoregressive=False, 
                markovian=True),
        debug=None
)

gj_p2 = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=100,
        obs_to_action_ratio=0.5,
        interval_save_weights=3000,
        interval_print=5,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(
              name="arithmetic_explanations.jsonl", 
              task=None, peek_every=2),
        training_type=EI(
                prev_action=False, prev_observation=False, action=False, num_samples=3, 
                reinforce=False, rf_baseline=False, autoregressive=False, 
                markovian=True),
        debug=None
)


def gen_eval(model_name, num_evals, wandb, use_gptj):
        # for GptEval, only model_name,  num_evals are used
        return InitialConfig(
                        model_name=model_name,
                        lr=1e-3,
                        batch_size=2,
                        num_batches=1000,
                        obs_to_action_ratio=2,
                        interval_save_weights=1000,
                        interval_print=2,
                        wandb=wandb,
                        load_model=False,
                        do_lora=True,
                        training_ctxt_size=300,
                        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
                        training_type=GptEval(num_evals=num_evals, use_gptj=use_gptj),
                        debug=None                        
        )

def test_debug_template(debug_type):
    return InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=10,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=True,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=False),
        debug=debug_type
    ) 

debug_types = [
        RepeatNPoints(num_points=1) , RepeatNPoints(num_points=2), 
        RepeatPointNTimes(num_times=1), RepeatPointNTimes(num_times=2), 
        ReplaceWithRandomTokens(), NoWeightUpdates()
]

#example_configs = [test_debug_template(x) for x in debug_types]
#example_configs =  [gen_eval("gptj", 10, False, use_gptj=True)]
#example_configs = [gpt2_arith_AO_local]#, gpt2_wiki_AO_local]
#example_configs = [gpt2_arith_EI_local]
example_configs = [gj_p2, g2_p2]