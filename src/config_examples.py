from src.training_types import *
import os

gpt2_AO = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=1,
        num_batches=10,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=None
)

gpt2_bb_AO = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=10,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="bigbench", task=None, peek_every=5),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=None
)



gj_OA_wk_20k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=20000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=False),
        debug=None
)

gj_O_wk_20k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=20000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)

gj_O_arith_20k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=20000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset=InitDatasetType(name="arithmetic_explanations.jsonl", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)

mst_O_wk_20k = InitialConfig(
        model_name="mistral",
        lr=1e-3,
        batch_size=1,
        num_batches=20000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset = InitDatasetType(name="wikipedia", task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)

gpt2_arith_AO_local = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=15,
        obs_to_action_ratio=1,
        interval_save_weights=3000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(
              name=os.getcwd()+"/arithmetic_explanations.jsonl", 
              task=None, peek_every=1),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=None
)

gpt2_arith_EI_local = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=15,
        obs_to_action_ratio=1,
        interval_save_weights=3000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(
              name=os.getcwd()+"/arithmetic_explanations.jsonl", 
              task=None, peek_every=1),
        training_type=EI(ignore_first_action=True, ignore_second_action=True, num_samples=3),
        debug=None
)


gpt2_wiki_AO_local = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=10,
        obs_to_action_ratio=1,
        interval_save_weights=3000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(
              name="wikipedia",
              task=None, peek_every=None),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=None
)


gpt2_arith_O = InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=16,
        num_batches=400,
        obs_to_action_ratio=0.5,
        interval_save_weights=3000,
        interval_print=100,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(name=os.getcwd()+"/src/arithmetic_explanations.jsonl", task=None, peek_every=1),
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=NoWeightUpdates()
)

gptj_arith_O = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=10,
        obs_to_action_ratio=0.5,
        interval_save_weights=3000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=125,
        dataset=InitDatasetType(name=os.getcwd()+"/src/arithmetic_explanations.jsonl", task=None, peek_every=1),
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=False),
        debug=None
)


# should only use model name, lr, observation_size, and wandb
gpt2_arith_AR =  InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=1,
        num_batches=10,
        obs_to_action_ratio=15.0/40.0,
        interval_save_weights=10000,
        interval_print=500,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=0, #not used
        dataset=InitDatasetType(name="arithmetic_explanations.jsonl", task=None, peek_every=None),
        training_type=AR(),
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
example_configs = [gpt2_arith_EI_local]
#example_configs = [gpt2_AO]
#example_configs = [gj_OA_wk_20k]
#example_configs = [gj_O_wk_20k]
#example_configs = [mst_O_wk_20k]
#example_configs = [gj_AR_20k]
#example_configs = [gpt2_arith_O_local]