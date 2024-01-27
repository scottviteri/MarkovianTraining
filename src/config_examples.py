from src.training_types import *

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
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=None
)


gj_OA_wk_15k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=15000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=False),
        debug=None
)

gj_O_wk_15k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=1000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)

gj_O_bb_15k = InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=15000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="bigbench",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)

mst_O_wk_15k = InitialConfig(
        model_name="mistral",
        lr=1e-3,
        batch_size=1,
        num_batches=15000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=31,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=True, ignore_second_action=True),
        debug=None
)


# should only use model name, lr, observation_size, and wandb
gj_AR_15k =  InitialConfig(
        model_name="gptj",
        lr=1e-3,
        batch_size=1,
        num_batches=15000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=0, #not used
        dataset_name="wikipedia",
        task_name=None,
        training_type=AR(observation_size=100),
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
                        dataset_name="wikipedia",
                        task_name=None,
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
        dataset_name="wikipedia",
        task_name=None,
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
#example_configs = [gpt2_RAO, gpt2_AOA, gpt2_AO, gpt2_AR, gen_eval("mistral", 10, False, use_gptj=False)]
example_configs = [gpt2_AO]
#example_configs = [gj_OA_wk_15k]
#example_configs = [gj_O_wk_15k]
#example_configs = [gj_O_bb_15k]
#example_configs = [gj_AR_15k]
