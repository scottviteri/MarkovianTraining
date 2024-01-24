from src.training_types import *

gpt2_AOA = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=2,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=False),
        debug=RepeatNPoints(num_points=1)
)

gpt2_OA = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=2,
        num_batches=100,
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
        debug=RepeatNPoints(num_points=1)
)

gpt2_bb = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=2,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="bigbench",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=False),
        debug=None
)



phi2_AOA = InitialConfig(
        model_name="phi2",
        lr=1e-4,
        batch_size=16,
        num_batches=10000,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=False),
        debug=None
)


# only run this on large GPU
gpt2_AOA_gumbel = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=5,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=True,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=True, ignore_first_action=False, ignore_second_action=False),
        debug=None
)


gpt2_AO = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=2,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=10,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False, ignore_first_action=False, ignore_second_action=True),
        debug=RepeatNPoints(num_points=1)
)


gpt2_RAO_nr0_obwu4 = InitialConfig(
                model_name="distilgpt2",
                lr=1e-3,
                batch_size=2,
                num_batches=50,
                obs_to_action_ratio=1,
                interval_save_weights=1000,
                interval_print=10,
                wandb=False,
                load_model=False,
                do_lora=False,
                training_ctxt_size=300,
                dataset_name="wikipedia",
                task_name=None,
                training_type=RAOInit(
                        num_rao=0,
                        obs_between_weight_updates=4,
                        use_loss_difference=False,
                        use_multirao_for_action_gen=False,
                        use_rewards_to_go=False
                ),
                debug=RepeatNPoints(num_points=1)
)

gpt2_RAO_nr0_obwu0 = InitialConfig(
                model_name="distilgpt2",
                lr=1e-3,
                batch_size=2,
                num_batches=100,
                obs_to_action_ratio=1,
                interval_save_weights=100,
                interval_print=10,
                wandb=False,
                load_model=False,
                do_lora=False,
                training_ctxt_size=300,
                dataset_name="wikipedia",
                task_name=None,
                training_type=RAOInit(
                        num_rao=0,
                        obs_between_weight_updates=4,
                        use_loss_difference=False,
                        use_multirao_for_action_gen=False,
                        use_rewards_to_go=False
                ),
                debug=RepeatNPoints(num_points=1)
)


# should only use model name, lr, observation_size, and wandb
gpt2_AR =  InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=2,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=10000, #not used
        dataset_name="wikipedia",
        task_name=None,
        training_type=AR(observation_size=300),
        debug=RepeatNPoints(num_points=1)
)

def gen_eval(model_name, num_evals, wandb):
        # for GptEval, only model_name,  num_evals are used
        return InitialConfig(
                        model_name=model_name,
                        lr=1e-3,
                        batch_size=2,
                        num_batches=1000,
                        obs_to_action_ratio=2,
                        interval_save_weights=2,
                        interval_print=2,
                        wandb=wandb,
                        load_model=False,
                        do_lora=True,
                        training_ctxt_size=300,
                        dataset_name="wikipedia",
                        task_name=None,
                        training_type=GptEval(num_evals=num_evals),
                        debug=None                        
        )

def test_debug_template(debug_type):
    return InitialConfig(
        model_name="distilgpt2",
        lr=1e-3,
        batch_size=2,
        num_batches=10,
        obs_to_action_ratio=1,
        interval_save_weights=20,
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

#example_configs =  [gen_eval("mistral", 10, False)]
#example_configs = [gpt2_RAO, gpt2_AOA, gpt2_AO, gpt2_AR, gen_eval("mistral", 10, False)]
#example_configs = [gpt2_RAO]
#example_configs = [gpt2_AR]
example_configs = [gpt2_AO]
#example_configs = [gpt2_RAO_nr0_obwu0]
#gpt2_RAO_nr0_obwu4,