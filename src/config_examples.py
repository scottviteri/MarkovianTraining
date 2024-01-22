from src.types_and_utilities import InitialConfig, InitTrainingType, AR, GptEval, AO, AOA, RAOInit

gpt2_AOA = InitialConfig(
        model_name="distilgpt2",
        lr=1e-4,
        batch_size=1,
        num_batches=100,
        obs_to_action_ratio=1,
        interval_save_weights=1000,
        interval_print=1,
        wandb=False,
        load_model=False,
        do_lora=False,
        training_ctxt_size=300,
        dataset_name="wikipedia",
        task_name=None,
        training_type=AOA(use_gumbel=False)
)

gpt2_RAO = InitialConfig(
                model_name="distilgpt2",
                lr=1e-3,
                batch_size=2,
                num_batches=5,
                obs_to_action_ratio=2,
                interval_save_weights=2,
                interval_print=2,
                wandb=False,
                load_model=False,
                do_lora=True,
                training_ctxt_size=300,
                dataset_name="wikipedia",
                task_name=None,
                training_type=RAOInit(
                        num_rao=3,
                        obs_between_weight_updates=4,
                        use_loss_difference=True,
                        use_multirao_for_action_gen=False,
                        use_rewards_to_go=True
                )
)

# for GptEval, only model_name,  num_evals are used
gpt2_eval = InitialConfig(
                model_name="distilgpt2",
                lr=1e-3,
                batch_size=2,
                num_batches=1000,
                obs_to_action_ratio=2,
                interval_save_weights=2,
                interval_print=2,
                wandb=False,
                load_model=False,
                do_lora=True,
                training_ctxt_size=300,
                dataset_name="wikipedia",
                task_name=None,
                training_type=GptEval(num_evals=10)
)

gptj_eval = InitialConfig(
                model_name="gptj",
                lr=1e-3,
                batch_size=2,
                num_batches=100,
                obs_to_action_ratio=2,
                interval_save_weights=2,
                interval_print=2,
                wandb=False,
                load_model=False,
                do_lora=True,
                training_ctxt_size=300,
                dataset_name="wikipedia",
                task_name=None,
                training_type=GptEval(num_evals=100)
)


example_configs =  [gptj_eval]
#[gpt2_eval]  #[gpt2_RAO, gpt2_AOA]

