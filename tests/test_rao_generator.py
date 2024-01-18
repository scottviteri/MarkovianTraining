from src.rao_generator import RaoGenerator
from src.rao_tools import RaoConfig
import json

# Load the sweep configuration
with open('sweep_config.json') as f:
    sweep_config = json.load(f)

# Extract the parameters
parameters = sweep_config["parameters"]

# Create the RaoConfig object
cfg = RaoConfig(
    model_name=parameters["model_name"]["values"][0],
    lr=parameters["lr"]["values"][0],
    num_rao=parameters["num_rao"]["values"][0],
    batch_size=parameters["batch_size"]["values"][0],
    num_batches=parameters["num_batches"]["values"][0],
    obs_between_weight_updates=parameters["obs_between_weight_updates"]["values"][0],
    obs_to_action_ratio=parameters["obs_to_action_ratio"]["values"][0],
    interval_save_weights=parameters["interval_save_weights"]["values"][0],
    interval_print=parameters["interval_print"]["values"][0],
    wandb=parameters["wandb"]["values"][0],
    load_model=parameters["load_model"]["values"][0],
    do_lora=parameters["do_lora"]["values"][0],
    use_loss_difference=parameters["use_loss_difference"]["values"][0],
    training_ctxt_size=parameters["training_ctxt_size"]["values"][0],
)

rao_generator = RaoGenerator(cfg)  # Assuming RaoGenerator is the class containing the intersperse_lists method

def test_dataloader_starts_with_observation():
    rao_generator = RaoGenerator(cfg)  
    dataset = rao_generator.dataset
    for _ in range(10):
        data = next(dataset)
        for i in range(data.shape[1]):
            decoded_data = cfg.tokenizer.batch_decode(data[:, i, :])
            for batch_string in decoded_data:
                assert batch_string.startswith("\nObservation:")

test_dataloader_starts_with_observation()
