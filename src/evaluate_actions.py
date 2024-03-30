import torch
import torch.distributed as dist
import json
import matplotlib.pyplot as plt
import copy
import numpy as np
import tqdm
import random
from scipy.ndimage.filters import gaussian_filter1d

from utilities import extend_initial_config, predict_observation
from config_examples import configs
from training_types import PerturbationConfig


def perturb_action(action, cfg, perturbation_cfg):
    action_out = copy.deepcopy(action).to(cfg.device)
    offset = cfg.prefix_tensors.action_prefix_tensor.shape[-1]
    # PERTURBATION 1
    # Given n <= cfg.pure_ctxt_sizes.action_size, change token through randomization
    frac_randomize = perturbation_cfg.frac_of_tokens_to_randomize
    assert 1.0 >= frac_randomize >= 0.0, f"frac_randomize is {frac_randomize}"
    perturb_target_inds = torch.randint(
        low=offset,
        high=action.shape[-1],
        size=[int(frac_randomize * (action.shape[-1] - offset))],
    )
    action_out[:, perturb_target_inds] = torch.randint(
        low=0,
        high=cfg.tokenizer.vocab_size,
        size=[int(frac_randomize * (action.shape[-1] - offset))],
    ).to(cfg.device)

    # PERTURBATION 2
    # Given a fraction of cfg.pure_ctxt_sizes.action_size, replace with spaces/padding
    frac_spaces = perturbation_cfg.frac_of_tokens_to_pad
    assert 1.0 >= frac_spaces >= 0.0, f"frac_randomize is {frac_spaces}"
    token_id_space = cfg.tokenizer.encode(" ")[-1]
    action_out[:, offset + int((1.0 - frac_spaces) * (action.shape[-1] - offset)) :] = (
        token_id_space
    )

    # PERTURBATION 3
    # For probability p_digit_change, flip each individual digit
    p_digit_change = perturbation_cfg.p_digit_change
    assert 1.0 >= p_digit_change >= 0.0, f"p_digit_change is {p_digit_change}"
    digit_tokens = cfg.tokenizer("0123456789", add_special_tokens=False)["input_ids"]
    for i in range(action_out.shape[0]):  # Loop over batch dimension
        for j in range(action_out.shape[1]):  # Loop over sequence dimension
            if action_out[i, j] in digit_tokens:
                if random.random() < p_digit_change:
                    action_out[i, j] = torch.tensor(
                        random.choice(digit_tokens), device=cfg.device
                    )

    # p_digit_change = perturbation_cfg.p_digit_change
    # assert 1.0 >= p_digit_change >= 0.0, f"p_digit_change is {p_digit_change}"
    # digit_tokens = cfg.tokenizer("0123456789", add_special_tokens=False)["input_ids"]
    ## Create a mask for digit tokens
    # is_digit_mask = torch.isin(
    #    action_out, torch.tensor(digit_tokens, device=cfg.device)
    # )
    ## Generate a random mask for digit tokens to be changed
    # change_mask = torch.rand_like(action_out, dtype=torch.float32) < p_digit_change
    ## Combine the masks to get the final mask for digit tokens to be changed
    # change_digit_mask = is_digit_mask & change_mask
    ## Generate random digit tokens for the selected positions
    # random_digits = torch.tensor(
    #    random.choices(digit_tokens, k=change_digit_mask.sum().item()),
    #    device=cfg.device,
    # )
    ## Update the action_out tensor with the random digit tokens
    # action_out[change_digit_mask] = random_digits
    # new = []
    # for act in action_out_detok:
    #    new_act = randomize_numbers_with_probability(act, p_digit_change)
    #    new.append(cfg.tokenizer.encode(new_act))
    # action_out = torch.tensor(new).to(cfg.device)

    return action_out


def perturb_action_str(action, cfg, perturbation_cfg):
    action_out = copy.deepcopy(action)
    offset = len(cfg.tokenizer.decode(cfg.prefix_tensors.action_prefix_tensor[0]))

    # PERTURBATION 1
    # Given n <= cfg.pure_ctxt_sizes.action_size, change token through randomization
    frac_randomize = perturbation_cfg.frac_of_tokens_to_randomize
    assert 1.0 >= frac_randomize >= 0.0, f"frac_randomize is {frac_randomize}"
    char_lib = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "+", " ", "-", "*", "/", ",", ".", "(", ")",
    ]
    new = []
    for i, c in enumerate(action_out):
        # Should be equivalent (randomizing every char with prob p instead of randomizing fraction p of chars)
        if random.random() < frac_randomize:
            new.append(random.choice(char_lib))
        else:
            new.append(c)
    action_out = "".join(new)

    # PERTURBATION 2
    # Given a fraction of cfg.pure_ctxt_sizes.action_size, replace with spaces/padding
    frac_spaces = perturbation_cfg.frac_of_tokens_to_pad
    assert 1.0 >= frac_spaces >= 0.0, f"frac_randomize is {frac_spaces}"
    new = []
    for i, c in enumerate(action_out):
        if i > offset + int((1.0 - frac_spaces) * (len(action_out) - offset)):
            new.append(" ")
        else:
            new.append(c)
    action_out = "".join(new)

    # PERTURBATION 3
    # For probability p_digit_change, flip each individual digit
    p_digit_change = perturbation_cfg.p_digit_change
    assert 1.0 >= p_digit_change >= 0.0, f"p_digit_change is {p_digit_change}"
    new = []
    for act in action_out:
        new_act = randomize_numbers_with_probability(act, p_digit_change)
        new.append(new_act)
    action_out = "".join(new)

    return action_out


def randomize_numbers_with_probability(input_string, probability=0.5):
    output_string = ""
    for char in input_string:
        if char.isdigit():
            if random.random() < probability:
                output_string += str(random.randint(0, 9))
            else:
                output_string += char
        else:
            output_string += char
    return output_string


class ActionEvaluator:
    """Class to evaluate training trajectories (json files)"""

    def __init__(
            self,
            f_name: str,
            configs: list,
            perturbations: dict = None,
            n_max: int = None,
            n_step: int = None,
            path_to_dir: str = "saved_weights_and_losses/",
    ):
        self._f_name = f_name
        self._configs = configs
        self._n_max = n_max
        self._n_step = n_step
        self._file_path = path_to_dir + self._f_name
        self._load_traj_data()
        if perturbations is None:
            self._set_default_perturbations()
        else:
            self._perts = perturbations

    def _load_traj_data(self):
        """Load trajectory and other data from json file."""

        with open(self._file_path, "r") as file:
            self._data = json.load(file)

    def _set_default_perturbations(self):
        """"""

        self.eval_every = 1
        self._perts = {
            "15%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.15,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "10%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.10,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "5%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.05,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "75%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.75,
                p_digit_change=0.0,
            ),
            "50%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.5,
                p_digit_change=0.0,
            ),
            "25%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.25,
                p_digit_change=0.0,
            ),
            "50%Digits": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.5,
            ),
            "25%Digits": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.25,
            ),
            "10%Digits": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.1,
            ),
        }

    def evaluate(self):
        """"""

        if self._n_max is None:
            self._n_max = len(self._data["trajectory"])
        if self._n_step is None:
            self._n_step = 1

        eval_results = {}
        for cfg in self._configs:
            tok = cfg.tokenizer
            mod = cfg.causal_lm
            mod.eval()

            eval_loss = {"Training": [], "Pure": []}
            for keys in self._perts:
                eval_loss[keys] = []

            for step_i, step in enumerate(tqdm.tqdm(self._data["trajectory"])):
                if step_i >= self._n_max:
                    break
                if step_i % self._n_step != 0:
                    continue

                action_str = step["action"]
                obs_str = step["obs"]

                # Do perturbations here
                actions_tok_pert = {}
                actions_str_pert = {}
                for key, perturbation_cfg in self._perts.items():
                    action_str_pert = [perturb_action_str(a, cfg, perturbation_cfg) for
                                       a in action_str]
                    actions_str_pert[key] = action_str_pert

                    # Tokenize
                    action_tok_pert = tok(
                        action_str_pert,
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding=True,
                    )["input_ids"].to(cfg.device)
                    actions_tok_pert[key] = action_tok_pert

                action_tok = tok(
                    action_str,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )["input_ids"].to(cfg.device)

                obs_tok = tok(
                    obs_str,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                )["input_ids"].to(cfg.device)

                with torch.no_grad():
                    unperturbed_loss = (
                        predict_observation(cfg, action_tok, obs_tok, add_q_head=False)
                        .mean()
                        .item()
                    )
                    eval_loss["Pure"].append([step_i, unperturbed_loss])
                    eval_loss["Training"].append(
                        [step_i, torch.tensor(step["observation_losses"]).mean().item()]
                    )

                    for key, act in actions_tok_pert.items():
                        perturbed_loss = (
                            predict_observation(cfg, act, obs_tok, add_q_head=False)
                            .mean()
                            .item()
                        )
                        eval_loss[key].append([step_i, perturbed_loss])

            eval_results[cfg.model_name] = eval_loss

        return eval_results

    @staticmethod
    def plot_results(results, model_name, train_model, file_name, x_max_data=None):
        """"""

        fig, axs = plt.subplots(1, 1, figsize=(9, 4))
        fig2, axs2 = plt.subplots(1, 1, figsize=(9, 4))
        fig3, axs3 = plt.subplots(1, 1, figsize=(9, 4))

        # fig.suptitle(f"Eval: {model_name}, Train: {train_model}")
        # fig2.suptitle(f"Eval: {model_name}, Train: {train_model}")
        # fig3.suptitle(f"Eval: {model_name}, Train: {train_model}")
        x_max = 0.0
        x_min = -1.0

        colors = {
            "darkblue": (0., 0., 0.5),
            "Training": (0.05, 0.03, 0.53),
            "Pure": (0.99, 0.2, 0.36),
            "plasmagreen": (0.14, 0.92, 0.14),
            "plasmaorange": (0.97, 0.58, 0.25),
        }
        f_sp = 1.
        f_dig = 1.
        f_rand = 1.

        for keys in results:
            if x_max_data:
                mask = np.array(results[keys])[:, 0] <= x_max_data
                x, y = np.array(results[keys])[mask].T
                x_max = x_max_data
            else:
                x, y = np.array(results[keys]).T
                x_max = np.max([x_max, np.max(x)])
            x_min = np.min([x_min, np.min(x)])

            if "Spaces" in keys:
                axs.plot(x, y, ".", label=None, alpha=0.15,
                         color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp))
                y_smoothed = gaussian_filter1d(y, sigma=2)
                axs.plot(x, y_smoothed, "-", label=keys,
                         color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp))
                f_sp = f_sp * 1.5
            elif "Digit" in keys:
                axs2.plot(x, y, ".", label=None, alpha=0.15,
                          color=(0.3 * f_dig, 0.3 * f_dig, 0.3 * f_dig))
                y_smoothed = gaussian_filter1d(y, sigma=2)
                axs2.plot(x, y_smoothed, "-", label=keys,
                          color=(0.3 * f_dig, 0.3 * f_dig, 0.3 * f_dig))
                f_dig = f_dig * 1.6
            elif "Rand" in keys:
                axs3.plot(x, y, ".", label=None, alpha=0.15,
                          color=(0.3 * f_rand, 0.3 * f_rand, 0.3 * f_rand))
                y_smoothed = gaussian_filter1d(y, sigma=2)
                axs3.plot(x, y_smoothed, "-", label=keys,
                          color=(0.3 * f_rand, 0.3 * f_rand, 0.3 * f_rand))
                f_rand = f_rand * 1.6
            else:  # "Training and Pure"
                axs.plot(x, y, ".", label=None, alpha=0.2, color=colors[keys])
                y_smoothed = gaussian_filter1d(y, sigma=2)
                axs.plot(x, y_smoothed, "-", label=keys, color=colors[keys])
                axs2.plot(x, y, ".", label=None, alpha=0.2, color=colors[keys])
                axs2.plot(x, y_smoothed, "-", label=keys, color=colors[keys])
                axs3.plot(x, y, ".", label=None, alpha=0.2, color=colors[keys])
                axs3.plot(x, y_smoothed, "-", label=keys, color=colors[keys])

        for a in [axs, axs2, axs3]:
            a.set_xlabel("Training Steps [ ]")
            a.set_ylabel("Average Prediction Loss [a.u.]")
            a.legend(loc="upper right")
            a.set_xlim(x_min - x_max * 0.05, x_max * 1.3)

        # plt.tight_layout()
        fig.savefig(f"{file_name[:-5]}_1.pdf", dpi=300)
        fig2.savefig(f"{file_name[:-5]}_2.pdf", dpi=300)
        fig3.savefig(f"{file_name[:-5]}_3.pdf", dpi=300)

    # @staticmethod
    # def plot_results(results, model_name, train_model, file_name):
    #    """"""

    #    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    #    fig.suptitle(f"Eval: {model_name}, Train: {train_model}")
    #    x_max = 0.0
    #    x_min = -1.0

    #    for keys in results:
    #        x, y = np.array(results[keys]).T
    #        x_max = np.max([x_max, np.max(x)])
    #        x_min = np.min([x_min, np.min(x)])

    #        if "Spaces" in keys:
    #            axs[0].plot(x, y, ".", label=keys)
    #        elif "Rand" in keys:
    #            axs[1].plot(x, y, ".", label=keys)
    #        elif "Training" in keys:
    #            axs[0].plot(x, y, "", label=keys)
    #            axs[1].plot(x, y, "", label=keys)
    #        else:
    #            # This case includes the "Pure" action results
    #            axs[0].plot(x, y, ".", label=keys)
    #            axs[1].plot(x, y, ".", label=keys)

    #    axs[0].set_title("Replacing with Spaces")
    #    axs[1].set_title("Swapping with Random Token")
    #    for i in [0, 1]:
    #        axs[i].set_xlabel("Training Steps [ ]")
    #        axs[i].set_ylabel("Average Prediction Loss [a.u.]")
    #        axs[i].legend(loc="upper right")
    #        axs[i].set_xlim(x_min - x_max * 0.1, x_max * 1.5)

    #    # plt.tight_layout()
    #    plt.savefig(f"results/{file_name[:-5]}.png")
    #    plt.show()


def main():
    # file_name = "gpt2_traj_1709608868.json"
    file_name = "mistral_traj_20240329_051532.json"

    # Set model
    init_cfg = configs[0]
    assert init_cfg.model_name == "mistral"  # data["model"]
    # init_cfg.use_mac = False  # data["model"]
    assert init_cfg.perturbation_cfg is None

    cfg = extend_initial_config(configs[0])

    eval_class = ActionEvaluator(configs=[cfg], f_name=file_name, n_step=50)
    res = eval_class.evaluate()
    print(res)
    eval_class.plot_results(
        res[cfg.model_name],
        model_name="mistral7b",
        train_model="mistral7b",
        file_name=file_name,
    )

    with open('result.json', 'w') as fp:
        json.dump(res, fp)


if __name__ == "__main__":
    main()
