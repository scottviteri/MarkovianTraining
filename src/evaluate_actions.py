import torch
import torch.distributed as dist
import json
import matplotlib.pyplot as plt
import copy
import numpy as np
import tqdm
import random
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from utilities import extend_initial_config, predict_observation
from config_examples import configs, lma
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
    action_out[:, offset + int((1.0 - frac_spaces) * (action.shape[-1] - offset)):] = (
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
            "30%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.30,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "20%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.25,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "10%Rand": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.20,
                frac_of_tokens_to_pad=0.0,
                p_digit_change=0.0,
            ),
            "80%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.8,
                p_digit_change=0.0,
            ),
            "65%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.65,
                p_digit_change=0.0,
            ),
            "50%Spaces": PerturbationConfig(
                eval_every=self.eval_every,
                frac_of_tokens_to_randomize=0.0,
                frac_of_tokens_to_pad=0.5,
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

colors = {
    "darkblue": (0., 0., 0.5),
    "Training": (0.05, 0.03, 0.53),
    "Pure": (0.99, 0.2, 0.36),
    "plasmagreen": (0.14, 0.92, 0.14),
    "plasmaorange": (0.97, 0.58, 0.25),
}

def plot_results(results, model_name, train_model, file_name, x_max_data=None):
    """"""

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 5))
    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 5))
    fig4, axs4 = plt.subplots(1, 1, figsize=(8, 5))

    # fig.suptitle(f"Eval: {model_name}, Train: {train_model}")
    # fig2.suptitle(f"Eval: {model_name}, Train: {train_model}")
    # fig3.suptitle(f"Eval: {model_name}, Train: {train_model}")
    x_max = 0.0
    x_min = -1.0

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
            axs.plot(x, y, ".", label=None, alpha=0.12,
                     color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp))
            y_smoothed = gaussian_filter1d(y, sigma=3)
            axs.plot(x, y_smoothed, "-", label=keys,
                     color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp), lw=2)
            f_sp = f_sp * 1.45
        elif "Digit" in keys:
            axs2.plot(x, y, ".", label=None, alpha=0.12,
                      color=(0.3 * f_dig, 0.3 * f_dig, 0.3 * f_dig))
            y_smoothed = gaussian_filter1d(y, sigma=3)
            axs2.plot(x, y_smoothed, "-", label=keys,
                      color=(0.3 * f_dig, 0.3 * f_dig, 0.3 * f_dig), lw=2)
            f_dig = f_dig * 1.45
        elif "Rand" in keys:
            axs3.plot(x, y, ".", label=None, alpha=0.12,
                      color=(0.3 * f_rand, 0.3 * f_rand, 0.3 * f_rand))
            y_smoothed = gaussian_filter1d(y, sigma=3)
            axs3.plot(x, y_smoothed, "-", label=keys,
                      color=(0.3 * f_rand, 0.3 * f_rand, 0.3 * f_rand), lw=2)
            f_rand = f_rand * 1.45
        elif "Pure" in keys:
            axs.plot(x, y, ".", label=None, alpha=0.12, color=colors[keys])
            y_smoothed = gaussian_filter1d(y, sigma=3)
            axs.plot(x, y_smoothed, "-", label="Unperturbed", color=colors[keys], lw=2)
            axs2.plot(x, y, ".", label=None, alpha=0.12, color=colors[keys])
            axs2.plot(x, y_smoothed, "-", label="Unperturbed", color=colors[keys], lw=2)
            axs3.plot(x, y, ".", label=None, alpha=0.12, color=colors[keys])
            axs3.plot(x, y_smoothed, "-", label="Unperturbed", color=colors[keys], lw=2)

        else:
            y_smoothed = gaussian_filter1d(y, sigma=3)
            axs4.plot(x, y, ".", label=None, alpha=0.12, color=colors[keys])
            axs4.plot(x, y_smoothed, "-", label=keys, color=colors[keys], lw=2)

    for a in [axs, axs2, axs3, axs4]:
        a.set_xlabel("Training Steps [ ]", fontsize=18)
        a.set_ylabel("Prediction Loss [a.u.]", fontsize=18)
        a.legend(loc="upper right")
        a.set_xlim(x_min - x_max * 0.05, x_max * 1.3)
        a.tick_params(axis='both', which='major', labelsize=18)

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig.savefig(f"{file_name[:-5]}_1.pdf", dpi=300)
    fig2.savefig(f"{file_name[:-5]}_2.pdf", dpi=300)
    fig3.savefig(f"{file_name[:-5]}_3.pdf", dpi=300)
    fig4.savefig(f"{file_name[:-5]}_4.pdf", dpi=300)


def plot_result_differences(results, model_name, train_model, file_name, tar_key, x_max_data=None):
    """"""

    val_keys = ["Digit", "Spaces"]
    assert tar_key in val_keys, f"{tar_key} not in {val_keys}"

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    fig2, axs2 = plt.subplots(1, 1, figsize=(8, 5))
    fig3, axs3 = plt.subplots(1, 1, figsize=(8, 5))
    x_max = 0.0
    x_min = -1.0

    f_sp = 1.
    f_dig = 1.
    f_rand = 1.

    # do max x val for Pure
    x_pure, y_pure = np.array(results["Pure"]).T
    x_max = np.max([x_max, np.max(x_pure)])
    x_min = np.min([x_min, np.min(y_pure)])

    a_list = [axs, axs2, axs3]

    for keys in results:
        if "Pure" in keys:
            continue
        if tar_key in keys:
            if x_max_data:
                mask = np.array(results[keys])[:, 0] <= x_max_data
                x, y = np.array(results[keys])[mask].T
                x_max = x_max_data
            else:
                x, y = np.array(results[keys]).T
                x_max = np.max([x_max, np.max(x)])
            x_min = np.min([x_min, np.min(x)])

            a = a_list.pop()

            scaler = StandardScaler()
            x_loss_gp = x_pure
            loss_gp = y_pure - y
            X_normalized = scaler.fit_transform(x_loss_gp.reshape(-1, 1))
            # kernel = Matern(nu=1.5) + ConstantKernel(1.0) * RBF(length_scale=50.0)
            kernel = Matern(nu=1.5) + WhiteKernel(noise_level=2., noise_level_bounds=[0.7, 5.])
            gp_signal = GaussianProcessRegressor(kernel=kernel, alpha=0.7,
                                                 n_restarts_optimizer=10)
            gp_signal.fit(X_normalized, loss_gp)
            y_pred, sigma = gp_signal.predict(X_normalized, return_std=True)

            a.plot(x_loss_gp, y_pred, label=f"Unperturbed - {keys} (Predicted)",
                     color=colors["Pure"], lw=2)
            a.fill_between(
                x_loss_gp,
                y_pred - 1.96 * sigma,
                y_pred + 1.96 * sigma,
                color=colors["Pure"],
                alpha=0.5,
                label=r"95% Conf.",
            )
            a.plot(x_pure, [0.] * x_pure.shape[-1], "--", color="k", alpha=0.5)

            a.plot(x_pure, y_pure - y, ".", label=None, alpha=0.12, color=colors["Training"])
                     # color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp))
            y_smoothed = gaussian_filter1d(y_pure - y, sigma=3)
            a.plot(x, y_smoothed, "-", label=f"Unperturbed - {keys} (Smoothed)", color=colors["Training"])
                     # color=(0.3 * f_sp, 0.3 * f_sp, 0.3 * f_sp), lw=2)
            # f_sp = f_sp * 1.45


    for a in [axs, axs2, axs3]:
        a.set_xlabel("Training Steps [ ]", fontsize=18)
        a.set_ylabel("Loss Difference [a.u.]", fontsize=18)
        a.legend(loc="upper right", prop={'size': 12})
        a.set_xlim(x_min - x_max * 0.05, x_max * 1.1)
        a.tick_params(axis='both', which='major', labelsize=18)

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig.savefig(f"{file_name[:-5]}_diff{tar_key}_1.pdf", dpi=300)
    fig2.savefig(f"{file_name[:-5]}_diff{tar_key}_2.pdf", dpi=300)
    fig3.savefig(f"{file_name[:-5]}_diff{tar_key}_3.pdf", dpi=300)

def main():
    re_evaluate = False
    do_diffs = True
    if re_evaluate:
        # file_name = "gpt2_traj_1709608868.json"
        file_name = "mistral_traj_20240329_234001.json"

        # Set model
        # init_cfg = configs[0]
        init_cfg = lma
        print(init_cfg.model_name)
        assert init_cfg.model_name == "llama"  # data["model"]
        # init_cfg.use_mac = False  # data["model"]
        assert init_cfg.perturbation_cfg is None

        cfg = extend_initial_config(init_cfg)

        eval_class = ActionEvaluator(configs=[cfg], f_name=file_name, n_step=1)
        res = eval_class.evaluate()
        print(res)
        plot_results(
            res[cfg.model_name],
            model_name="mistral7b",
            train_model="mistral7b",
            file_name=file_name,
        )

        with open('result.json', 'w') as fp:
            json.dump(res, fp)

    elif not do_diffs:
        file_name = "mistral_traj_20240329_234001.json"
        with open('result_mistralfinal.json', 'r') as fp:
            res = json.load(fp)

        plot_results(
            res["mistral"],
            model_name="mistral7b",
            train_model="mistral7b",
            file_name=file_name,
        )
    else:
        file_name = "mistral_traj_20240329_234001.json"
        with open('result_mistralfinal.json', 'r') as fp:
            res = json.load(fp)

        plot_result_differences(
            res["mistral"],
            model_name="mistral7b",
            train_model="mistral7b",
            file_name=file_name,
            tar_key="Spaces",
        )
        plot_result_differences(
            res["mistral"],
            model_name="mistral7b",
            train_model="mistral7b",
            file_name=file_name,
            tar_key="Digit",
        )


if __name__ == "__main__":
    main()
