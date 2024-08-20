
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


def plot_loss(x_loss, loss):
    """"""

    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    colors = {
        "darkblue": (0., 0., 0.5),
        "Training": (0.05, 0.03, 0.53),
        "Pure": (0.99, 0.2, 0.36),
        "plasmagreen": (0.14, 0.92, 0.14),
        "plasmaorange": (0.97, 0.58, 0.25),
    }


    scaler = StandardScaler()
    x_loss_gp = x_loss[:420]
    loss_gp = loss[:420]
    X_normalized = scaler.fit_transform(x_loss_gp.reshape(-1, 1))
    # kernel = Matern(nu=1.5) + ConstantKernel(1.0) * RBF(length_scale=50.0)
    kernel = Matern(nu=1.5) + WhiteKernel(noise_level=1)
    gp_signal = GaussianProcessRegressor(kernel=kernel, alpha=0.6,
                                         n_restarts_optimizer=10)
    gp_signal.fit(X_normalized, loss_gp)
    y_pred, sigma = gp_signal.predict(X_normalized, return_std=True)

    axs.plot(x_loss_gp, y_pred, label="Normalized Loss (Predicted)", color=colors["Pure"], lw=2)
    axs.fill_between(
        x_loss_gp,
        y_pred - 1.96 * sigma,
        y_pred + 1.96 * sigma,
        color=colors["Pure"],
        alpha=0.5,
        label=r"95% Conf.",
    )
    axs.plot(x_loss, [0.] * x_loss.shape[-1], "--", color="k", alpha=0.5)

    axs.plot(x_loss, loss, ".", label=None, alpha=0.12, color=colors["Training"])
    y_smoothed = gaussian_filter1d(loss, sigma=3)
    axs.plot(x_loss, y_smoothed, "-", label="Normalized Loss (Smoothed)", color=colors["Training"],
             lw=2)

    for a in [axs]:
        a.set_xlabel("Training Steps [ ]", fontsize=18)
        a.set_ylabel("Prediction Loss [a.u.]", fontsize=18)
        a.legend(loc="upper right", fontsize=14)
        # a.set_xlim(x_min - x_max * 0.05, x_max * 1.3)
        a.tick_params(axis='both', which='major', labelsize=18)

    fig.tight_layout()
    fig.savefig(f"loss_gp.pdf", dpi=300)
    # plt.show()


def main():
    file_path = 'wandb_export_2024-03-30T03 43 00.509-07 00.csv'
    loss = pd.read_csv(file_path, usecols=[1]).squeeze().to_numpy()
    x_loss = pd.read_csv(file_path, usecols=[0]).squeeze().to_numpy()
    plot_loss(x_loss, loss)


if __name__ == "__main__":
    main()
