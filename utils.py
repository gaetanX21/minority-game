import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

FIGSIZE = (6, 4) # in inches, single column
DPI = 200
# Use PGF backend for LaTeX styling
plt.style.use(['science', 'no-latex'])  # Use 'science' with or without LaTeX


def plot_attendance(df: pd.DataFrame, window: int=None, ax=None) -> None:
    A = df["Attendance"]
    N = df["N"].iloc[0]
    x = df["Step"]
    y = A/np.sqrt(N)
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(x, y, alpha=0.5)
    ax.axhline(0, color="k", alpha=0.5)
    if window is not None: # we overlay std bands
        std = (y**2).rolling(window).mean() # rolling variance
        ax.fill_between(x, -std, +std, alpha=0.5, color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$A/\sqrt{N}$")
    return ax


def plot_var_attendance(df: pd.DataFrame, window: int=100, ax=None) -> None:
    A = df["Attendance"]
    N = df["N"].iloc[0]
    x = df["Step"]
    A2 = (A**2).rolling(window).mean() # rolling variance
    var = (A**2).mean() # variance of the attendance over the whole run
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(x, A2/N)
    ax.axhline(var/N, linestyle='--')
    ax.axhline(1, color='black', alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\langle A^2 \rangle / N$")
    return ax


def plot_full_attendance(df: pd.DataFrame, window: int=100) -> None:
    figsize = (1*FIGSIZE[0], 2*FIGSIZE[1])
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=DPI)
    ax[0] = plot_attendance(df, window=window, ax=ax[0])
    ax[1] = plot_var_attendance(df, window=window, ax=ax[1])
    fig.tight_layout()
    return fig


# def compute_volatility(M: int) -> pd.DataFrame:
#     df = pd.read_pickle(f'data/df_M{M}.pkl')
#     df = df[df["Step"]>1500]
#     out = df.groupby(["M", "N", "iteration"])["Attendance"].apply(lambda x: (x**2).mean()).reset_index(name="sigma2")
#     out = out.groupby(["M", "N"])["sigma2"].mean().reset_index(name="sigma2") # average over the iterations
#     out["alpha"] = 2**out["M"] / out["N"]
#     out["sigma2/N"] = out["sigma2"] / out["N"]
#     return out


def compute_volatility(M: int) -> pd.DataFrame:
    df = pd.read_pickle(f'data/df_markovian_M{M}_steps10000_it10.pkl')
    df = df[df["Step"]>8000]
    out = df.groupby(["P", "N", "iteration"])["Attendance"].apply(lambda x: (x**2).mean()).reset_index(name="sigma2")
    out = out.groupby(["P", "N"])["sigma2"].mean().reset_index(name="sigma2") # average over the iterations
    out["alpha"] = out["P"] / out["N"]
    out["sigma2/N"] = out["sigma2"] / out["N"]
    return out


def scatter_volatility(outs: dict[str, pd.DataFrame], fname: str=None) -> None:
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.axhline(1, color='black', alpha=0.5)
    for M, out in outs.items():
        plt.scatter(out["alpha"], out["sigma2/N"], label=f"$M={M}$", marker='.')
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\sigma^2/N$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.1, 1, 10, 100], [0.1, 1, 10, 100])
    plt.yticks([0.1, 1, 10], [0.1, 1, 10])
    plt.legend()
    if fname is not None:
        plt.savefig(fname, dpi=DPI)
    else:
        plt.show()


def compute_predictability_markovian(M: int, max_steps: int, iterations: int) -> pd.DataFrame:
    df = pd.read_pickle(f'data/df_markovian_M{M}_steps{max_steps}_it{iterations}.pkl')
    df["signA"] = np.sign(df["Attendance"])
    df["alpha"] = df["P"] / df["N"]
    out = df.groupby(["RunId", "alpha", "mu_idx"])["signA"].apply(lambda x: x.mean()**2).reset_index(name="E[(signA|mu)^2]")
    out = out.groupby(["alpha"])["E[(signA|mu)^2]"].mean().reset_index(name="predictability") # average over the iterations and the mu for each alpha
    return out


def scatter_predictibility(outs: dict[str, pd.DataFrame], fname: str=None) -> None:
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    plt.axhline(0, color='black', alpha=0.5)
    for M, out in outs.items():
        plt.scatter(out["alpha"], out["predictability"], label=f"$M={M}$", marker='.')
    plt.xscale("log")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\theta^2$")
    plt.xticks([0.1, 1, 10, 100], [0.1, 1, 10, 100])
    plt.legend()
    if fname is not None:
        plt.savefig(fname, dpi=DPI)
    else:
        plt.show()