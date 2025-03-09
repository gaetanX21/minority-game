import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (3.5, 2.5) # in inches, single column

def compute_volatility(M: int) -> pd.DataFrame:
    df = pd.read_pickle(f'data/df_M{M}.pkl')
    out = df.groupby(["M", "N"])["Attendance"].apply(lambda x: (x**2).mean()).reset_index(name="sigma2")
    out["alpha"] = 2**out["M"] / out["N"]
    out["sigma2/N"] = out["sigma2"] / out["N"]
    return out

def scatter_volatility(outs: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=FIGSIZE, dpi=200)
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
    plt.show()

def compute_predictability_markovian(M: int, max_steps: int, iterations: int) -> pd.DataFrame:
    df = pd.read_pickle(f'data/df_markovian_M{M}_steps{max_steps}_it{iterations}.pkl')
    df["signA"] = np.sign(df["Attendance"])
    df["alpha"] = df["P"] / df["N"]
    out = df.groupby(["RunId", "alpha", "mu_idx"])["signA"].apply(lambda x: x.mean()**2).reset_index(name="E[(signA|mu)^2]")
    out = out.groupby(["alpha"])["E[(signA|mu)^2]"].mean().reset_index(name="predictability") # average over the iterations and the mu for each alpha
    return out

def scatter_predictibility(outs: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=FIGSIZE, dpi=200)
    plt.axhline(0, color='black', alpha=0.5)
    for M, out in outs.items():
        plt.scatter(out["alpha"], out["predictability"], label=f"$M={M}$", marker='.')
    plt.xscale("log")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\langle \text{sign}(A|\mu)^2 \rangle$")
    plt.xticks([0.1, 1, 10, 100], [0.1, 1, 10, 100])
    plt.legend()
    plt.show()