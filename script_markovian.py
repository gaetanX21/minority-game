import models
import mesa
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-M", type=int, required=True)
parser.add_argument("--max_steps", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True)
args = parser.parse_args()


def get_n_range(M, num, alpha_min, alpha_max):
    """
    Get a range of odd N values for a given M, such that the range is logarithmically spaced
    and yields alpha values from alpha_min to alpha_max.
    """
    N_min, N_max = 2**M / alpha_max, 2**M / alpha_min
    N_min = max(11, N_min)
    n_range = np.logspace(np.log10(N_min), np.log10(N_max), num=num, dtype=int)
    n_range = 2*np.floor(n_range/2).astype(int)+1
    n_range = np.unique(n_range)
    return n_range


alpha_min = 0.1
alpha_max = 100
num = 60 # because we have 20 CPU cores


def main():
    M = args.M
    max_steps = args.max_steps
    iterations = args.iterations
    print(f"M: {args.M}, max_steps: {max_steps}, iterations: {iterations}")
    n_list = get_n_range(M, num, alpha_min, alpha_max)
    P = 2**M
    params = {"N": n_list, "P": P}
    results = mesa.batch_run(
        models.MarkovianMinorityGame,
        parameters=params,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=None, # use all available cores
        data_collection_period=1,
        display_progress=True,
    )
    df = pd.DataFrame(results)
    fname = f"data/df_markovian_M{M}_steps{max_steps}_it{iterations}.pkl"
    df.to_pickle(fname)
    print(f"Results saved to {fname}")

if __name__ == "__main__":
    main()
