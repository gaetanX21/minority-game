import models
import mesa
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-M", type=int, required=True)
args = parser.parse_args()


def get_n_range(M, num, alpha_min, alpha_max):
    N_min, N_max = 2**M / alpha_max, 2**M / alpha_min
    N_min = max(11, N_min)
    n_range = np.logspace(np.log10(N_min), np.log10(N_max), num=num, dtype=int)
    n_range = 2*np.floor(n_range/2).astype(int)+1
    n_range = np.unique(n_range)
    return n_range


alpha_min = 0.1
alpha_max = 100
num = 100
max_steps = 20_000
iterations = 1


def main():
    M = args.M
    print(f"M: {args.M}")
    n_list = get_n_range(M, num, alpha_min, alpha_max)
    params = {"N": n_list, "M": M, "S": 2}
    results = mesa.batch_run(
        models.ClassicMinorityGame,
        parameters=params,
        iterations=iterations,
        max_steps=max_steps,
        number_processes=None, # use all available cores
        data_collection_period=1,
        display_progress=True,
    )
    df = pd.DataFrame(results)
    df.to_pickle(f"data/df_single_long_M{M}.pkl")
    print(f"Results saved to data/df_single_long_M{M}.pkl")

if __name__ == "__main__":
    main()
