import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import utils # custom utils for manipulating data and plotting

sns.set_theme() # for prettier plots

# ensure we're using the correct version of Python
import sys
assert sys.version.startswith('3.12.7')

FOLDER = "plots/" # folder where the data is stored
DPI = 200 # dpi for the plots
plt.style.use(['science', 'no-latex'])


#### Dynamics ####
window = 100 # window for rolling mean
couples = [(187,6), (639,6), (31,10)] # (N, M) couples to plot
for N, M in couples:
    df = pd.read_pickle(f"data/df_M{M}.pkl")
    df = df[(df["N"] == N) & (df["iteration"]==0)]
    fig = utils.plot_full_attendance(df, window)
    fig.savefig(FOLDER + f"M{M}_N{N}.png", dpi=DPI)


#### Volatility ####
M_list = [6,7,8]
outs = {}
for M in M_list:
    outs[M] = utils.compute_volatility(M)
fname = FOLDER + f"volatility.png"
utils.scatter_volatility(outs, fname)

#### Predictability ####
M_list = [6,7,8] # M=9,10 not computed because they take too long too simulate
max_steps = 10_000
iterations = 10
outs = {}
for M in M_list:
    outs[M] = utils.compute_predictability_markovian(M, max_steps, iterations)
fname = FOLDER + f"predictability.png"
utils.scatter_predictibility(outs, fname)