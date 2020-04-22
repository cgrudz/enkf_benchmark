from ipyparallel import Client
import sys
import numpy as np
import glob
import pickle

########################################################################################################################
# set up parallel client

rc = Client()
dview = rc[:]

with dview.sync_imports():
    from smoother import classic_state

# [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
exps = []
time_series = './data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt'
methods = ['enks', 'etks']
seed = 0
lag = range(1, 52, 5)
shift = 1
obs_un = 1.0
obs_dim = 40
N_ens = np.arange(14,42)
inflation = np.linspace(1.0, 1.2, 21)

for meth in method:
    for l in lag:
        for N in N_ens:
            for inf in inflation:
                exps.append([time_series, meth, seed, l, shift, obs_un, obs_dim, N, infl])

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
