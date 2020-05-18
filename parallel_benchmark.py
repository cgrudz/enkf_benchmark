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
    from filter_exps import filter_param 

## [time_series, method, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
exps = []
time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt'
methods = ['enkf', 'etkf']
seed = 0
obs_un = 1.0
obs_dim = 40
N_ens = np.arange(14,42)
inflation = np.linspace(1.0, 1.2, 21)
wlks = [0.0000, 0.0100, 0.0010, 0.0001]

for meth in methods:
    for wlk in wlks:
        for N in N_ens:
            for inf in inflation:
                exps.append([time_series, meth, seed, obs_un, obs_dim, N, infl, 1.0])

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
