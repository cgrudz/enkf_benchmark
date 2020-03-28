import os
import sys
import numpy as np
import glob
import pickle
from ensemble_kalman_schemes import ensemble_filter 
import ipdb

########################################################################################################################


exps = []

## Arguments of experiment are as follows
## [time_series, method, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args 

#fnames = sorted(glob.glob('./data/timeseries_obs/*'))
fnames = ['./data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt',
          './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt']

analysis = ['enkf', 'etkf']
seed = 0
obs_un = 1.0
obs_dim = 40
param_err = 0.2
param_wlk = [0, 0.01]
N_ens = range(14, 42, 3)
state_inflation = np.linspace(1.0, 1.2, 21)
#param_inflation = np.linspace(1.0, 1.1, 11)
p_infl = 1.0

for name in fnames:
    for anal in analysis:
        for ens in N_ens:
            for wlk in param_wlk:
                for s_infl in state_inflation:
                    exps.append([name, anal, seed, obs_un, obs_dim, param_err, wlk, ens, s_infl, p_infl])

f = open('./data/input_data/benchmark_filter_param.txt','wb')
pickle.dump(exps, f)
f.close()


# submit the experiments given the parameters and write to text files over the initializations
#for j in range(len(exps)):
for j in range(len(exps)):
    f = open('./submit_job.sl', 'w')
    f.writelines('#!/bin/bash\n')
    f.writelines('#SBATCH -n 1\n')
    f.writelines('#SBATCH -t 0-40:00\n')
    f.writelines('#SBATCH --mem-per-cpu=3500M\n')
    f.writelines('#SBATCH -o filter.out\n')
    f.writelines('#SBATCH -e filter.err\n')
    f.writelines('#SBATCH --account=cpu-s2-mathstat_trial-0\n')
    f.writelines('#SBATCH --partition=cpu-s2-core-0\n')
    f.writelines('python benchmark_filter_param_est.py ' + str(j))

    f.close()

    os.system('sbatch  submit_job.sl')

