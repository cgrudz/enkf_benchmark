import os
import sys
import numpy as np
import glob
import pickle
import ipdb

########################################################################################################################

exps = []

## Arguments of experiment are as follows
## [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args

#fnames = sorted(glob.glob('./data/timeseries_obs/*'))
fnames = ['./data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt']#,
         # './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt']

analysis = ['etks', 'enks']
seed = 0
lag = range(1, 52, 5)
obs_un = 1.0
obs_dim = 40
N_ens = range(14, 42)
inflation = np.linspace(1.0, 1.2, 21)
param_err = 0.03
param_wlk = [0.0100, 0.0010, 0.0001, 0.0000]

for name in fnames:
    for wlk in param_wlk:
        for anal in analysis:
            for l in lag:
                for ens in N_ens:
                    for infl in inflation:
                        exps.append([name, anal, seed, l, 1, obs_un, obs_dim, param_err, wlk,  ens, infl, 1.0])

f = open('./data/input_data/benchmark_smoother_param.txt','wb')
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
    f.writelines('#SBATCH -o smoother.out\n')
    f.writelines('#SBATCH -e smoother.err\n')
    f.writelines('#SBATCH --account=cpu-s2-mathstat_trial-0\n')
    f.writelines('#SBATCH --partition=cpu-s2-core-0\n')
    f.writelines('python batch_experiment_driver.py ' + str(j))
    f.close()

    os.system('sbatch  submit_job.sl')

