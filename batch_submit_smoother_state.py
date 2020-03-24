import os
import sys
import numpy as np
import glob
import pickle
from ensemble_kalman_schemes import enks
import ipdb

########################################################################################################################

exps = []

## Arguments of experiment are as follows
## [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args

#fnames = sorted(glob.glob('./data/timeseries_obs/*'))
fnames = ['./data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt']#,
         # './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt']

analysis = [etks]
seed = 0
lag = range(1, 52, 5)
# shift will be varied between = lag or = 0
obs_un = 1.0
obs_dim = 40
N_ens = range(14, 42)
inflation = np.linspace(1.0, 1.2, 21)


for name in fnames:
    for anal in analysis:
        for l in lag:
            for i in range(2):
                for ens in N_ens:
                    for infl in inflation:
                        if i == 0:
                            exps.append([name, anal, seed, l, l, obs_un, obs_dim, ens, infl])

                        else:
                            if l != 1:
                                exps.append([name, anal, seed, l, 1, obs_un, obs_dim, ens, infl])

f = open('./data/input_data/benchmark_smoother_state.txt','wb')
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
    f.writelines('python benchmark_smoother_state_est.py ' + str(j))

    f.close()

    os.system('sbatch  submit_job.sl')

