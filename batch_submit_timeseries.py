import os
import sys
import numpy as np
import glob
import pickle
from l96 import rk4_step, l96s_tay2_step
import ipdb

########################################################################################################################


exps = []
# [seed, N_ens, forward_step, h, diffusion, sys_dim, nanl, spin, anal_int]
seed = 0
N_ens = 1
sys_dim = 40
nanl = 50000
spin = 2500
anal_int = [0.05, 0.1]

diffusion = [0.1, 0.25, 0.5]
exps.append([seed, 1, rk4_step, 0.01, 0, sys_dim, nanl, spin, anal_int[0]])
exps.append([seed, 1, rk4_step, 0.01, 0, sys_dim, nanl, spin, anal_int[1]])


for diff in diffusion:
    for anal in anal_int:
        exps.append([seed, 1, l96s_tay2_step, 0.005, diff, sys_dim, nanl, spin, anal])


f = open('./data/input_data_generate_timeseries.txt','wb')
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
    f.writelines('#SBATCH -o time_series.out\n')
    f.writelines('#SBATCH -e time_series.err\n')
    f.writelines('#SBATCH --account=cpu-s2-mathstat_trial-0\n')
    f.writelines('#SBATCH --partition=cpu-s2-core-0\n')
    f.writelines('python generate_l96_timeseries.py ' + str(j))

    f.close()
   
    os.system('sbatch  submit_job.sl')


########################################################################################################################
