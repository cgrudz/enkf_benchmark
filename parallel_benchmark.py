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
    from benchmark_ensemble_methods import experiment

exps = []

fnames = sorted(glob.glob('./data/l96_ts*'))
seed = 0
obs_dim = 10
obs_un = [1.0]
N_ens = np.arange(5,41)
inflation = np.linspace(1.0, 1.3, 31)

for name in fnames:
    for N in N_ens:
        for infl in inflation:
            for r in obs_un:
                exps.append([name, seed, r, 10, N, infl])

# run the experiments given the parameters and write to text files, in parallel over the initializations

completed = dview.map_sync(experiment, exps)

print(completed)

sys.exit()


########################################################################################################################
