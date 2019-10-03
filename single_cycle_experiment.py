import numpy as np
from ensemble_kalman_schemes import enkf_stoch_analysis, enkf_trans_analysis

Nx = 3
Ne = 8
Ny = 2

ens = np.random.multivariate_normal(np.ones(Nx), np.eye(Nx), Ne).transpose()

H = np.eye(Nx)[:Ny]

obs = np.random.multivariate_normal(np.ones(Nx), np.eye(Nx), 1).transpose()[:Ny]

# Make a random -- but SPD -- true obs_cov
L = np.random.multivariate_normal(np.zeros(Ny), np.eye(Ny), 60)
obs_cov = L.transpose() @ L / 60

# Should be equal
answer2 = enkf_stoch_analysis(ens, H, obs, obs_cov)
answer1 = enkf_trans_analysis(ens, H, obs, obs_cov)
