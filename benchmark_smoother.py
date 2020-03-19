import numpy as np
from l96 import l96_rk4_stepV as dyn_model
from ensemble_kalman_schemes import analyze_ensemble
from ensemble_kalman_schemes import enks, etks 
import pickle
import copy
import ipdb

########################################################################################################################
# functionalized experiment, will generate ensemble random seed for the initialization by the ensemble number and
# save the data to the associated ensemble number


def experiment(args):
    ####################################################################################################################
    # Define experiment parameters

    [time_series, method, seed, obs_un, obs_dim, N_ens, infl] = args

    # load the timeseries and associated parameters
    f = open(time_series, 'rb')
    tmp = pickle.load(f)
    f.close()
    [f, sys_dim, h, spin, tanl] = tmp['params']
    f_steps = int(tanl / h)

    # define the dynamic model parameters
    params = [h, f]

    # number of analyses
    nanl = 1000

    # set seed 
    np.random.seed(seed)
    
    # define the initial ensembles
    obs = tmp['obs']
    init = obs[:, 0]

    init = np.random.multivariate_normal(init, np.eye(sys_dim), size=N_ens).transpose()
    ens = copy.copy(init)

    # define the observation sequence
    obs = obs[:, 1:nanl + 1]
    [sys_dim, nanl] = np.shape(obs)
    truth = copy.copy(obs)
    obs = obs + np.sqrt(obs_un) * np.random.standard_normal([sys_dim, nanl])
    obs_cov = obs_un * np.eye(obs_dim)
    H = np.eye(sys_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl)
    anal_rmse = np.zeros(nanl)
    
    fore_spread = np.zeros(nanl)
    anal_spread = np.zeros(nanl)

    for i in range(nanl):
        for j in range(f_steps):
            # loop over the integration steps between observations
            ens = dyn_model(ens, params)

        # compute the forecast statistics
        fore_rmse[i], fore_spread[i] = analyze_ensemble(ens, truth[:, i])

        # after the forecast step, perform assimilation of the observation
        ens = method(ens, H, obs[:, i], obs_cov, inflation=infl)

        # compute the analysis statistics
        anal_rmse[i], anal_spread[i] = analyze_ensemble(ens, truth[:, i])

    data = {
            'fore_rmse': fore_rmse,
            'anal_rmse': anal_rmse,
            'fore_spread': fore_spread,
            'anal_spread': anal_spread,
            'params' : [seed, obs_un, obs_dim, N_ens, np.around(infl, 2), h]
            }

    fname = './data/' + method.__name__ + '_benchmark_seed_' + str(seed).zfill(2) + '_sys_dim_' + str(sys_dim) + \
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).zfill(3) + \
            '_nanl_' + str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + \
            '_N_ens_' + str(N_ens).zfill(3) + '_inflation_' + str(np.around(infl, 2)).zfill(2) + '.txt'

    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return(args)

########################################################################################################################

fname = './data/l96_ts_sys_dim_10_h_0.01_nanl_100000_anal_int_0.05_seed_0.txt'


# [time_series, seed, obs_un, obs_dim, N_ens, infl] = args
experiment([fname, etkf, 0, 1.0, 10, 10, 1.03])
