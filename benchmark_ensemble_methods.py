import numpy as np
from l96 import l96_rk4_stepV
from ensemble_kalman_schemes import analyze_ensemble, enkf_stoch_analysis, enkf_deter_analysis, ienkf_analysis, ietlm_analysis 
import pickle
import copy
import ipdb

########################################################################################################################
# functionalized experiment, will generate ensemble random seed for the initialization by the ensemble number and
# save the data to the associated ensemble number


def experiment(args):
    ####################################################################################################################
    # Define experiment parameters

    [seed, obs_un, obs_dim, N_ens] = args

    # load the timeseries and associated parameters
    f = open('./data/l96_ts_sys_dim_10_h_0.01_nanl_10000_anal_int_0.1_seed_0.txt', 'rb')
    tmp = pickle.load(f)
    f.close()
    [f, sys_dim, h, spin, tanl] = tmp['params']
    f_steps = int(tanl / h)

    # number of analyses
    nanl = 1000 

    # set seed 
    np.random.seed(seed)
    
    # define the initial ensembles
    obs = tmp['obs']
    init = obs[:, 0]

    init = np.random.multivariate_normal(init, np.eye(sys_dim), size=N_ens).transpose()
    X_stoch = copy.copy(init)
    X_deter = copy.copy(init)
    X_ienkf = copy.copy(init)
    X_ietlm = copy.copy(init)


    # define the observation sequence
    obs = obs[:, 1:nanl + 1]
    [sys_dim, nanl] = np.shape(obs)
    truth = copy.copy(obs)
    obs = obs + np.sqrt(obs_un) * np.random.standard_normal([sys_dim, nanl])
    obs_cov = obs_un * np.eye(obs_dim)
    H = np.eye(sys_dim)

    # create storage for the forecast and analysis statistics
    stoch_fore_rmse = np.zeros(nanl)
    stoch_anal_rmse = np.zeros(nanl)
    stoch_fore_spread = np.zeros(nanl)
    stoch_anal_spread = np.zeros(nanl)

    deter_fore_rmse = np.zeros(nanl)
    deter_anal_rmse = np.zeros(nanl)
    deter_fore_spread = np.zeros(nanl)
    deter_anal_spread = np.zeros(nanl)
    
    ienkf_fore_rmse = np.zeros(nanl)
    ienkf_anal_rmse = np.zeros(nanl)
    ienkf_fore_spread = np.zeros(nanl)
    ienkf_anal_spread = np.zeros(nanl)
    
    ietlm_fore_rmse = np.zeros(nanl)
    ietlm_anal_rmse = np.zeros(nanl)
    ietlm_fore_spread = np.zeros(nanl)
    ietlm_anal_spread = np.zeros(nanl)
    
    for i in range(nanl):
        # loop over the analysis cycles
        
        # keep the initial conditions of the smoothers for later, as this is only to generate
        # forecat statistics
        X_ienkf_0 = copy.copy(X_ienkf)
        X_ietlm_0 = copy.copy(X_ietlm)

        for j in range(f_steps):
            # loop over the integration steps between observations
            X_stoch = l96_rk4_stepV(X_stoch, h, f)
            X_deter = l96_rk4_stepV(X_deter, h, f)
            X_ienkf = l96_rk4_stepV(X_ienkf, h, f)
            X_ietlm = l96_rk4_stepV(X_ietlm, h, f)

        # compute the forecast statistics
        stoch_fore_rmse[i], stoch_fore_spread[i] = analyze_ensemble(X_stoch, truth[:, i])
        deter_fore_rmse[i], deter_fore_spread[i] = analyze_ensemble(X_deter, truth[:, i])
        ienkf_fore_rmse[i], ienkf_fore_spread[i] = analyze_ensemble(X_ienkf, truth[:, i])
        ietlm_fore_rmse[i], ietlm_fore_spread[i] = analyze_ensemble(X_ietlm, truth[:, i])

        # after the forecast step, perform assimilation of the observation
        X_stoch = enkf_stoch_analysis(X_stoch, H, obs[:, i], obs_cov)
        X_deter = enkf_deter_analysis(X_deter, H, obs[:, i], obs_cov)
        
        X_ienkf = ienkf_analysis(X_ienkf_0, H, obs[:, i], obs_cov, f_steps, f, h)
        X_ietlm = ietlm_analysis(X_ietlm_0, H, obs[:, i], obs_cov, f_steps, f, h)

        # compute the analysis statistics
        stoch_anal_rmse[i], stoch_anal_spread[i] = analyze_ensemble(X_stoch, truth[:, i])
        deter_anal_rmse[i], deter_anal_spread[i] = analyze_ensemble(X_deter, truth[:, i])
        ienkf_anal_rmse[i], ienkf_anal_spread[i] = analyze_ensemble(X_ienkf, truth[:, i])
        ietlm_anal_rmse[i], ietlm_anal_spread[i] = analyze_ensemble(X_ietlm, truth[:, i])

    data = {
            'stoch_fore_rmse': stoch_fore_rmse,
            'stoch_anal_rmse': stoch_anal_rmse,
            'stoch_fore_spread': stoch_fore_spread,
            'stoch_anal_spread': stoch_anal_spread,
            'deter_fore_rmse': deter_fore_rmse,
            'deter_anal_rmse': deter_anal_rmse,
            'deter_fore_spread': deter_fore_spread,
            'deter_anal_spread': deter_anal_spread,
            'ienkf_fore_rmse': ienkf_fore_rmse,
            'ienkf_anal_rmse': ienkf_anal_rmse,
            'ienkf_fore_spread': ienkf_fore_spread,
            'ienkf_anal_spread': ienkf_anal_spread,
            'ietlm_fore_rmse': ietlm_fore_rmse,
            'ietlm_anal_rmse': ietlm_anal_rmse,
            'ietlm_fore_spread': ietlm_fore_spread,
            'ietlm_anal_spread': ietlm_anal_spread,
            'params' : [obs_un, obs_dim, N_ens, h]
            }

    fname = './data/enkf_benchmark_seed_' + str(seed).zfill(2) + '_nanl_' + str(nanl).zfill(3) + \
            '_tanl_' + str(tanl).zfill(3) + '_obs_un_' + str(obs_un).zfill(3) + \
            '_N_ens_' + str(N_ens).zfill(3) + '.txt'

    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

########################################################################################################################

experiment([0, .25, 10, 100])
