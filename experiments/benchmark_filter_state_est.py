import numpy as np
from l96 import rk4_step as step_model, l96 as dx_dt
from ensemble_kalman_schemes import analyze_ensemble
from ensemble_kalman_schemes import ensemble_filter 
import pickle
import copy
import sys
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
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01
    
    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # define kwargs for lag-1 smoothing
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model, 
              'dx_params': [f],
              'h': h,
              'diffusion': diffusion,
              'shift': 1,
              'mda': False
             }

    # number of analyses
    nanl = 100

    # set seed 
    np.random.seed(seed)
    
    # define the initial ensembles, squeezing the sys_dim times 1 array from the timeseries generation
    obs = np.squeeze(tmp['obs'])
    init = obs[:, 0]
    sys_dim = len(init)
    ens = np.random.multivariate_normal(init, np.eye(sys_dim), size=N_ens).transpose()

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 1]
    [sys_dim, nanl] = np.shape(obs)
    truth = copy.copy(obs)
    
    # define the observation operator here, change if needed for different configurations
    H = np.eye(obs_dim, sys_dim)
    obs = H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl])
    
    # define the associated time invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl)
    anal_rmse = np.zeros(nanl)
    
    fore_spread = np.zeros(nanl)
    anal_spread = np.zeros(nanl)

    for i in range(nanl):
        # copy the initial ensemble for lag-1, shift-1 smoothing
        ens_0 = copy.copy(ens)
        kwargs['ens_0'] = ens_0

        for j in range(f_steps):
            # loop over the integration steps between observations
            ens = step_model(ens, **kwargs)

        # compute the forecast statistics
        fore_rmse[i], fore_spread[i] = analyze_ensemble(ens, truth[:, i])

        # after the forecast step, perform assimilation of the observation
        analysis = ensemble_filter(method, ens, H, obs[:, [i]], obs_cov, infl, **kwargs)
        ens = analysis['ens']

        # compute the analysis statistics
        anal_rmse[i], anal_spread[i] = analyze_ensemble(ens, truth[:, i])

    data = {
            'fore_rmse': fore_rmse,
            'anal_rmse': anal_rmse,
            'fore_spread': fore_spread,
            'anal_spread': anal_spread,
            'method':method,
            'seed' : seed, 
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'nanl': nanl,
            'tanl':tanl,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(infl, 2)
            }
    
    fname = './data/' + method + '/' + method + '_filter_l96_state_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) +\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + '_nanl_' + str(nanl).zfill(3) +\
            '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_inflation_' + str(np.around(infl, 2)).ljust(4, '0') + '.txt'

    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return(args)

########################################################################################################################

## SINGLE EXPERIMENT DEBUGGING
fname = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt'


# [time_series, analysis, seed, obs_un, obs_dim, N_ens, infl] = args
experiment([fname, 'enkf', 0, 1.0, 40, 41, 1.04])


### FUNCTIONALIZED EXPERIMENT CALL OVER PARAMETER MAP
#j = int(sys.argv[1])
#f = open('./data/input_data/benchmark_filter_state.txt', 'rb')
#data = pickle.load(f)
#args = data[j]
#f.close()
#
#experiment(args)


