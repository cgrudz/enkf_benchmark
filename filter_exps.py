import numpy as np
import copy
import sys
import ipdb
from common import picopen, picwrite
from methods.l96 import rk4_step as step_model, l96 as dx_dt
from methods.ensemble_kalman_schemes import analyze_ensemble, analyze_ensemble_parameters
from methods.ensemble_kalman_schemes import ensemble_filter 

########################################################################################################################
# Main filtering experiments, debugged and validated for use with schemes in methods directory
########################################################################################################################


def filter_state(args):
    # Define experiment parameters
    [time_series, method, seed, obs_un, obs_dim, N_ens, infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
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
    nanl = 450

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

    picwrite(data, fname)
    return(args)

########################################################################################################################


def filter_param(args):
    # Define experiment parameters
    [time_series, method, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01

    # unpack the observations and the initial true state of the dynamic variables
    obs = np.squeeze(tmp['obs'])
    init = obs[:, 0]
    
    # define the state dynamic state dimension and the extended state parameters to be estimated
    state_dim = len(init) 
    param_truth = np.array([f])
    sys_dim = state_dim + len(param_truth)

    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # define kwargs, note the possible exclusion of dx_params if this is the only parameter for
    # dx_dt and this is the parameter to be estimated
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model,
              'h': h,
              'diffusion': diffusion,
              'state_dim': state_dim,
              'param_infl': param_infl
             }
    
    # number of analyses
    nanl = 450

    # set seed 
    np.random.seed(seed)
    
    # define the initial ensembles
    ens = np.random.multivariate_normal(init, np.eye(state_dim), size=N_ens).transpose()
    
    if len(param_truth) > 1:
        param_ens = np.random.multivariate_normal(np.squeeze(param_truth), np.diag(param_truth * param_err)**2, size=N_ens)
    else:
        param_ens = np.reshape(np.random.normal(np.squeeze(param_truth), scale=np.squeeze(param_truth)*param_err, size=N_ens), [1, N_ens])
    
    # defined the extended state ensemble
    ens = np.concatenate([ens, param_ens], axis=0)

    # define the observation sequence where we project the true state into the observation space and
    # perturb by white-in-time-and-space noise with standard deviation obs_un
    obs = obs[:, 1:nanl + 1]
    truth = copy.copy(obs)
    
    # define the observation operator for the dynamic state variables
    H = np.eye(obs_dim, state_dim)
    obs =  H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl])
    
    # define the observation operator on the extended state, used for the ensemble
    H_ens = np.eye(obs_dim, sys_dim)

    # define the associated time invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # create storage for the forecast and analysis statistics
    state_fore_rmse = np.zeros(nanl)
    state_anal_rmse = np.zeros(nanl)
    param_anal_rmse = np.zeros(nanl)
    
    state_fore_spread = np.zeros(nanl)
    state_anal_spread = np.zeros(nanl)
    param_anal_spread = np.zeros(nanl)

    for i in range(nanl):
        # copy the initial ensemble for lag-1 smoothing
        ens_0 = copy.copy(ens)
        kwargs['ens_0'] = ens_0

        for j in range(f_steps):
            # loop over the integration steps between observations
            ens = step_model(ens, **kwargs)

        # compute the forecast statistics
        state_fore_rmse[i], state_fore_spread[i] = analyze_ensemble(ens[:state_dim, :], truth[:, i])

        # after the forecast step, perform assimilation of the observation
        analysis = ensemble_filter(method, ens, H_ens, obs[:, [i]], obs_cov, state_infl, **kwargs)
        ens = analysis['ens']

        # compute the analysis statistics
        state_anal_rmse[i], state_anal_spread[i] = analyze_ensemble(ens[:state_dim, :], truth[:, i])
        param_anal_rmse[i], param_anal_spread[i] = analyze_ensemble_parameters(ens[state_dim:, :], param_truth)

        # include random walk for the ensemble of parameters
        param_ens = param_ens + param_wlk * np.random.standard_normal(np.shape(param_ens))
        ens[state_dim:, :] = param_ens

    data = {
            'state_fore_rmse': state_fore_rmse,
            'state_anal_rmse': state_anal_rmse,
            'param_anal_rmse': param_anal_rmse,
            'state_fore_spread': state_fore_spread,
            'state_anal_spread': state_anal_spread,
            'param_anal_spread': param_anal_spread,
            'method': method,
            'seed' : seed, 
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'state_dim': state_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'param_err': param_err,
            'param_wlk': param_wlk,
            'nanl': nanl,
            'tanl': tanl,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(state_infl, 2),
            'param_infl': np.around(param_infl, 2)
            }
    
    fname = './data/' + method + '/' + method + '_filter_l96_param_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) + '_state_dim_' + str(state_dim)+\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + '_param_err_' + str(param_err).ljust(4, '0') +\
            '_param_wlk_' + str(param_wlk).ljust(4, '0') +\
            '_nanl_' + str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_N_ens_' + str(N_ens).zfill(3) + '_state_inflation_' + str(np.around(state_infl, 2)).ljust(4, '0') +\
            '_param_infl_' + str(np.around(param_infl, 2)).ljust(4, '0') + '.txt'

    picwrite(data, fname)
    return(args)

########################################################################################################################
