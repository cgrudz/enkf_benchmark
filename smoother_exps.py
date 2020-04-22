import numpy as np
import copy
import sys
import ipdb
from common import picopen, picwrite
from methods.l96 import rk4_step as step_model, l96 as dx_dt
from methods.ensemble_kalman_schemes import analyze_ensemble, analyze_ensemble_parameters
from methods.ensemble_kalman_schemes import lag_shift_smoother_classic, lag_shift_smoother_hybrid

########################################################################################################################
# Main smoothing experiments, debugged and validated for use with schemes in methods directory
########################################################################################################################
# All experiments are funcitonalized so that they can be called from an array of parameter values which will typically
# be varied with naive parallelism.  Relevant arguments and experimental results are dumped as a side effect to a
# dictionary in a pickle binary file.  Arguments are retuned from the function for keeping track of successul runs.
########################################################################################################################

def classic_state(args):
    # Define experiment parameters
    [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01
    
    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # define kwargs
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model, 
              'dx_params': [f],
              'h': h,
              'diffusion': diffusion,
              'shift': shift,
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
    # perturb by white-in-time-and-space noise with standard deviation obs_un  NOTE: we will include 
    # obs at time zero and pad the obs sequence with 2*lag to cut down and align all statistics in time
    obs = obs[:, :nanl + 2 * lag + 1]
    truth = copy.copy(obs)
    
    # define the observation operator here, change if needed for different configurations
    H = np.eye(obs_dim, sys_dim)
    obs = H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl + 2 *  lag + 1])
    
    # define the associated time-invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl + 2 * lag + 1)
    filt_rmse = np.zeros(nanl + 2 * lag + 1)
    anal_rmse = np.zeros(nanl + 2 * lag + 1)
    
    fore_spread = np.zeros(nanl + 2 * lag + 1)
    filt_spread = np.zeros(nanl + 2 * lag + 1)
    anal_spread = np.zeros(nanl + 2 * lag + 1)

    # make a first forecast of length lag, this will become the first re-analyzed "posterior"
    filt = np.zeros([sys_dim, N_ens, shift])
    for l in range(lag):
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        if l >= lag - shift:
            filt[:, :, l - lag + shift] = ens

    # we will run through nanl + 2 * lag total analyses but discard the first lag and last lag values
    # such that the statistics align on the same absolute time points
    for i in range(lag + 1, nanl + 2 * lag + 1 - shift, shift):
        # perform assimilation of the DAW
        # we use the observation window from time +1 of the ensemble to time of ensemble +shift
        # last filtered state becomes new posterior for reanalysis
        kwargs['posterior'] = filt
        analysis = lag_shift_smoother_classic(method, ens, H, obs[:, i: i + shift], obs_cov, infl, **kwargs)
        ens = analysis['ens']
        fore = analysis['fore']
        filt = analysis['filt']
        post = analysis['post']
        
        for j in range(shift):
            # compute the forecast, filter and analysis statistics
            # forward index the true state by 1, because the sequence starts at time zero for which there is no
            # observation
            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
            fore_rmse[i + j], fore_spread[i + j] = analyze_ensemble(fore[:, :, j], truth[:, i + j])
            filt_rmse[i + j], filt_spread[i + j] = analyze_ensemble(filt[:, :, j], truth[:, i + j])
            anal_rmse[i - shift + j], anal_spread[i - shift + j] = analyze_ensemble(post[:, :, j], 
                                                                                truth[:, i - shift  + j])

    # cut the statistics so that they align on the same time points
    fore_rmse = fore_rmse[lag + 1: lag + nanl + 1]
    fore_spread = fore_spread[lag + 1: lag + nanl + 1]
    filt_rmse = filt_rmse[lag + 1: lag + nanl + 1]
    filt_spread = filt_spread[lag + 1: lag + nanl + 1]
    anal_rmse = anal_rmse[lag + 1: lag + nanl + 1]
    anal_spread = anal_spread[lag + 1: lag + nanl + 1]

    data = {
            'fore_rmse': fore_rmse,
            'filt_rmse': filt_rmse,
            'anal_rmse': anal_rmse,
            'fore_spread': fore_spread,
            'filt_spread': filt_spread,
            'anal_spread': anal_spread,
            'method': method,
            'seed' : seed, 
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'nanl': nanl,
            'tanl': tanl,
            'lag': lag,
            'shift': shift,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(infl, 2)
            }
    
    fname = './data/' + method + '_classic/' + method + '_classic_smoother_l96_state_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) +\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + '_nanl_' +\
            str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_lag_' + str(lag).zfill(3) + '_shift_' + str(shift).zfill(3) +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_inflation_' + str(np.around(infl, 2)).ljust(4, '0') + '.txt'

    picwrite(data, fname)
    return(args)

########################################################################################################################


def classic_param(args):
    # Define experiment parameters
    [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01
    
    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # unpack the observations and the initial true state of the dynamic variables
    obs = np.squeeze(tmp['obs'])
    init = obs[:, 0]

    # define the state dynamic state dimension and the extended state parameters to be estimated
    state_dim = len(init)
    sys_dim = state_dim
    param_truth = np.array([f])
    sys_dim = state_dim + len(param_truth)

    # define kwargs
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model, 
              'h': h,
              'diffusion': diffusion,
              'shift': shift,
              'mda': False,
              'state_dim': state_dim,
              'param_infl': param_infl,
              'param_wlk': param_wlk
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
    # NOTE: we will include obs at time zero and pad the obs sequence with lag on front and back
    # to align the analysis and forecast statistics
    obs = obs[:, :nanl + 2 * lag + 1]
    truth = copy.copy(obs)
    
    # define the observation operator here, change if needed for different configurations
    H = np.eye(obs_dim, state_dim)
    obs = H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl + 2 *  lag + 1])
    
    # define the associated time-invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # define the observation operator on the extended state, used for the ensemble
    H_ens = np.eye(obs_dim, sys_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl + 2 * lag + 1)
    filt_rmse = np.zeros(nanl + 2 * lag + 1)
    anal_rmse = np.zeros(nanl + 2 * lag + 1)
    param_rmse = np.zeros(nanl + 2 * lag + 1)
    
    fore_spread = np.zeros(nanl + 2 * lag + 1)
    filt_spread = np.zeros(nanl + 2 * lag + 1)
    anal_spread = np.zeros(nanl + 2 * lag + 1)
    param_spread = np.zeros(nanl + 2 * lag + 1)

    # make a first forecast of length lag, this will become the first re-analyzed "posterior"
    filt = np.zeros([sys_dim, N_ens, shift])
    for l in range(lag):
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        if l >= lag - shift:
            filt[:, :, l - lag + shift] = ens

    # we will run through nanl + 2 * lag total analyses but discard the first lag and last lag values
    # such that the statistics align on the same absolute time points
    for i in range(lag + 1, nanl + 2 * lag + 1 - shift, shift):
        # perform assimilation of the DAW
        # we use the observation window from time +1 of the ensemble to time of ensemble +shift
        # last filtered state becomes new posterior for reanalysis
        kwargs['posterior'] = filt
        analysis = lag_shift_smoother_classic(method, ens, H_ens, obs[:, i: i + shift], obs_cov, state_infl, **kwargs)
        ens = analysis['ens']
        fore = analysis['fore']
        filt = analysis['filt']
        post = analysis['post']
        
        for j in range(shift):
            # compute the forecast, filter and analysis statistics
            # forward index the true state by 1, because the sequence starts at time zero for which there is no
            # observation
            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
            fore_rmse[i + j], fore_spread[i + j] = analyze_ensemble(fore[:state_dim, :, j], truth[:, i + j])
            filt_rmse[i + j], filt_spread[i + j] = analyze_ensemble(filt[:state_dim, :, j], truth[:, i + j])
            anal_rmse[i - shift + j], anal_spread[i - shift + j] = analyze_ensemble(post[:state_dim, :, j], 
                                                                                truth[:, i - shift  + j])
            param_rmse[i - shift + j], param_spread[i - shift + j] = analyze_ensemble_parameters(post[state_dim:, :, j], 
                                                                                param_truth)

    # cut the statistics so that they align on the same time points
    fore_rmse = fore_rmse[lag + 1: lag + nanl + 1]
    fore_spread = fore_spread[lag + 1: lag + nanl + 1]
    filt_rmse = filt_rmse[lag + 1: lag + nanl + 1]
    filt_spread = filt_spread[lag + 1: lag + nanl + 1]
    anal_rmse = anal_rmse[lag + 1: lag + nanl + 1]
    anal_spread = anal_spread[lag + 1: lag + nanl + 1]
    param_rmse = param_rmse[lag + 1: lag + nanl + 1]
    param_spread = param_spread[lag + 1: lag + nanl + 1]

    data = {
            'fore_rmse': fore_rmse,
            'filt_rmse': filt_rmse,
            'anal_rmse': anal_rmse,
            'param_rmse': param_rmse,
            'fore_spread': fore_spread,
            'filt_spread': filt_spread,
            'anal_spread': anal_spread,
            'param_spread': param_spread,
            'seed' : seed, 
            'method': method,
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'state_dim': state_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'param_err': param_err,
            'param_wlk': param_wlk,
            'nanl': nanl,
            'tanl': tanl,
            'lag': lag,
            'shift': shift,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(state_infl, 2),
            'param_infl': np.around(param_infl, 2)
            }
    
    fname = './data/' + method + '_classic/' + method + '_classic_smoother_l96_param_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) + '_state_dim_' + str(state_dim)+\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + \
            '_param_err_' + str(param_err).ljust(4, '0') + '_param_wlk_' + str(param_wlk).ljust(4, '0') +\
            '_nanl_' + str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_lag_' + str(lag).zfill(3) + '_shift_' + str(shift).zfill(3) +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_infl_' + str(np.around(state_infl, 2)).ljust(4, '0') +\
            '_param_infl_' + str(np.around(param_infl, 2)).ljust(4, '0') + '.txt'

    picwrite(data, fname)
    return(args)

########################################################################################################################


def hybrid_state(args):
    # Define experiment parameters
    [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01
    
    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # define kwargs
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model, 
              'dx_params': [f],
              'h': h,
              'diffusion': diffusion,
              'shift': shift,
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
    # NOTE: we will include obs at time zero and pad the obs sequence with lag on front and back
    # to align the analysis and forecast statistics
    obs = obs[:, :nanl + 2 * lag + 1]
    truth = copy.copy(obs)
    
    # define the observation operator here, change if needed for different configurations
    H = np.eye(obs_dim, sys_dim)
    obs = H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl + 2 *  lag + 1])
    
    # define the associated time-invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl + 2 * lag + 1)
    filt_rmse = np.zeros(nanl + 2 * lag + 1)
    anal_rmse = np.zeros(nanl + 2 * lag + 1)
    
    fore_spread = np.zeros(nanl + 2 * lag + 1)
    filt_spread = np.zeros(nanl + 2 * lag + 1)
    anal_spread = np.zeros(nanl + 2 * lag + 1)

    # we will run through nanl + 2 * lag total analyses but discard the last-lag forecast values and
    # first-lag and second-lag posterior values at the end so that the statistics align on the same time points
    for i in range(lag, nanl + 2 * lag + 1, shift):
        # perform assimilation of the DAW
        # we use the observation window from time zero to time lag
        analysis = lag_shift_smoother_hybrid(method, ens, H, obs[:, i-lag: i+1], obs_cov, infl, **kwargs)
        ens = analysis['ens']
        fore = analysis['fore']
        filt = analysis['filt']
        post = analysis['post']
        
        for j in range(shift):
            # compute the forecast, filter and analysis statistics
            # forward index the true state by 1, because the sequence starts at time zero for which there is no
            # observation
            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
            fore_rmse[i - shift + j + 1], fore_spread[i - shift + j + 1] = analyze_ensemble(fore[:, :, j], 
                                                                                    truth[:, i - shift + j + 1])
            
            filt_rmse[i - shift + j + 1], filt_spread[i - shift + j + 1] = analyze_ensemble(filt[:, :, j], 
                                                                                    truth[:, i - shift + j + 1])
            
            anal_rmse[i - lag + j], anal_spread[i - lag + j] = analyze_ensemble(post[:, :, j], 
                                                                                truth[:, i - lag  + j])

    # cut the statistics so that they align on the same time points
    fore_rmse = fore_rmse[lag: lag + nanl]
    fore_spread = fore_spread[lag: lag + nanl]
    filt_rmse = filt_rmse[lag: lag + nanl]
    filt_spread = filt_spread[lag: lag + nanl]
    anal_rmse = anal_rmse[lag: lag + nanl]
    anal_spread = anal_spread[lag: lag + nanl]

    data = {
            'fore_rmse': fore_rmse,
            'filt_rmse': filt_rmse,
            'anal_rmse': anal_rmse,
            'fore_spread': fore_spread,
            'filt_spread': filt_spread,
            'anal_spread': anal_spread,
            'method': method,
            'seed' : seed, 
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'nanl': nanl,
            'tanl': tanl,
            'lag': lag,
            'shift': shift,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(infl, 2)
            }
    
    fname = './data/' + method + '_hybrid/' + method + '_hybrid_smoother_l96_state_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) +\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + '_nanl_' +\
            str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_lag_' + str(lag).zfill(3) + '_shift_' + str(shift).zfill(3) +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_inflation_' + str(np.around(infl, 2)).ljust(4, '0') + '.txt'

    picwrite(data, fname)
    return(args)

########################################################################################################################


def hybrid_param(args):
    # Define experiment parameters

    [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args

    # load the timeseries and associated parameters
    tmp = picopen(time_series)
    diffusion = tmp['diffusion']
    f = tmp['f']
    tanl = tmp['tanl']
    h = 0.01

    
    # number of discrete forecast steps
    f_steps = int(tanl / h)

    # unpack the observations and the initial true state of the dynamic variables
    obs = np.squeeze(tmp['obs'])
    init = obs[:, 0]

    # define the state dynamic state dimension and the extended state parameters to be estimated
    state_dim = len(init)
    sys_dim = state_dim
    param_truth = np.array([f])
    sys_dim = state_dim + len(param_truth)

    # define kwargs
    kwargs = {
              'dx_dt': dx_dt,
              'f_steps': f_steps,
              'step_model': step_model, 
              'h': h,
              'diffusion': diffusion,
              'shift': shift,
              'mda': False,
              'state_dim': state_dim,
              'param_infl': param_infl,
              'param_wlk': param_wlk
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
    # NOTE: we will include obs at time zero and pad the obs sequence with lag on front and back
    # to align the analysis and forecast statistics
    obs = obs[:, :nanl + 2 * lag + 1]
    truth = copy.copy(obs)
    
    # define the observation operator here, change if needed for different configurations
    H = np.eye(obs_dim, state_dim)
    obs = H @ obs + obs_un * np.random.standard_normal([obs_dim, nanl + 2 *  lag + 1])
    
    # define the associated time-invariant observation error covariance
    obs_cov = obs_un**2 * np.eye(obs_dim)

    # define the observation operator on the extended state, used for the ensemble
    H_ens = np.eye(obs_dim, sys_dim)

    # create storage for the forecast and analysis statistics
    fore_rmse = np.zeros(nanl + 2 * lag + 1)
    filt_rmse = np.zeros(nanl + 2 * lag + 1)
    anal_rmse = np.zeros(nanl + 2 * lag + 1)
    param_rmse = np.zeros(nanl + 2 * lag + 1)
    
    fore_spread = np.zeros(nanl + 2 * lag + 1)
    filt_spread = np.zeros(nanl + 2 * lag + 1)
    anal_spread = np.zeros(nanl + 2 * lag + 1)
    param_spread = np.zeros(nanl + 2 * lag + 1)

    # we will run through nanl + shift total analyses but discard the last-shift forecast values and
    # first-shift and second-shift posterior values at the end so that the statistics align on the same time points
    for i in range(lag, nanl + 2 * lag + 1, shift):
        # perform assimilation of the DAW
        # we use the observation windo from time zero to time lag
        analysis = lag_shift_smoother_hybrid(method, ens, H_ens, obs[:, i-lag: i+1], obs_cov, state_infl, **kwargs)
        ens = analysis['ens']
        fore = analysis['fore']
        filt = analysis['filt']
        post = analysis['post']
        
        for j in range(shift):
            # compute the forecast, filter and analysis statistics
            # forward index the true state by 1, because the sequence starts at time zero for which there is no
            # observation
            # indices for the forecast, filter, analysis and truth arrays are in absolute time, not relative
            fore_rmse[i - shift + j + 1], fore_spread[i - shift + j + 1] = analyze_ensemble(fore[:state_dim, :, j], 
                                                                                    truth[:, i - shift + j + 1])
            
            filt_rmse[i - shift + j + 1], filt_spread[i - shift + j + 1] = analyze_ensemble(filt[:state_dim, :, j], 
                                                                                    truth[:, i - shift + j + 1])
            
            anal_rmse[i - lag + j], anal_spread[i - lag + j] = analyze_ensemble(post[:state_dim, :, j], 
                                                                                truth[:, i - lag  + j])

            param_rmse[i - lag + j], param_spread[i - lag + j] = analyze_ensemble_parameters(post[state_dim:, :, j], 
                                                                                param_truth)

            
    # cut the statistics so that they align on the same time points
    fore_rmse = fore_rmse[lag: lag + nanl]
    fore_spread = fore_spread[lag: lag + nanl]
    filt_rmse = filt_rmse[lag: lag + nanl]
    filt_spread = filt_spread[lag: lag + nanl]
    anal_rmse = anal_rmse[lag: lag + nanl]
    anal_spread = anal_spread[lag: lag + nanl]
    param_rmse = param_rmse[lag: lag + nanl]
    param_spread = param_spread[lag: lag + nanl]

    data = {
            'fore_rmse': fore_rmse,
            'filt_rmse': filt_rmse,
            'anal_rmse': anal_rmse,
            'param_rmse': param_rmse,
            'fore_spread': fore_spread,
            'filt_spread': filt_spread,
            'anal_spread': anal_spread,
            'param_spread': param_spread,
            'seed' : seed, 
            'method': method,
            'diffusion': diffusion,
            'sys_dim': sys_dim,
            'state_dim': state_dim,
            'obs_dim': obs_dim, 
            'obs_un': obs_un,
            'param_err': param_err,
            'param_wlk': param_wlk,
            'nanl': nanl,
            'tanl': tanl,
            'lag': lag,
            'shift': shift,
            'h': h,
            'N_ens': N_ens, 
            'state_infl': np.around(state_infl, 2),
            'param_infl': np.around(param_infl, 2)
            }
    
    fname = './data/' + method + '_hybrid/' + method + '_hybrid_smoother_l96_param_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) + '_state_dim_' + str(state_dim)+\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + \
            '_param_err_' + str(param_err).ljust(4, '0') + '_param_wlk_' + str(param_wlk).ljust(4, '0') +\
            '_nanl_' + str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_lag_' + str(lag).zfill(3) + '_shift_' + str(shift).zfill(3) +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_infl_' + str(np.around(state_infl, 2)).ljust(4, '0') +\
            '_param_infl_' + str(np.around(param_infl, 2)).ljust(4, '0') + '.txt'

    picwrite(data, fname)
    return(args)

########################################################################################################################
