import numpy as np
from l96 import rk4_step as step_model, l96 as dx_dt
from ensemble_kalman_schemes import analyze_ensemble
from ensemble_kalman_schemes import lag_shift_smoother 
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

    [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args

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

    # we will run through nanl + shift total analyses but discard the last-shift forecast values and
    # first-shift and second-shift posterior values at the end so that the statistics align on the same time points
    for i in range(lag, nanl + 2 * lag + 1, shift):
        # perform assimilation of the DAW
        # we use the observation windo from time zero to time lag
        analysis = lag_shift_smoother(method, ens, H, obs[:, i-lag: i+1], obs_cov, infl, **kwargs)
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
    
    fname = './data/' + method + '/' + method + '_smoother_l96_param_benchmark_seed_' +\
            str(seed).zfill(2) + '_diffusion_' + str(diffusion).ljust(4, '0') + '_sys_dim_' + str(sys_dim) +\
            '_obs_dim_' + str(obs_dim) + '_obs_un_' + str(obs_un).ljust(4, '0') + '_nanl_' +\
            str(nanl).zfill(3) + '_tanl_' + str(tanl).zfill(3) + '_h_' + str(h).ljust(4, '0') + \
            '_lag_' + str(lag).zfill(3) + '_shift_' + str(shift).zfill(3) +\
            '_N_ens_' + str(N_ens).zfill(3) + '_state_inflation_' + str(np.around(infl, 2)).ljust(4, '0') + '.txt'

    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

    return(args)

########################################################################################################################

### SINGLE EXPERIMENT DEBUGGING
#fname = './data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt'
#
## [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
#experiment([fname, 'enks', 0, 4, 1, 1.0, 40, 20, 1.05])
#
#
### FUNCTIONALIZED EXPERIMENT CALL OVER PARAMETER MAP
#j = int(sys.argv[1])
#f = open('./data/input_data/benchmark_smoother_state.txt', 'rb')
#data = pickle.load(f)
#args = data[j]
#f.close()
#
#experiment(args)
