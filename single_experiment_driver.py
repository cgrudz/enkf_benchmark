import ipdb
from smoother_exps import classic_state, classic_param, hybrid_state, hybrid_param
from filter_exps import filter_state, filter_param

########################################################################################################################
## Timeseries data 
########################################################################################################################
# observation timeseries to load into the experiment as truth twin
# timeseries are named by the model, seed to initialize, the integration scheme used to produce, number of analyses,
# the spinup length, and the time length between observation points
#
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.05.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.1_nanl_50000_spin_2500_anal_int_0.1.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.25_nanl_50000_spin_2500_anal_int_0.05.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.25_nanl_50000_spin_2500_anal_int_0.1.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.5_nanl_50000_spin_2500_anal_int_0.05.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_l96s_tay2_step_sys_dim_40_h_0.005_diffusion_0.5_nanl_50000_spin_2500_anal_int_0.1.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt'
#time_series = './data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.1.txt'
########################################################################################################################

########################################################################################################################
## Experiments to run as a single function call
########################################################################################################################

########################################################################################################################
# Filters
########################################################################################################################
## filter_state single run for degbugging, arguments are
## [time_series, method, seed, obs_un, obs_dim, N_ens, infl] = args
#
#args = [time_series, 'etkf', 0, 1.0, 40, 25, 1.10]
#print(filter_state(args))
########################################################################################################################
## filter_param single run for degbugging, arguments are
## [time_series, method, seed, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#args = [time_series, 'etkf', 0, 1.0, 40, 0.03, 0.00, 25, 1.10, 1.0]
#print(filter_param(args))
########################################################################################################################

########################################################################################################################
# Classic smoothers
########################################################################################################################
## classic_state single run for degbugging, arguments are
## [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
#
#args = [time_series, 'etks', 0, 1, 1, 1.0, 40, 25, 1.05]
#print(classic_state(args))
########################################################################################################################
## classic_param single run for debugging, arguments are
## [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#args = [time_series, 'etks', 0, 1, 1, 1.0, 40, 0.03, 0.01, 25, 1.05, 1.0] 
#print(classic_param(args))
########################################################################################################################

########################################################################################################################
# Hybrid smoothers
########################################################################################################################
# hybrid_state single run for degbugging, arguments are
# [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
#
#args = [time_series, 'etks', 0, 1, 1, 1.0, 40, 25, 1.05]
#print(hybrid_state(args))
########################################################################################################################
## hybrid_param single run for debugging, arguments are
## [time_series, method, seed, lag, shift, obs_un, obs_dim, param_err, param_wlk, N_ens, state_infl, param_infl] = args
#
#args = [time_series, 'etks', 0, 1, 1, 1.0, 40, 0.03, 0.01, 25, 1.05, 1.0] 
#print(hybrid_param(args))
########################################################################################################################


