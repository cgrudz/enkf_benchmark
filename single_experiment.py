import ipdb
from smoother import classic_state, hybrid_state, hybrid_param

## classic_state single run for degbugging
# [time_series, method, seed, lag, shift, obs_un, obs_dim, N_ens, infl] = args
time_series = './data/timeseries_obs/timeseries_l96_seed_0_rk4_step_sys_dim_40_h_0.01_diffusion_000_nanl_50000_spin_2500_anal_int_0.05.txt'
args = [time_series, 'enks', 0, 5, 2, 1.0, 40, 40, 1.05]
print(classic_state(args))
