import numpy as np
import pickle
from l96 import rk4_step, l96, rho, alpha, l96s_tay2_step
import ipdb
import sys

def experiment(args):
    """This experiment will spin a "truth" trajectory of the stochastic l96 and store a series of analysis points

    This will pickle the associated trajectories for processing in data assimilation experiments."""

    ####################################################################################################################
    
    # unpack arguments for the experiment
    [seed, N_ens, forward_step, h, diffusion, sys_dim, nanl, spin, anal_int]  = args
    dx_dt = l96


    kwargs = {
              'dx_params': [8],
              'dx_dt': dx_dt,
              'h': h,
              'diffusion': diffusion,
              #'state_dim':10
             }

    h = kwargs['h']
    
    if forward_step.__name__ == 'l96s_tay2_step':
        p = 1
        kwargs['rho'] = rho(p)
        kwargs['alpha'] = alpha(p)
        kwargs['p'] = p
    
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']

    else:
        state_dim = sys_dim

    # number of discrete forecast steps
    fore_steps = int(anal_int / h)

    # define the initialization of the model
    np.random.seed(seed)
    xt = np.random.multivariate_normal(np.zeros(state_dim), np.eye(state_dim) * state_dim, size=N_ens).transpose()
    truth = np.zeros([sys_dim, nanl, N_ens])
    
    if 'state_dim' in kwargs: 
        tmp = np.zeros([sys_dim, N_ens])
        tmp[:state_dim, :] = xt
        tmp[-1, :] = 8
        xt = tmp

    # spin is the length of the spin period in discrete time steps where spin is in continuous time
    for i in range(int(spin / h)):
        # recursively integrate one step forward
        xt = forward_step(xt, **kwargs)

    # after spin, store all analysis intervals
    for i in range(nanl):
        for j in range(int(anal_int/h)):
            xt = forward_step(xt, **kwargs)

        truth[:, i] = xt

    data = {'obs': truth, 'h': h, 'diffusion': diffusion, 'f': dx_params[0], 'tanl': anal_int}

    f = open('./data/'+ 'timeseries_' + dx_dt.__name__ + '_seed_' + str(seed) + '_' + \
             forward_step.__name__ + '_sys_dim_' + str(sys_dim) + '_h_' + str(h).zfill(2) + \
            '_diffusion_' + str(diffusion).zfill(3) + '_nanl_' + str(nanl) + '_spin_' + str(spin) + \
            '_anal_int_' + str(anal_int) + '.txt','wb')
    pickle.dump(data, f)
    f.close()


########################################################################################################################


# Arguments for debugging, experiment built functionalized for taking array maps of parameter values
# [seed, N_ens, dx_dt, forward_step, diffuison, sys_dim, nanl, spin, anal_int, kwargs] = args


#args = [0, 1, rk4_step, 0, 10, 100, 100, 0.01]
#
#experiment(args)
#


j = int(sys.argv[1])
f = open('./data/input_data_generate_timeseries.txt', 'rb')
data = pickle.load(f)
args = data[j]
f.close()

experiment(args)

