import numpy as np
import pickle
from methods.l96 import rk4_step, l96, rho, alpha, l96s_tay2_step
import ipdb
import sys
import time

def experiment():
    """This experiment will spin a "truth" trajectory of the stochastic l96 and store a series of analysis points

    This will pickle the associated trajectories for processing in data assimilation experiments."""

    ####################################################################################################################
    
    # unpack arguments for the experiment
    [seed, N_ens, forward_step, h, diffusion, sys_dim, nanl, spin, anal_int]  = [0, 25, rk4_step, 
                                                                                 0.01, 0.1, 40, 5000, 1000, 0.05]
    dx_dt = l96


    dx_params = [8.0]
    kwargs = {
              'dx_params': dx_params,
              'dx_dt': dx_dt,
              'h': h,
              'diffusion': diffusion,
             }

    h = kwargs['h']
    
    state_dim = sys_dim

    # number of discrete forecast steps
    f_steps = int(anal_int / h)

    # define the initialization of the model
    xt = np.random.multivariate_normal(np.zeros(state_dim), np.eye(state_dim), size=N_ens).transpose()
    
    # spin is the length of the spin period in discrete time steps where spin is in continuous time
    for i in range(int(spin)):
        for j in range(f_steps):
            # recursively integrate one step forward
            xt = forward_step(xt, **kwargs)

    # after spin, store all analysis intervals
    for i in range(nanl):
        for j in range(int(f_steps)):
            xt = forward_step(xt, **kwargs)

########################################################################################################################


# Arguments for debugging, experiment built functionalized for taking array maps of parameter values
# [seed, N_ens, forward_step, h, diffusion, sys_dim, nanl, spin, anal_int]  = args

start = time.time()
experiment()
end = time.time()
print(end-start)

