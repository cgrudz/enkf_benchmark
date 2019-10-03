import numpy as np
import pickle
from l96 import l96_rk4_step

def experiment(seed):
    """This experiment will spin a "truth" trajectory of the stochastic l96 and store a series of analysis points

    This is a function of the ensemble number which initializes the random seed.  This will pickle the associated
    trajectories for processing in data assimilation experiments."""

    ####################################################################################################################
    # static parameters
    f = 8

    # model dimension
    sys_dim = 10

    # time step
    h = .01

    # number of observations
    nanl = 10000

    # spin onto random attractor in continous time
    spin = 5000

    # interval between analysis steps
    analint = 0.1

    # number of discrete forecast steps
    fore_steps = int(analint / h)

    # define the initialization of the model
    np.random.seed(seed)
    xt = np.random.multivariate_normal(np.zeros(sys_dim), np.eye(sys_dim) * sys_dim)

    truth = np.zeros([sys_dim, nanl])
    # spin is the length of the spin period in the continuous time variable
    for i in range(int(spin / h)):
        # recursively integrate one step forward
        xt = l96_rk4_step(xt, h, f) 

    # after spin, store all analysis intervals
    for i in range(nanl):
        for j in range(int(analint/h)):
            xt = l96_rk4_step(xt, h, f)

        truth[:, i] = xt

    data = {'obs': truth, 'params': [f, sys_dim, h, spin, analint]}

    f = open('./data/l96_ts_sys_dim_' + str(sys_dim) + '_h_' + str(h).zfill(2) + '_nanl_' + str(nanl) +  \
            '_anal_int_' + str(analint) + '_seed_' + str(seed) + '.txt','wb')
    pickle.dump(data, f)
    f.close()


experiment(0)
