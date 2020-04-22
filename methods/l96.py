import numpy as np
import copy
import ipdb

# Module containing integration schemes for Lorenz 96

########################################################################################################################
# Non-linear model vectorized for ensembles


def l96(x, dx_params):
    """"This describes the derivative for the non-linear Lorenz 96 Model of arbitrary dimension n.

    This will take the state vector x of shape sys_dim X ens_dim and return the equation for dxdt.  For
    one dimensional vector, this can be enforced as a sys_dim X 1 dimensional array."""

    [f] = dx_params

    # shift minus and plus indices with concatenate, array compute the derivative over arbitrary ensemble sizes
    x_m_2 = np.concatenate([x[-2:, :], x[:-2, :]])
    x_m_1 = np.concatenate([x[-1:, :], x[:-1, :]])
    x_p_1 = np.concatenate([x[1:,:], np.reshape(x[0,:], [1, len(x[0, :])])], axis=0)
    
    return (x_p_1-x_m_2)*x_m_1 - x + f


########################################################################################################################
# Jacobian


def l96_jacobian(x, dx_params):
    """"This computes the Jacobian of the Lorenz 96, for arbitrary dimension, equation about the point x."""

    x_dim = len(x)

    dxF = np.zeros([x_dim, x_dim])

    for i in range(x_dim):
        i_m_2 = np.mod(i - 2, x_dim)
        i_m_1 = np.mod(i - 1, x_dim)
        i_p_1 = np.mod(i + 1, x_dim)

        dxF[i, i_m_2] = -x[i_m_1]
        dxF[i, i_m_1] = x[i_p_1] - x[i_m_2]
        dxF[i, i] = -1.0
        dxF[i, i_p_1] = x[i_m_1]

    return dxF


########################################################################################################################
# Stochastic Runge-Kutta, 4 step
# This is the four step runge kutta scheme for stratonovich calculus, described in Hansen and Penland 2005
# The rule has strong convergence order 1.0 for generic SDEs and order 4.0 for ODEs

def rk4_step(x, **kwargs):
    """One step of integration rule for l96 4 stage Runge-Kutta as discussed in Grudzien et al. 2020

    Arguments are given as
    x          -- array with ensemble of possibly extended state varaibles including parameter values
    kwargs     -- this should include dx_dt, the paramters for the dx_dt and optional arguments
    dx_dt      -- time derivative provided as function of x and dx_params
    dx_params  -- list of parameters necessary to resolve dx_dt, not including parameters we estimate
    h          -- integration step size
    diffusion  -- tunes the standard deviation of the noise process, equal to sqrt(h) * diffusion
    state_dim  -- indicates parameter estimation, this is the value of the dimension of the dynamic state
    xi         -- random array size state_dim X 1, can be defined in kwargs to provide a particular realization
    """

    # unpack the integration scheme arguments and the parameters of the derivative
    h = kwargs['h']
    diffusion = kwargs['diffusion']
    dx_dt = kwargs['dx_dt']

    if 'dx_params' in kwargs:
        # get parameters for dx_dt
        params = kwargs['dx_params']

    # infer the dimensions
    [sys_dim, N_ens] = np.shape(x)

    # check for extended state vector
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_est = True

    else:
        state_dim = sys_dim
        param_est = False

    if diffusion != 0:
        if 'xi' in kwargs:
            # pre-computed perturbation is provided, use this
            xi = kwargs['xi']
        
        else:
            # generate perturbation for brownian motion
            xi = np.random.multivariate_normal(np.zeros(state_dim), np.eye(state_dim), size=N_ens).transpose()

    else:
        # if no diffusion load dummy xi
        xi = np.zeros([state_dim, N_ens])

    # rescale the standard normal to variance h
    W = xi * np.sqrt(h)

    if param_est:
        # define the vectorized runge-kutta terms
        k1 = np.zeros([state_dim, N_ens])
        k2 = copy.copy(k1)
        k3 = copy.copy(k1)
        k4 = copy.copy(k1)

        #evolve ensemble members according to their parameter sample
        for i in range(N_ens):
            if 'dx_params' in kwargs:
                # extract the parameter sample
                params_i = [params[:], x[state_dim:, i]]
        
            else:
                params_i = [x[state_dim:, i]]

            # Define the four terms of the RK scheme recursively to evolve the state
            # components alone
            k1[:, [i]] = dx_dt(x[:state_dim, [i]], params_i) * h + diffusion * W[:, [i]]
            k2[:, [i]] = dx_dt(x[:state_dim, [i]] + .5 * k1[:, [i]], params_i) * h + diffusion * W[:, [i]]
            k3[:, [i]] = dx_dt(x[:state_dim, [i]] + .5 * k2[:, [i]], params_i) * h + diffusion * W[:, [i]]
            k4[:, [i]] = dx_dt(x[:state_dim, [i]] + k3[:, [i]], params_i) * h + diffusion * W[:, [i]]
            
            # compute the update to the dynamic variables
            x_step = x[:state_dim, [i]]+ (1 / 6) * (k1[:, [i]] + 2*k2[:, [i]] + 2*k3[:, [i]] + k4[:, [i]])
            
            # repack the parameter in the extended state
            x[:state_dim, [i]] = x_step  

    else:
        # Define the four terms of the RK scheme recursively
        k1 = dx_dt(x, params) * h + diffusion * W
        k2 = dx_dt(x + .5 * k1, params) * h + diffusion * W
        k3 = dx_dt(x + .5 * k2, params) * h + diffusion * W
        k4 = dx_dt(x + k3, params) * h + diffusion * W

        # compute the update to the dynamic variables
        x = x + (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x


########################################################################################################################
# 2nd order Taylor Method


def tay2_step(x, **kwargs):
    """Second order Taylor method for step size h"""

    # unpack dx_params
    h = kwargs['h']
    params = kwargs['dx_params']
    dx_dt = kwargs['dx_dt']
    jacobian = kwargs['jacobian']

    # calculate the evolution of x one step forward via the second order Taylor expansion

    # first derivative
    dx = dx_dt(x, params)

    # second order taylor expansion
    
    return x + dx * h + .5 * jacobian(x, params) @ dx * h**2


########################################################################################################################
# auxiliary functions for the 2nd order taylor expansion
# these need to be computed once, only as a function of the order of truncation of the fourier series, p

def rho(p):
        return 1/12 - .5 * np.pi**(-2) * np.sum(1 / np.arange(1, p+1)**2)

def alpha(p):
        return (np.pi**2) / 180 - .5 * np.pi**(-2) * np.sum(1 / np.arange(1, p+1)**4)


########################################################################################################################
# 2nd order strong taylor SDE step
# This method is derived from page 359, NUMERICAL SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS, KLOEDEN & PLATEN;
# this uses the approximate statonovich integrals defined on page 202
# this depends on rho and alpha as above


def l96s_tay2_step(x, **kwargs):
    """One step of integration rule for l96 second order taylor rule

    The rho and alpha are to be computed by the auxiliary functions, depending only on p, and supplied for all steps.  
    This is the general formulation which includes, eg. dependence on the truncation of terms in the auxilliary 
    function C with respect to the parameter p.  In general, truncation at p=1 is all that is necessary for order
    2.0 convergence, and in this case C below is identically equal to zero.  This auxilliary function can be removed 
    (and is removed) in other implementations for simplicity."""

    # Infer system dimension
    sys_dim = len(x)


    dx_params = kwargs['dx_params']
    h = kwargs['h']
    diffusion = kwargs['diffusion']
    dx_dt = kwargs['dx_dt']
    p = kwargs['p']
    rho = kwargs['rho']
    alpha = kwargs['alpha']

    # Compute the deterministic dxdt and the jacobian equations, squeeze at the end
    # to eliminate the extra dimension from ensembles
    dx = np.squeeze(l96(x, dx_params))
   
    # x is always a single trajectory, we make into a 1 dimension array for the rest of code
    x = np.squeeze(x)
    Jac_x = l96_jacobian(x, dx_params)
    ### random variables
    
    # Vectors xi, mu, phi are sys_dim X 1 vectors of iid standard normal variables, 
    # zeta and eta are sys_dim X p matrices of iid standard normal variables. Functional relationships describe each
    # variable W_j as the transformation of xi_j to be of variace given by the length of the time step h. The functions
    # of random Fourier coefficients a_i, b_i are given in terms mu/ eta and phi/zeta respectively.
    
    # draw standard normal samples
    rndm = np.random.standard_normal([sys_dim, 2*p + 3])
    xi = rndm[:, 0]
    
    mu = rndm[:, 1]
    phi = rndm[:, 2]

    zeta = rndm[:, 3: p+3]
    eta = rndm[:, p+3:]
    
    ### define the auxiliary functions of random fourier coefficients, a and b
    
    # denominators for the a series
    tmp = np.tile(1 / np.arange(1, p+1), [sys_dim, 1])

    # vector of sums defining a terms
    a = -2 * np.sqrt(h * rho) * mu - np.sqrt(2*h) * np.sum(zeta * tmp, axis=1) / np.pi
    
    # denominators for the b series
    tmp = np.tile(1 / np.arange(1, p+1)**2, [sys_dim, 1]) 

    # vector of sums defining b terms
    b = np.sqrt(h * alpha) * phi + np.sqrt(h / (2 * np.pi**2) ) * np.sum(eta * tmp, axis=1)

    # vector of first order Stratonovich integrals
    J_pdelta = (h / 2) * (np.sqrt(h) * xi + a)

    
    ### auxiliary functions for higher order stratonovich integrals ###

    # the triple stratonovich integral reduces in the lorenz 96 equation to a simple sum of the auxiliary functions, we
    # define these terms here abstractly so that we may efficiently compute the terms
    def C(l, j):
        C = np.zeros([p, p])
        # we will define the coefficient as a sum of matrix entries where r and k do not agree --- we compute this by a
        # set difference
        indx = set(range(1, p+1))

        for r in range(1, p+1):
            # vals are all values not equal to r
            vals = indx.difference([r])
            for k in vals:
                # and for row r, we define all columns to be given by the following, inexing starting at zero
                C[r-1, k-1] = (r / (r**2 - k**2)) * ((1/k) * zeta[l, r-1] * zeta[j, k-1] + (1/r) * eta[l, r-1] * eta[j, k-1] )

        # we return the sum of all values scaled by -1/2pi^2
        return .5 * np.pi**(-2) * np.sum(C)

    def Psi(l, j):
        # psi will be a generic function of the indicies l and j, we will define psi plus and psi minus via this
        psi = h**2 * xi[l] * xi[j] / 3 + h * a[l] * a[j] / 2 + h**(1.5) * (xi[l] * a[j] + xi[j] * a[l]) / 4 \
              - h**(1.5) * (xi[l] * b[j] + xi[j] * b[l]) / (2 * np.pi) - h**2 * (C(l,j) + C(j,l))
        return psi

    # we define the approximations of the second order Stratonovich integral
    psi_plus = np.array([Psi((i-1) % sys_dim, (i+1) % sys_dim) for i in range(sys_dim)])
    psi_minus = np.array([Psi((i-2) % sys_dim, (i-1) % sys_dim) for i in range(sys_dim)])

    # the final vectorized step forward is given as
    x_step = x + dx * h + h**2 * .5 * Jac_x @ dx    # deterministic taylor step 
    x_step += diffusion * np.sqrt(h) * xi           # stochastic euler step
    x_step += + diffusion * Jac_x @ J_pdelta        # stochastic first order taylor step
    x_step += diffusion**2 * (psi_plus - psi_minus) # stochastic second order taylor step

    return np.reshape(x_step, [sys_dim, 1])

########################################################################################################################
# Euler-Murayama step

def l96_em_sde(x, params):
    """This will propagate the state x one step forward by euler-murayama

    step size is h and the weiner process is assumed to have a scalar diffusion coefficient"""
    
    # unpack the arguments for the integration step
    [h, f, diffusion] = params

    # infer dimension and draw realizations of the wiener process
    sys_dim = len(x)
    W =  np.sqrt(h) * np.random.standard_normal([sys_dim])

    # step forward by interval h
    x_step = x +  h * l96(x, f) + diffusion * W

    return x_step


########################################################################################################################
# Step the tangent linear model


def l96_step_TLM(x, Y, h, nonlinear_step, params):
    """"This function describes the step forward of the tangent linear model for Lorenz 96 via RK-4

    Input x is for the non-linear model evolution, while Y is the matrix of perturbations, h is defined to be the
    time step of the TLM.  This returns the forward non-linear evolution and the forward TLM evolution as
    [x_next,Y_next]"""

    h_mid = h/2

    # calculate the evolution of x to the midpoint
    x_mid = nonlinear_step(x, h_mid, params)

    # calculate x to the next time step
    x_next = nonlinear_step(x_mid, h_mid, params)

    k_y_1 = l96_jacobian(x).dot(Y)
    k_y_2 = l96_jacobian(x_mid).dot(Y + k_y_1 * (h / 2.0))
    k_y_3 = l96_jacobian(x_mid).dot(Y + k_y_2 * (h / 2.0))
    k_y_4 = l96_jacobian(x_next).dot(Y + k_y_3 * h)

    Y_next = Y + (h / 6.0) * (k_y_1 + 2 * k_y_2 + 2 * k_y_3 + k_y_4)

    return [x_next, Y_next]


########################################################################################################################
