import numpy as np
import copy
from numpy.linalg import solve
import ipdb

########################################################################################################################
# Main methods, debugged and validated
########################################################################################################################
# alternating id obs

def alternating_obs_operator(sys_dim, obs_dim, **kwargs):
    """Defines observation operator by alternating state vector components.

    If obs_dim == state_dim, this returns the identity matrix, otherwise alternating observations of the state
    components.  For parameter estimation, state_dim is an optional kwarg to define the operator to only observe
    the regular state vector, not the extended one."""

    if 'state_dim' in kwargs:
        # performing parameter estimation, load the dynamic state dimension
        state_dim = kwargs['state_dim']
        
        # load observation operator for the extended state, without observing extended state components
        H = np.eye(state_dim, sys_dim)
        
        # proceed with alternating observations of the regular state vector
        sys_dim = state_dim

    else:
        H = np.eye(sys_dim)

    if sys_dim == obs_dim:
        pass

    elif (obs_dim / sys_dim) > .5:
        # the observation dimension is greater than half the state dimension, so we
        # remove only the trailing odd-index rows from the identity matrix, equal to the difference
        # of the state and observation dimension
        R = int(sys_dim - obs_dim)
        H = np.delete(H, np.s_[-2*R::2], 0)

    elif (obs_dim / sys_dim) == .5:
        # the observation dimension is equal to half the state dimension so we remove exactly
        # half the rows, corresponding to those with odd-index
        H = np.delete(H, np.s_[1::2], 0)

    else:
        # the observation dimension is less than half of the state dimension so that we
        # remove all odd rows and then the R=sys_dim/2 - obs_dim  trailing even rows
        H = np.delete(H, np.s_[1::2], 0)
        R = int(sys_dim/2 - obs_dim)
        H = np.delete(H, np.s_[-R:], 0)

    return H


########################################################################################################################

def analyze_ensemble(ens, truth):
    """This will compute the ensemble RMSE as compared with the true twin, and the ensemble spread."""

    # infer the shapes
    [sys_dim, N_ens] = np.shape(ens)

    # compute the ensemble mean
    mean = np.mean(ens, axis=1)

    # compute the RMSE of the ensemble mean
    rmse = np.sqrt( np.mean( (truth - mean)**2 ) )

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square 
    # deviation of each member from the ensemble mean 
    spread = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (mean - ens.transpose())**2, axis=1)))

    return [rmse, spread]


########################################################################################################################


def analyze_ensemble_parameters(ens, truth):
    """This will compute the ensemble RMSE as compared with the true twin, and the ensemble spread."""

    # infer the shapes
    [sys_dim, N_ens] = np.shape(ens)

    # compute the ensemble mean
    mean = np.mean(ens, axis=1)
    # compute the RMSE of the ensemble mean, where each value is computed relative to the magnitude of the parameter
    rmse = np.sqrt( np.mean( (truth - mean)**2 / truth**2 ) )

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the 
    # ensemble with the weight by the size of the parameter square
    spread = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (mean - ens.transpose())**2 / 
                                                            (np.ones([N_ens, sys_dim]) * truth**2), axis=1)))
    
    return [rmse, spread]

########################################################################################################################
# random mean preserving orthogonal matrix, auxilliary function for determinstic EnKF schemes


def rand_orth(N_ens):
    """This generates a mean preserving random orthogonal matrix as in sakov oke 08"""
    
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    Q, R = np.linalg.qr(Q)
    U_p =  np.zeros([N_ens, N_ens])
    U_p[0,0] = 1
    U_p[1:, 1:] = Q

    b_1 = np.ones(N_ens)/np.sqrt(N_ens)
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    B = np.zeros([N_ens, N_ens])
    B[:,0] = b_1
    B, R = np.linalg.qr(B)
    U = B @ U_p @ B.transpose()

    return U

########################################################################################################################
# dynamic state variable inflation


def inflate_state(ens, inflation, sys_dim, state_dim):
    """State variables are assumed to be in the leading rows, while extended
    state variables, parameter variables are after.
    
    Multiplicative inflation is performed only in the leading components."""

    X_mean = np.mean(ens, axis=1)
    A_t = ens.transpose() - X_mean
    tmp = np.eye(state_dim) * inflation
    infl = np.eye(sys_dim)
    infl[:state_dim, :state_dim] = tmp
    ens = (X_mean + A_t @ infl).transpose()

    return ens

########################################################################################################################
# parameter multiplicative inflation


def inflate_param(ens, inflation, sys_dim, state_dim):
    """State variables are assumed to be in the leading rows, while extended
    state, parameter variables are after.
    
    Multiplicative inflation is performed only in the trailing components."""

    X_mean = np.mean(ens, axis=1)
    A_t = ens.transpose() - X_mean
    tmp = np.eye(sys_dim - state_dim) * inflation
    infl= np.eye(sys_dim)
    infl[state_dim:, state_dim:] = tmp
    ens = (X_mean + A_t @ infl).transpose()

    return ens

########################################################################################################################
# transform auxilliary function for EnKF, ETKF, EnKS, ETKS


def transform(analysis, ens, H, obs, obs_cov):
    """Computes transform and related values for EnKF, ETKF, EnkS, ETKS

    analysis is a string which determines the type of transform update."""

    if analysis=='enkf' or analysis=='enks':
        ## This computes the stochastic transform for the EnKF/S as in Carrassi, et al. 2018
        # step 0: infer the ensemble, obs, and state dimensions
        [sys_dim, N_ens] = np.shape(ens)
        obs_dim = len(obs)

        # step 1: compute the ensemble mean and normalized anomalies
        X_mean = np.mean(ens, axis=1)

        # step 2: compute the normalized anomalies, transposed
        A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)

        # step 3: generate the unbiased perturbed observations, note, we use the actual instead of ensemble
        # obs_cov to handle rank degeneracy
        obs_perts = np.random.multivariate_normal(np.zeros(obs_dim), obs_cov, N_ens)
        obs_perts = obs_perts - np.mean(obs_perts, axis=0)

        # step 4: compute the observation ensemble
        obs_ens = (obs + obs_perts).transpose()

        # step 5: generate the ensemble transform matrix, note, transform is missing normalization
        # of sqrt(N_ens-1) in paper
        Y_t = A_t @ H.transpose()
        C = Y_t.transpose() @ Y_t + obs_cov
        T = np.eye(N_ens) + Y_t @ np.linalg.inv(C) @ (obs_ens - H @ ens) / np.sqrt(N_ens - 1)
        
        # step 6: package the transform output
        trans = [T]

    elif analysis=='etkf' or analysis=='etks':
        ## This computes the transform of the ETKF update as in Asch, Bocquet, Nodet
        # step 0: infer the system, observation and ensemble dimensions 
        [sys_dim, N_ens] = np.shape(ens)
        obs_dim = len(obs)

        # step 1: compute the ensemble mean
        x_mean = np.mean(ens, axis=1)

        # step 2: compute the normalized anomalies, transposed
        A_t = (ens.transpose() - x_mean) / np.sqrt(N_ens - 1)
        
        # step 3: compute the ensemble in observation space
        Z = H @ ens

        # step 4: compute the ensemble mean in observation space
        y_mean = np.mean(Z, axis=1)
        
        # step 5: compute the weighted anomalies in observation space
        
        # first we find the observation error covariance inverse
        V, Sigma, V_t = np.linalg.svd(obs_cov)
        obs_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
        
        # then compute the weighted anomalies
        S = (Z.transpose() - y_mean) / np.sqrt(N_ens - 1)
        S = obs_sqrt_inv @ S.transpose()

        # step 6: compute the weighted innovation
        delta = obs_sqrt_inv @ ( obs - y_mean )
       
        # step 7: compute the transform matrix
        T = np.linalg.inv(np.eye(N_ens) + S.transpose() @ S)
        
        # step 8: compute the analysis weights
        w = T @ S.transpose() @ delta

        # step 9: compute the square root of the transform
        V, Sigma, V_t = np.linalg.svd(T)
        T_sqrt = V @ np.diag(np.sqrt(Sigma)) @ V_t
        
        # step 10:  generate mean preserving random orthogonal matrix as in sakov oke 08
        U = rand_orth(N_ens)

        # step 11: package the transform output
        trans = [w, T_sqrt, U]

    return trans

########################################################################################################################
# auxilliary function for updating transform flavors of ensemble kalman filter 


def ens_update(analysis, ens, transform):

    if analysis=='enkf' or analysis=='enks':
        # step 0: unpack the tranform matrix
        [T] = transform

        # step 1: update the ensemble with right transform
        ens = ens @ T
    
    elif analysis=='etkf' or analysis=='etks':
        # step 0: unpack transform agruments and infer dimensions
        [w, T_sqrt, U] = transform
        [sys_dim, N_ens] = np.shape(ens)
        
        # step 1: compute the ensemble mean
        X_mean = np.mean(ens, axis=1)

        # step 2: compute the normalized anomalies, transposed
        A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)

        # step 3: compute the update, reshape for proper broadcasting
        ens = np.reshape(w, [N_ens, 1]) + T_sqrt @ U * np.sqrt(N_ens - 1)
        ens = (X_mean + ens.transpose() @ A_t).transpose()

    return ens

########################################################################################################################
# general filter code 


def ensemble_filter(analysis, ens, H, obs, obs_cov, state_infl, **kwargs):

    """General filter analysis step

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""

    # step 0: infer the system, observation and ensemble dimensions 
    [sys_dim, N_ens] = np.shape(ens)
    obs = np.squeeze(obs)
    obs_dim = len(obs)

    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_infl = kwargs['param_infl']

    else:
        state_dim = sys_dim

    # step 1: compute the tranform, update ensemble
    trans = transform(analysis, ens, H, obs, obs_cov)
    ens = ens_update(analysis, ens, trans)

    # step 2a: compute multiplicative inflation of state variables
    ens = inflate_state(ens, state_infl, sys_dim, state_dim)

    # step 2b: if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim:
        ens = inflate_param(ens, param_infl, sys_dim, state_dim)

    return {'ens': ens}

########################################################################################################################
# classical version lag_shift_smoother


def lag_shift_smoother_classic(analysis, ens, H, obs, obs_cov, state_infl, **kwargs):

    """Lag-shift ensemble kalman smoother analysis step, classical version

    This version of the lag-shift enks uses the last filtered state for the forecast, differentiated from the hybrid
    and iterative schemes which will use the once or multiple-times re-analized posterior for the initial condition
    for the forecast of the states to the next shift.

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""
    
    # step 0: unpack kwargs, posterior contains length lag past states ending with ens as final entry
    f_steps = kwargs['f_steps']
    step_model = kwargs['step_model']
    posterior = kwargs['posterior']
    
    # infer the ensemble, obs, and system dimensions, observation sequence includes shift forward times
    [obs_dim, shift] = np.shape(obs)
    [sys_dim, N_ens, lag] = np.shape(posterior)

    # optional parameter estimation
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_infl = kwargs['param_infl']
        param_wlk = kwargs['param_wlk']

    else:
        state_dim = sys_dim

    # step 1: create storage for the forecast and filter values over the DAW
    forecast = np.zeros([sys_dim, N_ens, shift])
    filtered = np.zeros([sys_dim, N_ens, shift])

    # step 2: forward propagate the ensemble and analyze the observations
    for s in range(shift):

        # initialize posterior for the special case lag=shift
        if lag==shift:
            posterior[:, :, s] = ens
        
        # step 2a: propagate between observation times
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        # step 2b: store the forecast to compute ensemble statistics before observations become available
        forecast[:, :, s] = ens

        # step 2c: perform the filtering step
        trans = transform(analysis, ens, H, obs[:, s], obs_cov)
        ens = ens_update(analysis, ens, trans)

        # compute multiplicative inflation of state variables
        ens = inflate_state(ens, state_infl, sys_dim, state_dim)

        # if including an extended state of parameter values,
        # compute multiplicative inflation of parameter values
        if state_dim != sys_dim:
            ens = inflate_param(ens, param_infl, sys_dim, state_dim)

        # store the filtered states
        filtered[:, :, s] = ens
        
        # step 2e: re-analyze the posterior in the lag window of states
        for l in range(lag):
            posterior[:, :, l] = ens_update(analysis, posterior[:, :, l], trans)
            
    # step 3: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim:
        param_ens = ens[state_dim: , :]
        param_ens = param_ens + param_wlk * np.random.standard_normal(np.shape(param_ens))
        ens[state_dim:, :] = param_ens
    
    return {'ens': ens, 'post': posterior, 'fore': forecast, 'filt': filtered}

########################################################################################################################
# single iteration, correlation-based lag_shift_smoother


def lag_shift_smoother_hybrid(analysis, ens, H, obs, obs_cov, state_infl, **kwargs):

    """Lag-shift ensemble kalman smoother analysis step, hybrid version

    This version of the lag-shift enks uses the final re-analyzed posterior initial state for the forecast, 
    which is pushed forward in time from the initial conidtion to shift-number of observation times.

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""
    
    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(ens)
    
    # observation sequence ranges from time +1 to time +lag
    [obs_dim, lag] = np.shape(obs)

    # unpack kwargs
    f_steps = kwargs['f_steps']
    step_model = kwargs['step_model']
    shift = kwargs['shift']
    
    # spin to be used on the first lag-assimilations -- this makes the smoothed time-zero re-analized prior
    # the first initial condition for the future iterations regardless of sda or mda settings
    spin = kwargs['spin']
    
    # multiple data assimilation (mda) is optional, read as boolean variable
    mda = kwargs['mda']
    if mda:
        obs_weights = kwargs['obs_weights']
    
    else:
        obs_weights = np.ones([lag])

    # optional parameter estimation
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_infl = kwargs['param_infl']
        param_wlk = kwargs['param_wlk']

    else:
        state_dim = sys_dim

    # step 1: create storage for the posterior, forecast and filter values over the DAW
    # only the shift-last and shift-first values are stored as these represent the newly forecasted values and
    # last-iterate posterior estimate respectively
    forecast = np.zeros([sys_dim, N_ens, shift])
    posterior = np.zeros([sys_dim, N_ens, shift])
    filtered = np.zeros([sys_dim, N_ens, shift])
    ens_0 = copy.copy(ens)

    # step 2: forward propagate the ensemble and analyze the observations
    for l in range(lag):
        
        # step 2a: propagate between observation times
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        # step 2b: store the forecast to compute ensemble statistics before observations become available
        if l >= (lag - shift):
            forecast[:, :, l - (lag - shift)] = ens

        # step 2c: perform the filtering step if in spin, multiple DA (mda=True) or
        # whenever the lag-forecast steps take us to new observations (l>=(lag - shift))
        if spin or mda or l >= (lag - shift):
            # observation sequence starts from the time of the inital condition
            # though we do not assimilate time zero observations
            trans = transform(analysis, ens, H, obs[:, l], obs_cov * obs_weights[l])
            ens = ens_update(analysis, ens, trans)

            if spin:
                # compute multiplicative inflation of state variables
                ens = inflate_state(ens, state_infl, sys_dim, state_dim)

                # if including an extended state of parameter values,
                # compute multiplicative inflation of parameter values
                if state_dim != sys_dim:
                    ens = inflate_param(ens, param_infl, sys_dim, state_dim)

            if l >= (lag - shift):
                # store the filtered states alone, not mda values
                filtered[:, :, l - (lag - shift)] = ens
        
        # step 2d: compute the re-analyzed initial condition if we have an assimilation update
        if spin or mda or l >= (lag - shift):
            ens_0 = ens_update(analysis, ens_0, trans)
            
    # step 3: propagate the posterior initial condition forward to the shift-forward time
    ens = copy.copy(ens_0)

    # step 3a: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim:
        param_ens = ens[state_dim: , :]
        param_ens = param_ens + param_wlk * np.random.standard_normal(np.shape(param_ens))
        ens[state_dim:, :] = param_ens

    # step 3b: propagate the re-analyzed, resampled-in-parameter-space ensemble up by shift
    # observation times
    for s in range(shift):
        posterior[:, :, s] = ens
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

    ens = inflate_state(ens, state_infl, sys_dim, state_dim)
        
    if state_dim != sys_dim:
        ens = inflate_param(ens, param_infl, sys_dim, state_dim)

    return {'ens': ens, 'post': posterior, 'fore': forecast, 'filt': filtered}

########################################################################################################################
########################################################################################################################
# Additional methods, non-standard, may have remaining bugs
########################################################################################################################
########################################################################################################################
# iterative_lag_shift_smoother


def lag_shift_smoother_iterative(analysis, ens, H, obs, obs_cov, state_infl, **kwargs):

    """Lag, shift iterative ensemble kalman smoother analysis step

    Optional keyword argument includes state dimension if there is an extended state including parameters.  In this
    case, a value for the parameter covariance inflation should be included in addition to the state covariance
    inflation."""
    
    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(ens)
    
    # observation sequence includes time zero, length lag + 1 
    [obs_dim, lag] = np.shape(obs)
    lag -= 1

    # unpack kwargs
    f_steps = kwargs['f_steps']
    step_model = kwargs['step_model']
    shift = kwargs['shift']
    mda = kwargs['mda']
    
    # optional parameter estimation
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_infl = kwargs['param_infl']
        param_wlk = kwargs['param_wlk']

    else:
        state_dim = sys_dim

    # step 1: create storage for the posterior, forecast and filter values over the DAW
    # only the shift-last and shift-first values are stored as these represent the newly forecasted values and
    # last-iterate posterior estimate respectively
    forecast = np.zeros([sys_dim, N_ens, shift])
    posterior = np.zeros([sys_dim, N_ens, shift])
    filtered = np.zeros([sys_dim, N_ens, shift])
    posterior[:, :, 0] = ens

    # step 2: forward propagate the ensemble and analyze the observations
    for l in range(1, lag+1):
        
        # step 2a: propagate between observation times
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        # step 2b: store the forecast to compute ensemble statistics before observations become available
        if l > (lag - shift):
            forecast[:, :, l - (lag - shift + 1)] = ens

        # step 2c: perform the filtering step if we do multiple data assimilation (mda=True) or
        # whenever the lag-forecast steps take us to new observations (l>(lag - shift))
        if mda or l > (lag - shift):
            # observation sequence starts from the time of the inital condition
            # though we do not assimilate time zero observations
            trans = transform(analysis, ens, H, obs[:, l], obs_cov)
            ens = ens_update(analysis, ens, trans)

            # compute multiplicative inflation of state variables
            ens = inflate_state(ens, state_infl, sys_dim, state_dim)

            # if including an extended state of parameter values,
            # compute multiplicative inflation of parameter values
            if state_dim != sys_dim:
                ens = inflate_param(ens, param_infl, sys_dim, state_dim)

            if l > (lag - shift):
                # store the filtered states alone, not mda values
                filtered[:, :, l - (lag - shift + 1)] = ens
        
        #  step 2d: we store the current posterior estimate for times within the initial shift window
        if l < shift:
            posterior[:, :, l] = ens
        
        # step 2e: find the re-analyzed posterior for the initial shift window
        # states, if we have an assimilation update
        if mda or l > (lag - shift):
            for m in range(shift):
                posterior[:, :, m] = ens_update(analysis, posterior[:, :, m], trans)
        
    # step 3: propagate the posterior initial condition forward to the shift-forward time
    # NOTE: MAY NEED TO SEND PYTHON A BUG REPORT, INEXPLICABLE CHANGE IN THE VALUE FOR THE
    # INITIAL POSTERIOR WHEN EVOLVING ENS WHEN USING THE PARAEMTER ENSEMBLE BUT NOT WHEN
    # ONLY ENSEMBLE OF DYNAMIC STATES, FIXED WITH COPY
    ens = copy.copy(np.squeeze(posterior[:, :, 0]))

    # step 3a: if performing parameter estimation, apply the parameter model
    if state_dim != sys_dim:
        param_ens = ens[state_dim: , :]
        param_ens = param_ens + param_wlk * np.random.standard_normal(np.shape(param_ens))
        ens[state_dim:, :] = param_ens

    for s in range(shift):
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

    # step 3b: compute multiplicative inflation of state variables
    ens = inflate_state(ens, state_infl, sys_dim, state_dim)

    # step 3c: if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim:
        ens = inflate_param(ens, param_infl, sys_dim, state_dim)

    return {'ens': ens, 'post': posterior, 'fore': forecast, 'filt': filtered}

########################################################################################################################
# IEnKF


def ienkf(ens, H, obs, obs_cov, state_infl, 
        epsilon=0.0001, tol=0.001, l_max=50, **kwargs):

    """Compute ienkf analysis as in algorithm 1, bocquet sakov 2014
    
    This should be considered always as a lag-1 smoother, the more general IEnKS is considered separately."""

    # step 0: infer the ensemble, obs, and state dimensions, and smoothing window,
    # unpack kwargs
    [sys_dim, N_ens] = np.shape(ens)
    obs = np.squeeze(obs)
    obs_dim = len(obs)
    f_steps = kwargs["f_steps"]
    step_model = kwargs["step_model"]

    # optional parameter estimation
    if 'state_dim' in kwargs:
        state_dim = kwargs['state_dim']
        param_infl = kwargs['param_infl']

    else:
        state_dim = sys_dim

    # lag-1 smoothing - an initial ensemble value should replace
    # the dummy ens argument in this case to be written consistently with other methods
    ens = kwargs['ens_0']
    
    # create storage for the posterior over the smoothing window
    posterior = np.zeros([sys_dim, N_ens, 2])

    # step 1: define the initial correction and interation count
    w = np.zeros(N_ens)
    l = 0

    # steps 2-3: compute the initial ensemble mean and non-normalized anomalies transposed
    X_mean_0 = np.mean(ens, axis=1)
    A_t = ens.transpose() - X_mean_0


    # define the initial iteration increment as dummy variable
    delta_w = np.ones(sys_dim)

    # step 4: begin while, break at max or reaching tolerance in the iterative step
    while np.sqrt(delta_w @ delta_w) >= tol:
        if l >= l_max:
            break

        # step 5: update the mean via the increment (always with the 0th iterate of X)
        X_mean = X_mean_0 + A_t.transpose() @ w

        # step 6: redefine the scaled ensemble in the bundle variant, with updated mean
        ens = (X_mean + epsilon * A_t).transpose()

        # step 7: compute the forward ensemble evolution
        for k in range(f_steps):
            ens = step_model(ens, **kwargs)

        # step 8: compute the mean of the forward ensemble in observation space
        Y_ens = H @ ens
        Y_mean = np.mean(Y_ens, axis=1)

        # step 9: compute the scaled Y ensemble of anomalies
        Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon

        # step 10: compute the approximate gradient of the cost function
        grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

        # step 11: compute the approximate hessian of the cost function
        hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

        # step 12: solve the system of equations for the update to w
        delta_w = solve(hess, grad_J)

        # steps 13 - 14: update w and the number of iterations
        w = w - delta_w
        l += 1

    # step 15: end while

    # step 16: update past ensemble with the current iterate of the ensemble mean, plus increment
    
    # generate mean preserving random orthogonal matrix as in sakov oke 08
    U = rand_orth(N_ens)
    
    # we compute the inverse square root of the hessian
    V, Sigma, V_t = np.linalg.svd(hess)
    hess_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # we use the current X_mean iterate, and transformed anomalies to define the new past ensemble
    ens = X_mean + np.sqrt(N_ens - 1) * (A_t.transpose() @ hess_sqrt_inv @ U).transpose()
    ens = ens.transpose()

    # store the posterior initial condition
    posterior[:, :, 0] = ens

    # step 17: forward propagate the ensemble
    for k in range(f_steps):
        ens = step_model(ens, **kwargs)
    
    # step 18 - 19: compute the inflated forward ensemble
    ens = inflate_state(ens, state_infl, sys_dim, state_dim)

    # step 19b: if including an extended state of parameter values,
    # compute multiplicative inflation of parameter values
    if state_dim != sys_dim:
        ens = inflate_param(ens, param_infl, sys_dim, state_dim)

    # store the analyzed and inflated current posterior - lag-1 filter
    posterior[:, :, 1] = ens

    return {'ens': ens, 'posterior': posterior}

########################################################################################################################
# Stochastic EnKF analysis step, using anomalies

def eakf(ens, H, obs, obs_cov, inflation=1.0):

    """This function performs the stochastic enkf analysis step as in Asch et al.

    This takes an ensemble, an observation operator, a matrix of (unbiased) perturbed observations and the ensemble
    estimated observational uncertainty, thereafter performing the analysis"""
    
    # step 0: infer the ensemble dimension, the system dimension, and observation dimension
    [sys_dim, N_ens] = np.shape(ens)
    obs_dim = len(obs)

    # step 1: generate perturbed observations
    obs_perts = np.random.multivariate_normal(np.zeros(obs_dim), obs_cov, N_ens)
    obs_ens = (obs + obs_perts).transpose()

    # step 2: compute the ensemble means and normalized anomalies
    X_mean = np.mean(ens, axis=1)
    y_mean = np.mean(H @ ens, axis=1)
    p_mean = np.mean(obs_perts, axis=0)

    A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)
    Y_t = (ens.transpose() @ H.transpose() - obs_perts - y_mean + p_mean) / np.sqrt(N_ens - 1)

    # step 3: compute the gain
    U, Sigma, V_t = np.linalg.svd(Y_t.transpose() @ Y_t)
    K_gain = A_t.transpose() @ Y_t @ U @ np.diag( 1 / Sigma) @ V_t

    # step 4: update the ensemble
    ens = ens + K_gain @ (obs_ens - H @ ens)

    # step 5: compute multiplicative inflation
    X_mean = np.mean(ens, axis=1) 
    A_t = ens.transpose() - X_mean
    infl = np.eye(N_ens) * inflation
    ens = (X_mean + infl @  A_t).transpose()
    
    return ens


########################################################################################################################
# Stochastic EnKF analysis step

def enkf_direct(ens, H, obs, obs_cov, inflation=1.0):

    """This function performs the stochastic enkf analysis step 

    This takes an ensemble, an observation operator, a matrix of (unbiased) perturbed observations and the ensemble
    estimated observational uncertainty, thereafter performing the analysis"""
    # first infer the ensemble dimension, the system dimension, and observation dimension
    [sys_dim, N_ens] = np.shape(ens)
    obs_dim = len(obs)

    # we compute the ensemble mean and normalized anomalies
    X_mean = np.mean(ens, axis=1)

    A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)

    # and the ensemble covariances
    S = A_t.transpose() @ A_t

    ## generate the unbiased perturbed observations
    obs_perts = np.random.multivariate_normal(np.zeros(obs_dim), obs_cov, N_ens)
    obs_perts = obs_perts - np.mean(obs_perts, axis=0)

    ## compute the empirical observation error covariance and the observation ensemble
    #obs_cov = (obs_perts.transpose() @ obs_perts) / (N_ens - 1)
    obs_ens = (obs + obs_perts).transpose()

    # we compute the ensemble based gain and the analysis ensemble
    K_gain = S @ H.transpose() @ np.linalg.inv(H @ S @ H.transpose() + obs_cov)
    ens = ens + K_gain @ (obs_ens - H @ ens)

    # the ensemble may be rank deficient, so we compute the pseudo inverse
    U, Sigma, V_t = np.linalg.svd(H @ S @ H.transpose() + obs_cov)
    K_gain = S @ H.transpose() @ U @ np.diag(1 / Sigma) @ V_t

    # compute the new ensemble mean
    X_mean = np.mean(ens, axis=1)

    # compute the inflated ensemble
    A_t = ens.transpose() - X_mean
    infl = np.eye(N_ens) * inflation
    ens = (X_mean + infl @  A_t).transpose()
    
    return ens


########################################################################################################################
# square root transform EnKF analysis step

def etkf_direct(ens, H, obs, obs_cov, inflation=1.0):

    """This function performs the ensemble transform ennkf analysis step
    
    This follows the direct implementation of the ETKF, without taking square roots of the observation
    error covariance, etc..."""

    # step 0: infer the system, observation and ensemble dimensions 
    [sys_dim, N_ens] = np.shape(ens)
    obs_dim = len(obs)

    # step 1: compute the ensemble mean
    x_mean = np.mean(ens, axis=1)

    # step 2: compute the normalized anomalies, transposed
    A_t = (ens.transpose() - x_mean) / np.sqrt(N_ens - 1)
    
    # step 3: compute the ensemble in observation space
    Z = H @ ens

    # step 4: compute the ensemble mean in observation space
    y_mean = np.mean(Z, axis=1)
    
    # then compute the normalized anomalies
    Y_t = (Z.transpose() - y_mean) / np.sqrt(N_ens - 1)

    # step 6: compute the innovation
    delta = obs - y_mean
   
    # step 7: compute the transform matrix
    T = np.linalg.inv(np.eye(N_ens) + Y_t @ np.linalg.inv(obs_cov) @ Y_t.transpose())
    
    # step 8: compute the analysis weights
    w = T @ Y_t @ np.linalg.inv(obs_cov) @ delta

    # step 9: update the ensemble
   
    # compute the square root of the transform
    V, Sigma, V_t = np.linalg.svd(T)
    T_sqrt = V @ np.diag(np.sqrt(Sigma)) @ V_t

    # generate mean preserving random orthogonal matrix as in sakov oke 08
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    Q, R = np.linalg.qr(Q)
    U_p =  np.zeros([N_ens, N_ens])
    U_p[0,0] = 1
    U_p[1:, 1:] = Q

    b_1 = np.ones(N_ens)/np.sqrt(N_ens)
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    B = np.zeros([N_ens, N_ens])
    B[:,0] = b_1
    B, R = np.linalg.qr(B)
    U = B @ U_p @ B.transpose()
    
    # compute the update, reshape for proper broadcasting
    ens = np.reshape(w, [N_ens, 1]) + T_sqrt @ U * np.sqrt(N_ens - 1)
    ens = (x_mean + ens.transpose() @ A_t).transpose()

    # compute the new ensemble mean
    X_mean = np.mean(ens, axis=1)

    # compute the inflated ensemble
    A_t = ens.transpose() - X_mean
    infl = np.eye(N_ens) * inflation
    ens = (X_mean + infl @  A_t).transpose()
    
    return ens



########################################################################################################################
# square root transform EnKF analysis step

def etkf_svd(ens, H, obs, obs_cov, inflation=1.0):

    """This function performs the ensemble transform ennkf analysis step
    
    This follows the version proposed by Raanes in DAPPER for better stability"""

    # step 0: infer the system and ensemble dimensions 
    [sys_dim, N_ens] = np.shape(ens)

    # step 1: compute the ensemble mean
    x_mean = np.mean(ens, axis=1)

    # step 2: compute the normalized anomalies, transposed
    A_t = (ens.transpose() - x_mean) / np.sqrt(N_ens - 1)
    
    # step 3: compute the ensemble in observation space
    Z = H @ ens

    # step 4: compute the ensemble mean in observation space
    y_mean = np.mean(Z, axis=1)
    
    # step 5: compute the weighted anomalies in observation space
    
    # first we find the observation error covariance inverse
    V, Sigma, V_t = np.linalg.svd(obs_cov)
    obs_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
    
    # then compute the weighted anomalies
    S = (Z.transpose() - y_mean) / np.sqrt(N_ens - 1)
    S = obs_sqrt_inv @ S.transpose()

    # finally we compute the SVD of the weighed anomalies to define the transform
    U, Sigma, V_t = np.linalg.svd(S)

    # step 6: compute the weighted innovation
    delta = obs_sqrt_inv @ ( obs - y_mean )
   
    # step 7: compute the transform matrix
    tmp = np.zeros(N_ens)
    tmp[:len(Sigma)] = Sigma[:]**2
    T = V_t.transpose() @ np.diag( 1 / (tmp + np.ones(N_ens))) @ V_t
    T_sqrt = V_t.transpose() @ np.diag(1 / np.sqrt(tmp + np.ones(N_ens))) @ V_t

    # step 8: compute the analysis weights
    w = T @ S.transpose() @ delta

    # step 9: update the ensemble
    
    # generate mean preserving random orthogonal matrix as in sakov oke 08
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    Q, R = np.linalg.qr(Q)
    U_p =  np.zeros([N_ens, N_ens])
    U_p[0,0] = 1
    U_p[1:, 1:] = Q

    b_1 = np.ones(N_ens)/np.sqrt(N_ens)
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    B = np.zeros([N_ens, N_ens])
    B[:,0] = b_1
    B, R = np.linalg.qr(B)
    U = B @ U_p @ B.transpose()

    # reshape w to correct for the broadcasting so that w acts as a matrix with columns of the original w 
    ens = np.reshape(w, [N_ens, 1]) + T_sqrt @ U * np.sqrt(N_ens - 1)
    ens = (x_mean + ens.transpose() @ A_t).transpose()
    
    # compute the new ensemble mean
    X_mean = np.mean(ens, axis=1)

    # compute the inflated ensemble
    A_t = ens.transpose() - X_mean
    infl = np.eye(N_ens) * inflation
    ens = (X_mean + infl @  A_t).transpose()
    
    return ens


########################################################################################################################
# IEnKF-T-LM


def ietlm(X_ext_ens, H, obs, obs_cov, f_steps, f, h, tau=0.001, e1=0,
         inflation=1.0, tol=0.001, l_max=40):

    """This produces an analysis ensemble via transform as in algorithm 3, bocquet sakov 2012"""

    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(X_ext_ens)
    obs_dim = len(obs)

    # step 1: we compute the ensemble mean and non-normalized anomalies
    X_mean_0 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_0

    # step 2: we define the initial iterative minimization parameters
    l = 0
    nu = 2
    w = np.zeros(N_ens)
    
    # step 3: update the mean via the w increment
    X_mean_1 = X_mean_0 + A_t.transpose() @ w
    X_mean_tmp = copy.copy(X_mean_1)

    # step 4: evolve the ensemble mean forward in time, and transform into observation space
    for k in range(f_steps):
        # propagate ensemble mean one step forward
        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)

    # define the observed mean by the propagated mean in the observation space
    Y_mean = H @ X_mean_tmp

    # step 5: Define the initial transform
    T = np.eye(N_ens)
    
    # step 6: redefine the ensemble with the updated mean and the transform
    X_ext_ens = (X_mean_1 + T @ A_t).transpose()

    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
    for k in range(f_steps):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)

    # step 8: compute the forecast anomalies in the observation space, via the observed, evolved mean and the 
    # observed, forward ensemble, conditioned by the transform
    Y_ens = H @ X_ext_ens
    Y_ens_t = np.linalg.inv(T).transpose() @ (Y_ens.transpose() - Y_mean) 

    # step 9: compute the cost function in ensemble space
    J = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens - 1) * w @ w
    
    # step 10: compute the approximate gradient of the cost function
    grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

    # step 11: compute the approximate hessian of the cost function
    hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

    # step 12: compute the infinity norm of the jacobian and the max of the hessian diagonal
    flag = np.max(np.abs(grad_J)) > e1
    mu = tau * np.max(np.diag(hess))

    # step 13: while loop
    while flag: 
        if l > l_max:
            break

        # step 14: set the iteration count forward
        l+= 1
        
        # step 15: solve the system for the w increment update
        delta_w = solve(hess + mu * np.eye(N_ens),  -1 * grad_J)

        # step 16: check if the increment is sufficiently small to terminate
        if np.sqrt(delta_w @ delta_w) < tol:
            # step 17: flag false to terminate
            flag = False

        # step 18: begin else
        else:
            # step 19: reset the ensemble adjustment
            w_prime = w + delta_w
            
            # step 20: reset the initial ensemble with the new adjustment term
            X_mean_1 = X_mean_0 + A_t.transpose() @ w_prime
            
            # step 21: forward propagate the new ensemble mean, and transform into observation space
            X_mean_tmp = copy.copy(X_mean_1)
            for k in range(f_steps):
                X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
            
            Y_mean = H @ X_mean_tmp

            # steps 22 - 24: define the parameters for the confidence region
            L = 0.5 * delta_w @ (mu * delta_w - grad_J)
            J_prime = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens -1) * w_prime @ w_prime
            theta = (J - J_prime) / L

            # step 25: evaluate if new correction needed
            if theta > 0:
                
                # steps 26 - 28: update the cost function, the increment, and the past ensemble, conditioned with the
                # transform
                J = J_prime
                w = w_prime
                X_ext_ens = (X_mean_1 + T.transpose() @ A_t).transpose()

                # step 29: integrate the ensemble forward in time
                for k in range(f_steps):
                    X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)

                # step 30: compute the forward anomlaies in the observation space, by the forward evolved mean and forward evolved
                # ensemble
                Y_ens = H @ X_ext_ens
                Y_ens_t = np.linalg.inv(T).transpose() @ (Y_ens.transpose() - Y_mean)

                # step 31: compute the approximate gradient of the cost function
                grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

                # step 32: compute the approximate hessian of the cost function
                hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

                # step 33: define the transform as the inverse square root of the hessian
                V, Sigma, V_t = np.linalg.svd(hess)
                T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

                # steps 34 - 35: compute the tolerance and correction parameters
                flag = np.max(np.abs(grad_J)) > e1
                mu = mu * np.max([1/3, 1 - (2 * theta - 1)**3])
                nu = 2

            # steps 36 - 37: else statement, update mu and nu
            else:
                mu = mu * nu
                nu = nu * 2

            # step 38: end if
        # step 39: end if
    # step 40: end while

    # step 41: perform update to the initial mean with the new defined anomaly transform 
    X_mean_1 = X_mean_0 + A_t.transpose() @ w

    # step 42: define the transform as the inverse square root of the hessian, bundle version only
    #V, Sigma, V_t = np.linalg.svd(hess)
    #T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # step 43: compute the updated ensemble by the transform conditioned anomalies and updated mean
    X_ext_ens = (T.transpose() @ A_t + X_mean_1).transpose()
    
    # step 44: forward propagate the ensemble to the observation time 
    for k in range(f_steps):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
   
    # step 45: compute the ensemble with inflation
    X_mean_2 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_2
    infl = np.eye(N_ens) * inflation
    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()

    return X_ext_ens

########################################################################################################################
# IEnKF-B-LM


def ieblm(X_ext_ens, H, obs, obs_cov, f_steps, f, h, tau=0.001, e1=0, epsilon=0.0001,
         inflation=1.0, tol=0.001, l_max=40):

    """This produces an analysis ensemble as in algorithm 3, bocquet sakov 2012"""

    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(X_ext_ens)
    obs_dim = len(obs)

    # step 1: we compute the ensemble mean and non-normalized anomalies
    X_mean_0 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_0

    # step 2: we define the initial iterative minimization parameters
    l = 0
    
    # NOTE: MARC'S VERSION HAS NU SET TO ONE FIRST AND THEN ITERATES ON THIS IN PRODUCTS
    # OF TWO    
    #nu = 2
    nu = 1

    w = np.zeros(N_ens)
    
    # step 3: update the mean via the w increment
    X_mean_1 = X_mean_0 + A_t.transpose() @ w
    X_mean_tmp = copy.copy(X_mean_1)

    # step 4: evolve the ensemble mean forward in time, and transform into observation space
    for k in range(f_steps):
        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)

    Y_mean = H @ X_mean_tmp

    # step 5: Define the initial transform, transform version only
    # T = np.eye(N_ens)
    
    # step 6: redefine the ensemble with the updated mean, rescaling by epsilon
    X_ext_ens = (X_mean_1 + epsilon * A_t).transpose()

    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
    for k in range(f_steps):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)

    # step 8: compute the anomalies in the observation space, via the observed, evolved mean and the observed, 
    # forward ensemble, rescaling by epsilon
    Y_ens = H @ X_ext_ens
    Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon

    # step 9: compute the cost function in ensemble space
    J = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens - 1) * w @ w
    
    # step 10: compute the approximate gradient of the cost function
    grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

    # step 11: compute the approximate hessian of the cost function
    hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

    # step 12: compute the infinity norm of the jacobian and the max of the hessian diagonal
    # NOTE: MARC'S VERSION DOES NOT HAVE A FLAG BASED ON THE INFINITY NORM OF THE GRADIENT
    # THIS IS ALSO PROBABLY A TRIVIAL FLAG
    # flag = np.max(np.abs(grad_J)) > e1
    
    # NOTE: MARC'S FLAG
    flag = True

    # NOTE: MARC'S VERSION USES MU=1 IN THE FIRST ITERATION AND NEVER MAKES
    # THIS DECLARATION IN TERMS OF TAU AND HESS
    # mu = tau * np.max(np.diag(hess))
    mu = 1
    
    # step 13: while loop
    while flag: 
        if l > l_max:
            print(l)
            break

        # step 14: set the iteration count forward
        l+= 1
        
        # NOTE: MARC'S RE-DEFINITION OF MU AND NU
        mu *= nu
        nu *= 2

        # step 15: solve the system for the w increment update
        delta_w = solve(hess + mu * np.eye(N_ens),  -1 * grad_J)

        # step 16: check if the increment is sufficiently small to terminate
        # NOTE: MARC'S VERSION NORMALIZES THE LENGTH RELATIVE TO THE ENSEMBLE SIZE
        if np.sqrt(delta_w @ delta_w) < tol:
            # step 17: flag false to terminate
            flag = False
            print(l)

        # step 18: begin else
        else:
            # step 19: reset the ensemble adjustment
            w_prime = w + delta_w
            
            # step 20: reset the initial ensemble with the new adjustment term
            X_mean_1 = X_mean_0 + A_t.transpose() @ w_prime
            
            # step 21: forward propagate the new ensemble mean, and transform into observation space
            X_mean_tmp = copy.copy(X_mean_1)
            for k in range(f_steps):
                X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)
            
            Y_mean = H @ X_mean_tmp

            # steps 22 - 24: define the parameters for the confidence region
            L = 0.5 * delta_w @ (mu * delta_w - grad_J)
            J_prime = 0.5 * (obs - Y_mean) @ np.linalg.inv(obs_cov) @ (obs - Y_mean) + 0.5 * (N_ens -1) * w_prime @ w_prime
            theta = (J - J_prime) / L
            
            # step 25: evaluate if new correction needed
            if theta > 0:
                
                # steps 26 - 28: update the cost function, the increment, and the past ensemble, rescaled with epsilon
                J = J_prime
                w = w_prime
                X_ext_ens = (X_mean_1 + epsilon * A_t).transpose()

                # step 29: integrate the ensemble forward in time
                for k in range(f_steps):
                    X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)

                # step 30: compute the forward anomlaies in the observation space, by the forward evolved mean and forward evolved
                # ensemble
                Y_ens = H @ X_ext_ens
                Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon

                # step 31: compute the approximate gradient of the cost function
                grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

                # step 32: compute the approximate hessian of the cost function
                hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

                # step 33: define the transform as the inverse square root of the hessian, transform version only
                #V, Sigma, V_t = np.linalg.svd(hess)
                #T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

                # steps 34 - 35: compute the tolerance and correction parameters
                # NOTE: TRIVIAL FLAG?
                # flag = np.max(np.abs(grad_J)) > e1
                
                mu = mu * np.max([1/3, 1 - (2 * theta - 1)**3])
                
                # NOTE: ADJUSTMENT HERE TO MATCH NU TO MARC'S CODE
                # nu = 2
                nu = 1

            # steps 36 - 37: else statement, update mu and nu
            #else:
            #    mu = mu * nu
            #    nu = nu * 2

            # step 38: end if
        # step 39: end if
    # step 40: end while
    
    # step 41: perform update to the initial mean with the new defined anomaly transform 
    X_mean_1 = X_mean_0 + A_t.transpose() @ w

    # step 42: define the transform as the inverse square root of the hessian
    V, Sigma, V_t = np.linalg.svd(hess)
    T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # step 43: compute the updated ensemble by the transform conditioned anomalies and updated mean
    X_ext_ens = (T.transpose() @ A_t + X_mean_1).transpose()
    
    # step 44: forward propagate the ensemble to the observation time 
    for k in range(f_steps):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
    
    # step 45: compute the ensemble with inflation
    X_mean_2 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_2
    infl = np.eye(N_ens) * inflation
    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()

    return X_ext_ens


########################################################################################################################
# IEnKF

def ienkf_f(ens, H, obs, obs_cov, f_steps, h, 
        epsilon=0.0001, infl=1.0, tol=0.001, l_max=50):

    """Compute Ienkf analysis as in algorithm 1, bocquet sakov 2014"""

    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(ens)
    sys_dim = sys_dim - 1
    obs_dim = len(obs)

    # step 1: we define the initial iterative minimization parameters
    l = 0
    w = np.zeros(N_ens)
    delta_w = np.ones(N_ens)
    
    # step 2: compute the ensemble mean
    X_mean_0 = np.mean(ens, axis=1)
    
    # step 3: compute the non-normalized ensemble of anomalies (transposed)
    A_t = ens.transpose() - X_mean_0

    # step 4: begin while
    while np.sqrt(delta_w @ delta_w) >= tol:
        if l >= l_max:
            break

        # step 5: update the mean via the increment (always with the 0th iterate)
        X_mean = X_mean_0 + A_t.transpose() @ w

        # step 6: redefine the scaled ensemble in the bundle variant, with updated mean
        ens = (X_mean + epsilon * A_t).transpose()

        # step 7: compute the forward ensemble evolution
        for j in range(N_ens):
            for k in range(f_steps):
                ens[:sys_dim, j] = l96_rk4_step(ens[:sys_dim, j], h, ens[sys_dim, j])

        # step 8: compute the mean of the forward ensemble in observation space
        Y_ens = H @ ens
        Y_mean = np.mean(Y_ens, axis=1)

        # step 9: compute the scaled Y ensemble of anomalies
        Y_ens_t = (Y_ens.transpose() - Y_mean) / epsilon

        # step 10: compute the approximate gradient of the cost function
        grad_J = (N_ens - 1) * w - Y_ens_t @ np.linalg.inv(obs_cov) @ (obs - Y_mean)

        # step 11: compute the approximate hessian of the cost function
        hess = (N_ens - 1) * np.eye(N_ens) + Y_ens_t @  np.linalg.inv(obs_cov) @ Y_ens_t.transpose()

        # step 12: solve the system of equations for the update to w
        delta_w = solve(hess, grad_J)

        # steps 13 - 14: update w and the number of iterations
        w = w - delta_w
        l += 1


    # step 15: end while

    # step 16: update past ensemble with the current iterate of the ensemble mean, plus increment
    
    # generate mean preserving random orthogonal matrix as in sakov oke 08
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    Q, R = np.linalg.qr(Q)
    U_p =  np.zeros([N_ens, N_ens])
    U_p[0,0] = 1
    U_p[1:, 1:] = Q

    b_1 = np.ones(N_ens)/np.sqrt(N_ens)
    Q = np.random.standard_normal([N_ens -1, N_ens -1])
    B = np.zeros([N_ens, N_ens])
    B[:,0] = b_1
    B, R = np.linalg.qr(B)
    U = B @ U_p @ B.transpose()
    
    # we compute the inverse square root of the hessian
    V, Sigma, V_t = np.linalg.svd(hess)
    hess_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # we use the current X_mean iterate, and transformed anomalies to define the new past ensemble
    ens = X_mean + np.sqrt(N_ens - 1) * (A_t.transpose() @ hess_sqrt_inv @ U).transpose()
    ens = ens.transpose()

    # step 17: forward propagate the ensemble
    for j in range(N_ens):
        for k in range(f_steps):
            ens[:sys_dim, j] = l96_rk4_step(ens[:sys_dim, j], h, ens[sys_dim, j])
    
    # step 18: compute the forward ensemble mean
    X_mean = np.mean(ens, axis=1)

    # step 19: compute the inflated forward ensemble
    A_t = ens.transpose() - X_mean
    infl = np.eye(N_ens) * infl
    ens = (X_mean + infl @  A_t).transpose()

    return ens

