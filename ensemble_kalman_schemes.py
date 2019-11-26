import numpy as np
import copy
#import scipy as sp
from numpy.linalg import solve
import ipdb
from l96 import l96_rk4_stepV, l96_rk4_step

########################################################################################################################

def analyze_ensemble(ens, truth):
    """This will compute the ensemble RMSE as compared with the true twin, and the spread."""

    # infer the shapes
    [sys_dim, N_ens] = np.shape(ens)

    # compute the ensemble mean
    mean = np.mean(ens, axis=1)

    # compute the RMSE of the ensemble mean
    rmse = np.sqrt( np.mean( (truth - mean)**2 ) )

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the ensemble
    spread = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (mean - ens.transpose())**2, axis=1)))

    return [rmse, spread]

########################################################################################################################

def analyze_ensemble_parameters(ens, truth):
    """This will compute the ensemble RMSE as compared with the true twin, and the spread."""

    # infer the shapes
    [sys_dim, N_ens] = np.shape(ens)

    # compute the ensemble mean
    mean = np.mean(ens, axis=1)
    # compute the RMSE of the ensemble mean, where each value is computed relative to the magnitude of the parameter
    rmse = np.sqrt( np.mean( (truth - mean)**2 / truth**2 ) )

    # we compute the spread as in whitaker & louge 98 by the standard deviation of the mean square deviation of the ensemble,
    # with the variation in which we weight each approximate sample of the variance by the size of the parameter square
    spread = np.sqrt( ( 1 / (N_ens - 1) ) * np.sum(np.mean( (mean - ens.transpose())**2 / (np.ones([N_ens, sys_dim]) * truth**2), axis=1)))

    return [rmse, spread]



########################################################################################################################
# Stochastic EnKF analysis step

def enkf_stoch_analysis(ens, H, obs, obs_cov):

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
    obs_cov = (obs_perts.transpose() @ obs_perts) / (N_ens - 1)
    obs_ens = (obs + obs_perts).transpose()

    # we compute the ensemble based gain and the analysis ensemble
    K_gain = S @ H.transpose() @ np.linalg.inv(H @ S @ H.transpose() + obs_cov)
    ens = ens + K_gain @ (obs_ens - H @ ens)

    return ens


########################################################################################################################
# square root transform EnKF analysis step

def enkf_deter_analysis(ens, H, obs, obs_cov):

    """This function performs the ensemble transform enkf analysis step"""


    # first infer the ensemble dimension and the system dimension 
    [sys_dim, N_ens] = np.shape(ens)

    # we compute the ensemble mean and normalized anomalies
    x_mean = np.mean(ens, axis=1)
    A_t = (ens.transpose() - x_mean) / np.sqrt(N_ens - 1)
    
    # compute the observed anomalies and the ensemble mean in observation space
    Y_t = A_t @ H.transpose()
    y_mean = H @ x_mean
    
    ## We compute the transform as in asch et al. 
    
    # compute the square root of the observation error covariance inverse
    V, Sigma, V_t = np.linalg.svd(obs_cov)
    obs_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t
    
    # we compute the square root of the obs_cov_inv with the observed anomalies
    S = obs_sqrt_inv @ Y_t.transpose()

    # this defines the T matrix, which we compute the square root of
    T = np.linalg.inv(np.eye(N_ens) + S.transpose() @ S)
    V, Sigma, V_t = np.linalg.svd(T)
    T_sqrt = V @ np.diag(np.sqrt(Sigma)) @ V_t
    
    # weighted innovation
    delta = obs_sqrt_inv @ ( obs - y_mean )
    
    # compute the analysis weights
    w = T @ S.transpose() @ delta

    # correct for the broadcasting so that w acts as a matrix with columns of the original w 
    w = np.reshape(w, [N_ens, 1]) + np.sqrt(N_ens - 1) * T_sqrt

    # transform the anomalies, then combine with the mean to reconstruct the ensemble
    ens = (A_t.transpose() @ w)
    ens = (ens.transpose() + x_mean).transpose()
    
    return ens



########################################################################################################################
# stochastic transform matrix

def stoch_transform(ens, H, obs, obs_cov):

    # infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(ens)
    obs_dim = len(obs)

    # we compute the ensemble mean and normalized anomalies
    X_mean = np.mean(ens, axis=1)

    # A_t will be the normalized anomaly matrix transposed
    A_t = (ens.transpose() - X_mean) / np.sqrt(N_ens - 1)

    # generate the unbiased perturbed observations
    obs_perts = np.random.multivariate_normal(np.zeros(obs_dim), obs_cov, N_ens)
    obs_perts = obs_perts - np.mean(obs_perts, axis=0)

    # compute the empirical observation error covariance and the observation ensemble
    obs_cov = (obs_perts.transpose() @ obs_perts) / (N_ens - 1)
    obs_ens = (obs + obs_perts).transpose()

    # create the ensemble transform matrix
    Y_t = A_t @ H.transpose()
    C = Y_t.transpose() @ Y_t + obs_cov
    T = np.eye(N_ens) + Y_t @ np.linalg.inv(C) @ (obs_ens - H @ ens) / np.sqrt(N_ens - 1)
    
    return T

########################################################################################################################
# stochastic transform analysis

def enkf_trans_analysis(ens, H, obs, obs_cov):

    T = stoch_transform(ens, H, obs, obs_cov)

    return ens @ T


########################################################################################################################
# IEnKF

def ienkf_analysis(X_ext_ens, H, obs, obs_cov, tanl, f, h, 
        epsilon=0.0001, inflation=1.0, tol=0.001, l_max=10):

    """Compute Ienkf analysis as in algorithm 1, bocquet sakov 2014"""

    # step 0: infer the ensemble, obs, and state dimensions
    [sys_dim, N_ens] = np.shape(X_ext_ens)
    obs_dim = len(obs)

    # step 1: we define the initial iterative minimization parameters
    l = 0
    w = np.zeros(N_ens)
    delta_w = np.ones(N_ens)
    
    # step 2: compute the ensemble mean
    X_mean_0 = np.mean(X_ext_ens, axis=1)
    
    # step 3: compute the non-normalized ensemble of anomalies (transposed)
    A_t = X_ext_ens.transpose() - X_mean_0

    # step 4: begin while
    while np.sqrt(delta_w @ delta_w) >= tol:
        if l >= l_max:
            break

        # step 5: update the mean via the increment (always with the 0th iterate)
        X_mean = X_mean_0 + A_t.transpose() @ w

        # step 6: redefine the scaled ensemble in the bundle variant, with updated mean
        X_ext_ens = (X_mean + epsilon * A_t).transpose()

        # step 7: compute the forward ensemble evolution
        for k in range(tanl):
            X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)

        # step 8: compute the mean of the forward ensemble in observation space
        Y_ens = H @ X_ext_ens
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
    
    # we compute the inverse square root of the hessian
    V, Sigma, V_t = np.linalg.svd(hess)
    hess_sqrt_inv = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # we use the current X_mean iterate, and transformed anomalies to define the new past ensemble
    X_ext_ens = X_mean + np.sqrt(N_ens - 1) * (A_t.transpose() @ hess_sqrt_inv).transpose()
    X_ext_ens = X_ext_ens.transpose()

    # step 17: forward propagate the ensemble
    for k in range(tanl):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
    
    # step 18: compute the forward ensemble mean
    X_mean = np.mean(X_ext_ens, axis=1)

    # step 19: compute the inflated forward ensemble
    A_t = X_ext_ens.transpose() - X_mean
    infl = np.eye(N_ens) * inflation
    X_ext_ens = (X_mean + infl @  A_t).transpose()

    return X_ext_ens

########################################################################################################################
# IEnKF-T-LM


def ietlm_analysis(X_ext_ens, H, obs, obs_cov, tanl, f, h, tau=0.001, e1=0,
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
    for k in range(tanl):
        # propagate ensemble mean one step forward
        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)

    # define the observed mean by the propagated mean in the observation space
    Y_mean = H @ X_mean_tmp

    # step 5: Define the initial transform
    T = np.eye(N_ens)
    
    # step 6: redefine the ensemble with the updated mean and the transform
    X_ext_ens = (X_mean_1 + T @ A_t).transpose()

    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
    for k in range(tanl):
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
            for k in range(tanl):
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
                for k in range(tanl):
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
    for k in range(tanl):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
   
    # step 45: compute the ensemble with inflation
    X_mean_2 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_2
    infl = np.eye(N_ens) * inflation
    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()

    return X_ext_ens

########################################################################################################################
# IEnKF-B-LM


def ieblm_analysis(X_ext_ens, H, obs, obs_cov, tanl, f, h, tau=0.001, e1=0, epsilon=0.0001,
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
    nu = 2
    w = np.zeros(N_ens)
    
    # step 3: update the mean via the w increment
    X_mean_1 = X_mean_0 + A_t.transpose() @ w
    X_mean_tmp = copy.copy(X_mean_1)

    # step 4: evolve the ensemble mean forward in time, and transform into observation space
    for k in range(tanl):
        X_mean_tmp = l96_rk4_step(X_mean_tmp, h, f)

    Y_mean = H @ X_mean_tmp

    # step 5: Define the initial transform, transform version only
    # T = np.eye(N_ens)
    
    # step 6: redefine the ensemble with the updated mean, rescaling by epsilon
    X_ext_ens = (X_mean_1 + epsilon * A_t).transpose()

    # step 7: loop over the discretization steps between observations to produce a forecast ensemble
    for k in range(tanl):
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
            for k in range(tanl):
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
                for k in range(tanl):
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

    # step 42: define the transform as the inverse square root of the hessian
    V, Sigma, V_t = np.linalg.svd(hess)
    T = V @ np.diag( 1 / np.sqrt(Sigma) ) @ V_t

    # step 43: compute the updated ensemble by the transform conditioned anomalies and updated mean
    X_ext_ens = (T.transpose() @ A_t + X_mean_1).transpose()
    
    # step 44: forward propagate the ensemble to the observation time 
    for k in range(tanl):
        X_ext_ens = l96_rk4_stepV(X_ext_ens, h, f)
   
    # step 45: compute the ensemble with inflation
    X_mean_2 = np.mean(X_ext_ens, axis=1)
    A_t = X_ext_ens.transpose() - X_mean_2
    infl = np.eye(N_ens) * inflation
    X_ext_ens = (X_mean_2 + infl @  A_t).transpose()

    return X_ext_ens

