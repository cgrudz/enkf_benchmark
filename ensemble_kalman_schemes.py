import numpy as np
import copy
#import scipy as sp

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
# stochastic ensemble kalman smoother
#
#def enks_stoch_analysis(ens, H, obs, obs_cov):
#    """This function performs the ensemble kalman smoother analysis step
#
#    """
#    # first infer the ensemble dimension, system dimension and lag dimension
#    # observation time points will be given from earliest to latest observations
#    # in the index increasing in the second dimension
#    [sys_dim, N_ens] = np.shape(ens)
#    [obs_dim, lag] = np.shape(obs)
#    
#    ens_init = copy.copy(ens)
#
#    # loop over the observation time series
#    for i in range(lag):
#
#    
