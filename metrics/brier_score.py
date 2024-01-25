import properscoring as ps
import numpy as np
import metrics.wind_comp as wc
import copy

def brier_score(obs_data, fake_data, parameters):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation

        parameters : 2 x 6 array with thresholds for calculating the brier score
        
    Returns :
        
        brier score  : N_brier x C x H x W array containing the result N_brier number of thresholds
    
    """
    N, C, H, W  = fake_data.shape
    
    N_brier = parameters.shape[1]
    
    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)
    
    brier = np.zeros((N_brier,  C, H, W))
    
    
    for i in range(parameters.shape[1]):    

    
        T_brier = parameters[1, i]
        ff_brier = parameters[0, i]
    
        X_brier = np.zeros((C,N,H,W))
        O_brier = np.zeros((C,H,W))
        O_brier[:] = np.nan
        
        """
        Converting forecasts and observation
        """
    
        X_brier[0, fake_data_p[:,0] > ff_brier] = 1.0
        X_brier[2, fake_data_p[:,2] > T_brier] = 1.0
    
        X_brier_prob = X_brier.sum(axis = 1) / N
        O_brier[0, obs_data_p[0] > ff_brier] = 1
        O_brier[2, obs_data_p[2] > T_brier] = 1
        O_brier[0, obs_data_p[0] < ff_brier] = 0
        O_brier[2, obs_data_p[2] < T_brier] = 0
        
        brier[i] = ps.brier_score(O_brier, X_brier_prob)
        
        #print(np.nanmax(obs_data_p[0]), np.nanmin(obs_data_p[0]))
        
    return brier
