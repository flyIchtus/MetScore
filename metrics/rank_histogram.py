"""


AROME-specific version of skill_spread

"""

import copy

import numpy as np


def rank_histo(obs_data, fake_data):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation
        
    Returns :
        
        bins (C, N + 1)
    """
    
    N, C, H, W  = fake_data.shape
    
    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)

    bins = np.zeros((C, fake_data.shape[0] + 1))
    for i in range(C):
        
        obs_data_var = obs_data_p[i]
        fake_data_var = fake_data_p[:,i]
   
        obs = obs_data_var[~np.isnan(obs_data_var)]
        ens = fake_data_var[:,~np.isnan(obs_data_var)]

        for j in range(obs.shape[0]):
        
            ens_sort = ens[:,j]
            ens_sort.sort()
            ens_sort = np.concatenate((ens_sort, [9999999.]))
            out = np.where((ens_sort < obs[j]), True, False)
            bins[i, np.argmax(out==False)] += 1 / obs.shape[0]
        
    return bins


def unreliability(rankHisto, N_obs):
    """
    Inputs :
        
        rankHisto :  C x N_bins array with C different variables    
        
    Returns :
        
        Delta :  (C,) shaped array, estimation of sum(n - Delta0)**2 for each bin n onf rankHisto
        Delta0 : flot, the flat histogram value
        unreliability : (C,) shaped array, Delta / Delta0
    """
    C, N_bins = rankHisto.shape

    delta0 = N_obs * (N_bins - 1)  / (N_bins)

    delta = np.zeros((C,))

    for var_idx in range(C):
        delta[var_idx] = ((rankHisto[var_idx,:] - (N_obs /N_bins)) ** 2).mean()

    return delta, delta0, delta / delta0
