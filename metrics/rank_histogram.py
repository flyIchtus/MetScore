"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics.wind_comp as wc
import copy


def rank_histo(obs_data, fake_data):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation
        
    Returns :
        
        bins (C, 121) max number of members in the ensemble  
    
    """
    
    N, C, H, W  = fake_data.shape
    
    
    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)
    

    bins = np.zeros((C, 121)) # 121 since N=120 is the biggest ensemble... Maybe use the biggest number that we will use 1000? 
    for i in range(C):
        
        obs_data_var = copy.deepcopy(obs_data_p[i])
        fake_data_var = copy.deepcopy(fake_data_p[:,i])
        
        obs = obs_data_var[ ~np.isnan(obs_data_var)]
        ens = fake_data_var[:, ~np.isnan(obs_data_var)]
        

        for j in range(obs.shape[0]):
        
            ens_sort = ens[:,j]
            ens_sort.sort()
            ens_sort = np.concatenate((ens_sort, [9999999.]))
            out = np.where((ens_sort < obs[j]), True, False)
            bins[i, np.argmax(out==False)] +=1 
            

        
    return bins
