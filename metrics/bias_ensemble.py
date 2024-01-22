"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics.wind_comp as wc
import copy


def bias_ens(obs_data, fake_data):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation
        
    Returns :
        
        bias : avg(fake_data) - obs_data  
    
    """
    
    N, C, H, W  = fake_data.shape
    
    
    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)


    
    #print(fake_data_p.shape, obs_data_p.shape)
    
    #print(fake_data_p[:,0].max(), fake_data_p[:,1].max(), fake_data_p[:,2].max())

    
    
    
    fake_data_p_mean = np.nanmean(fake_data_p, axis=0)
    X_bias = fake_data_p_mean - obs_data_p


    return X_bias


def abs_bias_penalty(fake, real,debiasing=False):
   
   bias = bias_ens(real, fake, real, debiasing=debiasing)

   
