"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics.wind_comp as wc
import copy

def skill_spread(obs_data, fake_data):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation
        
    Returns :
        
        sp_out : 2 x C x H x W array containing the result 0 is skill and 1 is spread
    
    """
    N, C, H, W  = fake_data.shape
    
    sp_out = np.zeros((2,C,H,W))
    

    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)

    skill = fake_data_p.mean(axis=0) - obs_data_p
    
    spread = fake_data_p.std(axis=0)
    
    
    sp_out[0] = skill
    
    sp_out[1] = spread


        
    return sp_out
