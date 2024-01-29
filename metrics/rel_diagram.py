"""


AROME-specific version of skill_spread

"""

import copy

import numpy as np


def rel_diag(obs_data, fake_data, parameters):
    """
    
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : C x H x W array observation
        
    Returns :
        
        rel : N_param x 2 x C x H x W   
    
    """

    N, C, H, W = fake_data.shape

    N_param = parameters.shape[1]

    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)

    rel = np.zeros((N_param, 2, C, H, W))

    for i in range(parameters.shape[1]):
        T_tr = parameters[1, i]
        ff_tr = parameters[0, i]

        X_tr = np.zeros((C, N, H, W))
        O_tr = np.zeros((C, H, W))
        O_tr[:] = np.nan

        """
        Converting forecasts and observation
        """

        X_tr[0, fake_data_p[:, 0] > ff_tr] = 1.0
        X_tr[2, fake_data_p[:, 2] > T_tr] = 1.0
        O_tr[0, obs_data_p[0] > ff_tr] = 1
        O_tr[2, obs_data_p[2] > T_tr] = 1
        O_tr[0, obs_data_p[0] < ff_tr] = 0
        O_tr[2, obs_data_p[2] < T_tr] = 0

        X_prob = X_tr.sum(axis=1) / N
        rel[i, 0] = X_prob
        rel[i, 1] = O_tr

    return rel
