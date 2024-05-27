"""


AROME-specific version of skill_spread

"""

import copy

import numpy as np


def skill_spread(obs_data, fake_data):
    """

    Inputs :

        fake_data : N x C x H x W array with N samples

        obs_data : C x H x W array observation

    Returns :

        sp_out : 2 x C x H x W array containing the result 0 is skill and 1 is spread

    """
    N, C, H, W = fake_data.shape

    sp_out = np.zeros((2, C, H, W))

    fake_data_p = copy.deepcopy(fake_data)
    obs_data_p = copy.deepcopy(obs_data)

    skill = fake_data_p.mean(axis=0) - obs_data_p

    var = fake_data_p.var(axis=0, ddof=1)

    sp_out[0] = skill

    sp_out[1] = (
        var  # var is actually needed for correctly calculating the spread, which is done in the vizualization code
    )

    return sp_out
