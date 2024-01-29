#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:15:29 2022

@author: brochetc

Metric version of quantiles calculation

"""

import numpy as np


def quantiles(data, qlist):
    """
    compute quantiles of data shape on first axis using numpy 'primitive'
    
    Inputs :
        
        data : np.array, shape B x C x H x W
        
        qlist : iterable of size N containing quantiles to compute 
        (between 0 and 1 inclusive)
        
    Returns :
        
        np.array of shape N x C x H x W
    """

    return np.quantile(data, qlist, axis=0)


def quantile_score(real_data, fake_data, qlist=[0.99]):
    """
    compute rmse of quantiles maps as outputted by quantiles function
    
    Inputs :
        
        real_data : np.array of shape B x C x H x W
        
        fake_data : np.array of shape B x C x H x W
    
        qlist : iterable of length N containing quantiles to compute 
        ((between 0 and 1 inclusive)
    Returns :
        
        q_score : np.array of length N x C
    
    """

    q_real = quantiles(real_data, qlist)
    q_fake = quantiles(fake_data, qlist)

    q_score = np.sqrt((q_fake - q_real) ** 2).mean(axis=(2, 3))

    return q_score
