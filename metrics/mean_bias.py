#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:14:51 2023

@author: brochetc
"""

import numpy as np

def mean_absolute_bias(cond,X) :
    
    bias = 0.0
    
    N0 = X.shape[0]
    
    for x in X :
        
        bias = bias + np.abs(x - cond).mean(axis = 0)
    
    return bias / N0

def mean_bias(cond,X,real_ens) :
    
    bias = 0.0
    
    N0 = X.shape[0]
    
    X_mean = X.mean(axis=0)
    
    for x in X :
        
        bias = bias + (x - X_mean).mean(axis = 0)
    
    return bias / N0
        