import numpy as np
import random
import logging

def mse(X, obs):
    
    random_idx = random.sample(range(X.shape[0]), 1)[0]   
    _mse = np.nanmean(((X[random_idx] - obs.squeeze())**2),axis=(-2,-1))
    return _mse