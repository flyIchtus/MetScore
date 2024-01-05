import torch as torch

from metrics.metrics import Metric
import numpy as np


class OrographyRMSE(Metric):

    def __init__(self, usetorch=False):
        super().__init__(isBatched=True)
        self.usetorch = False

    def _preprocess(self, fake_data, real_data=None, obs_data=None):
        # No additional preprocessing is needed for orography_RMSE
        return {'real_data': real_data, 'fake_data': fake_data, 'obs_data': obs_data}

    def _calculateCore(self, processed_data):
        real_orog = processed_data['real_data'][0, -1:, :, :]
        fake_orog = processed_data['fake_data'][:, -1:, :, :]

        if self.usetorch:
            res = torch.sqrt(((fake_orog - real_orog) ** 2).mean())
        else:
            res = np.sqrt(((fake_orog - real_orog) ** 2).mean())
        return res


############################ General simple metrics ###########################

def simple_variance(X) : 
    """
    X :  N x C H x W array
    
    Returns variance of X along the first dimension
    """
    
    return X.var(axis = 0)

def variance_diff(X, cond):
    """
    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        Diff : C x H x W array
    """
    
    Diff =  (X.var(axis=0) - cond.var(axis = 0))
    
    return Diff

def std_diff(X, cond):
    """
    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        Diff : C x H x W array
    """
    
    Diff =  (X.std(axis=0) - cond.std(axis = 0))
    
    return Diff
    
def relative_std_diff(X, cond) :
    """

    Maps of variance difference between ensemble X and condition cond
    
    X :  N x C x H x W array
    
    cond  : N_c x C x H x W array
    
    Returns :
        
        ratio : C x H x W array

    """
    
    ratio =  (X.std(axis=0) - cond.std(axis=0)) / (X.std(axis =0))
    
    return ratio
