"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics4ensemble.wind_comp as wc
import copy

def skill_spread(cond, X,real_ens, debiasing = False):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        skill spread  :2 x C x H x W array containing the result 0 is skill and 1 is spread
    
    """
    N, C, H, W  = X.shape
    
    sp_out = np.zeros((2,C,H,W))
    

    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)
    real_ens_p = copy.deepcopy(real_ens)

    ############# DEBIASING U v t2m

    # N_a=int(X_p.shape[0]/real_ens_p.shape[0])
    # for i in range(int(real_ens_p.shape[0])):
        
    #     Gan_avg_mem = np.mean(X_p[i*N_a:(i+1)*N_a], axis = 0)
    #     Bias = real_ens_p[i] - Gan_avg_mem
    #     #Bias[1] = 0.
    #     X_p[i*N_a:(i+1)*N_a] = X_p[i*N_a:(i+1)*N_a] + Bias 
        
        
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    real_ens_p[:,0], real_ens_p[:,1] = wc.computeWindDir(real_ens_p[:,0], real_ens_p[:,1])
    
    
    ############# DEBIASING ################
    #X_p_mean = X_p.mean(axis=0)
    #real_ens_p_mean = real_ens_p.mean(axis=0)
    #Bias = real_ens_p_mean - X_p_mean
    #Bias[1] = 0.
    
    #print(Bias.shape, real_ens_p_mean.shape, real_ens_p.shape, X_p_mean.shape)
    #X_p = X_p + Bias
    
    #N_a=int(X_p.shape[0]/real_ens_p.shape[0])
    #for i in range(int(real_ens_p.shape[0])):
        
    #    Gan_avg_mem = np.mean(X_p[i*N_a:(i+1)*N_a], axis = 0)
    #    Bias = real_ens_p[i] - Gan_avg_mem
    #    Bias[1] = 0.
    #    X_p[i*N_a:(i+1)*N_a] = X_p[i*N_a:(i+1)*N_a] + Bias 
    if debiasing == True : 
        X_p = wc.debiasing(X_p, real_ens_p)
    # ############# DEBIASING ################
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.
    
    skill = X_p.mean(axis=0) - cond_p
    
    spread = X_p.std(axis=0)
    
    
    sp_out[0] = skill
    
    sp_out[1] = spread


        
    return sp_out
