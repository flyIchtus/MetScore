"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics.wind_comp as wc
import copy


def bias_ens(cond, X,real_ens, debiasing = False):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        bias : avg(X) - cond  
    
    """
    
    N, C, H, W  = X.shape
    
    
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
    if debiasing == True : 

        X_p = wc.debiasing(X_p, real_ens_p)
    #N_a=int(X_p.shape[0]/real_ens_p.shape[0])
    #for i in range(int(real_ens_p.shape[0])):
        
    #    Gan_avg_mem = np.mean(X_p[i*N_a:(i+1)*N_a], axis = 0)
    #    Bias = real_ens_p[i] - Gan_avg_mem
    #    Bias[1] = 0.
    #    X_p[i*N_a:(i+1)*N_a] = X_p[i*N_a:(i+1)*N_a] + Bias 
        
    ############# DEBIASING ################
    
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    #print(angle_dif)
    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.
    
    
    X_p_mean = X_p.mean(axis=0)
    X_bias = X_p_mean - cond_p

    return X_bias


def abs_bias_penalty(fake, real,debiasing=False):
   
   bias = bias_ens(real, fake, real, debiasing=debiasing)

   
