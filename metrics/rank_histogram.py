"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics4ensemble.wind_comp as wc
import copy


def rank_histo(cond, X,real_ens, debiasing = False):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        bins (C, 121) max number of members in the ensemble  
    
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
    
    ##################################################################    
    
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
    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.
    
    bins = np.zeros((C, 121)) # 121 since N=120 is the biggest ensemble... Maybe use the biggest number that we will use 1000? 
    for i in range(C):
        
        cond_var = copy.deepcopy(cond_p[i])
        X_var = copy.deepcopy(X_p[:,i])
        
        obs = cond_var[ ~np.isnan(cond_var)]
        ens = X_var[:, ~np.isnan(cond_var)]
        #print(obs.shape, ens.shape, i)
        

    
        for j in range(obs.shape[0]):
        
            ens_sort = ens[:,j]
            ens_sort.sort()
            ens_sort = np.concatenate((ens_sort, [9999999.]))
            out = np.where((ens_sort < obs[j]), True, False)
            bins[i, np.argmax(out==False)] +=1 
            #print(ens_sort, np.argmax(out==False), obs[j])
            

        
    return bins
