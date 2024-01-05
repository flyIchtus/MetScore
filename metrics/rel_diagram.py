"""


AROME-specific version of skill_spread

"""

import numpy as np
import metrics4ensemble.wind_comp as wc
import copy


def rel_diag(cond, X,real_ens, parameters, debiasing = False):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        rel diagram x and y   
    
    """
    
    N, C, H, W  = X.shape
    
    N_param = parameters.shape[1]
    
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
    #angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    
    #rel = np.zeros((N_param,3,2,10))
    rel = np.zeros((N_param, 2,  C, H, W))
    #forecast_prob = np.linspace(0, 1, num=11)
    for i in range(parameters.shape[1]):    

    
        T_tr = parameters[1, i]
        ff_tr = parameters[0, i]
    
        X_tr = np.zeros((C,N,H,W))
        O_tr = np.zeros((C,H,W))
        O_tr[:] = np.nan
        
        """
        Converting forecasts and observation
        """



    
        X_tr[0, X_p[:,0] > ff_tr] = 1.0
        X_tr[2, X_p[:,2] > T_tr] = 1.0
        O_tr[0, cond_p[0] > ff_tr] = 1
        O_tr[2, cond_p[2] > T_tr] = 1
        O_tr[0, cond_p[0] < ff_tr] = 0
        O_tr[2, cond_p[2] < T_tr] = 0
    
        X_prob = X_tr.sum(axis = 1) / N
        rel[i,0] = X_prob
        rel[i,1] = O_tr
        # for j in range(forecast_prob.shape[0]-1):
            
        #     for k in range(C): 
        #         obs = copy.deepcopy(O_tr[k, np.where((X_prob[k] >= forecast_prob[j]) & (X_prob[k] < forecast_prob[j+1]), True, False)])
                
        #         obs = obs[~np.isnan(obs)]
        #         #print(obs, forecast_prob[j], forecast_prob[j+1])
    
        #         freq_obs = obs.sum()/obs.shape[0]
                
        #         rel[i,k,0,j] = forecast_prob[j]
        #         rel[i,k,1,j] = freq_obs
            #print(freq_obs,i, forecast_prob[j], forecast_prob[j+1])
        # for i in range(128):
        #     for j in range(128):
        #         if np.isnan(O_tr[0,i,j]) == False and X_prob[0,i,j] < 0.2:
        #             print(O_tr[0,i,j], X_prob[0,i,j])

        
        
    return rel
