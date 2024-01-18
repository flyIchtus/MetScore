#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:11 2023

@author: brochetc

AROME-specific version of CRPS

"""

import properscoring as ps
import numpy as np
import metrics.wind_comp as wc
import copy
import CRPS.CRPS as psc
from multiprocessing import Pool

def ensemble_crps(cond, real_ens, X, debiasing = False):
    """
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : N_c x C x H x W array with N_c members
        
    Returns :
        
        avg_crps : C x H x W array containing the result
    
    """

    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond[0])
    real_ens_p = copy.deepcopy(real_ens)

    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    real_ens_p[:,0], real_ens_p[:,1] = wc.computeWindDir(real_ens_p[:,0], real_ens_p[:,1])
    if debiasing == True : 
        X_p = wc.debiasing(X_p, real_ens_p)

    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])


    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.

    
    ################################################## CRPS with another method ##################################
    print(cond_p.shape)
    cond_p_ff = cond_p[0,~np.isnan(cond_p[0])]
    cond_p_dd = cond_p[1,~np.isnan(cond_p[1])]
    cond_p_t2m = cond_p[2,~np.isnan(cond_p[2])]
    
    X_p_ff = X_p[:,0,~np.isnan(cond_p[0])]
    X_p_dd = X_p[:,1,~np.isnan(cond_p[1])]
    X_p_t2m = X_p[:,2,~np.isnan(cond_p[2])]
    
    print(X_p_ff.max(), X_p_dd.max(), X_p_t2m.max())
    
    crps_res = np.zeros((3,1))
    sm = 0.
    for i in range(len(cond_p_ff)):
        
        crps,fcrps,acrps = psc(X_p_ff[:,i],cond_p_ff[i]).compute()   
        sm = sm + fcrps
    crps_res[0] = sm / len(cond_p_ff) 
    sm = 0.
    
    for i in range(len(cond_p_dd)):
        
        crps,fcrps,acrps = psc(X_p_dd[:,i],cond_p_dd[i]).compute()   
        sm = sm + fcrps
    crps_res[1] = sm / len(cond_p_dd)
    sm = 0.

    for i in range(len(cond_p_t2m)):
        
        crps,fcrps,acrps = psc(X_p_t2m[:,i],cond_p_t2m[i]).compute()   
        sm = sm + fcrps
    crps_res[2] = sm / len(cond_p_t2m)    

    print(crps_res)
   
    return crps_res


def fcrps_calc(data):
    cond_p, X_p = data[0], data[1]
    crps,fcrps,acrps = psc(X_p,cond_p).compute()


    return fcrps

def crps_multi_dates(cond, X, real_ens, debiasing = False):
    """
    Inputs :
        
        X : D x N x C x H x W array with N samples and D dates
        
        cond : D x N_c x C x H x W array with N_c members
        
    Returns :
        
        avg_crps :  C array containing the result
    
    """
    
    D, N, C, H, W  = X.shape    
            
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)
    real_ens_p = copy.deepcopy(real_ens)

    if debiasing:
        
        X_p = wc.debiasing_multi_dates(X_p, real_ens_p)

    X_p[:,:,0], X_p[:,:,1] = wc.computeWindDir(X_p[:,:,0], X_p[:,:,1])

    condpangle = np.stack([cond_p[:,1] for i in range(N)], axis=1)

    angle_dif = wc.angle_diff(X_p[:,:,1], condpangle)
    X_p[:,:,1] = angle_dif
    cond_p[:,1,:] = 0.
    
    ################################################## CRPS with another method ##################################

    cond_p_ff = []
    cond_p_t2m = []
    cond_p_dd = []

    X_p_ff = []
    X_p_dd = []
    X_p_t2m = []
    
    # flattening dates and localisations on single dimension to parallelize better
    for d in range(D):
        cpd_ff = cond_p[d,0,~np.isnan(cond_p[d,0])]
        #cpd_dd = cond_p[d,1,~np.isnan(cond_p[d,1])]
        cpd_t2m = cond_p[d,2,~np.isnan(cond_p[d,2])]

        Xpd_ff = X_p[d,:,0,~np.isnan(cond_p[d,0])]
        #Xpd_dd = X_p[d,:,1,~np.isnan(cond_p[d,1])]
        Xpd_t2m = X_p[d,:,2,~np.isnan(cond_p[d,2])]

        cond_p_ff = cond_p_ff + [cpd_ff[i] for i in range(len(cpd_ff))]
        #cond_p_dd = cond_p_dd + [cpd_dd[i] for i in range(len(cpd_dd))]
        cond_p_t2m = cond_p_t2m + [cpd_t2m[i] for i in range(len(cpd_t2m))]
        
        X_p_ff = X_p_ff + [Xpd_ff[i] for i in range(len(Xpd_ff))]
        #X_p_dd = X_p_dd + [Xpd_dd[i] for i in range(len(Xpd_dd))]
        X_p_t2m = X_p_t2m + [Xpd_t2m[i] for i in range(len(Xpd_t2m))]

    # parallelizing for each variable separately (obs data break occur separately so lists have different lengths)
    data = [ (cf, Xf) for cf, Xf in zip(cond_p_ff, X_p_ff)]
    with Pool(32) as p:
       res = p.map(fcrps_calc, data)
    res_ff = np.nanmean(np.array(res), axis=0)
    
    data = [ (cf, Xf) for cf, Xf in zip(cond_p_t2m, X_p_t2m)]
    with Pool(32) as p:
        res = p.map(fcrps_calc, data)
        #print(len(res), res[0].shape)
    res_t2m = np.nanmean(np.array(res), axis=0)

    crps_res = np.array([res_ff, np.nan, res_t2m])

    print(crps_res, flush=True)
    
    return crps_res


def crps_vs_aro_multi_dates(cond, X, real_ens, debiasing = False):
    """
    Inputs :
        
        X : D x N x C x H x W array with N samples and D dates
        
        cond : D x N_c x C x H x W array with N_c members
        
    Returns :
        
        avg_crps :  C array containing the result
    
    """
    print("aro", flush=True)
    crps_aro = crps_multi_dates(cond,real_ens,real_ens,debiasing = False)
    print("gan", flush=True)
    crps_fake = crps_multi_dates(cond,X, real_ens, debiasing=debiasing)

    return  - crps_aro + crps_fake