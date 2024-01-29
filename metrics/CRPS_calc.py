#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:11 2023

@author: brochetc

AROME-specific version of CRPS

"""
import copy
import logging
from multiprocessing import Pool

import CRPS.CRPS as psc
import numpy as np

import metrics.wind_comp as wc


def ensemble_crps(obs_data, fake_data):
    """
    Inputs :
        
        fake_data : N x C x H x W array with N samples
        
        obs_data : N_c x C x H x W array with N_c members
        
    Returns :
        
        crps_res : 1 x C array containing average CRPS results    
    """


    
    ################################################## CRPS with another method ##################################
    logging.debug(obs_data.shape)
    obs_data_ff = obs_data[0,~np.isnan(obs_data[0])]
    obs_data_dd = obs_data[1,~np.isnan(obs_data[1])]
    obs_data_t2m = obs_data[2,~np.isnan(obs_data[2])]
    
    fake_data_ff = fake_data[:,0,~np.isnan(obs_data[0])]
    fake_data_dd = fake_data[:,1,~np.isnan(obs_data[1])]
    fake_data_t2m = fake_data[:,2,~np.isnan(obs_data[2])]
    
    #logging.debug(fake_data_ff.max(), fake_data_dd.max(), fake_data_t2m.max())
    
    crps_res = np.zeros((3,1))
    sm = 0.
    for i in range(len(obs_data_ff)):
        
        crps,fcrps,acrps = psc(fake_data_ff[:,i],obs_data_ff[i]).compute()   
        sm = sm + fcrps
    crps_res[0] = sm / len(obs_data_ff) 
    sm = 0.
    
    for i in range(len(obs_data_dd)):
        
        crps,fcrps,acrps = psc(fake_data_dd[:,i],obs_data_dd[i]).compute()   
        sm = sm + fcrps
    crps_res[1] = sm / len(obs_data_dd)
    sm = 0.

    for i in range(len(obs_data_t2m)):
        
        crps,fcrps,acrps = psc(fake_data_t2m[:,i],obs_data_t2m[i]).compute()   
        sm = sm + fcrps
    crps_res[2] = sm / len(obs_data_t2m)    

    logging.debug(f"CRPS results : {crps_res}")
   
    return crps_res


def fcrps_calc(data):
    cond_p, fake_data = data[0], data[1]
    crps,fcrps,acrps = psc(fake_data,cond_p).compute()


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
        #logging.debug(len(res), res[0].shape)
    res_t2m = np.nanmean(np.array(res), axis=0)

    crps_res = np.array([res_ff, np.nan, res_t2m])

    logging.debug(f"crps_res {crps_res}")
    
    return crps_res


def crps_vs_aro_multi_dates(cond, X, real_ens, debiasing = False):
    """
    Inputs :
        
        X : D x N x C x H x W array with N samples and D dates
        
        cond : D x N_c x C x H x W array with N_c members
        
    Returns :
        
        avg_crps :  C array containing the result
    
    """
    crps_aro = crps_multi_dates(cond,real_ens,real_ens,debiasing = False)
    crps_fake = crps_multi_dates(cond,X, real_ens, debiasing=debiasing)

    return  - crps_aro + crps_fake