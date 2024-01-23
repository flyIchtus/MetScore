
"""


AROME-specific version of skill_spread

"""
import logging

import numpy as np
import metrics.wind_comp as wc
import copy

def skill_spread_deviation_multi_dates(cond, X, real_ens, debiasing = False, print_skill_aro=False):
    """
    
    Inputs :
        
        X : D x N x C x H x W array with D different times and N samples per time
        
        cond : D x C x H x W array observation
        
    Returns :
        
        devaition :  C x H x W array containing the result 
    
    """
    D, N, C, H, W  = X.shape
    #correction factor according to Fortin et al., 2014
    #Why should ensemble Spread match RMSE of the ensemble mean ?
    correction_factor = np.sqrt((N+1)/N)

    sp_out = np.zeros((D,2,C,H,W))


    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)


    if debiasing:
        real_ens_p = copy.deepcopy(real_ens)
        X_p = wc.debiasing_multi_dates(X_p, real_ens_p)
        logging.debug(X_p[:,:,2].mean(), real_ens[:,:,2].mean())

        if print_skill_aro:
            cond_p0 = copy.deepcopy(cond_p)
            real_ens_p[:,:,0], real_ens_p[:,:,1] = wc.computeWindDir(real_ens_p[:,:,0], real_ens_p[:,:,1])


            condpangle = np.stack([cond_p0[:,1] for i in range(16)], axis=1)

            angle_dif = wc.angle_diff(real_ens_p[:,:,1], condpangle)
            real_ens_p[:,:,1] = angle_dif
            cond_p0[:,1,:] = 0.

            skill_aro = (real_ens_p.mean(axis=1) - cond_p0)**2

            logging.debug('skill aro', np.nanmean(np.sqrt(np.nanmean(skill_aro, axis=0)), axis=(-2,-1)))
            #spread_aro = np.sqrt(17/16) * np.sqrt()

    X_p[:,:,0], X_p[:,:,1] = wc.computeWindDir(X_p[:,:,0], X_p[:,:,1])


    condpangle = np.stack([cond_p[:,1] for i in range(N)], axis=1)

    angle_dif = wc.angle_diff(X_p[:,:,1], condpangle)
    X_p[:,:,1] = angle_dif
    cond_p[:,1,:] = 0.

    skill = (X_p.mean(axis=1) - cond_p)**2

    spread =  X_p.var(axis=1,ddof=1) # .std operation needs ddof=1 (delta degrees of freedom) to be unbiased

    spread_aggregated = correction_factor * np.sqrt(np.nanmean(spread, axis=(0,-2,-1)))

    skill_aggregated = np.sqrt(np.nanmean(skill, axis=(0,-2,-1)))

    logging.debug('spread',spread_aggregated, 'skill', skill_aggregated, flush=True)

    deviation = np.abs(skill_aggregated - spread_aggregated)

    return deviation


def skill_spread_vs_aro_multi_dates(cond, X, real_ens, debiasing = False):
    """
    
    Inputs :
        
        X : D x N x C x H x W array with D different times and N samples per time
        
        cond : D x C x H x W array observation
        
    Returns :
        
        devaition :  C x H x W array containing the result 
    
    """
    D, N, C, H, W  = X.shape
    #correction factor according to Fortin et al., 2014
    #Why should ensemble Spread match RMSE of the ensemble mean ?
    correction_factor = np.sqrt((N+1)/N)

    sp_out = np.zeros((D,2,C,H,W))


    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)
    real_ens_p = copy.deepcopy(real_ens)

    if debiasing:

        X_p = wc.debiasing_multi_dates(X_p, real_ens_p)
        logging.debug(X_p[:,:,2].mean(), real_ens[:,:,2].mean())


    cond_p0 = copy.deepcopy(cond_p)
    real_ens_p[:,:,0], real_ens_p[:,:,1] = wc.computeWindDir(real_ens_p[:,:,0], real_ens_p[:,:,1])


    condpangle = np.stack([cond_p0[:,1] for i in range(16)], axis=1)

    angle_dif = wc.angle_diff(real_ens_p[:,:,1], condpangle)
    real_ens_p[:,:,1] = angle_dif
    cond_p0[:,1,:] = 0.

    skill_aro = (real_ens_p.mean(axis=1) - cond_p0)**2
    skill_aggregated_aro = np.sqrt(np.nanmean(skill_aro, axis=(0,-2,-1)))

    spread_aro = real_ens_p.var(axis=1, ddof=1)
    spread_aggregated_aro = correction_factor * np.sqrt(np.nanmean(spread_aro, axis=(0,-2,-1)))


    X_p[:,:,0], X_p[:,:,1] = wc.computeWindDir(X_p[:,:,0], X_p[:,:,1])


    condpangle = np.stack([cond_p[:,1] for i in range(N)], axis=1)

    angle_dif = wc.angle_diff(X_p[:,:,1], condpangle)
    X_p[:,:,1] = angle_dif
    cond_p[:,1,:] = 0.

    skill = (X_p.mean(axis=1) - cond_p)**2

    spread =  X_p.var(axis=1,ddof=1) # .std operation needs ddof=1 (delta degrees of freedom) to be unbiased

    spread_aggregated = correction_factor * np.sqrt(np.nanmean(spread, axis=(0,-2,-1)))

    skill_aggregated = np.sqrt(np.nanmean(skill, axis=(0,-2,-1)))


    deviation_fake = np.abs(skill_aggregated - spread_aggregated)

    deviation_aro = np.abs(skill_aggregated_aro - spread_aggregated_aro)

    logging.debug('spread',spread_aggregated, 'skill', skill_aggregated, flush=True)
    logging.debug('spread aro',spread_aggregated_aro, 'skill aro', skill_aggregated_aro, flush=True)

    return - deviation_aro + deviation_fake

