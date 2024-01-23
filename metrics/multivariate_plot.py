#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:42:53 2022

@author: brochetc

multivariate_plot

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import multivariate as mlt

path='/home/brochetc/Bureau/Thèse/présentations_thèse/multivariables/Set_38_32/'

if __name__=="__main__":
    
    res=pickle.load(open(path+'multivar0distance_metrics_16384.p', 'rb'))
    
    RES=res['multivar'].squeeze()
    
    data_r,data_f=RES[22,0], RES[22,1]
    logging.debug(data_r.shape, data_f.shape)
    
    
    levels=mlt.define_levels(data_r,5)
    ncouples2=data_f.shape[0]*(data_f.shape[0]-1)
    bins=np.linspace(tuple([-1 for i in range(ncouples2)]), tuple([1 for i in range(ncouples2)]),101, axis=1)
    
    var_r=(np.log(data_r), bins)
    var_f=(np.log(data_f), bins)
    
    mlt.plot2D_histo(var_f, var_r, levels)