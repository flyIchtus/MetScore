#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:08:13 2022

@author: brochetc

Multivariate correlations
"""
import logging

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from matplotlib.ticker import FormatStrFormatter

def plot2D_histo(var2var_f, var2var_r, levels, output_dir, add_name):
    """
    
    better use density based histograms rather than counting histograms for this function
    
    Inputs :
        data : numpy array, B x C
        
    """
    Xs=['u', 'u', 'v']
    Ys=['v','t2m', 't2m']
    bivariates_f, bins_f=var2var_f
    bivariates_r, bins_r=var2var_r
    
    assert bivariates_f.shape==bivariates_r.shape
    
    ncouples=bivariates_f.shape[0]
    
    fig,axs=plt.subplots(1, ncouples, figsize=(4*ncouples,2*ncouples), sharex=True, sharey=True)
    for i in range(ncouples):
       cs=axs[i].contourf(bins_r[2*i][:-1], bins_r[2*i+1][:-1],bivariates_r[i], cmap='plasma', levels=levels[i])
       axs[i].contour(bins_f[2*i][:-1], bins_f[2*i+1][:-1], bivariates_f[i],cmap='Greys', levels=levels[i])
       axs[i].set_xlabel(Xs[i], fontsize='large', fontweight='bold')
       axs[i].set_ylabel(Ys[i], fontsize='large', fontweight='bold')

       if i==ncouples-1:
          cbax=fig.add_axes([0.9,0.1,0.02,0.83])
          cb=fig.colorbar(cs, cax=cbax)
          cb.ax.tick_params(labelsize=10)
          cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
          cb.set_label('Density (log scale)', fontweight='bold', fontsize='large', rotation=270)
    fig.tight_layout(rect=(0.0,0.0,0.9,0.95))
    plt.savefig(output_dir+'multi_plot_'+add_name+'.png')
    
    plt.close()
    
    

def var2Var_hist(data, bins, density=True):
    """
    provide 2D histograms with pairs of variables present in the data channels
    if bins are provided, performs the histogram with the bins given as arguments
    
    Inputs :
        data : numpy array, shape : B x C (number of samples x channels)
        
        bins : int -> in this case, bins is number of bins used for histogram counting
               numpy array, shape -> already calculated bin edges as output by numpy.histogram2d
                if numpy array, shape of bins[0], bins[1] is thus nbins+1
    Returns :
        
        bivariates : numpy array, shape C*(C-1)/2 x Nb, where Nb in either bins 
                     if bins is int, or (bins[0].shape[0]-1) if bins is array tuple
                     
                     bivariate count histograms
                     
        bins : the bins either outputted by histogram2d or those
                passed as array inputs
    """
    channels=data.shape[1]
    var_couples=combinations_with_replacement(range(channels), 2)
    ncouples=channels*(channels-1)//2
    
    if type(bins)==int :
        Bins=np.zeros((ncouples*2,bins+1))
        bivariates=np.zeros((ncouples,bins,bins))
        
    
    
    elif type(bins)==np.ndarray:
        Bins=bins
        bivariates=np.zeros((ncouples,Bins.shape[1]-1, Bins.shape[1]-1))
        
    k=0
    for tu in var_couples:
        i,j=tu

        if i!=j:
            if type(bins)==int:
                bivariates[k], Bins[2*k], Bins[2*k+1]=\
                                np.histogram2d(data[:,i], data[:,j], bins=bins, density=density)
            elif type(bins)==np.ndarray:
                bivariates[k], _, _=\
                                np.histogram2d(data[:,i], data[:,j], bins=Bins[k], density=density)
            k+=1
    return bivariates, Bins

def batchingvar2Var(data_list, bins, load_func, density=True):
    """
    
    perform var2Var_hist in a batched manner
    
    Inputs :
        data_list : list(str), with each str allowing a "load" like function to load
                   a batch of data
                   
                   each data bacth being structured B x C as in var2Var_hist
                   
        bins : int -> in this case, bins is number of bins used for histogram counting
               numpy array, shape -> already calculated bin edges as output by numpy.histogram2d
                if numpy array, shape of bins[0], bins[1] is thus nbins+1
        
        load_func : function used to load data batches
        
    Returns :
        
        bivariates : numpy array, shape C*(C-1)/2 x nbins; bivariate density histograms
        
        Bins : numpy array, shape
    """
    N_tot=0 #total number of samples
    for i, data_name in enumerate(data_list):
        
        data=load_func(data_name)
        N_tot+=data.shape[0]
        if i==0:
            bivariates, Bins=var2Var_hist(data, bins, density=False)
            
        else :
            
            bivar, _ = var2Var_hist(data,Bins, density=False)
            bivariates = bivariates+bivar
    
    if density :
        
        
        binsizes=np.diff(Bins, axis=1)
        binareas=np.zeros((Bins.shape[0]//2,binsizes.shape[1], binsizes.shape[1]))
        for i in range(binsizes.shape[0]//2):
            
            binareas[i]=np.transpose(binsizes[2*i:2*i+1])*binsizes[2*i+1:2*i+2]
            
        return (1/N_tot)*bivariates/binareas, Bins
    
    else :
        
        
        return bivariates, Bins

def define_levels(bivariates, nlevels):
    """
    
    Define a logairthmic scales of levels to be used for histogram-2d plots,
    with the given "bivariates" data.
    
    Inputs :
        
        bivariates : np.array, shape is C*(C-1)/2 x nbins : bivariate density/count histograms
        
        nlevels :  number of desired levels
    
    Returns :
        
        levels : np.array, shape is C*(C-1)//2 x nlevels : sets of levels, with nlevels for eahc variable couple.
    
    """
    
    Shape=bivariates.shape
    assert len(Shape)==3
    inter=bivariates.reshape(Shape[0],Shape[1]*Shape[2])
    
    
    levels=np.zeros((Shape[0],nlevels))
    
    for i in range(Shape[0]):
        b=np.sort(inter[i])

        usable_data=b[b>0].shape[0]
        
        N_values=usable_data//nlevels
        assert N_values>2
        levels[i]=np.log(b[b>0][::N_values])[:nlevels]
    
    return levels



def space2batch(data):

    Shape=data.shape
    assert len(Shape)==4
    
    data_list = []
    for i in range(Shape[1]):
        data_list.append(np.expand_dims(data[:,i,:,:].reshape(Shape[0] * Shape[2] * Shape[3]), axis= 1))
    
    a = np.concatenate(data_list, axis = 1)
    return a


def multi_variate_correlations(data_real, data_fake):
    """
    To be used in the metrics evaluation framework
    data_r, data_f : numpy arrays, shape B xC x H xW
    
    Returns :
        
        Out_rf : numpy array, shape 2 x C*(C-1)//2 x nbins 
          bivariates histograms for [0,:,:] -> real samples
                                    [1,:,:] -> fake samples
    
    """
    
    channels=data_fake.shape[1]
    ncouples2=channels*(channels-1)
    
    
    bins=np.linspace(tuple([-1 for i in range(ncouples2)]), tuple([1 for i in range(ncouples2)]),101, axis=1)
    
    data_f=space2batch(data_fake)
    data_r=space2batch(data_real)
    logging.debug('data_f shape',data_f.shape)
    logging.debug('data_r shape',data_r.shape)
    
   
    bivariates_f, bins_f=var2Var_hist(data_f,bins)
    bivariates_r, bins_r=var2Var_hist(data_r,bins)
    
    out_rf=np.zeros((2, ncouples2//2,bivariates_f.shape[-1],bivariates_f.shape[-1]))
    out_rf[0]=bivariates_r
    out_rf[1]=bivariates_f
    logging.debug('out shape',out_rf.shape)
    return out_rf



    

