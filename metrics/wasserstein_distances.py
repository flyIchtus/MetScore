#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:41:23 2022

@author: brochetc

Diverse Wasserstein distances computations
"""

import scipy.stats as sc
import torch
import numpy as np
from multiprocessing import Pool
from math import sqrt

###############################################################################
################### Wasserstein distances ############################
###############################################################################

def wasserstein_wrap(data):
    real_data, fake_data, real_weights, fake_weights=data
    return sc.wasserstein_distance(real_data, fake_data, \
                                   real_weights, fake_weights)

def W1_on_image_samples(real_data, fake_data, num_proc=4,\
                        Crop_Size=64,
                        real_weights=None, fake_weights=None):
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    data is cropped at the center so as to reduce comput. overhead
    
    """
    Side_Size=fake_data.shape[2]
    HALF=Side_Size//2-1
    half=Crop_Size//2
    real_data=real_data[:,:-1,:,:]  #dropping last channel as it's orography
    fake_data=fake_data[:,:-1,HALF-half:HALF+half, HALF-half:HALF+half]
    Channel_size=real_data.shape[1]
    Lists=[]
    
    for ic in range(Channel_size):
        for i_x in range(Crop_Size):
            for j_y in range(Crop_Size):
                Lists.append((real_data[:,ic,i_x,j_y],fake_data[:,ic,i_x,j_y],\
                              real_weights, fake_weights))
    #logging.debug(Lists[0:2][0][:2], Lists[0:2][1][:2])
    with Pool(num_proc) as p:
        W_list=p.map(wasserstein_wrap,Lists)
    return [np.array(W_list).mean()]

def W1_center(real_data, fake_data,Crop_Size=64):
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    data is cropped at the center so as to reduce comput. overhead
    
    """
    Side_Size=fake_data.shape[2]
    HALF=Side_Size//2-1
    half=Crop_Size//2
    real_data=real_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    fake_data=fake_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    Channel_size=real_data.shape[1]
    
    dist=torch.tensor([0.], dtype=torch.float32).cuda()
    for i in range(Crop_Size):
        for j in range(Crop_Size):
            for c in range(Channel_size):
                real,_=torch.sort(real_data[:,c,i,j],dim=0)
                fake,_=torch.sort(fake_data[:,c,i,j],dim=0)
                dist=dist+torch.abs(real-fake).mean()
    return dist*(1e3/(Crop_Size**2*Channel_size))

def W1_random(real_data, fake_data, pixel_num=4096):
    
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    random pixels of data are selected so as to reduce comput. overhead
    
    """
    
    h,w = real_data.shape[2], real_data.shape[3]
    
    x_ind = np.random.randint(0,h,size=pixel_num)
    y_ind = np.random.randint(0,w,size=pixel_num)
    
    real_data = real_data[:,:, x_ind, y_ind]
    fake_data = fake_data[:,:, x_ind, y_ind]
    
    Channel_size = real_data.shape[1]
    
    dist=torch.tensor([0.], dtype=torch.float32).cuda()
    for i in range(pixel_num) :
        for c in range(Channel_size):
            real,_ = torch.sort(real_data[:,c,i],dim=0)
            fake,_ = torch.sort(fake_data[:,c,i],dim=0)
            dist = dist+torch.abs(real-fake).mean()
    return dist*(1e3/(pixel_num**2*Channel_size))
    
def W1_random_NUMPY(real_data, fake_data, pixel_num=4096):
    
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    random pixels of data are selected so as to reduce comput. overhead
    
    """
    
    h,w=real_data.shape[2], real_data.shape[3]
    
    x_ind=np.random.randint(0,h,size=pixel_num)
    y_ind=np.random.randint(0,w,size=pixel_num)
    
    real_data=real_data[:,:, x_ind, y_ind]
    fake_data=fake_data[:,:, x_ind, y_ind]
        
    Channel_size = real_data.shape[1]
    
    dist = np.array([0.], dtype=np.float32)
    for i in range(pixel_num):
        for c in range(Channel_size):
            real = np.sort(real_data[:,c,i],axis=0)
            fake = np.sort(fake_data[:,c,i],axis=0)
            dist = dist+np.abs(real-fake).mean()
    return dist*(1e3/(pixel_num*Channel_size))

class pixel_W1():
    """
    wrapper class to compute W1 distance on 1 pixel of bounded maps
    """
    
    def __init__(self,real, fake):
        self.real=real
        self.fake=fake
        self.channelSize=real.shape[1]
        self.SideSize=fake.shape[2]
        
    def distance(self,indices):
        c,i,j=indices
        r=np.sort(self.real[:,c,i,j],axis=0)
        f=np.sort(self.fake[:,c,i,j],axis=0)
        return np.abs(r-f).mean()
    
def pointwise_W1(real_data, fake_data):
    """
    compute pixel-wise and channel-wise Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    """
    assert real_data.shape==fake_data.shape
    if len(real_data.shape)==4:
        Height_Size=fake_data.shape[2]
        Width_Size=fake_data.shape[3]
        Channel_Size=fake_data.shape[1]
        
        dist=np.zeros((Channel_Size, Height_Size, Width_Size), dtype=np.float32)
    
        for i in range(Height_Size):
            for j in range(Width_Size):
                for c in range(Channel_Size):
                    real=np.sort(real_data[:,c,i,j],axis=0)
                    fake=np.sort(fake_data[:,c,i,j],axis=0)
                    dist[c,i,j]=np.abs(real-fake).mean()
    elif len(real_data.shape)==3:
        Height_Size=real_data.shape[-1]
        Channel_Size=real_data.shape[1]
        
        dist=np.zeros((Channel_Size, Height_Size), dtype=np.float32)
        
        for i in range(Height_Size):
            for c in range(Channel_Size):
                real=np.sort(real_data[:,c,i], axis=0)
                fake=np.sort(fake_data[:,c,i], axis=0)
                dist[c,i]=np.abs(real-fake).mean()
    else :
        raise ValueError('Data format not accounted for')
        
    return dist
                
def W1_center_numpy(real_data, fake_data, Crop_Size=64):
    """
    compute the Wasserstein distance between real_data and fake_data
    using real_weights and fake weights as importance weights
    
    data is cropped at the center so as to reduce comput. overhead
    
    NUMPY VERSION
    
    """
    Side_Size=fake_data.shape[2]
    HALF=Side_Size//2-1
    half=Crop_Size//2
    real_data=real_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    fake_data=fake_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
    Channel_size=real_data.shape[1]
    
    dist=np.array([0.], dtype=np.float32)
    for i in range(Crop_Size):
        for j in range(Crop_Size):
            for c in range(Channel_size):
                real=np.sort(real_data[:,c,i,j],axis=0)
                fake=np.sort(fake_data[:,c,i,j],axis=0)
                dist=dist+np.abs(real-fake).mean()
    return dist*(1e3/(Crop_Size**2*Channel_size))

class W1_center_class():
    def __init__(self,channel_wise,module, Crop_Size=64):
        self.channel_wise=channel_wise
        self.module=module
        self.Crop_Size=Crop_Size
    
    def compute_W1(self, real_data, fake_data):
        Side_Size=fake_data.shape[2]
        HALF=Side_Size//2-1
        half=self.Crop_Size//2
        real_data=real_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
        fake_data=fake_data[:,:, HALF-half:HALF+half, HALF-half:HALF+half]
        Channel_size=real_data.shape[1]
        
        if self.module==np:
            if self.channel_wise:
                dist=np.array([0. for var in range(Channel_size)], dtype=np.float32)
        if self.channel_wise:
            factor=1e3/(self.Crop_Size**2)
        else :
            factor=1e3/(self.CropSize**2*Channel_size)
        for i in range(self.Crop_Size):
            for j in range(self.Crop_Size):
                for c in range(Channel_size):
                    
                    if self.module==np:
                        real=np.sort(real_data[:,c,i,j],axis=0)
                        fake=np.sort(fake_data[:,c,i,j],axis=0)
                        if self.channel_wise:
                            dist=dist+np.abs(real-fake).mean(axis=(1,2))
                        else:
                            dist=dist+np.abs(real-fake).mean(axis=(1,2))
                    else :
                        real,_=torch.sort(real_data[:,c,i,j],dim=0)
                        fake,_=torch.sort(fake_data[:,c,i,j],dim=0)
                        dist=dist+torch.abs(real-fake).mean()
        return dist*factor

