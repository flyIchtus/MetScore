#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:17:55 2022

@author: brochetc

# DCT transform and spectral energy calculation routines


Include :
    -> 2D dct and idct transforms
    -> 'radial' energy spectrum calculation
    -> spectrum plotting configuration


"""

from scipy.fftpack import dct, idct,fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import metrics.wind_comp as wc
from copy import deepcopy

################## DCT ########################################################

def dct2D(x):
    """
    2D dct transform for 2D (square) numpy array
    or for each sample b of BxNxN numpy array
    
    """
    assert x.ndim in [2,3]
    if x.ndim==3:
        res=dct(dct(x.transpose((0,2,1)), norm='ortho').transpose((0,2,1)),\
                norm='ortho')
    else :
        res=dct(dct(x.T, norm='ortho').T, norm='ortho')
    return res

def idct2D(f):
    """
    2D iverse dct transform for 2D (square) numpy array
    
    or for each sample b of BxNxN numpy array
    """
    
    assert f.ndim in [2,3]
    if f.ndim==3:
        res=idct(idct(f.transpose(0,2,1), norm='ortho').transpose(0,2,1),\
                 norm='ortho')
    else :
        res=dct(dct(f.T,norm='ortho').T, norm='ortho')
    return res

def dct_var(x):
    """
    compute the bidirectional variance spectrum of the (square) numpy array x
    """
    N=x.shape[-1]
    
    fx=dct2D(x)
    Sigma=(1/N**2)*fx**2
    return Sigma


#################### FFT ######################################################

def fft2D(x):
    """
    2D FFT transform for 2D (square) numpy array
    or for each sample b of BxNxN numpy array
    
    """
    assert x.ndim in [2,3]
    if x.ndim==3:
        res=fft(fft(x.transpose((0,2,1)), norm='ortho').transpose((0,2,1)), \
                norm='ortho')
    else :
        res=fft(fft(x.T, norm='ortho').T, norm='ortho')
    return res


def ifft2D(x):
    
    """
    2D iverse fft transform for 2D (square) numpy array
    
    or for each channel of CxNxN numpy array
    """
    return 0
    
################## Radial binning and computing ###############################

def radial_bin_dct(dct_sig,center):
    
    y, x= np.indices(dct_sig.shape)
    r=np.sqrt((x-center[0])**2+(y-center[1])**2)
    r=r.astype(int)
    
    Rmax=min(x.max(),y.max(),r.max())//2
    
    
    #### double binning for dct
    dct=dct_sig.ravel()[2*r.ravel()]+0.5*dct_sig.ravel()[2*r.ravel()-1]\
                                    +0.5*dct_sig.ravel()[2*r.ravel()+1]
    
    tbin=np.bincount(r.ravel()[r.ravel()<Rmax], dct[r.ravel()<Rmax])
    nr=np.bincount(r.ravel()[r.ravel()<Rmax])
    
    radial_profile=tbin/nr
    
    return radial_profile

def PowerSpectralDensitySlow(x):
    """
    compute the radially-binned, sample-averaged power spectral density 
    and radially-binned, sample-standardized power spectral density
    of the data x
    
    Inputs :
        x : numpy array, shape is B x N x N
    
    Returns :
        
        out : numpy array, shape is (Rmax,3), defined in radial_bin_dct function
               [:, 0] : contains average spectrum
               [:, 1] : contains q90 of spectrum
               [:, 2] : contains q10 of spectrum
               
    Slow but should be more robust
    """
    
    sig = dct_var(x)

    center = (sig.shape[1]//2, sig.shape[2]//2)
    N_samples = sig.shape[0]
    out_list = []
    
    for i in range(N_samples):
        out_list.append(radial_bin_dct(sig[i], center))
        
    out_list = np.array(out_list)
    out = out_list.mean(axis=0)
    out_90 = np.quantile(out_list,0.9, axis=0)
    out_10 = np.quantile(out_list,0.1, axis=0)
    return np.concatenate((np.expand_dims(out, axis=-1),\
                            np.expand_dims(out_90, axis=-1),\
                            np.expand_dims(out_10, axis=-1)),axis=-1)
    
def PowerSpectralDensity_Distrib(x) :
    """
    Channel-wise call to PowerSpectralDensitySlow
    
    
    x is array of shape B x C x N x N
    
    return array of shape C x Rmax x  3
    """
    
    C = x.shape[1]
    
    
    out_list = []
    
    for i in range(C):
        out_list.append(PowerSpectralDensitySlow(x[:,i]))
    
    return np.array(out_list)
    

def PowerSpectralDensity(x):
    """
    compute the radially-averaged, sample-averaged power spectral density 
    of the data x
    
    Inputs :
        x : numpy array, shape is B x C x N x N
    
    Return :
        
        out : numpy array, shape is (C, Rmax), with R_max defined in radial_bin_dct function
    
    """
    
    out_list = []
    channels = x.shape[1]    
    
    for c in range(channels) :
        x_c = x[:,c,:,:]
        sig = dct_var(x_c).mean(axis=0)
    
        center = (sig.shape[0]//2, sig.shape[1]//2)
        out_list.append(radial_bin_dct(sig, center))
    
    out=np.concatenate([np.expand_dims(o, axis = 0) for o in out_list], axis = 0)
    
    return out


def PSD_compare(real_data,fake_data):
    
    """
    compute the RSME of real_data and fake_data average spectrograms
    
    Inputs : 
        real_data, fake_data : numpy arrays, shape B x C x N x N
        with C the different channels where spectrograms are independently 
        computed
    
    Returns :
        
        res : numpy array, shape C (output array)
    """
    channels = real_data.shape[1]
    res = np.zeros((channels,))
    for c in range(channels):
        psd_real = PowerSpectralDensity(real_data[:,c,:,:])
        psd_fake = PowerSpectralDensity(fake_data[:,c,:,:])
        res[c] = np.sqrt(np.mean((np.log10(psd_real)-np.log10(psd_fake))**2))
    return res

def PSD_compare_multidates(obsdata, real_data, fake_data, debiasing=False):

    channels = real_data.shape[2]
    H,W = real_data.shape[-2], real_data.shape[-1]

    real_data, fake_data = deepcopy(real_data), deepcopy(fake_data)

    real_data[:,:,0], real_data[:,:,1] = wc.computeWindDir(real_data[:,:,0], real_data[:,:,1])
    fake_data[:,:,0], fake_data[:,:,1] = wc.computeWindDir(fake_data[:,:,0], fake_data[:,:,1])

    if debiasing:
        fake_data = wc.debiasing_multi_dates(fake_data, real_data)

    resh_real = real_data.reshape(real_data.shape[0] * real_data.shape[1],channels, H, W)

    resh_fake = fake_data.reshape(fake_data.shape[0] * fake_data.shape[1], channels, H, W)

    print(resh_real.shape, resh_fake.shape)

    return 10 * PSD_compare(resh_real, resh_fake)

    
################### Simple Testing ############################################    

if __name__=="__main__":
    ranges=[2.0,3.0,5.0,10.0]
    
    for r in ranges[-2:] :
        print(r/0.25)
        t,p=np.ogrid[-r:r:0.01, -r:r:0.01]
        x=np.cos((2*np.pi/0.1)*np.sqrt(t**2+p**2))+ \
                np.cos((2*np.pi/0.0333333)*np.sqrt(t**2+p**2))
        x=x.reshape(1,x.shape[0], x.shape[1])
        psd=PowerSpectralDensity(x)
        N=psd.shape[0]
        plt.plot(np.arange(N)/N,np.log(psd))
        plt.show()
        
######################## OLD ##################################################
        
#### this old code is deprecated since the binning version is considerably 
# slower than the one used  above

def radial_bin_dct_old(dct_sig, UniRad=None, Rad=None,Inds=None):
    """
    compute radial binning sum of dct_sig array
    Inputs :
        dct_sig : the signal to be binned (eg variance) : np array of square size
        UniRad : unique values of radii to be binned on (array)
        Rad : values of radii according to x,y int location
        Inds : indexes following the sie of dct_sig
            if the 3 latter are not provided, they are computed and returned
        
    Returns  :
        
        Binned_Sigma : binned dct signal along UniRad
        UniRad, Rad, Inds
        
    """
    N=dct_sig.shape[0]

    if Inds==None and UniRad==None:
        Inds=np.array([[[i,j] for i in range(N)]for j in range(N)])
        
        Rad=np.linalg.norm(Inds, axis=-1)/(N)
        UniRad=np.unique(Rad)[1:]
    
    
    Binned_Sigma=np.zeros(UniRad[UniRad<0.5].size)
    for i,r in enumerate(UniRad[UniRad<0.5]):
        #sum the contributions of dct_sig positions for which radius==r
        Binned_Sigma[i]=0.5*dct_sig[Rad==2*r-1].sum()+dct_sig[Rad==2*r].sum()+0.5*dct_sig[Rad==2*r+1].sum()
        
    return UniRad, Binned_Sigma, Inds, Rad
    
def PowerSpectralDensity_old(x, UniRad=None, Rad=None, Inds=None):
    """
    collating previous functions
    """
    
    UniRad, Binned_Sigma, Inds, Rad=radial_bin_dct(dct_var(x).mean(axis=0),UniRad,Rad,Inds)
    return UniRad, Binned_Sigma, Inds, Rad

def PSD_wrap_old(x):
     UniRad, Binned_Sigma, Inds, Rad=PowerSpectralDensity(x)
     return UniRad[UniRad<0.5], Binned_Sigma
 
    
def plot_spectrum(rad, binned_spectrum, name, delta, unit):
        plt.plot((1/delta)*rad, binned_spectrum)
        plt.yscale('log')
        plt.xlabel('Wavenumber (km^{-1})')
        plt.ylabel(name+' ({})'.format(unit))
        plt.title('Power Spectral Density, '+name)
        #plt.savefig('./PSD_'+name+'.png')
        plt.show()
