#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:23:53 2023

@author: brochetc
"""

import numpy as np

import metrics.spectrum_analysis as spec


def spectrum_deviation(X):
    X_avg = X.mean(axis=0)

    X_dev = X - X_avg

    psd = spec.PowerSpectralDensity(X_dev)
    return psd


def spectrum_variance(X):
    X_var = np.expand_dims(X.var(axis=0), axis=0)

    psd = spec.PowerSpectralDensity(X_var)

    return psd


def spectrum_std(X):
    X_var = np.expand_dims(X.std(axis=0), axis=0)

    psd = spec.PowerSpectralDensity(X_var)

    return psd
