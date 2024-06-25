#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:15:29 2022

@author: brochetc

Metric version of quantiles calculation

"""

import numpy as np
import warnings


def quantiles(data, qlist=[0.99]):
    """
    compute quantiles of data shape on first axis using numpy 'primitive'

    Inputs :

        data : np.array, shape B x C x H x W

        qlist : iterable of size N containing quantiles to compute
        (between 0 and 1 inclusive)

    Returns :

        np.array of shape N x C x H x W
    """

    return np.quantile(data, qlist, axis=0)


def quantiles_non_zero(data, qlist, threshold=1.0):
    """
    compute quantiles of data shape on first axis using numpy 'primitive'
    drop the data below a threshold to evaluate the quantiles
    quantiles are represented in qlist ; data maximum (=q100) is added afterwardss

    Inputs :

        data : np.array, shape B x C x H x W

        qlist : iterable of size N containing quantiles to compute
        (between 0 and 1 inclusive)

        threshold : defines the threshold under which data is considered to be 0

    Returns :

        np.array of shape N x C x H x W
    """

    non_zero_mask = data[:, 0, :, :] >= threshold  # Rapid
    non_zero_data = np.where(non_zero_mask, data[:, 0, :, :], np.nan)  # Rapid
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
        quantiles = np.nanquantile(
            non_zero_data, qlist, axis=0, keepdims=True
        )  # This takes time

    maxi = np.max(data[:, 0, :, :], axis=0, keepdims=True)

    quantiles = np.concatenate((quantiles, maxi.reshape(1, *maxi.shape)))
    return quantiles


def quantile_score(real_data, fake_data, qlist=[0.99]):
    """
    compute rmse of quantiles maps as outputted by quantiles function

    Inputs :

        real_data : np.array of shape B x C x H x W

        fake_data : np.array of shape B x C x H x W

        qlist : iterable of length N containing quantiles to compute
        ((between 0 and 1 inclusive)
    Returns :

        q_score : np.array of length N x C

    """

    q_real = quantiles(real_data, qlist)
    q_fake = quantiles(fake_data, qlist)

    q_score = np.sqrt((q_fake - q_real) ** 2).mean(axis=(2, 3))

    return q_score
