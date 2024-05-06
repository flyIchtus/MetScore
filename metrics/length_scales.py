#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:48:51 2022

@author: brochetc

correlation lengths

"""

import logging

import numpy as np


def get_metric_tensor(eps, sca):
    """

    Compute the metric correlation tensor of a given field eps
    with a unit length scale sca
    Inputs :

        eps : array of shape B X C X H x W
        sca : float
    Returns :

        g : array of shape 2 x 2 x C  x (H-1) x (W-1)

    """
    print(eps.shape)
    C, H, W = eps.shape[1], eps.shape[2], eps.shape[3]

    d_eps_x = np.diff(eps, axis=2)[:, :, :, 1:] / sca
    d_eps_y = (
        np.diff(eps, axis=3)[
            :,
            :,
            1:,
        ]
        / sca
    )

    dx_dx = np.expand_dims(np.mean(d_eps_x * d_eps_x, axis=0), axis=0)
    dx_dy = np.expand_dims(np.mean(d_eps_x * d_eps_y, axis=0), axis=0)

    dx = np.concatenate((dx_dx, dx_dy), axis=0)

    dy_dx = np.expand_dims(np.mean(d_eps_y * d_eps_x, axis=0), axis=0)
    dy_dy = np.expand_dims(np.mean(d_eps_y * d_eps_y, axis=0), axis=0)

    dy = np.concatenate((dy_dx, dy_dy), axis=0)

    logging.debug(f"dx shape : {dx.shape}, dy shape : {dy.shape}")
    g = np.concatenate((dx, dy), axis=0)

    return g.reshape(2, 2, C, H - 1, W - 1)


def get_normalized_field(eps):
    """

    Normalizes a given field with respect to Batch and spatial dimensions

    Inputs :

        eps : array of shape B x C x H x W

    Returns :

        array of shape B X C x H x W

    """
    sig = np.std(eps, axis=(0, 2, 3), keepdims=True)
    mean = np.mean(eps, axis=(0, 2, 3), keepdims=True)

    return (eps - mean) / sig


def correlation_length(g, sca):
    """

    Give an estimate of the correlation length present in a metric tensor g
    with a given length scale sca

    Inputs :

        g : array of shape  2 x 2 x C x H x W
        sca : float

    Returns :

        ls : array of shape C x H x W

    """

    correl = 0.5 * (np.trace(np.sqrt(np.abs(g))))

    ls = 1.0 / correl

    return ls


def correlation_length_coord(g, sca):
    """

    Give an estimate of the per-direction
    correlation lengths present in a metric tensor g
    with a given length scale sca

    Inputs :

        g : array of shape 2 x 2 x C x H x W
        sca : float

    Returns :

        ls : array of shape 2 x C x H x W

    """

    correl = np.sqrt(np.diag(g))

    ls = (1.0 / correl) * sca

    return ls


def compute_anisotropy(g, sca):
    """

    Give an estimate of the anisotropy in a metric tensor g
    with a given length scale sca

    Inputs :

        g : array of shape 2 x 2 x C x H x W
        sca : float

    Returns :

        ani  : array of shape C x H x W

    """

    ani = 0.5 * (np.sqrt(np.abs(g[0, 1])) + np.sqrt(np.abs(g[1, 0])))

    return ani / sca


def length_scale(eps, sca=1.0):
    """
    Give an estimate of correlation length maps given a field eps and
    a scale sca

    Inputs :

        eps : array of shape B x C x H x W
        sca : float

    Returns :

        ls : array of shape C x H x W

    """

    eps_0 = get_normalized_field(eps)

    g = get_metric_tensor(eps_0, sca)

    ls = correlation_length(g, sca)

    return ls


def length_scale_abs(eps_real, eps_fake, sca=1.0):
    """
    Give an estimate of correlation length maps given a field eps and
    a scale sca

    Inputs :

        eps : array of shape B x C x H x W
        sca : float

    Returns :

        ls : array of shape C x H x W

    """
    ls_real = length_scale(eps_real, sca=sca)
    ls_fake = length_scale(eps_fake, sca=sca)

    return np.abs(ls_real - ls_fake)
