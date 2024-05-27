#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:39:22 2022

@author: brochetc


Sliced Wasserstein Distance API and functions
"""

####################### NVIDIA implementation for Sliced wasserstein Distance##
# ----------------------------------------------------------------------------
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import copy
from math import log2

import numpy as np
import scipy.ndimage
from torch import tensor


def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    r"""Compute the descriptors from the minibatch

            Parameters
            ----------
            minibatch : np.array
            nhood_size : int
            nhoods_per_image : int

            Returns
            -------
            descriptors : np.array
            

        """
    S = minibatch.shape  # (minibatch, channel, height, width)
    assert len(S) == 4
    N = nhoods_per_image * S[0]
    H = nhood_size // 2
    nhood, chan, x, y = np.ogrid[0:N, 0 : S[1], -H : H + 1, -H : H + 1]
    img = nhood // nhoods_per_image
    x = x + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1))
    y = y + np.random.randint(H, S[2] - H, size=(N, 1, 1, 1))
    idx = ((img * S[1] + chan) * S[2] + y) * S[3] + x
    return minibatch.flat[idx]


# ----------------------------------------------------------------------------


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)  # normalizing on each channel
    desc /= np.std(
        desc, axis=(0, 2, 3), keepdims=True
    )  # similar to batch+instance norm
    # actually this is weird and not quite justified ?
    desc = desc.reshape(desc.shape[0], -1)  # reshaping
    return desc


# ----------------------------------------------------------------------------


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    assert A.ndim == 2 and A.shape == B.shape  # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(
            A.shape[1], dirs_per_repeat
        )  # (descriptor_component, direction)
        dirs /= np.sqrt(
            np.sum(np.square(dirs), axis=0, keepdims=True)
        )  # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(A, dirs)  # (neighborhood, direction)
        projB = np.matmul(B, dirs)
        projA = np.sort(
            projA, axis=0
        )  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)  # pointwise wasserstein distances
        results.append(np.mean(dists))  # average over neighborhoods and directions
    return np.mean(results)  # average over repeats


# ----------------------------------------------------------------------------


def downscale_minibatch(minibatch, lod):
    if lod == 0:
        return minibatch
    t = minibatch.astype(np.float32)
    for i in range(lod):
        t = (
            t[:, :, 0::2, 0::2]
            + t[:, :, 0::2, 1::2]
            + t[:, :, 1::2, 0::2]
            + t[:, :, 1::2, 1::2]
        ) * 0.25
    return t


# ----------------------------------------------------------------------------

gaussian_filter = (
    np.float32(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    / 256.0
)


def pyr_down(minibatch):  # matches cv2.pyrDown()
    assert minibatch.ndim == 4
    return scipy.ndimage.convolve(
        minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :], mode="mirror"
    )[:, :, ::2, ::2]


def pyr_up(minibatch):  # matches cv2.pyrUp()
    assert minibatch.ndim == 4
    S = minibatch.shape
    if log2(S[2]) - round(log2(S[2])) != 0:
        res = np.zeros((S[0], S[1], S[2] * 2 - 1, S[3] * 2 - 1), minibatch.dtype)
    else:
        res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2), minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(
        res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode="mirror"
    )


def generate_laplacian_pyramid(minibatch, num_levels):
    if isinstance(minibatch, np.ndarray):
        pyramid = [np.float32(minibatch)]
    else:
        pyramid = [np.float32(minibatch.cpu())]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


# ----------------------------------------------------------------------------
"""
this API is quite strange, but should work
"""


class SWD_API:
    def __init__(self, image_shape, numpy=False):
        self.nhood_size = 7
        self.nhoods_per_image = 128
        self.dir_repeats = 4
        self.dirs_per_repeat = 128
        self.resolutions = []
        self.numpy = numpy
        res = image_shape[1]
        while res >= 16:
            self.resolutions.append(res)
            res //= 2

    def get_metric_names(self):
        return ["SWDx1e3_%d" % res for res in self.resolutions] + ["SWDx1e3_avg"]

    def get_metric_formatting(self):
        return ["%-13.4f"] * len(self.get_metric_names())

    def begin(self, mode):
        assert mode in ["warmup", "reals", "fakes"]
        self.descriptors = [[] for res in self.resolutions]

    def feed(self, minibatch):
        for lod, level in enumerate(
            generate_laplacian_pyramid(minibatch, len(self.resolutions))
        ):
            desc = get_descriptors_for_minibatch(
                level, self.nhood_size, self.nhoods_per_image
            )
            self.descriptors[lod].append(desc)

    def end(self):
        self.desc_real = [
            finalize_descriptors(self.descriptors[lod][0])
            for lod, _ in enumerate(self.descriptors)
        ]
        self.desc_fake = [
            finalize_descriptors(self.descriptors[lod][1])
            for lod, _ in enumerate(self.descriptors)
        ]

        del self.descriptors
        dist = [
            sliced_wasserstein(dreal, dfake, self.dir_repeats, self.dirs_per_repeat)
            for dreal, dfake in zip(self.desc_real, self.desc_fake)
        ]

        del self.desc_real, self.desc_fake

        dist = [d * 1e3 for d in dist]  # multiply by 10^3

        return dist + [np.mean(dist)]

    def End2End(self, real, fakes):
        real = copy.deepcopy(real)
        fakes = copy.deepcopy(fakes)

        self.begin("fakes")

        self.feed(fakes)

        self.feed(real)

        if self.numpy:
            return np.array(self.end())
        else:
            return tensor(self.end())
