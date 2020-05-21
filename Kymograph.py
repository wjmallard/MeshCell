#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:04:04 2020

@author: wmallard
"""
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

def gauss(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset

def fit_gaussian(X, Y):
    popt, pcov = curve_fit(gauss, X, Y, p0=[1, 1, 1, 1])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def collapse_kymo_to_1d(K):

    num_rows, num_cols = K.shape
    assert num_rows > num_cols

    X = np.arange(num_cols)
    Y = K.sum(axis=0)

    G = fit_gaussian(X, Y)

    mu = G[0][1]
    mu = int(round(mu))

    return K[:,mu]

def find_dominant_frequency(Y, sampling_period):

    n_timepoints = len(Y)
    ff = rfftfreq(n_timepoints, sampling_period)

    F = abs(rfft(Y))
    F[0] = 0

    freq = abs(ff[np.argmax(F)])
    amp = Y.std() * np.sqrt(2.)
    offset = Y.mean()

    print(f'amplitude = {amp}')
    print(f'frequency = {freq}')
    print(f'period = {1 / freq}')
    print(f'offset = {offset}')

    return (freq, amp, offset)

def calculate_velocity(K, sampling_period, cell_diameter):
    '''
    Calculate the velocity of a treadmilling filament from a kymograph.

    Find the dominant frequency of the dominant bin of the kymograph.
    Calculate velocity via simple harmonic motion.

    Parameters
    ----------
    K : 2D np.array
        Kymograph.
    sampling_period : scalar, float32
        Time between exposures, in seconds.
    cell_diameter : scalar, float32
        Cell diameter, in microns.

    Returns
    -------
    velocity : scalar, float32
        Velocity, in microns per second.

    '''
    k = collapse_kymo_to_1d(K)
    F = find_dominant_frequency(k, sampling_period)
    frequency = F[0]

    return np.pi * cell_diameter * frequency
