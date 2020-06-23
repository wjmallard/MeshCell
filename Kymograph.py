#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:04:04 2020

@author: wmallard
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks

from numpy.fft import rfft, rfftfreq
from scipy.signal import detrend, windows
from scipy.optimize import curve_fit

import Util

def make_kymograph(filename, P1, P2, kymo_width):

    shape = Util.read_image_shape(filename)
    assert len(shape) == 3
    _, im_height, im_width = shape

    x1, y1 = P1
    x2, y2 = P2

    llx, lly, urx, ury = calc_bbox(P1, P2, kymo_width, im_height, im_width)

    stack = Util.load_cropped_image(filename, llx, lly, urx, ury)

    y_mesh = np.arange(ury - lly)
    x_mesh = np.arange(urx - llx)

    splines = [RectBivariateSpline(y_mesh, x_mesh, frame) for frame in stack]

    y_points = np.linspace(y1 - lly, y2 - lly, kymo_width)
    x_points = np.linspace(x1 - llx, x2 - llx, kymo_width)

    kymograph = np.array([sp.ev(y_points, x_points) for sp in splines])

    return kymograph

def calc_bbox(P1, P2, size, im_height, im_width):

    # Find the center.
    Cx, Cy = np.mean((P1, P2), axis=0)

    # Find box half-width.
    dxy = size / 2 + 1

    # Find bbox coords.
    llx = int(np.floor(Cx - dxy))
    lly = int(np.floor(Cy - dxy))
    urx = int(np.ceil(Cx + dxy))
    ury = int(np.ceil(Cy + dxy))

    llx = max(llx, 0)
    lly = max(lly, 0)
    urx = min(urx, im_width)
    ury = min(ury, im_height)

    return llx, lly, urx, ury

def integrate_along_ribs(im, mesh, kymo_width):

    Sy, Sx = im.shape
    x_mesh = np.arange(Sx)
    y_mesh = np.arange(Sy)

    spline = RectBivariateSpline(y_mesh, x_mesh, im)

    _, top_intersections, bot_intersections = mesh

    rib_sums = []

    for P1, P2 in zip(top_intersections, bot_intersections):

        x1, y1 = P1
        x2, y2 = P2

        x_points = np.linspace(x1, x2, kymo_width)
        y_points = np.linspace(y1, y2, kymo_width)

        result = spline.ev(y_points, x_points).sum()
        rib_sums.append(result)

    rib_sums = np.array(rib_sums)

    return rib_sums

def find_intensity_peaks(im, mesh, kymo_width):

    rib_sums = integrate_along_ribs(im, mesh, kymo_width)

    X = detrend(rib_sums)
    peaks, _ = find_peaks(X, height=0)

    return rib_sums, peaks

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

    Y = detrend(Y)
    Y *= windows.hann(n_timepoints)

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
