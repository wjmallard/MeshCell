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

def make_line_endpoints(center, theta, length):
    '''
    Generate endpoint xy-coords of a line with the specified parameters.

    Parameters
    ----------
    center : 2-tuple, float
        Coordinates of the center of the line.
    theta : float
        Angle of the line, ccw from the x-axis.
    length : float
        Length, in pixels.

    Returns
    -------
    (float, float), (float, float)
        P1, P2. xy-coords for each end of the generated line.

    '''
    X = np.array((-length/2, length/2))
    Y = np.zeros(2)

    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) - Y * np.cos(theta)

    x1, x2 = center[0] + X_rot
    y1, y2 = center[1] + Y_rot

    return (x1, y1), (x2, y2)

def normalize_rib(rib, length):
    '''
    Length-normalize a rib.

    The interpolation path should be a fixed length for all ribs,
    regardless of cell's width at that point.

    Parameters
    ----------
    rib : ((float, float), (float, float), (float, float))
        xy-coords of: rib_start (on skeleton), rib_top, rib_bottom.
    length : float
        Length of new rib, in pixels.

    Returns
    -------
    new_rib : ((float, float), (float, float), (float, float))
        xy-coords of: rib_start (from input), rib_top, rib_bottom.

    '''
    C, P1, P2 = rib

    dx, dy = P2 - P1
    theta = np.arctan2(dy, dx)

    new_P1, new_P2 = make_line_endpoints(C, theta, length)

    return C, new_P1, new_P2

def make_kymograph(filename, rib, kymo_width):

    shape = Util.read_image_shape(filename)
    assert len(shape) == 3
    _, im_height, im_width = shape

    _, P1, P2 = normalize_rib(rib, kymo_width)

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

def fit_gaussian(Y, X=None):

    if X is None:
        X = np.arange(len(Y))

    # Initial guess:
    a = Y.max() - Y.min()
    m = X.mean()
    s = X.std()
    o = Y.min()

    popt, pcov = curve_fit(gauss, X, Y, p0=[a, m, s, o])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def sine(x, A, w, p, C):
    return A * np.sin(w * x + p) + C

def fit_sine(Y, X=None):

    if X is None:
        X = np.arange(len(Y))

    # Find dominant frequency.
    n_timepoints = len(Y)
    dt = X[1] - X[0]
    ff = abs(rfftfreq(n_timepoints, dt))

    F = abs(rfft(Y))
    F[0] = 0
    f_dominant = ff[np.argmax(F)]

    # Initial guess:
    a = Y.std() * 2**.5
    w = 2 * np.pi * f_dominant
    p = 0.
    c = Y.mean()

    popt, pcov = curve_fit(sine, X, Y, p0=[a, w, p, c])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def find_kymo_trace(kymograph):

    params = [fit_gaussian(row) for row in kymograph]
    values = np.array([p[0] for p in params])
    errors = np.array([p[1] for p in params])

    mu = values[:,1]

    return mu
