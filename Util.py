#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:53:08 2020

@author: wmallard
"""
import numpy as np
from scipy.stats import linregress
from skimage import measure
import re

def natural_sort(L):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(L, key=alphanum_key)

def find_cell_bbox(object_labels, cell_id):

    cell = (object_labels == cell_id).astype(np.uint8)
    props = measure.regionprops(cell)
    assert len(props) == 1

    lly, llx, ury, urx = props[0]['bbox']

    return llx, lly, urx, ury

def find_contour_bbox(contour):

    llx, lly = contour.min(axis=0)
    urx, ury = contour.max(axis=0)

    return llx, lly, urx, ury

def find_square_contour_bbox(contour, pad=5):

    llx, lly = contour.min(axis=0)
    urx, ury = contour.max(axis=0)

    cx = np.mean((llx, urx))
    cy = np.mean((lly, ury))

    w = urx - llx
    h = ury - lly

    size = np.max((w, h)) + pad

    llx = cx - size // 2
    urx = cx + size // 2
    lly = cy - size // 2
    ury = cy + size // 2

    return llx, lly, urx, ury

def extend_line(line, ext_len):
    '''
    Extend a line by projecting a point outward from its end.

    Fit via linear regression, handling the case of near-vertical lines.

    Returns a single point projected outward from the line's end.

    Adapted from: https://stats.stackexchange.com/a/182893
    '''
    VERT_THRESHOLD = 1

    # Calculate the verticality of the line.
    X_std, Y_std = line.std(axis=0)
    verticality = (Y_std / X_std) if (X_std > 0) else np.inf

    X, Y = line.T

    # If the line is too vertical,
    # rotate it 90 degrees.
    if verticality > VERT_THRESHOLD:
        X, Y = Y, X

    # Fit via linear regression.
    res = linregress(X, Y)

    # Create a unit vector along the regression line.
    xa = X[0]
    xb = X[-1]
    yb = Y[-1]

    fa = res.slope * xa + res.intercept
    fb = res.slope * xb + res.intercept

    vector = np.array([xb - xa, fb - fa])
    vector /= np.linalg.norm(vector)

    # Project a point outward along the unit vector
    # starting from the end of the original line.
    x_ext, y_ext = ext_len * vector + np.array([xb, yb])

    # If we rotated the line 90 degrees,
    # rotate the projected point back.
    if verticality > VERT_THRESHOLD:
        x_ext, y_ext = y_ext, x_ext

    return x_ext, y_ext
