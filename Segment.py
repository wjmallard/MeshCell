#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:42:35 2020

@author: wmallard xies
"""

#%% Step 0: Load libraries and image.

import numpy as np
from skimage import io
from skimage import util
from skimage.morphology import reconstruction
from skimage.morphology import watershed
from skimage.measure import regionprops
from skimage.measure import find_contours

FILENAME = '/Users/wmallard/Desktop/Microscopy/test.tif'

h_min_to_fill = 0.1 # relative to 1.0 maximum

im = io.imread(FILENAME, as_gray=True)

#%% Step 1: Segment Cells

def fill_image_to_min_height(image, h_min):
    '''
    Reproduces Matlab's imhmin() function.
    
    Fill local minima in image up to depth h_min.
    
    Arguments:
        image - 2D image
        h_min - scalar
    '''
    im1 = util.invert(image)
    im2 = reconstruction((im1 - h_min), im1)
    im3 = util.invert(im2)
    return im3

im_filled = fill_image_to_min_height(im, h_min_to_fill)
object_labels = watershed(im_filled, connectivity=8)

#%% Step 2: Fit Contours

def get_cell_boundary_coords(object_labels, cell_id):
    '''
    Find the boundary coordinates of a particular cell.
    
    Arguments:
        object_labels - int matrix; labels which object each pixel belongs to
        cell_id - specifies which object to find the boundary coords of
    
    Returns:
        X, Y - coordinates of the cell's boundary,
               ordered counter clockwise around the contour
    '''
    cell_mask = object_labels == cell_id

    c = find_contours(cell_mask, level=.9)
    Y, X = c[0].T

    return X, Y

# TODO: Pre-compute smoothed image force components.

# TODO: Loop over all cells.
#for cell_id in range(num_cells):
#    pass
cell_id = 2

X, Y = get_cell_boundary_coords(object_labels, cell_id)

# Interpolate along curve to obtain evenly spaced spline anchors.
#
# X -- cartesian x-coordinates
# Y -- cartesian y-coordinates
# C -- 1D path coordinate along the contour
#
# NOTE: Assumes X-Y coordinates are sorted along the contour.
X_old, Y_old = X, Y

# Compute old coordinates along the contour with uneven spacing.
dx = X_old[1:] - X_old[:-1]
dy = Y_old[1:] - Y_old[:-1]
C_old = np.cumsum(np.sqrt(dx**2 + dy**2))

# Create new coordinates along the contour with even 1-pixel spacing.
C_length = np.int(np.ceil(C_old[-1]))
C_new = np.arange(C_length)

# Find the distance between each new (evenly spaced) anchor point
# and the next-smallest old (unevenly spaced) anchor point.
anchor_distance = C_old[:,None] - C_new[None,:]
anchor_distance[anchor_distance > 0] = -np.inf

# For each new anchor point,
# find the index of the next-smallest old anchor point.
J = np.argmax(anchor_distance, axis=0)

# Update step.
alpha = (C_new - C_old[J]) / (C_old[J+1] - C_old[J])
X_new = X_old[J] + (X_old[J+1] - X_old[J]) * alpha
Y_new = Y_old[J] + (Y_old[J+1] - Y_old[J]) * alpha
