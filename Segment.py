#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:42:35 2020

@author: wmallard
"""

#%% Step 0: Load libraries and image.

from skimage import io
from skimage import util
from skimage.morphology import reconstruction
from skimage.morphology import watershed
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

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

def get_cell_boundary_coords(object_labels, boundaries, cell_id):
    '''
    Find the boundary coordinates of a particular cell.

    Arguments:
        object_labels - int matrix; labels which object each pixel belongs to
        boundaries - bool matrix; labels whether a pixel is a boundary
        cell_id - specifies which object to find the boundary coords of
    '''
    cell_mask = object_labels == cell_id
    cell_boundary = boundaries & cell_mask

    props = regionprops(cell_boundary.astype(int))
    assert len(props) == 1
    p = props[0]

    X = p.coords.T[0]
    Y = p.coords.T[1]

    return X, Y

boundaries = find_boundaries(object_labels)
