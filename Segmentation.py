#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
from skimage import util
from skimage import morphology
from skimage import segmentation

h_min_to_fill = 0.1  # relative to 1.0 maximum

def fill_image_to_min_height(im, h_min):
    '''
    Reproduces Matlab's imhmin() function.

    Fill local minima in image up to depth h_min.

    Parameters
    ----------
    im : 2D np.array, float32
        A TIFF image to be filled.
    h_min : scalar, float32
        Percentile for clipping. (out of 100)

    '''
    im = util.invert(im)
    im = morphology.reconstruction((im - h_min), im, method='dilation')
    im = util.invert(im)
    return im

def segment_deepcell_masks(im):
    '''
    Segment DeepCell cell masks via watershed.

    Parameters
    ----------
    im : 2D np.array, float32
        A TIFF image to be segmented.

    Returns
    -------
    object_labels : 2D np.array, int32
        Assigns each pixel to a cell_id.

    '''
    im_filled = fill_image_to_min_height(im, h_min_to_fill)
    object_labels = segmentation.watershed(im_filled, connectivity=8)
    # TODO: Filter out objects that are too small or too big.
    return object_labels
