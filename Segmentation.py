#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np

from skimage import util
from skimage import exposure
from skimage import filters
from skimage import morphology
from skimage import segmentation

h_min_to_fill = 0.1  # relative to 1.0 maximum
deepcell_threshold = 0.5  # relative to 1.0 maximum

def contrast_stretch(im, clip=0):
    '''
    Enhance image contrast via linear scaling.

    Stretch pixel values to cover the full dynamic range of the image format.
    Optionally clip pixel values at a given percentile, saturating any values
    below the Xth percentile or above the (100-X)th percentile.

    Parameters
    ----------
    im : 2D np.array, float32
        A TIFF image to be segmented.
    clip : scalar, float32
        Percentile for clipping, in [0, 100).

    '''
    assert(0 <= clip < 100)
    v_min, v_max = np.percentile(im, (clip, 100 - clip))
    im = exposure.rescale_intensity(im, in_range=(v_min, v_max))
    return im

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

def segment_phase_image(im):
    '''
    Segment a phase image via watershed.

    Parameters
    ----------
    im : 2D np.array, float32
        A TIFF image to be segmented.

    Returns
    -------
    object_labels : 2D np.array, int32
        Assigns each pixel to a cell_id.

    '''
    im = contrast_stretch(im, clip=2)
    im = filters.sobel(im)
    im = fill_image_to_min_height(im, h_min_to_fill)
    object_labels = segmentation.watershed(im, connectivity=8)
    return object_labels

def segment_deepcell_masks(im):
    '''
    Segment DeepCell cell masks.

    Parameters
    ----------
    im : 2D np.array, float32
        A TIFF image to be segmented.

    Returns
    -------
    object_labels : 2D np.array, int32
        Assigns each pixel to a cell_id.

    '''
    im = im >= deepcell_threshold
    object_labels = morphology.label(im, connectivity=2)
    # TODO: Filter out objects that are too small or too big.
    return object_labels

def identify_background(im, object_labels, im_type=None):
    '''
    Infer the background label based on mean intensity.

    If im_type is 'phase', find the label with the maximum mean intensity.
    Otherwise, use minimum mean intensity.

    Parameters
    ----------
    im : 2D np.array, float32
        An image.
    object_labels : 2D np.array, int32
        Assigns each pixel to a cell_id.
    im_type : str
        None (default), 'phase'

    Returns
    -------
    bg_label : scalar, int32
        The cell_id of the background.

    '''
    labels = np.unique(object_labels)
    mean_intensities = [im[object_labels == label].mean()
                        for label in labels]

    if im_type == 'phase':
        bg_label = labels[np.argmax(mean_intensities)]
    else:
        bg_label = labels[np.argmin(mean_intensities)]

    return bg_label

def size_filter(object_labels, min_area):

    labels, areas = np.unique(object_labels, return_counts=True)
    labels_to_remove = labels[areas < min_area]

    return set(labels_to_remove)

def intensity_filter(im, object_labels, min_intensity):

    mean_intensity = lambda label: im[object_labels == label].mean()

    labels_to_remove = [label for label in np.unique(object_labels)
                        if mean_intensity(label) < min_intensity]

    return set(labels_to_remove)

def sort_by_intensity(im, object_labels):

    mean_intensity = lambda label: im[object_labels == label].mean()

    labels = list(np.unique(object_labels))
    labels.sort(key=mean_intensity, reverse=True)

    return labels
