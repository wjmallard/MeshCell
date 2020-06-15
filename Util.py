#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:53:08 2020

@author: wmallard
"""
import numpy as np
import warnings
from skimage import io
from nd2reader import ND2Reader
from tifffile import TiffFile
from scipy.signal import fftconvolve
from skimage import measure

def load_image(filename):

    if filename.endswith('.nd2'):
        return load_nd2(filename)
    else:
        return io.imread(filename)

def load_nd2(filename):

    # Suppress ND2 metadata warnings.
    warnings.filterwarnings('ignore', category=UserWarning)

    with ND2Reader(filename) as im:

        assert 'num_frames' in im.metadata
        num_frames = im.metadata['num_frames']

        frames = [im[i] for i in range(num_frames)]
        frames = np.stack(frames)
        frames = frames.squeeze()

    warnings.resetwarnings()

    return frames

def read_image_shape(filename):

    with TiffFile(filename) as im:

        num_frames = len(im.pages)
        frame = im.pages[0]
        stack_shape = (num_frames, *frame.shape)

    return stack_shape

def load_cropped_image(filename, x1, y1, x2, y2):

    assert y2 > y1
    assert x2 > x1
    dy = y2 - y1
    dx = x2 - x1

    with TiffFile(filename) as im:

        num_frames = len(im.pages)
        frame = im.pages[0]
        stack_shape = (num_frames, dy, dx)

        crop_stack = np.zeros(stack_shape, dtype=frame.dtype)
        full_frame = np.zeros(frame.shape, dtype=frame.dtype)

        for i, frame in enumerate(im.pages):
            frame.asarray(out=full_frame)
            crop_stack[i] = full_frame[y1:y2,x1:x2]

    return crop_stack

def maximum_intensity_projection(filename):

    if filename.endswith('.nd2'):
        return make_mip_nd2(filename)
    elif filename.endswith('.tif') or filename.endswith('.tiff'):
        return make_mip_tiff(filename)
    else:
        return make_mip_generic(filename)

def make_mip_nd2(filename):

    # Suppress ND2 metadata warnings.
    warnings.filterwarnings('ignore', category=UserWarning)

    with ND2Reader(filename) as im:

        frame = im[0]
        mip = np.zeros(frame.shape, dtype=frame.dtype)

        for frame in im:
            np.maximum(mip, frame, out=mip)

    warnings.resetwarnings()

    return mip

def make_mip_tiff(filename):

    with TiffFile(filename) as im:

        frame = im.pages[0]
        mip = np.zeros(frame.shape, dtype=frame.dtype)

        for frame in im.pages:
            np.maximum(mip, frame.asarray(), out=mip)

    return mip

def make_mip_generic(filename):

    im = io.imread(filename)

    assert len(im.shape) == 3
    mip = im.max(axis=0)

    return mip

def align_images(im1, im2):

    corr = fftconvolve(im1, im2[::-1,::-1], mode='same')

    max_corr = np.unravel_index(np.argmax(corr), corr.shape)
    midpoint = np.array(corr.shape) // 2

    dy, dx = max_corr - midpoint

    return dy, dx

def shift_image(im, dx, dy):

    im = np.roll(im, dy, axis=0)
    im = np.roll(im, dx, axis=1)

    if dy > 0: im[:dy,:] = 0
    if dy < 0: im[dy:,:] = 0
    if dx > 0: im[:,:dx] = 0
    if dx < 0: im[:,dx:] = 0

    return im

def find_cell_bbox(object_labels, cell_id):

    cell = (object_labels == cell_id).astype(np.uint8)
    props = measure.regionprops(cell)
    assert len(props) == 1

    lly, llx, ury, urx = props[0]['bbox']

    return llx, lly, urx, ury
