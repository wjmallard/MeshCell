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
