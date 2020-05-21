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

    # Suppress ND2 metadata warnings.
    warnings.filterwarnings('ignore', category=UserWarning)

    with ND2Reader(filename) as im:

        mip = np.zeros_like(im[0])

        for frame in im:
            np.maximum(mip, frame, out=mip)

    warnings.resetwarnings()

    return np.array(mip)
