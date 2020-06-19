#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:42:52 2020

@author: wmallard
"""
import os
import sys
import numpy as np
from tifffile import TiffFile
from skimage import io
from tqdm import tqdm

try:
    filenames = sys.argv[1:]
    assert filenames
except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} STACK', file=sys.stderr)
    sys.exit(1)

class ProgressBar(tqdm):

    def __init__(self, num_frames):
        super().__init__(total=num_frames, unit='frames')

    def update_to(self, current_iteration):
        current_frame = current_iteration + 2
        self.update(current_frame - self.n)

def count_frames(filename):

    with TiffFile(filename) as im:
        num_frames = len(im.pages)

    return num_frames

def make_mip(filename, progress_callback):

    with TiffFile(filename) as im:

        frame = im.pages[0]
        mip = np.zeros(frame.shape, dtype=frame.dtype)

        for n, frame in enumerate(im.pages):
            np.maximum(mip, frame.asarray(), out=mip)
            progress_callback(n)

    return mip

for n, src in enumerate(filenames):

    print(f'[{n+1}/{len(filenames)}] {src}')

    base, _ = os.path.splitext(src)
    dst = f'{base} MIP.tif'

    num_frames = count_frames(src)

    with ProgressBar(num_frames) as pb:
        mip = make_mip(src, progress_callback=pb.update_to)

    io.imsave(dst, mip, bigtiff=True, check_contrast=False)
