#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:42:52 2020

@author: wmallard
"""
import os
import sys

try:
    filenames = sys.argv[1:]
    assert filenames
except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} STACK', file=sys.stderr)
    sys.exit(1)

from skimage import io
import Util

for n, src in enumerate(filenames):

    print(f'[{n+1}/{len(filenames)}] {src}')

    base, _ = os.path.splitext(src)
    dst = f'{base} MIP.tif'

    mip = Util.maximum_intensity_projection(src)
    io.imsave(dst, mip, bigtiff=True, check_contrast=False)
