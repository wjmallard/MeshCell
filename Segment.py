#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:42:35 2020

@author: wmallard
"""

#%% Step 0: Load libraries and image.

from skimage import io

FILENAME = '/Users/wmallard/Desktop/Microscopy/test.tif'

im = io.imread(FILENAME, as_gray=True)
