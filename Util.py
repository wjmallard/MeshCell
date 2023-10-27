#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 23:53:08 2020

@author: wmallard
"""
import numpy as np
from skimage import measure
import re

def natural_sort(L):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(L, key=alphanum_key)

def find_cell_bbox(object_labels, cell_id):

    cell = (object_labels == cell_id).astype(np.uint8)
    props = measure.regionprops(cell)
    assert len(props) == 1

    lly, llx, ury, urx = props[0]['bbox']

    return llx, lly, urx, ury

def find_contour_bbox(contour):

    llx, lly = contour.min(axis=0)
    urx, ury = contour.max(axis=0)

    return llx, lly, urx, ury

def find_square_contour_bbox(contour, pad=5):

    llx, lly = contour.min(axis=0)
    urx, ury = contour.max(axis=0)

    cx = np.mean((llx, urx))
    cy = np.mean((lly, ury))

    w = urx - llx
    h = ury - lly

    size = np.max((w, h)) + pad

    llx = cx - size // 2
    urx = cx + size // 2
    lly = cy - size // 2
    ury = cy + size // 2

    return llx, lly, urx, ury
