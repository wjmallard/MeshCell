#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
from skimage import io
import matplotlib.pyplot as plt

import Segmentation
import Contour
import Skeleton
import Mesh

FILENAME = '/Users/wmallard/Desktop/Microscopy/test.tif'

im = io.imread(FILENAME, as_gray=True)
object_labels = Segmentation.segment_cells(im)

Contours = Contour.ContourGenerator(im, object_labels)

# TODO: Loop over all cells.
# num_cells = object_labels.max() + 1
# for cell_id in range(num_cells):
#     X, Y = get_cell_boundary_coords(object_labels, cell_id)
#     if X is None: continue
cell_id = 6

contour = Contours.generate(cell_id)
skeleton = Skeleton.generate(contour)
rib_starts, top_intersections, bot_intersections = Mesh.make_ribs(contour, skeleton)


plt.close('all')
plt.plot(*contour.T, 'ko-')
plt.plot(*skeleton.T, 'rx-')

for i in range(len(rib_starts)):

    xs, ys = rib_starts[i]
    xt, yt = top_intersections[i]
    xb, yb = bot_intersections[i]

    plt.plot([xs, xt], [ys, yt], 'b')
    plt.plot([xs, xb], [ys, yb], 'b')

