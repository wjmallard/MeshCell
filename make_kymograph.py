#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date May 2020
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

import Segmentation
import Contour
import Skeleton
import Mesh

PHASE = 'test_phase.tif'
TIRF_MIP = 'test_tirf_mip.tif'
TIRF_REG = 'test_tirf_reg.tif'

#%% Generate meshes for phase image.
im = io.imread(PHASE, as_gray=True)
object_labels = Segmentation.segment_phase_image(im)

Contours = Contour.ContourGenerator(im, object_labels)

# TODO: Loop over all cells.
# num_cells = object_labels.max() + 1
# for cell_id in range(num_cells):
#     X, Y = get_cell_boundary_coords(object_labels, cell_id)
#     if X is None: continue
cell_id = 3

contour = Contours.generate(cell_id)
skeleton = Skeleton.generate(contour)
rib_starts, top_intersections, bot_intersections = Mesh.make_ribs(contour, skeleton)


plt.close('all')
plt.imshow(im, cmap='gray')
plt.plot(*contour.T, 'ko-')
plt.plot(*skeleton.T, 'rx-')

for i in range(len(rib_starts)):

    xs, ys = rib_starts[i]
    xt, yt = top_intersections[i]
    xb, yb = bot_intersections[i]

    plt.plot([xs, xt], [ys, yt], 'b')
    plt.plot([xs, xb], [ys, yb], 'b')

#%% Generate kymographs along meshes.
movie = io.imread(TIRF_REG)
tirf_mip = io.imread(TIRF_MIP)

num_points = 20

Sy, Sx = movie[0].shape
x_mesh = np.arange(Sx)
y_mesh = np.arange(Sy)

for i in range(len(top_intersections)):

    print(f'Processing rib {i}.')

    # Construct rib.
    x1, y1 = top_intersections[i]
    x2, y2 = bot_intersections[i]

    x_points = np.linspace(x1, x2, num_points)
    y_points = np.linspace(y1, y2, num_points)

    # Generate kymograph.
    F = [RectBivariateSpline(y_mesh, x_mesh, frame) for frame in movie]
    K = np.array([f.ev(y_points, x_points) for f in F])

    # Plot.
    plt.close('all')
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    fig.suptitle(f'Rib {i}')
    grid = fig.add_gridspec(2, 2,
                            wspace=0.0,
                            hspace=0.0,
                            width_ratios=[10, 1],
                            height_ratios=[1, 50])

    ax = fig.add_subplot(grid[1,0])
    ax.imshow(tirf_mip, cmap='gray')
    ax.plot((x1, x2), (y1, y2), 'r-', linewidth=2)
    ax.plot(*contour.T, 'g:')
    ax.plot(*skeleton.T, 'y:')
    ax.set_xlim([0, Sx])
    ax.set_xlim([0, Sy])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid[1,1])
    ax.imshow(K, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(f'Rib_{i:02}.png')
