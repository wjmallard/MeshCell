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
from scipy.signal import find_peaks

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

bg_id = Segmentation.identify_background(im, object_labels, im_type='phase')

Contours = Contour.ContourGenerator(im, object_labels)

# TODO: Loop over all cells.
# for cell_id in num_cells.unique():
#     if cell_id == bg_id: continue
#     X, Y = get_cell_boundary_coords(object_labels, cell_id)
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

interp_mip = RectBivariateSpline(y_mesh, x_mesh, tirf_mip)
interp_movie = [RectBivariateSpline(y_mesh, x_mesh, frame) for frame in movie]

# Find intensity peaks along skeleton.
rib_sums = []

for P1, P2 in zip(top_intersections, bot_intersections):

    x1, y1 = P1
    x2, y2 = P2

    x_points = np.linspace(x1, x2, num_points)
    y_points = np.linspace(y1, y2, num_points)

    result = interp_mip.ev(y_points, x_points).sum()
    rib_sums.append(result)

rib_sums = np.array(rib_sums)
peaks, _ = find_peaks(rib_sums, height=rib_sums.mean())

for i in peaks:

    print(f'Processing rib {i}.')

    # Construct rib.
    x1, y1 = top_intersections[i]
    x2, y2 = bot_intersections[i]

    x_points = np.linspace(x1, x2, num_points)
    y_points = np.linspace(y1, y2, num_points)

    # Generate kymograph.
    K = np.array([interp_frame.ev(y_points, x_points)
                  for interp_frame in interp_movie])

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
    ax.set_ylim([0, Sy])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(grid[1,1])
    ax.imshow(K, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(f'Rib_{i:02}.png')
