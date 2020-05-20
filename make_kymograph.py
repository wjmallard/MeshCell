#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date May 2020
"""
import numpy as np
from skimage import io
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks

import Segmentation
import Contour
import Skeleton
import Mesh
import Kymograph

PHASE = 'test_phase.tif'
TIRF_REG = 'test_tirf_reg.tif'
TIRF_MIP = 'test_tirf_mip.tif'

phase = io.imread(PHASE)
movie = io.imread(TIRF_REG)
tirf_mip = io.imread(TIRF_MIP)

# Prepare kymograph interpolation.
num_points = 20

Sy, Sx = movie[0].shape
x_mesh = np.arange(Sx)
y_mesh = np.arange(Sy)

interp_mip = RectBivariateSpline(y_mesh, x_mesh, tirf_mip)
interp_movie = [RectBivariateSpline(y_mesh, x_mesh, frame) for frame in movie]

# Generate meshes for phase image.

# Segment cells.
object_labels = Segmentation.segment_phase_image(phase)
bg_id = Segmentation.identify_background(phase, object_labels, im_type='phase')

# Prepare contour generator.
Contours = Contour.ContourGenerator(phase, object_labels)

Velocities = []

for cell_id in np.unique(object_labels):

    print(f'Processing cell {cell_id}.')

    if cell_id == bg_id:
        print(' - Background, skipping.')
        continue

    contour = Contours.generate(cell_id)
    skeleton = Skeleton.generate(contour)
    mesh = Mesh.make_ribs(contour, skeleton)
    rib_starts, top_intersections, bot_intersections = mesh

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

    # Generate a kymograph at each peak.
    for i in peaks:

        print(f'Processing cell {cell_id} rib {i}.')

        # Construct rib.
        x1, y1 = top_intersections[i]
        x2, y2 = bot_intersections[i]

        x_points = np.linspace(x1, x2, num_points)
        y_points = np.linspace(y1, y2, num_points)

        # Generate kymograph.
        K = np.array([interp_frame.ev(y_points, x_points)
                      for interp_frame in interp_movie])

        # Calculate velocity.
        sampling_period = 1  # second
        cell_diameter = 11 * .065  # microns
        velocity = Kymograph.calculate_velocity(K, sampling_period, cell_diameter)

        Velocities.append(velocity)

Velocities = np.array(Velocities)
