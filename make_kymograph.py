#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date May 2020
"""
import os
import numpy as np

import Util
import Segmentation
import Contour
import Skeleton
import Mesh
import Kymograph
import Diagnostics

SAMP_TIRF = 'bAB185 T0 TIRF 001 registered.tif'
SAMP_CELL = 'bAB185 T0 feature_0 001.tif'
SAMP_EDGE = 'bAB185 T0 feature_1 001.tif'

BASEDIR = '/Volumes/Delphium/Microscopy/Vabam/2019.07.16 bWM100-34-38 T0-T3 S750 TIRF'

TIRF_IMG = os.path.join(BASEDIR, 'registered_tirf', SAMP_TIRF)
CELL_IMG = os.path.join(BASEDIR, 'segmented_phase', SAMP_CELL)
EDGE_IMG = os.path.join(BASEDIR, 'segmented_phase', SAMP_EDGE)

kymo_width = 20

# Load images.
print(f'Loading images.')
tirf_mip = Util.maximum_intensity_projection(TIRF_IMG)
cells = Util.load_image(CELL_IMG)
edges = Util.load_image(EDGE_IMG)

# Align cell boundaries with TIRF movie.
print(f'Aligning phase and TIRF images.')
dy, dx = Util.align_images(tirf_mip, cells)
cells = Util.shift_image(cells, dx, dy)
edges = Util.shift_image(edges, dx, dy)

# Segment cells.
print(f'Extracting segmented regions.')
object_labels = Segmentation.segment_deepcell_masks(cells)
bg_id = Segmentation.identify_background(cells, object_labels)

# Prepare contour generator.
Contours = Contour.ContourGenerator(edges, object_labels)

for cell_id in np.unique(object_labels):

    print(f'Processing cell {cell_id}.')

    if cell_id == bg_id:
        print(' - Background, skipping.')
        continue

    # Generate mesh.
    try:
        contour = Contours.generate(cell_id)
        skeleton = Skeleton.generate(contour)
        mesh = Mesh.make_ribs(contour, skeleton)
        rib_starts, top_intersections, bot_intersections = mesh
    except:
        print(' - Failed, skipping.')
        continue

    # Find brighest ribs.
    peaks = Kymograph.find_intensity_peaks(tirf_mip, mesh, kymo_width)

    # Generate kymographs at these ribs.
    for i in peaks:

        print(f'Processing cell {cell_id} rib {i}.')

        # Generate kymograph.
        P1 = top_intersections[i]
        P2 = bot_intersections[i]

        kymograph = Kymograph.make_kymograph(TIRF_IMG, P1, P2, kymo_width)

        # Save results.
        title = f'Cell {cell_id}, Rib {i}'
        filename = f'Cell_{cell_id:04}_rib_{i:04}.png'

        bbox = Util.find_cell_bbox(object_labels, cell_id)

        Diagnostics.debug_kymograph(tirf_mip,
                                    bbox,
                                    top_intersections[i],
                                    bot_intersections[i],
                                    kymograph,
                                    contour=contour,
                                    skeleton=skeleton,
                                    title=title,
                                    filename=filename)
