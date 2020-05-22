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

BASEDIR = '.'
PHASE = 'test_phase.tif'
TIRF_REG = 'test_tirf_reg.tif'
TIRF_MIP = 'test_tirf_mip.tif'

kymo_width = 20

stack = os.path.join(BASEDIR, TIRF_REG)

phase = Util.load_image(os.path.join(BASEDIR, PHASE))
tirf_mip = Util.maximum_intensity_projection(os.path.join(BASEDIR, TIRF_REG))

# Segment cells.
object_labels = Segmentation.segment_phase_image(phase)
bg_id = Segmentation.identify_background(phase, object_labels, im_type='phase')

# Prepare contour generator.
Contours = Contour.ContourGenerator(phase, object_labels)

for cell_id in np.unique(object_labels):

    print(f'Processing cell {cell_id}.')

    if cell_id == bg_id:
        print(' - Background, skipping.')
        continue

    # Generate mesh.
    contour = Contours.generate(cell_id)
    skeleton = Skeleton.generate(contour)
    mesh = Mesh.make_ribs(contour, skeleton)
    rib_starts, top_intersections, bot_intersections = mesh

    # Find brighest ribs.
    peaks = Kymograph.find_intensity_peaks(tirf_mip, mesh, kymo_width)

    # Generate kymographs at these ribs.
    for i in peaks:

        print(f'Processing cell {cell_id} rib {i}.')

        # Generate kymograph.
        P1 = top_intersections[i]
        P2 = bot_intersections[i]

        kymograph = Kymograph.make_kymograph(stack, P1, P2, kymo_width)

        # Save results.
        title = f'Cell {cell_id}, Rib {i}'
        filename = f'Cell_{cell_id:04}_rib_{i:04}.png'

        Diagnostics.debug_kymograph(tirf_mip,
                                    top_intersections[i],
                                    bot_intersections[i],
                                    kymograph,
                                    contour=contour,
                                    skeleton=skeleton,
                                    title=title,
                                    filename=filename)
