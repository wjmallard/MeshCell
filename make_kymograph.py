#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date May 2020
"""
import os
import sys

try:
    BASEDIR = sys.argv[1]
    SAMPLE = sys.argv[2]
    REPLICATE = sys.argv[3]
except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} BASEDIR SAMPLE REPLICATE', file=sys.stderr)
    sys.exit(1)

import numpy as np

import Util
import Segmentation
import Contour
import Skeleton
import Mesh
import Kymograph
import Diagnostics

SAMP_STD = f'{SAMPLE} TIRF {REPLICATE} registered STD.tif'
SAMP_TIRF = f'{SAMPLE} TIRF {REPLICATE} registered.tif'
SAMP_CELL = f'{SAMPLE} feature_0 {REPLICATE}.tif'
SAMP_EDGE = f'{SAMPLE} feature_1 {REPLICATE}.tif'

BASEDIR = os.path.abspath(BASEDIR)

STD_IMG = os.path.join(BASEDIR, 'stdev', SAMP_STD)
TIRF_IMG = os.path.join(BASEDIR, 'registered_tirf', SAMP_TIRF)
CELL_IMG = os.path.join(BASEDIR, 'segmented_phase', SAMP_CELL)
EDGE_IMG = os.path.join(BASEDIR, 'segmented_phase', SAMP_EDGE)

assert os.path.exists(STD_IMG), f'Cannot open: {STD_IMG}'
assert os.path.exists(TIRF_IMG), f'Cannot open: {TIRF_IMG}'
assert os.path.exists(CELL_IMG), f'Cannot open: {CELL_IMG}'
assert os.path.exists(EDGE_IMG), f'Cannot open: {EDGE_IMG}'

kymo_width = 20

# Load images.
print(f'Loading images.')
tirf_std = Util.load_image(STD_IMG)
cells = Util.load_image(CELL_IMG)
edges = Util.load_image(EDGE_IMG)

# Align cell boundaries with TIRF movie.
print(f'Aligning phase and TIRF images.')
dy, dx = Util.align_images(tirf_std, cells)
cells = Util.shift_image(cells, dx, dy)
edges = Util.shift_image(edges, dx, dy)

# Segment cells.
print(f'Extracting segmented regions.')
object_labels = Segmentation.segment_deepcell_masks(cells)
bg_id = Segmentation.identify_background(cells, object_labels)

# Prepare contour generator.
Contours = Contour.ContourGenerator(edges, object_labels)

print(f'Processing cells.')
cell_ids = np.unique(object_labels)
for n, cell_id in enumerate(cell_ids):

    print(f'Cell {cell_id}. [{cell_id + 1}/{len(cell_ids)}]')

    if cell_id == bg_id:
        print(' - Background, skipping.')
        continue

    # Generate mesh.
    try:
        contour = Contours.generate(cell_id)
        skeleton = Skeleton.generate(contour)
        mesh = Mesh.make_ribs(contour, skeleton)
        ribs = np.array(mesh).transpose(1, 0, 2)
        assert not np.isnan(ribs).any()
    except:
        print(' - Failed, skipping.')
        continue

    # Find brighest ribs.
    rib_sums, peaks = Kymograph.find_intensity_peaks(tirf_std, mesh, kymo_width)

    # Generate kymographs at these ribs.
    for m, i in enumerate(peaks):

        print(f' - Rib {i}. [{m + 1}/{len(peaks)}]')

        # Generate kymograph.
        kymograph = Kymograph.make_kymograph(TIRF_IMG, ribs[i], kymo_width)

        # Save results.
        cx, cy = ribs[i][0].round().astype(np.int)
        title = f'{SAMPLE} {REPLICATE} - Cell {cell_id}, Rib {i} - ({cx}, {cy})'
        filename = f'{SAMPLE}_{REPLICATE}__Cell_{cell_id:04}_rib_{i:04}.png'

        Diagnostics.debug_kymograph(tirf_std,
                                    ribs[i],
                                    i,
                                    rib_sums,
                                    kymograph,
                                    contour=contour,
                                    skeleton=skeleton,
                                    title=title,
                                    filename=filename)

print(f'Done.')
