#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
MASK_FILE_SUFFIX = '.masks_edited.tif'
#
# Parse command line arguments.
#
import sys
from pathlib import Path
try:
    MASKS = sys.argv[1]

    assert MASKS.endswith(MASK_FILE_SUFFIX)
    basefile = MASKS[:-len(MASK_FILE_SUFFIX)]

    IMAGE = f'{basefile}.tif'

    assert Path(IMAGE).exists()
    assert Path(MASKS).exists()

except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} MASKS{MASK_FILE_SUFFIX}', file=sys.stderr)
    print()
    print('MASKS must also be accompanied by the original .tif file the masks were derived from.')
    sys.exit(1)

#
# Load libraries.
#
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import Mask
import Contour
import Skeleton
import Mesh

#
# Load images.
#
print('Loading images.')
print(f' - {IMAGE}')
print(f' - {MASKS}')

image = io.imread(IMAGE, as_gray=True)
cell_masks = io.imread(MASKS, as_gray=True)

#
# Generate chain masks.
#
print('Generating chain masks.')
chain_masks, chains_to_cells = Mask.assign_cells_to_chains(cell_masks)

#
# Generate contours, skeletons, and ribs.
#
print('Generating contours, skeletons, and ribs.')

Chain_Contours = Contour.ContourGenerator(image, chain_masks)
Cell_Contours = Contour.ContourGenerator(image, cell_masks)

Cells = {}

chain_ids = np.unique(chain_masks)
chain_ids = chain_ids[chain_ids > 0]

for chain_id in chain_ids:

    print(f' - Chain {chain_id}')

    chain_contour = Chain_Contours.generate(chain_id)
    chain_skeleton = Skeleton.generate(chain_contour)

    if chain_skeleton is None:
        print(' * Skeletonization failed. Skipping.')
        continue

    cell_ids = chains_to_cells[chain_id]

    for cell_id in cell_ids:

        print(f'   - Cell {cell_id}')

        cell_contour = Cell_Contours.generate(cell_id)
        cell_skeleton = Skeleton.extend_skeleton(chain_skeleton, cell_contour)
        rib_starts, top_intersections, bot_intersections = Mesh.make_ribs(cell_contour, cell_skeleton)

        Cells[cell_id] = {
            'Contour': cell_contour,
            'Skeleton': cell_skeleton,
            'Ribs': {
                'Start': rib_starts,
                'Top': top_intersections,
                'Bot': bot_intersections,
            }
        }

# Save contours to a .npz file.
outfile = basefile + '.contours'
print('Saving contours to disk.')
print(f' - {outfile}.npz')
np.savez(outfile, Cells=Cells)

#
# Display results.
#
plt.close('all')

dpi = 100
y, x = np.array(image.shape) / dpi

fig = plt.figure(figsize=(x, y), dpi=dpi)
axis_args = {
    'xticks': [],
    'yticks': [],
    'xticklabels': [],
    'yticklabels': [],
}

# Rectangle coordinates: [left, bottom, width, height]
ax = fig.add_axes([0, 0, 1, 1], **axis_args)
ax.imshow(image, cmap='Greys_r')

for cell_id, Cell in Cells.items():

    contour = Cell['Contour']
    skeleton = Cell['Skeleton']
    rib_starts = Cell['Ribs']['Start']
    top_intersections = Cell['Ribs']['Top']
    bot_intersections = Cell['Ribs']['Bot']

    for i in range(len(rib_starts)):

        xs, ys = rib_starts[i]
        xt, yt = top_intersections[i]
        xb, yb = bot_intersections[i]

        plt.plot([xs, xt], [ys, yt], 'green', alpha=.2)
        plt.plot([xs, xb], [ys, yb], 'green', alpha=.2)

    ax.plot(*contour.T, 'k-')
    ax.plot(*skeleton.T, 'r-')

    x, y = skeleton[len(skeleton)//2]
    s = str(cell_id)
    ax.text(x, y, s,
             size=12,
             ha='center',
             va='center')

# Save plot to disk.
outfile = basefile + '.png'
print('Saving plot to disk.')
print(f' - {outfile}')
plt.savefig(outfile)

print('Done.\n')
