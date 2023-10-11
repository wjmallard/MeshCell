#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
#
# Parse command line arguments.
#
import sys
from pathlib import Path
try:
    IMAGE = sys.argv[1]

    assert IMAGE.endswith('.tif')
    basefile = IMAGE[:-len('.tif')]

    MASKS = f'{basefile}.masks_edited.tif'

    assert Path(IMAGE).exists()
    assert Path(MASKS).exists()

except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} IMAGE.tif', file=sys.stderr)
    print()
    print('where IMAGE ends with .tif, and is accompanied by:')
    print(' - IMAGE.masks_edited.tif : curated cell masks')
    sys.exit(1)

#
# Load libraries.
#
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

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

import pandas as pd
from skimage.measure import label as label_connected

def assign_cells_to_chains(cell_masks):
    '''
    Merge cells into chains.
    '''
    cell_labels = np.unique(cell_masks)
    cell_labels = cell_labels[cell_labels > 0]

    # Create a unique label for each chain.
    chain_masks = label_connected(cell_masks > 0)

    # Assign each cell to a chain.
    chain_assignments = pd.Series(data=-1, index=cell_labels, dtype=int)

    for cell_label in cell_labels:
        chain_assignments[cell_label] = chain_masks[cell_masks == cell_label][0]

    return chain_masks, chain_assignments

chain_masks, chain_assignments = assign_cells_to_chains(cell_masks)

#
# Generate chain contours and skeletons.
#
print('Generating chain contours and skeletons.')

Contours = Contour.ContourGenerator(image, chain_masks)

chain_ids = np.unique(chain_masks)
chain_ids = chain_ids[chain_ids > 0]

Chains = {}

for chain_id in chain_ids:

    print(f' - Chain {chain_id}')

    contour = Contours.generate(chain_id)
    skeleton = Skeleton.generate(contour)
    if skeleton is None:
        print(' * Skeletonization failed. Skipping.')
        continue

    Chains[chain_id] = {
        'Contour': contour,
        'Skeleton': skeleton,
    }

#
# Generate cell contours.
#
print('Generating cell contours.')

Contours = Contour.ContourGenerator(image, cell_masks)

cell_ids = np.unique(cell_masks)
cell_ids = cell_ids[cell_ids > 0]

Cells = {}

for cell_id in cell_ids:

    print(f' - Cell {cell_id}')

    contour = Contours.generate(cell_id)
    skeleton = Skeleton.generate(contour)
    if skeleton is None:
        print(' * Skeletonization failed. Skipping.')
        continue
    rib_starts, top_intersections, bot_intersections = Mesh.make_ribs(contour, skeleton)

    Cells[cell_id] = {
        'Contour': contour,
        'Skeleton': skeleton,
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
