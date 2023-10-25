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
    MASKS = sys.argv[1:]
except:
    script = sys.argv[0].split('/')[-1]
    print(f'Usage: {script} MASKS{MASK_FILE_SUFFIX}', file=sys.stderr)
    print()
    print('where MASKS is a list of one or more images.')
    print('Each masks file must also be accompanied by the')
    print('original .tif file those masks were derived from.')
    sys.exit(1)

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import Mask
import Contour
import Skeleton
import Mesh
import Util

def process(mask_file):

    #
    # Validate input.
    #
    if not mask_file.endswith(MASK_FILE_SUFFIX):
        print(' - Invalid mask file name. Skipping.')
        return

    base_file = mask_file[:-len(MASK_FILE_SUFFIX)]
    image_file = f'{base_file}.tif'

    npz_file = f'{base_file}.contours.npz'
    png_file = f'{base_file}.png'

    if Path(npz_file).exists():
        print(f' - Already processed. Skipping.')
        return

    if not Path(mask_file).exists():
        print(' - Mask file does not exist. Skipping.')
        return

    if not Path(image_file).exists():
        print(' - Original image file does not exist. Skipping.')
        return

    #
    # Load images.
    #
    try:
        print('Loading images.')

        print(f' - {image_file}')
        image = io.imread(image_file, as_gray=True)

        print(f' - {mask_file}')
        cell_masks = io.imread(mask_file, as_gray=True)

    except:
        print('Cannot load image. Skipping.')
        return

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
    Failed_Chains = {}

    chain_ids = np.unique(chain_masks)
    chain_ids = chain_ids[chain_ids > 0]

    for chain_id in chain_ids:

        print(f' - Chain {chain_id}')

        cell_ids = chains_to_cells[chain_id]

        try:

            chain_contour = Chain_Contours.generate(chain_id)
            chain_skeleton = Skeleton.generate(chain_contour)

            for cell_id in cell_ids:

                print(f'   - Cell {cell_id}')

                cell_contour = Cell_Contours.generate(cell_id)
                cell_skeleton = Skeleton.extend_skeleton(chain_skeleton, cell_contour)
                cell_skeleton = Skeleton.crop_skeleton(cell_skeleton, cell_contour)
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

        except:
            print('   - Mesh generation failed. Skipping this chain.')
            Failed_Chains[chain_id] = cell_ids

    # Save contours to a .npz file.
    print('Saving contours to disk.')
    print(f' - {npz_file}')
    np.savez(npz_file, Cells=Cells, Failed_Chains=Failed_Chains)

    # Save diagnostic plot to a .png file.
    print('Saving plot to disk.')
    print(f' - {png_file}')
    save_diagnostic_plot(png_file, image, Cells)

def save_diagnostic_plot(outfile, image, Cells):

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
    plt.savefig(outfile)


filenames = Util.natural_sort(MASKS)

for n, filename in enumerate(filenames):

    print(f'[{n+1}/{len(MASKS)}] {filename}')
    process(filename)
    print()

print('Done.')
