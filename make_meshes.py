#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

import Contour
import Skeleton
import Mesh

IMAGE = '/Volumes/Delphium/Microscopy/Red_Segmentation/image001-red -- 2021.08.10 bWM98 CH No IPTG FM5-95__bWM98 CH No IPTG 002-001 FM5-95.tif'
EDGES = '/Volumes/Delphium/Microscopy/Red_Segmentation/image001-red -- 2021.08.10 bWM98 CH No IPTG FM5-95__bWM98 CH No IPTG 002-001 FM5-95-c0.tif'
MASKS = '/Volumes/Delphium/Microscopy/Red_Segmentation/image001-red -- 2021.08.10 bWM98 CH No IPTG FM5-95__bWM98 CH No IPTG 002-001 FM5-95-c1.cell_markers.tif'

image = io.imread(IMAGE, as_gray=True)
edges = io.imread(EDGES, as_gray=True)
masks = io.imread(MASKS, as_gray=True)

#
# Generate contours.
#
Contours = Contour.ContourGenerator(edges, masks)

cell_ids = np.unique(masks)
cell_ids = cell_ids[cell_ids > 0]

Results = {}

for cell_id in cell_ids:

    print(f'Processing cell #{cell_id}.')

    contour = Contours.generate(cell_id)
    skeleton = Skeleton.generate(contour)
    rib_starts, top_intersections, bot_intersections = Mesh.make_ribs(contour, skeleton)

    Results[cell_id] = (
        contour,
        skeleton,
        rib_starts,
        top_intersections,
        bot_intersections,
    )

# Save contours to a .npz file.
p = Path(IMAGE)
outfile = p.parent.joinpath(p.stem + '.contours')
np.savez(outfile, Results)

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


for cell_id, result in Results.items():

    (contour,
     skeleton,
     rib_starts,
     top_intersections,
     bot_intersections) = result

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

outfile = IMAGE + '.png'
plt.savefig(outfile)
