#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
import matplotlib.pyplot as plt

def save_image(filename, cell_id, im, contour, skeleton):
    
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 6), dpi=100)
    axes = fig.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    
    ax = axes[0][0]
    ax.imshow(im, cmap='Greys_r')
    
    ax = axes[0][1]
    #ax.imshow(im_filled, cmap='Greys_r')
    
    ax = axes[0][2]
    dc = .5 * np.ones((2,1))
    ax.imshow(im, cmap='Greys_r', alpha=.2)
    ax.plot(*contour.T + dc, 'r:')
    
    ax = axes[1][0]
    ax.plot(*contour.T, 'k--')
    for u, v in E:
        if u < 0: continue
        if v < 0: continue
        if not I[u]: continue
        if not I[v]: continue
        ax.plot(*list(zip(V[u], V[v])), linewidth=.5)
    
    ax = axes[1][1]
    ax.plot(*contour.T, 'k--', linewidth=1)
    ax.plot(*skeleton.T, 'r:', linewidth=1)
    
    ax = axes[1][2]
    ax.plot(*contour.T, 'k--', linewidth=1)
    ax.plot(*skeleton.T, 'b-', linewidth=1)
    for i in range(len(rib_starts)):
    
        xs, ys = rib_starts[i]
        xt, yt = top_intersections[i]
        xb, yb = bot_intersections[i]
    
        ax.plot([xs, xt], [ys, yt], 'r', linewidth=1)
        ax.plot([xs, xb], [ys, yb], 'r', linewidth=1)
    
    # Find width of longest dimension.
    wx = X.max() - X.min()
    wy = Y.max() - Y.min()
    dist_from_center_to_edge = max(wx, wy) / 2
    
    # Find center of square.
    cx = (X.min() + X.max()) / 2
    cy = (Y.min() + Y.max()) / 2
    
    # Find new xy limits.
    pad = 5  # pixels
    xmin = cx - dist_from_center_to_edge - pad
    xmax = cx + dist_from_center_to_edge + pad
    ymin = cy - dist_from_center_to_edge - pad
    ymax = cy + dist_from_center_to_edge + pad
    
    # Set xy limits.
    ax = axes[0][0]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    
    fig.tight_layout()
    
    tiff = filename.split('/')[-1]
    fig.savefig(f'{tiff}.cell_{cell_id}.png')

