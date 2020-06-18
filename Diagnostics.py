#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.signal import windows
from matplotlib.patches import Rectangle

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

def debug_contour(image, contour=None, skeleton=None, mesh=None, title=None, filename=None):

    plt.close('all')

    plt.imshow(image, cmap='gray')

    if contour is not None:
        plt.plot(*contour.T, 'ko-')

    if skeleton is not None:
        plt.plot(*skeleton.T, 'rx-')

    if mesh is not None:
        rib_starts, top_intersections, bot_intersections = mesh

        for i in range(len(rib_starts)):

            xs, ys = rib_starts[i]
            xt, yt = top_intersections[i]
            xb, yb = bot_intersections[i]

            plt.plot([xs, xt], [ys, yt], 'b')
            plt.plot([xs, xb], [ys, yb], 'b')

    if title:
        plt.title(title)

    if filename:
        plt.savefig(filename)

def debug_kymograph(image,
                    bbox,
                    P1,
                    P2,
                    i,
                    rib_sums,
                    kymograph,
                    contour=None,
                    skeleton=None,
                    title='',
                    filename=None):

    padding = 10

    x1, y1 = P1
    x2, y2 = P2
    Sy, Sx = image.shape

    llx, lly, urx, ury = bbox

    plt.close('all')

    fig = plt.figure(figsize=(12, 6), dpi=100)

    axis_args = {
        'xticks': [],
        'yticks': [],
        'xticklabels': [],
        'yticklabels': [],
    }

    # Rectangle coordinates: [left, bottom, width, height]
    ax1 = fig.add_axes([.00, .95, 1.0, .05], **axis_args)
    ax2 = fig.add_axes([.00, .00, .44, .95], **axis_args)
    ax3 = fig.add_axes([.45, .30, .44, .65], **axis_args)
    ax4 = fig.add_axes([.45, .00, .44, .30], **axis_args)
    ax5 = fig.add_axes([.90, .00, .10, .95], **axis_args)

    # Header
    ax = ax1
    ax.text(0.5, 0.5,
            title,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='center',
            horizontalalignment='center')

    # Entire image
    ax = ax2
    ax.imshow(image, cmap='gray')

    xy = llx - padding, lly - padding
    width = urx - llx + 2 * padding
    height = ury - lly + 2 * padding

    rect = Rectangle(xy,
                     width=width,
                     height=height,
                     fill=False,
                     linestyle='--',
                     linewidth=1,
                     edgecolor='red',
                     facecolor=None)
    ax.add_patch(rect)

    ax.set_xlim([0, Sx])
    ax.set_ylim([Sy, 0])

    # Cell close-up
    ax = ax3
    ax.imshow(image, cmap='gray')
    ax.plot((x1, x2), (y1, y2), 'r-', linewidth=2)

    if contour is not None:
        ax.plot(*contour.T, 'g:')

    if skeleton is not None:
        ax.plot(*skeleton.T, 'y:')

    ax.set_xlim([llx - padding, urx + padding])
    ax.set_ylim([lly - padding, ury + padding])
    ax.invert_yaxis()

    # Rib intensities
    ax = ax4
    ax.plot(rib_sums)
    ymin, ymax = ax.get_ylim()
    ax.vlines(i, ymin, ymax, colors='red')
    ax.set_xlim(0, len(rib_sums))
    ax.set_ylim(ymin, ymax)

    # Kymograph
    ax = ax5
    ax.imshow(kymograph, cmap='gray')

    if filename is not None:
        plt.savefig(filename)

def debug_fft(kymograph, kymo_trace, sampling_period):

    n_timepoints = len(kymo_trace)
    ff = rfftfreq(n_timepoints, sampling_period)

    F = abs(rfft(kymo_trace))
    F[0] = 0

    freq = abs(ff[np.argmax(F)])
    amp = kymo_trace.std() * np.sqrt(2.)
    offset = kymo_trace.mean()

    plt.close('all')
    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    grid = fig.add_gridspec(1, 3,
                            wspace=0.0,
                            hspace=0.0,
                            width_ratios=[1, 5, 5])

    ax = fig.add_subplot(grid[0,0])
    ax.imshow(kymograph, cmap='gray')

    ax = fig.add_subplot(grid[0,1])
    ax.plot(kymo_trace)

    ax = fig.add_subplot(grid[0,2])
    ax.plot(F)

def debug_detrend_window(k):

    plt.close('all')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    X = k
    axes[0][0].plot(X)
    axes[0][0].set_xlim((0, 240))
    axes[1][0].semilogy(abs(rfft(X))[:])
    axes[1][0].set_xlim((0, 120))
    axes[1][0].set_ylim((1e1, 1e5))

    X = detrend(k)
    axes[0][1].plot(X)
    axes[0][1].set_xlim((0, 240))
    axes[1][1].semilogy(abs(rfft(X))[:])
    axes[1][1].set_xlim((0, 120))
    axes[1][1].set_ylim((1e1, 1e5))

    X = windows.hann(k.shape[0]) * detrend(k)
    axes[0][2].plot(X)
    axes[0][2].set_xlim((0, 240))
    axes[1][2].semilogy(abs(rfft(X))[:])
    axes[1][2].set_xlim((0, 120))
    axes[1][2].set_ylim((1e1, 1e5))

    axes[0][0].set_ylabel('Kymograph dominant bin')
    axes[0][1].set_ylabel('Spectrum')

    axes[1][0].set_xlabel('Raw')
    axes[1][1].set_xlabel('Detrend')
    axes[1][2].set_xlabel('Detrend + Hann window')

    fig.tight_layout()
