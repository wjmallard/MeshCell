#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

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
                    rib,
                    i,
                    rib_sums,
                    kymograph,
                    contour=None,
                    skeleton=None,
                    title='',
                    filename=None):

    window = 100

    C, P1, P2 = rib

    x1, y1 = P1
    x2, y2 = P2
    Sy, Sx = image.shape

    cx, cy = C
    llx = cx - window // 2
    lly = cy - window // 2
    urx = cx + window // 2
    ury = cy + window // 2

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

    xy = llx, lly
    width = window
    height = window

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

    ax.set_xlim([llx, urx])
    ax.set_ylim([lly, ury])
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
