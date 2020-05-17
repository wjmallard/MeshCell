#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
from skimage.measure import find_contours
from skimage.filters import sobel_h, sobel_v
from scipy.interpolate import RectBivariateSpline

max_iter = 300
step_size = 0.1

def get_cell_boundary_coords(object_labels, cell_id):
    '''
    Find the boundary coordinates of a particular cell.
    
    Arguments:
        object_labels - int matrix; labels which object each pixel belongs to
        cell_id - specifies which object to find the boundary coords of
    
    Returns:
        X, Y - coordinates of the cell's boundary,
               ordered counter clockwise around the contour
    '''
    cell_mask = object_labels == cell_id

    c = find_contours(cell_mask, level=.9)

    # If no contours (or multiple contours) are found:
    if len(c) != 1: return None, None

    C = c[0]

    # If an object intersects the edge of an image,
    # find_contours() will leave the contour open.
    # Close it, so skeletonization does not fail.
    if (C[0] != C[-1]).any():
        C = np.vstack((C, C[0]))

    Y, X = C.T

    return X, Y

def evenly_distribute_contour_points(X_old, Y_old):

    # Compute old coordinates along the contour with uneven spacing.
    dx = X_old[1:] - X_old[:-1]
    dy = Y_old[1:] - Y_old[:-1]
    C_old = np.cumsum(np.sqrt(dx**2 + dy**2))
    C_old = np.concatenate(([0], C_old))

    # Create new coordinates along the contour with even 1-pixel spacing.
    C_length = C_old[-1]
    C_new = np.linspace(0, C_length, np.int(np.round(C_length)))
    
    # Find the distance between each new (evenly spaced) anchor point
    # and the next-smallest old (unevenly spaced) anchor point.
    anchor_distance = C_new[None,:] - C_old[:,None]
    anchor_distance[anchor_distance < 0] = np.inf
    
    # For each new anchor point,
    # find the index of the old anchor point immediately to its left.
    J = np.argmin(anchor_distance, axis=0)

    # Ensure the index falls between 0 and the second-to-last
    # element of the old contour.
    # (The final old anchor point cannot be left of any new anchor point.)
    J = np.clip(J, 0, len(C_old) - 2)
    
    # Interpolate X and Y coordinates along the contour.
    alpha = (C_new - C_old[J]) / (C_old[J+1] - C_old[J])
    X_new = X_old[J] + (X_old[J+1] - X_old[J]) * alpha
    Y_new = Y_old[J] + (Y_old[J+1] - Y_old[J]) * alpha

    return X_new, Y_new

def find_segment_intersection(L1, L2, C):
    '''
    Find intersection points between a list of line segments and a contour.

    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Retrieved: April 30, 2020.
    '''
    assert len(L1) == len(L2)

    X3, Y3 = C.T[:,:-1]
    X4, Y4 = C.T[:,1:]

    intersections = np.zeros((len(L1), 2))

    for i in range(len(L1)):

        x1, y1 = L1[i]
        x2, y2 = L2[i]

        D = (x1 - x2) * (Y3 - Y4) - (y1 - y2) * (X3 - X4)
        T = (x1 - X3) * (Y3 - Y4) - (y1 - Y3) * (X3 - X4)
        U = (x1 - x2) * (y1 - Y3) - (y1 - y2) * (x1 - X3)

        T /= D
        U /= -D

        intersection_criterion = (0. <= T) & (T <= 1.) & (0. <= U) & (U <= 1.)

        if intersection_criterion.any():
            # TODO: Assumes for now only a single intersection is found per
            # segment, but need to handle multiple intersection cases
            intersection_index = np.where(intersection_criterion)[0]

            # Calculate intersection point.
            t = T[intersection_index]

            Px = x1 + t * (x2 - x1)
            Py = y1 + t * (y2 - y1)

            intersections[i] = (Px, Py)

        else:
            intersections[i] = (np.nan, np.nan)

    return intersections

class ContourGenerator:
    '''
    Generates a smooth contour for any given cell_id.
    
    '''
    def __init__(self, image, object_labels):

        self.object_labels = object_labels

        # Compute gradient of smoothed images.
        Fx = sobel_v(image)
        Fy = sobel_h(image)
        
        # Compute smoothed image force components.
        Sy, Sx = image.shape
        x_mesh = np.arange(Sx)
        y_mesh = np.arange(Sy)
        
        self.Fx_smoothed = RectBivariateSpline(y_mesh, x_mesh, Fx)
        self.Fy_smoothed = RectBivariateSpline(y_mesh, x_mesh, Fy)

    def generate(self, cell_id):
        
        # Get the pixel coordinates of current cell's boundary,
        # returns X and Y that wraps around contour (no break b/w end and beginning)
        X, Y = get_cell_boundary_coords(self.object_labels, cell_id)
        X_old, Y_old = X, Y
        
        for i in range(max_iter):
            # Interpolate along curve to obtain evenly spaced spline anchors.
            #
            # X -- cartesian x-coordinates
            # Y -- cartesian y-coordinates
            # C -- 1D path coordinate along the contour
            #
            # NOTE: Assumes X-Y coordinates are sorted along the contour.
        
            X_new, Y_new = evenly_distribute_contour_points(X_old, Y_old)
            
            # Update the anchor positions using the image gradient (to maximize the
            # alignment of the contour with the image maxima)
            # 
            # Interpolate the smoothed image-forces at the current anchor points.
            Fx_new = self.Fx_smoothed.ev(Y_new, X_new)
            Fy_new = self.Fy_smoothed.ev(Y_new, X_new)
            
            X_old = X_new + step_size * Fx_new
            Y_old = Y_new + step_size * Fy_new
        
        contour = np.vstack((X_old, Y_old)).T

        return contour
