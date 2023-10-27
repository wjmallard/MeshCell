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

def get_cell_boundary_coords(masks, cell_id):
    '''
    Find the boundary coordinates of a particular cell.

    Arguments:
        masks - int matrix; labels which object each pixel belongs to
        cell_id - specifies which object to find the boundary coords of

    Returns:
        X, Y - coordinates of the cell's boundary,
               ordered counter clockwise around the contour
    '''
    mask = masks == cell_id

    contours = find_contours(mask, level=.9)

    if len(contours) == 0:
        return None, None

    C = select_longest_contour(contours)

    # If an object intersects the edge of an image,
    # find_contours() will leave the contour open.
    # Close it, so skeletonization does not fail.
    if (C[0] != C[-1]).any():
        C = np.vstack((C, C[0]))

    # Flip coordinates from (y,x) to (x,y).
    C = np.flip(C, 1)

    return C

def select_longest_contour(contours):

    return sorted(contours, key=len)[-1]

def evenly_distribute_contour_points(XY_old):

    assert XY_old.shape[1] == 2
    assert XY_old.shape[0] > 1

    # Compute old coordinates along the contour with uneven spacing.
    dxy = np.diff(XY_old, axis=0)
    dC = np.linalg.norm(dxy, axis=1)
    C_old = np.cumsum(dC)
    C_old = np.insert(C_old, 0, 0)

    # Create new coordinates along the contour with even 1-pixel spacing.
    C_length = C_old[-1]
    C_points = int(np.round(C_length))
    C_new = np.linspace(0, C_length, C_points)

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
    a = C_new - C_old[J]
    b = C_old[J+1] - C_old[J]
    alpha = np.divide(a, b, out=np.zeros_like(a), where=(b!=0))

    XY_new = XY_old[J] + (XY_old[J+1] - XY_old[J]) * alpha[:,None]

    return XY_new

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
            # TODO: Assumes there is only a single intersection per segment.
            # Should handle the case where multiple intersections are found.
            intersection_index = np.where(intersection_criterion)[0][0]

            # Calculate intersection point.
            t = T[intersection_index]

            Px = x1 + t * (x2 - x1)
            Py = y1 + t * (y2 - y1)

            intersections[i] = (Px, Py)

        else:
            intersections[i] = (np.nan, np.nan)

    return intersections

def find_contour_intersections(L, C):
    '''
    Find intersections between a line and a closed contour.

    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    Retrieved: April 30, 2020.
    '''
    # Load contour #1 points.
    X1, Y1 = L[:-1].T
    X2, Y2 = L[1:].T

    # Load contour #2 points.
    X3, Y3 = C[:-1].T
    X4, Y4 = C[1:].T

    # Add a dimension to contour #1 for broadcasting.
    X1 = X1[:,None]
    X2 = X2[:,None]
    Y1 = Y1[:,None]
    Y2 = Y2[:,None]

    # Test every pair of contour edges for intersection.
    D = (X1 - X2) * (Y3 - Y4) - (Y1 - Y2) * (X3 - X4)
    T = (X1 - X3) * (Y3 - Y4) - (Y1 - Y3) * (X3 - X4)
    U = (X1 - X2) * (Y1 - Y3) - (Y1 - Y2) * (X1 - X3)

    T /= D
    U /= -D

    intersection_criterion = (0. <= T) & (T <= 1.) & (0. <= U) & (U <= 1.)

    # Result: A boolean matrix with shape (|L| - 1) x (|C| - 1).
    # Each True element is an intersection between L and C.
    # If matrix position (r, c) is True, then L[r] and C[c] are
    # the (x,y) coordinates of points nearest that intersection.

    intersections = []

    if intersection_criterion.any():

        L_idx, C_idx = np.where(intersection_criterion)

        for l, c in zip(L_idx, C_idx):

            Px = X1[l,0] + T[l,c] * (X2[l,0] - X1[l,0])
            Py = Y1[l,0] + T[l,c] * (Y2[l,0] - Y1[l,0])

            intersections.append((Px, Py))

    else:
        intersections.append((np.nan, np.nan))

    return np.array(intersections)

def is_point_in_polygon(Point, Polygon):
    '''
    Test if a point lies inside a polygon.
    '''
    # Validate inputs.
    Point = np.array(Point)
    msg = f'Point must have shape (2,). Got: {Point.shape}'
    assert Point.shape == (2,), msg

    Polygon = np.array(Polygon)
    msg = f'Polygon must have shape (N,2) where N >= 3. Got: {Polygon.shape}'
    assert Polygon.shape[0] >= 3, msg
    assert Polygon.shape[1] == 2, msg

    # Select an exterior point.
    X_max, Y_max = Polygon.T.max(axis=1)
    x1, y1 = np.array([2 * X_max, 2 * Y_max])

    # Load the point.
    x2, y2 = Point

    # Load the polygon.
    X3, Y3 = Polygon[:-1].T
    X4, Y4 = Polygon[1:].T

    # Test for intersection with every polygon edge.
    #
    # Find intersection points between a list of line segments and a contour.
    #
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    # Retrieved: April 30, 2020.
    D = (x1 - x2) * (Y3 - Y4) - (y1 - y2) * (X3 - X4)
    T = (x1 - X3) * (Y3 - Y4) - (y1 - Y3) * (X3 - X4)
    U = (x1 - x2) * (y1 - Y3) - (y1 - y2) * (x1 - X3)

    T /= D
    U /= -D

    intersections = (0. <= T) & (T <= 1.) & (0. <= U) & (U <= 1.)

    # Test for interiority.
    #
    # Test for interiority via a ray casting algorithm.
    #
    # https://en.wikipedia.org/wiki/Point_in_polygon
    point_is_inside = intersections.sum() % 2 == 1

    return point_is_inside

class ContourGenerator:
    '''
    Generates a smooth contour for any given cell_id.
    '''
    def __init__(self, image, masks):

        self.masks = masks

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
        XY_old = get_cell_boundary_coords(self.masks, cell_id)

        for i in range(max_iter):
            # Interpolate along curve to obtain evenly spaced spline anchors.
            #
            # X -- cartesian x-coordinates
            # Y -- cartesian y-coordinates
            # C -- 1D path coordinate along the contour
            #
            # NOTE: Assumes X-Y coordinates are sorted along the contour.

            XY_new = evenly_distribute_contour_points(XY_old)

            # Update the anchor positions using the image gradient (to maximize the
            # alignment of the contour with the image maxima)
            #
            # Interpolate the smoothed image-forces at the current anchor points.
            X_new, Y_new = XY_new.T
            Fx_new = self.Fx_smoothed.ev(Y_new, X_new)
            Fy_new = self.Fy_smoothed.ev(Y_new, X_new)

            X_old = X_new + step_size * Fx_new
            Y_old = Y_new + step_size * Fy_new
            XY_old = np.vstack((X_old, Y_old)).T

        contour = np.vstack((X_old, Y_old)).T

        return contour
