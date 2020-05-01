#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:42:35 2020

@author: wmallard xies
"""

#%% Step 0: Load libraries and image.

import numpy as np
from skimage import io
from skimage import util
from skimage.morphology import reconstruction
from skimage.morphology import watershed
from skimage.measure import find_contours
from skimage.filters import sobel_h, sobel_v
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Voronoi
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import path

FILENAME = '/Users/wmallard/Desktop/Microscopy/test.tif'

h_min_to_fill = 0.1 # relative to 1.0 maximum
max_iter = 300
step_size = 0.1

im = io.imread(FILENAME, as_gray=True)

#%% Step 1: Segment Cells

def fill_image_to_min_height(image, h_min):
    '''
    Reproduces Matlab's imhmin() function.
    
    Fill local minima in image up to depth h_min.
    
    Arguments:
        image - 2D image
        h_min - scalar
    '''
    im1 = util.invert(image)
    im2 = reconstruction((im1 - h_min), im1)
    im3 = util.invert(im2)
    return im3

im_filled = fill_image_to_min_height(im, h_min_to_fill)
object_labels = watershed(im_filled, connectivity=8)

#%% Step 2: Fit Contours

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
    if len(c) != 1: return None, None

    Y, X = c[0].T

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

# Pre-compute smoothed image force components.
Fx = sobel_v(im)
Fy = sobel_h(im)

Sy, Sx = im.shape
x_mesh = np.arange(Sx)
y_mesh = np.arange(Sy)

Fx_smoothed = RectBivariateSpline(y_mesh, x_mesh, Fx)
Fy_smoothed = RectBivariateSpline(y_mesh, x_mesh, Fy)

# TODO: Loop over all cells.
# num_cells = object_labels.max() + 1
# for cell_id in range(num_cells):
#     X, Y = get_cell_boundary_coords(object_labels, cell_id)
#     if X is None: continue
cell_id = 6

# Get the pixel coordinates of current cell's boundary,
# returns X and Y that wraps around contour (no break b/w end and beginning)
X, Y = get_cell_boundary_coords(object_labels, cell_id)
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
    Fx_new = Fx_smoothed.ev(Y_new, X_new)
    Fy_new = Fy_smoothed.ev(Y_new, X_new)
    
    X_old = X_new + step_size * Fx_new
    Y_old = Y_new + step_size * Fy_new

contour = np.vstack((X_old, Y_old)).T

#%% Step 3: Construct Skeleton

# Construct a voronoi diagram.
# Extract its vertices and edges.
vor = Voronoi(contour)
V = vor.vertices
E = vor.ridge_vertices

# Identify which vertices are in the contour's interior.
p = path.Path(contour)
I = p.contains_points(V)

# Build a connectivity graph of all edges in the contour's interior.
# 
# Use a dictionary where:
#   - k: vertex index
#   - v: list of vertex indices
# These all index into V.
C = defaultdict(list)

for u, v in E:

    # Remove edges connecting to points at infinity.
    if u < 0: continue
    if v < 0: continue

    # Remove edges connecting to vertices outside the contour.
    if not I[u]: continue
    if not I[v]: continue

    # Connect vertex u to v and v to u.
    C[u].append(v)
    C[v].append(u)

# Find the vertex closest to the contour's centroid.
vertices = np.array(V)
centroid = contour.mean(axis=0)

distances_to_centroid = np.sum((vertices - centroid)**2, axis=1)
central_vertex_idx = np.argmin(distances_to_centroid)
central_vertex = V[central_vertex_idx]

# Build a list of all non-branching edges.
#
# Start from the vertex closest to the centroid and work outward.
# No need to check for:
#   - cycles because Voronoi diagrams are acyclic.
#   - disjoint subgraphs because Voronoi diagrams are fully connected.
Branches = []
vertices_to_walk = []

root = central_vertex_idx

next_v = root
next_v_neighbors = C[next_v]

for n in next_v_neighbors:
    C[n].remove(next_v)
    vertices_to_walk.append(n)

while vertices_to_walk:

    # Start a new branch.
    branch = []

    # Get the next vertex.
    next_v = vertices_to_walk.pop()

    # Find the next vertex.
    next_v_neighbors = C[next_v]

    while len(next_v_neighbors) == 1:

        # Advance to next vertex.
        this_v = next_v
        next_v = next_v_neighbors[0]

        # Add this vertex to the branch.
        branch.append(this_v)

        # Find the next vertex.
        next_v_neighbors = C[next_v]
        next_v_neighbors.remove(this_v)

    branch.append(next_v)

    # Add the forking neighbors to the list of vertices to walk.
    # Each is the start of a new branch.
    for n in next_v_neighbors:
        C[n].remove(next_v)
        vertices_to_walk.append(n)

    Branches.append(branch)

# Stitch together the two branches emenating from the root node.
n1, n2 = C[root]

b1 = [b for b in Branches if b[0] == n1][0]
b2 = [b for b in Branches if b[0] == n2][0]
main_branch = b1[::-1] + [root] + b2

Branches.remove(b1)
Branches.remove(b2)
Branches.append(main_branch)

# Find the longest branch. Make that the skeleton.
skel_length, skel_indices = sorted((len(b), b) for b in Branches)[-1]

skel = V[skel_indices]

#%% Step 4: Refine + Extend Skeleton

skeleton = evenly_distribute_contour_points(*skel.T)
skeleton = np.array(skeleton).T

# TODO: Figure out how to set the extension factor in a more principled way.
extension_factor = 20

left_end = skeleton[:3]
right_end = skeleton[-3:]

left_vector = left_end[1:] - left_end[:-1]
left_point = skeleton[1,:] - left_vector.mean(axis=0) * extension_factor

right_vector = right_end[1:] - right_end[:-1]
right_point = skeleton[-1,:] + right_vector.mean(axis=0) * extension_factor

skeleton_ext = np.concatenate((left_point[None,:], skeleton, right_point[None,:]))

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

# TODO: Only run intersection finder on the extended segments, not whole skeleton.
intersection = find_segment_intersection(skeleton_ext[:-1], skeleton_ext[1:], contour)

# Replace skeleton extension points with contour intersection points.
skeleton_trim = skeleton_ext
skeleton_trim[0] = intersection[0]
skeleton_trim[-1] = intersection[-1]
