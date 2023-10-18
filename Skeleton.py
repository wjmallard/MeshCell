#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
from scipy.spatial import Voronoi
from collections import defaultdict

import Contour
import Util

# TODO: Figure out how to set the extension factor in a more principled way.
extension_factor = 20

def generate(contour):
    try:
        skel = build_skeleton(contour)
        skel = extend_skeleton(skel, contour)
    except IndexError:
        return None
    return skel

def build_skeleton(contour):

    # Build a Voronoi diagram.
    # Extract the tree from its interior.
    V, E = get_voronoi_interior(contour)

    # Extract all branches.
    Branches = extract_branches(E)

    # Find the longest branch. Make that the skeleton.
    skel_indices = sorted(Branches, key=len)[-1]

    # Convert to skeleton.
    skeleton = V[skel_indices]

    return skeleton

def get_voronoi_interior(contour):

    # Construct a voronoi diagram.
    # Extract its vertices and edges.
    vor = Voronoi(contour)
    V = vor.vertices
    E = vor.ridge_vertices

    # Remove edges connecting to points at infinity.
    E = [e for e in E if min(e) >= 0]

    # Remove edges connecting to vertices outside the contour.
    E = [e for e in E
         if Contour.is_point_in_polygon(V[e[0]], contour) and
            Contour.is_point_in_polygon(V[e[1]], contour)]

    return V, E

def extract_branches(E):

    # Build a connectivity graph of all edges in the contour's interior.
    #
    # Use a dictionary where:
    #   - key: vertex index
    #   - val: list of vertex indices
    # These all index into V.
    Neighbors = defaultdict(list)

    for u, v in E:

        # Connect vertex u to v and v to u.
        Neighbors[u].append(v)
        Neighbors[v].append(u)

    # Build a list of all non-branching edges.
    #
    # Find all leaf nodes. Pick any one of them and start walking.
    #
    # No need to check for:
    #   - cycles because Voronoi diagrams are acyclic.
    #   - disjoint subgraphs because Voronoi diagrams are fully connected.
    Leaves = [k for k, v in Neighbors.items() if len(v) == 1]
    root = Leaves[0]

    Branches = []
    vertices_to_walk = []

    # Start at a leaf node.
    # Push the first two nodes onto the stack.
    this_v = root
    next_v = Neighbors[this_v].pop()
    Neighbors[next_v].remove(this_v)
    vertices_to_walk.append([this_v, next_v])

    # Walk the tree.
    while vertices_to_walk:

        # Start a new branch.
        branch = vertices_to_walk.pop()
        next_v = branch.pop()

        # Add vertices until we hit a fork.
        while len(Neighbors[next_v]) == 1:

            # Advance to next vertex.
            this_v = next_v
            next_v = Neighbors[this_v].pop()

            # Add this vertex to the branch.
            branch.append(this_v)

            # Remove this vertex from the
            # next vertex's neighbor list.
            Neighbors[next_v].remove(this_v)

        # Finish the branch.
        branch.append(next_v)
        Branches.append(branch)

        # Add the final vertex's neighbors
        # to the list of vertices to walk.
        while Neighbors[next_v]:
            n = Neighbors[next_v].pop()
            Neighbors[n].remove(next_v)
            vertices_to_walk.append([next_v, n])

    return Branches

def extend_skeleton(skeleton, contour):

    NUM_INTERP_POINTS = 10

    # Uniformly distribute points along the skeleton.
    skeleton = Contour.evenly_distribute_contour_points(*skeleton.T)
    skeleton = np.array(skeleton).T

    # Ensure the skeleton is ordered from left to right.
    if skeleton[0,0] > skeleton[-1,0]:
        skeleton = skeleton[::-1]

    # Make room for a new exterior point at both ends.
    skeleton_extended = np.concatenate((
        [[np.nan, np.nan]],
        skeleton,
        [[np.nan, np.nan]],
    ))

    # Calculate an extension length that guarantees a projected
    # point will definitely lie outside of the contour.
    extension_length = (contour.max(axis=0) - contour.min(axis=0)).max().round().astype(int)

    # Extrapolate a point guaranteed to be outside the left end.
    left_end = skeleton[:NUM_INTERP_POINTS]
    f = Util.fit_line(left_end)
    x = left_end[0][0] - extension_length
    y = f(x)
    skeleton_extended[0] = (x, y)

    # Extrapolate a point guaranteed to be outside the right end.
    right_end = skeleton[-NUM_INTERP_POINTS:]
    f = Util.fit_line(right_end)
    x = right_end[-1][0] + extension_length
    y = f(x)
    skeleton_extended[-1] = (x, y)

    # Identify all points on the skeleton inside the contour.
    is_inside = lambda p: Contour.is_point_in_polygon(p, contour)
    interior_points = list(map(is_inside, skeleton_extended))

    # Find the first and last skeleton indices inside the contour.
    a, b = 1 + np.where(np.diff(interior_points))[0]

    # Trim the skeleton to include points just outside the contour.
    skeleton_trimmed = skeleton_extended[a-1:b+1]

    # Trim the skeleton to precisely the edge of the contour.
    intersections = Contour.find_contour_intersections(skeleton_trimmed, contour)
    assert len(intersections) == 2
    skeleton_trimmed[0] = intersections[0]
    skeleton_trimmed[-1] = intersections[1]

    # Interpolate again on the whole extended skeleton.
    skeleton_final = Contour.evenly_distribute_contour_points(*skeleton_trimmed.T)
    skeleton_final = np.array(skeleton_final).T

    return skeleton_final
