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
from matplotlib import path

import Contour

# TODO: Figure out how to set the extension factor in a more principled way.
extension_factor = 20

def generate(contour):
    skel = build_skeleton(contour)
    skel = extend_skeleton(skel, contour)
    return skel

def build_skeleton(contour):

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
    Neighbors = defaultdict(list)
    
    for u, v in E:
    
        # Remove edges connecting to points at infinity.
        if u < 0: continue
        if v < 0: continue
    
        # Remove edges connecting to vertices outside the contour.
        if not I[u]: continue
        if not I[v]: continue
    
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
    #
    # TODO: Fork nodes do not appear on all branches emanating from them,
    #       so the skeleton is missing one vertex. Fix this!
    #       Rather than pushing all the fork node's neighbors,
    #       maybe push the fork node itself?
    Leaves = [k for k, v in Neighbors.items() if len(v) == 1]
    root = Leaves[0]
    
    Branches = []
    vertices_to_walk = [root]
    next_v = root
    
    while vertices_to_walk:
    
        # Start a new branch.
        branch = []
        next_v = vertices_to_walk.pop()
    
        while len(Neighbors[next_v]) == 1:
    
            # Advance to next vertex.
            this_v = next_v
            next_v = Neighbors[next_v].pop()
    
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
            vertices_to_walk.append(n)
    
    # Find the longest branch. Make that the skeleton.
    skel_length, skel_indices = sorted((len(b), b) for b in Branches)[-1]
    
    skeleton = V[skel_indices]

    return skeleton

def extend_skeleton(skeleton, contour):

    skeleton = Contour.evenly_distribute_contour_points(*skeleton.T)
    skeleton = np.array(skeleton).T
    
    left_end = skeleton[:3]
    right_end = skeleton[-3:]
    
    left_vector = left_end[1:] - left_end[:-1]
    left_point = skeleton[1,:] - left_vector.mean(axis=0) * extension_factor
    
    right_vector = right_end[1:] - right_end[:-1]
    right_point = skeleton[-1,:] + right_vector.mean(axis=0) * extension_factor
    
    skeleton_ext = np.concatenate((left_point[None,:], skeleton, right_point[None,:]))
    
    # TODO: Only run intersection finder on the extended segments, not whole skeleton.
    intersection = Contour.find_segment_intersection(skeleton_ext[:-1], skeleton_ext[1:], contour)
    
    # Replace skeleton extension points with contour intersection points.
    skeleton_trim = skeleton_ext
    skeleton_trim[0] = intersection[0]
    skeleton_trim[-1] = intersection[-1]
    # TODO: Handle the "edge" case:
    #       When a cell is flagged as hitting an edge,
    #       there is no contour to intersect, so one or
    #       both of the intersections comes back as NaN.
    #       If the "keep edge cells" flag is on,
    #       then truncate the cell perpendicularly.
    #       If it is off, then discard the cell.
    #       The flag should be off by default.
    
    # Interpolate again on the whole extended skeleton.
    skeleton_final = Contour.evenly_distribute_contour_points(skeleton_trim[:,0], skeleton_trim[:,1])
    skeleton_final = np.vstack(skeleton_final).T
    
    return skeleton_final
