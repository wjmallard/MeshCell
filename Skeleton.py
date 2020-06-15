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
from collections import Counter
from matplotlib import path

import Contour

# TODO: Figure out how to set the extension factor in a more principled way.
extension_factor = 20

def generate(contour):
    skel = build_skeleton(contour)
    skel = extend_skeleton(skel, contour)
    return skel

def find_leaves(Branches):

    # Find the degree of all branch-end vertices.
    Degree = Counter(branch[-1] for branch in Branches)
    Degree += Counter(branch[0] for branch in Branches)

    # Find all leaves.
    Leaves = {k for k, v in Degree.items() if v == 1}

    return Leaves

def find_forks(Branches):

    # Find the degree of all branch-end vertices.
    Degree = Counter(branch[-1] for branch in Branches)
    Degree += Counter(branch[0] for branch in Branches)

    # Find all leaves.
    Forks = {k for k, v in Degree.items() if v == 3}

    return Forks

def find_distal_vertices(Branches, fork):

    # Find all vertices distal to the specified fork.
    distal_vertices = {branch[-1] for branch in Branches if branch[0] == fork}
    distal_vertices |= {branch[0] for branch in Branches if branch[-1] == fork}

    return distal_vertices

def remove_spurs(Branches):
    '''
    Remove spurs from a skeleton.

    We define a leaf as a degree 1 vertex.
    We define a fork as a degree 3 vertex.

    A fork has a spur if two of its three branches terminate in other forks,
    while its third branch terminates in a leaf; this branch is the spur.

    For rod-shaped cells, there should never be enough bending to introduce
    complex branching in the Voronoi skeleton (ie, more than a spur) other
    than at the ends of the cell.
    '''
    for fork in find_forks(Branches):

        # Check if this vertex has a spur.
        leaves = find_leaves(Branches)
        distal_vertices = find_distal_vertices(Branches, fork)
        distal_leaves = distal_vertices.intersection(leaves)
        if len(distal_leaves) != 1: continue

        # Remove the spur.
        leaf = distal_leaves.pop()
        spur = [branch for branch in Branches
                if fork in branch and leaf in branch][0]
        Branches.remove(spur)

        # Merge the two remaining branches.
        A, B = [branch for branch in Branches if fork in branch]
        if A[0] == fork: A.reverse()
        if B[-1] == fork: B.reverse()
        assert A[-1] == B[0] == fork
        C = A + B[1:]

        # Replace the two branches with the merged one.
        Branches.remove(A)
        Branches.remove(B)
        Branches.append(C)

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

    while vertices_to_walk:

        # Start a new branch.
        branch = vertices_to_walk.pop()
        next_v = branch.pop()

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

    # Remove spurs.
    remove_spurs(Branches)

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
