#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from collections import defaultdict

import Contour
import Util

MAX_BRANCH_ANGLE = 20
NUM_INTERP_POINTS = 10

def generate(contour):
    skel = build_skeleton(contour)
    skel = extend_skeleton(skel, contour)
    return skel

def build_skeleton(contour):

    # Build a Voronoi diagram.
    # Extract the tree from its interior.
    V, E = get_voronoi_interior(contour)

    # Find longest path through graph.
    longest_path = find_longest_topological_path(E)

    # Extract coordinates.
    skeleton = V[longest_path]

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

def merge_collinear_branches(V, Branches):

    #
    # 1. Generate matrix of adjacent collinear branches.
    #

    # Generate branch collinearity matrix.
    Angles = build_angles_matrix(V, Branches)
    Collinear = Angles <= MAX_BRANCH_ANGLE

    # Generate branch connectivity matrix.
    Adjacent = build_adjacency_matrix(Branches)

    # Intersect collinearity and connectivity.
    AC = Adjacent & Collinear

    #
    # 2. Generate matrix of connected components.
    #

    # Find connected components.
    N = len(AC)
    Connected = np.linalg.matrix_power(AC, N)

    #
    # 3. Extract connected components from matrix.
    #

    # Merge redundant rows.
    X1 = np.unique(Connected, axis=0)

    # Remove singletons.
    X2 = X1[X1.sum(axis=1) > 1]

    # Extract lists of branch labels.
    X3 = [np.where(row)[0] for row in X2]

    # Extract lists of branch nodes.
    X4 = [[Branches[i] for i in branch_ids] for branch_ids in X3]

    #
    # 4. Stitch together branch segments.
    #
    Collinear_Branches = list(map(stitch_branch_segments, X4))

    return Collinear_Branches

def build_adjacency_matrix(Branches):

    # NOTE: We use numerical branch order
    #       as the label for each branch.

    # Build a branch adjacency list for the
    # vertices at the end of every branch.
    Forks = defaultdict(list)

    for e, Branch in enumerate(Branches):

        u = Branch[0]
        v = Branch[-1]

        Forks[u].append(e)
        Forks[v].append(e)

    # Remove all leaf vertices.
    Forks = {v: edges
             for v, edges in Forks.items()
             if len(edges) > 1}

    # Convert this into an adjacency matrix.
    N = len(Branches)
    Adjacency = np.zeros((N,N), dtype=bool)

    # For each fork vertex:
    for _, edges in Forks.items():

        # Enumerate all pairs of branches
        # that meet at this vertex.
        coords = find_all_pairs(edges, inclusive=True)

        # Mark their adjacency.
        for u, v in coords:
            Adjacency[u,v] = True
            Adjacency[v,u] = True

    return Adjacency

def find_all_pairs(L, inclusive=False):

    n = 1 if inclusive else 0

    pairs = [(L[i], L[j])
             for j in range(len(L))
             for i in range(j + n)]

    return pairs

def build_angles_matrix(V, Branches):

    # NOTE: We use numerical branch order
    #       as the label for each branch.

    # Map branches to list of coordinates.
    XA = np.array([V[B[0]] for B in Branches])
    XB = np.array([V[B[1]] for B in Branches])

    # Convert each edge to a vector.
    X = XA - XB

    # Find the angle between every pair of vectors.
    CD = pdist(X, metric='cosine')
    CS = np.abs(1 - CD)
    Theta = np.rad2deg(np.arccos(CS))
    Angles = squareform(Theta)

    return Angles

def find_line_angle(L1, L2):
    '''
    Find the angle between two lines, in degrees.
    '''
    AB = L1[0] - L2[0]
    CD = L1[1] - L2[1]

    # If either line is actually a point,
    # return an angle of zero degrees.
    if (AB == 0).all() or (CD == 0).all():
        return 0.

    AB /= np.linalg.norm(AB)
    CD /= np.linalg.norm(CD)

    abs_dot = np.abs(np.dot(AB, CD))
    theta = np.arccos(abs_dot)

    return np.rad2deg(theta)

def stitch_branch_segments(segments):
    '''
    Stitch a list of branch nodes into a single branch.

    Takes a list of lists.
    Each sub-list is a list of sequential nodes in a branch segment.
    Does not require the segments to be in order or in any particular
    orientation relative to one another.

    However, the branch segments:
     - should all be connected
     - should not form a branching structure
    '''
    G = nx.Graph()

    for nodes in segments:
        edges = zip(nodes, nodes[1:])
        G.add_edges_from(edges)

    start = min(v for v in G if len(G[v]) == 1)
    branch = list(nx.dfs_preorder_nodes(G, start))

    return branch

def find_longest_topological_path(E):

    G = nx.Graph(E)

    # Pick an arbitrary start node, v_init.
    v_init = min(G.nodes)

    # Run BFS starting from v_init,
    # and save the node farthest from it.
    # This is the end of the longest path.
    T1 = nx.bfs_tree(G, v_init)
    v_end = list(T1.nodes)[-1]

    # Run BFS starting from v_end,
    # and save the node farthest from it.
    # This is the start of the longest path.
    T2 = nx.bfs_tree(G, v_end)
    v_start = list(T2.nodes)[-1]

    # Find the shortest path from v_start to v_end.
    # This is the longest path in the graph.
    Longest_Path = nx.shortest_path(G, v_start, v_end)

    return Longest_Path

def extend_skeleton(skeleton, contour):

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
