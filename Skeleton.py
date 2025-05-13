#!/usr/bin/env python3
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np
import networkx as nx
from scipy.spatial import Voronoi
from scipy.stats import linregress

import Contour

END_TRIMMING_DIST = 20  # pixels
MAX_BRANCH_ANGLE = 30  # degrees
NUM_INTERP_POINTS = 10  # pixels

def generate(contour):

    skel = build_skeleton(contour)
    skel = smooth_skeleton(skel)
    skel = extend_skeleton(skel, contour)
    skel = crop_skeleton(skel, contour)
    skel = smooth_skeleton(skel)

    return skel

def build_skeleton(contour):

    # Build a Voronoi diagram.
    # Extract the tree from its interior.
    V, E = get_voronoi_interior(contour)

    # Find longest path through graph.
    longest_path = find_longest_path(V, E)

    # Extract coordinates.
    skeleton = V[longest_path]

    # Trim bent ends.
    skeleton = trim_bent_ends(skeleton)

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

def trim_bent_ends(skeleton):

    skeleton = trim_bent_end_at_front(skeleton)
    skeleton = skeleton[::-1]
    skeleton = trim_bent_end_at_front(skeleton)
    skeleton = skeleton[::-1]

    return skeleton

def trim_bent_end_at_front(skeleton):

    # Isolate vertices near the end of the skeleton.
    # Use the distance threshold specified above.
    dist_from_end = np.linalg.norm(skeleton[0] - skeleton, axis=1)
    close_to_end = dist_from_end < END_TRIMMING_DIST

    # For single cells, the entire cell may be "close to the end",
    # so just use the entire skeleton.
    transitions = np.where(np.diff(close_to_end))[0]
    last_to_keep = transitions[0] if len(transitions) > 0 else len(skeleton)

    skel_end = skeleton[:last_to_keep + 1]

    # Calculate the bending angle at each vertex.
    #
    # For each node, compute the angle between the two adjacent
    # vectors, in degrees. Leave the first and last values as 0.
    L = len(skel_end)
    angles = np.zeros(L, dtype=float)

    for a, b, c in zip(range(0,L), range(1,L), range(2,L)):
        angles[b] = find_line_angle(skel_end[[a,b]], skel_end[[b,c]])

    # Identify all nodes where the angle exceeds the
    # angle threshold specified above. Select the last one,
    # and trim all points before it from the skeleton.
    bent_vertices = np.where(angles > MAX_BRANCH_ANGLE)[0]
    new_start = bent_vertices[-1] if bent_vertices.size > 0 else 0

    skeleton = skeleton[new_start:]

    return skeleton

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

def find_longest_path(V, E):

    G = nx.Graph(E)

    # Pick an arbitrary start node, v_init.
    v_init = min(G.nodes)

    # Run BFS starting from v_init.
    # Annotate each node with its
    # cumulative Euclidean distance
    # from v_init along the search path.
    G.nodes[v_init]['dist'] = 0.

    for v, u in nx.bfs_predecessors(G, v_init):
        d = np.linalg.norm(V[u] - V[v])
        G.nodes[v]['dist'] = G.nodes[u]['dist'] + d

    # The node with the largest total distance
    # is the end of the longest path.
    v_end, _ = max(G.nodes(data=True), key=lambda x: x[1]['dist'])

    # Run BFS starting from v_end.
    # Annotate each node with its
    # cumulative Euclidean distance
    # from v_end along the search path.
    G.nodes[v_end]['dist'] = 0.

    for v, u in nx.bfs_predecessors(G, v_end):
        d = np.linalg.norm(V[u] - V[v])
        G.nodes[v]['dist'] = G.nodes[u]['dist'] + d

    # The node with the largest total distance
    # is the start of the longest path.
    v_start, _ = max(G.nodes(data=True), key=lambda x: x[1]['dist'])

    # Find the shortest path from v_start to v_end.
    # This is the longest path in the graph.
    Longest_Path = nx.shortest_path(G, v_start, v_end)

    return Longest_Path

def smooth_skeleton(skeleton):

    # Uniformly distribute points along the skeleton.
    skeleton = Contour.evenly_distribute_contour_points(skeleton)

    return skeleton

def extend_skeleton(skeleton, contour):

    # Calculate an extension length that guarantees a projected
    # point will definitely lie outside of the contour.
    contour_dimensions = np.ptp(contour, axis=0)
    extension_length = np.ceil(contour_dimensions.max()).astype(int)

    # If the "left" end is inside the contour, extend it.
    if Contour.is_point_in_polygon(skeleton[0], contour):

        # Extrapolate a point guaranteed to be outside the left end.
        left_end = skeleton[:NUM_INTERP_POINTS][::-1]
        P_ext = extend_line(left_end, extension_length)

        # Add this new exterior point to the left end.
        skeleton = np.insert(skeleton, 0, [P_ext], axis=0)

    # If the "right" end is inside the contour, extend it.
    if Contour.is_point_in_polygon(skeleton[-1], contour):

        # Extrapolate a point guaranteed to be outside the right end.
        right_end = skeleton[-NUM_INTERP_POINTS:]
        P_ext = extend_line(right_end, extension_length)

        # Add this new exterior point to the right end.
        skeleton = np.append(skeleton, [P_ext], axis=0)

    return skeleton

def extend_line(line, ext_len):
    '''
    Extend a line by projecting a point outward from its end.
    Fit via linear regression, handling the case of near-vertical lines.
    Returns a single point projected outward from the line's end.

    Adapted from: https://stats.stackexchange.com/a/182893
    '''
    VERT_THRESHOLD = 1

    # Calculate the verticality of the line.
    X_std, Y_std = line.std(axis=0)
    verticality = (Y_std / X_std) if (X_std > 0) else np.inf

    X, Y = line.T

    # If the line is too vertical,
    # rotate it 90 degrees.
    if verticality > VERT_THRESHOLD:
        X, Y = Y, X

    # Fit via linear regression.
    res = linregress(X, Y)

    # Create a unit vector along the regression line.
    xa = X[0]
    xb = X[-1]
    yb = Y[-1]

    fa = res.slope * xa + res.intercept
    fb = res.slope * xb + res.intercept

    vector = np.array([xb - xa, fb - fa])
    vector /= np.linalg.norm(vector)

    # Project a point outward along the unit vector
    # starting from the end of the original line.
    x_ext, y_ext = ext_len * vector + np.array([xb, yb])

    # If we rotated the line 90 degrees,
    # rotate the projected point back.
    if verticality > VERT_THRESHOLD:
        x_ext, y_ext = y_ext, x_ext

    return x_ext, y_ext

def crop_skeleton(skeleton, contour):

    # Identify all points on the skeleton inside the contour.
    interior_points = [Contour.is_point_in_polygon(p, contour) for p in skeleton]

    # Find the first and last skeleton indices inside the contour.
    a, b = 1 + np.where(np.diff(interior_points))[0]

    # Trim the skeleton to include points just outside the contour.
    skeleton = skeleton[a-1:b+1]

    # Trim the skeleton to precisely end at the contour boundary.
    #
    # This step handles a numerical precision issue:
    # If the skeleton endpoints are already close to the contour,
    # the intersection detection algorithm may fail to find exact
    # intersection points due to machine precision issues.
    #
    # To address this issue:
    # 1. We accept finding either one or two contour intersections.
    #    Ideally we would always find exactly two: one at each end.
    # 2. For each intersection we do find, we determine which end of
    #    the skeleton it belongs to by measuring its distance to the
    #    second-to-last point at both ends of the skeleton.
    # 3. We then replace each skeleton endpoint with the appropriate
    #    intersection point.
    intersections = Contour.find_contour_intersections(skeleton, contour)
    assert len(intersections) <= 2

    for intersection in intersections:

        dist_from_head = np.linalg.norm(skeleton[1] - intersection)
        dist_from_tail = np.linalg.norm(skeleton[-2] - intersection)

        if dist_from_head < dist_from_tail:
            skeleton[0] = intersection
        else:
            skeleton[-1] = intersection

    return skeleton
