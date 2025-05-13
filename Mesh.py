#!/usr/bin/env python3
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np

import Contour

def make_ribs(contour, skeleton):

    # Construct segments along skeleton that are perpendicular
    # to the skeleton and that intersect with the contour.

    # TODO: Find a more principled heuristic for this parameter.
    rib_projection_parameter = 20

    tangent_vectors = np.diff(skeleton, axis=0)
    tangent_vectors_smooth = (tangent_vectors[1:] + tangent_vectors[:-1]) / 2
    orthogonal_vectors = tangent_vectors_smooth[:,::-1]
    orthogonal_vectors[:,0] *= -1

    rib_starts = skeleton[1:-1]
    top_rib_ends = rib_starts + orthogonal_vectors * rib_projection_parameter
    bot_rib_ends = rib_starts - orthogonal_vectors * rib_projection_parameter

    top_intersections = Contour.find_segment_intersection(rib_starts, top_rib_ends, contour)
    bot_intersections = Contour.find_segment_intersection(rib_starts, bot_rib_ends, contour)

    return rib_starts, top_intersections, bot_intersections
