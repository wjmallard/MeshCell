#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author William Mallard (wmallard)
@author Shicong Xie (xies)
@date April 2020
"""
import numpy as np

from collections import defaultdict
from skimage.measure import label as label_connected

def assign_cells_to_chains(cell_masks):
    '''
    Merge cells into chains.
    '''
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids > 0]

    # Create a unique label for each chain.
    chain_masks = label_connected(cell_masks > 0)

    # Assign each cell to a chain.
    chain_to_cells = defaultdict(list)

    for cell_id in cell_ids:
        chain_id = chain_masks[cell_masks == cell_id][0]
        chain_to_cells[chain_id].append(cell_id)

    return chain_masks, chain_to_cells
