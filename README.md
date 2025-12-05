# MeshCell

A cell length analysis pipeline for fluorescence microscopy images of membrane-stained bacteria.

## Overview

This code processes fluorescence microscopy images of membrane-stained cells along with their segmentation masks to generate cell contours and midlines for downstream length analysis. While the workflow follows general principles established in tools like MicrobeTracker (Sliusarenko et al., 2011) and Morphometrics (Ursell et al., 2017), this code was specifically designed to measure the lengths of individual cells in chains of *B. subtilis*.

The pipeline proceeds as follows:

1. **Mask Processing**: Cell masks from image segmentation are filtered to remove small cells and those touching image edges. Cells are then grouped into chains based on connectivity.

2. **Contour Generation**: Boundaries are extracted from masks using the marching squares algorithm, then refined with an active contour model aligning with membrane signals.

3. **Midline Construction**: Chain midlines are constructed from their boundary contours by finding the longest path through Voronoi ridge vertices interior to the chain boundary. Chain midlines are then bisected with cell boundary contours to obtain cell midlines. By constructing midlines through entire chains rather than individual cells, the code avoids edge effects and artificial length biases that frequently occur at cell-cell junctions.

4. **Mesh Construction**: Perpendicular ribs are constructed from the midline skeleton to the boundary contour for visualization and validation of the cell shape model.

5. **Length Measurement**: Cell lengths are determined by measuring along midline skeletons.

The resulting cell mesh data structures are stored in `.npz` files containing all contours, midline skeletons, and meshes. Downstream analysis code can load these files and call `find_contour_length()` on each skeleton to determine precise cell lengths.

## Modules

- `make_meshes.py` — Main driver script that coordinates the pipeline
- `Mask.py` — Handles cell mask operations
- `Contour.py` — Generates contours from mask boundaries
- `Skeleton.py` — Generates centerline skeletons from contours
- `Mesh.py` — Generates meshes from skeletons and contours
- `Util.py` — Utility functions shared across modules

## Usage

```
./make_meshes.py image1.masks_edited.tif image2.masks_edited.tif ...
```

Each `*.masks_edited.tif` must have an accompanying `*.tif` in the same directory.

Outputs:
- `*.contours.npz` — contours, skeletons, and meshes
- `*.png` — diagnostic visualization
- `*.errors.txt` — any failed chains

## Input Requirements

Masks should be integer-labeled segmentation masks where each cell has a unique ID. Mask source does not matter (Cellpose, ilastik, manual segmentation, etc) as long as segmentation quality is reasonable.

## Dependencies

- Python 3.8+
- numpy
- scipy
- scikit-image
- networkx
- matplotlib

## Authors

- William Mallard ([@wjmallard](https://github.com/wjmallard))
- Shicong Xie ([@xies](https://github.com/xies))

## Citation

Published as Appendix G of:

> Mallard, W. (2025). *FtsZ phosphorylation modulates tail-core binding to tune cell division in Bacillus subtilis*. Doctoral dissertation, Harvard University.

## License

MIT License. See [LICENSE](LICENSE).
