"""Element node ordering conversion rules.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray

MESHIO_CELL_TYPES = {
    "tri3": "triangle",
    "quad4": "quad",
    "tet4": "tetra",
    "hex8": "hexahedron",
}

BOUNDARY_CELL_TYPES = {
    2: "line",
    3: "quad",
}

GMSH_TO_INTERNAL = {
    "tri3": np.array([0, 1, 2], dtype=np.int64),
    "quad4": np.array([0, 1, 3, 2], dtype=np.int64),
    "tet4": np.array([0, 1, 2, 3], dtype=np.int64),
    "hex8": np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64),
}

INTERNAL_TO_VTK = {
    "tri3": np.array([0, 1, 2], dtype=np.int64),
    "quad4": np.array([0, 1, 3, 2], dtype=np.int64),
    "tet4": np.array([0, 1, 2, 3], dtype=np.int64),
    "hex8": np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64),
}

INTERNAL_ORDERING_NAMES = {
    "tri3": "positive-simplex",
    "quad4": "tensor-product-xi-eta",
    "tet4": "positive-simplex",
    "hex8": "tensor-product-xi-eta-zeta",
}


def reorder_cells(cells: NDArray[np.int64], permutation: NDArray[np.int64]) -> NDArray[np.int64]:
    """Reorder element node connectivity with one permutation."""
    return cells[:, permutation]
