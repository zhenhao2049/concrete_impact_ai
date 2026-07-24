"""Linear algebra solver wrappers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def solve_sparse_direct(
    matrix: csr_matrix,
    right_hand_side: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve one sparse linear system with a direct solver."""
    solution = spsolve(matrix, right_hand_side)

    return np.asarray(solution, dtype=np.float64)
