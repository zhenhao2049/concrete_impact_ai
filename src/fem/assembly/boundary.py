"""Boundary-condition application for assembled systems.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def apply_dirichlet_to_linear_system(
    matrix: csr_matrix,
    right_hand_side: NDArray[np.float64],
    dofs: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Apply prescribed DOF values to a sparse linear system."""
    modified_rhs = right_hand_side - matrix[:, dofs] @ values
    modified_matrix = matrix.tolil()

    modified_matrix[dofs, :] = 0.0
    modified_matrix[:, dofs] = 0.0
    modified_matrix[dofs, dofs] = 1.0
    modified_rhs[dofs] = values

    return modified_matrix.tocsr(), modified_rhs


def impose_values_on_vector(
    vector: NDArray[np.float64],
    dofs: NDArray[np.int64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a vector with prescribed values inserted."""
    modified_vector = vector.copy()
    modified_vector[dofs] = values

    return modified_vector
