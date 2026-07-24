"""Shape function cache data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShapeFunctionCache:
    """Precomputed shape function and Jacobian data."""

    element_type: str
    quadrature_points: NDArray[np.float64]
    quadrature_weights: NDArray[np.float64]
    shape_values: NDArray[np.float64]
    shape_gradients_reference: NDArray[np.float64]
    jacobians: NDArray[np.float64]
    jacobian_determinants: NDArray[np.float64]
    shape_gradients_physical: NDArray[np.float64]
