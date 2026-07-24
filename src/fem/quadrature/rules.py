"""Quadrature rules for low-order Lagrange elements.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import itertools

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray


def make_quadrature_rule(
    element_type: str,
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build quadrature points and weights for one element type."""
    return QUADRATURE_BUILDERS[element_type](order)


def _tensor_product_rule(
    dimension: int,
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a tensor-product Gauss rule on the reference interval cube."""
    points_1d, weights_1d = leggauss(order)
    tensor_points = list(itertools.product(points_1d, repeat=dimension))
    tensor_weights = [np.prod(values) for values in itertools.product(weights_1d, repeat=dimension)]

    return np.asarray(tensor_points, dtype=np.float64), np.asarray(tensor_weights, dtype=np.float64)


def _triangle_rule(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the centroid quadrature rule for a linear triangle."""
    return np.asarray([[1.0 / 3.0, 1.0 / 3.0]]), np.asarray([0.5])


def _tetra_rule(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the centroid quadrature rule for a linear tetrahedron."""
    return np.asarray([[0.25, 0.25, 0.25]]), np.asarray([1.0 / 6.0])


QUADRATURE_BUILDERS = {
    "quad4": lambda order: _tensor_product_rule(2, order),
    "hex8": lambda order: _tensor_product_rule(3, order),
    "tri3": _triangle_rule,
    "tet4": _tetra_rule,
}
