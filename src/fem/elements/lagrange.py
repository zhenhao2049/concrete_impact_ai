"""Low-order Lagrange shape functions.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray


def evaluate_lagrange_shape_functions(
    element_type: str,
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate shape functions and reference gradients."""
    return SHAPE_FUNCTION_BUILDERS[element_type](points)


def _quad4_shape_functions(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate bilinear quadrilateral shape functions."""
    xi = points[:, 0]
    eta = points[:, 1]

    shape_values = np.column_stack(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
        ]
    )

    gradients = np.zeros((points.shape[0], 4, 2), dtype=np.float64)
    gradients[:, 0, 0] = -0.25 * (1.0 - eta)  # dN1/dxi
    gradients[:, 0, 1] = -0.25 * (1.0 - xi)  # dN1/deta
    gradients[:, 1, 0] = 0.25 * (1.0 - eta)  # dN2/dxi
    gradients[:, 1, 1] = -0.25 * (1.0 + xi)  # dN2/deta
    gradients[:, 2, 0] = -0.25 * (1.0 + eta)  # dN3/dxi
    gradients[:, 2, 1] = 0.25 * (1.0 - xi)  # dN3/deta
    gradients[:, 3, 0] = 0.25 * (1.0 + eta)  # dN4/dxi
    gradients[:, 3, 1] = 0.25 * (1.0 + xi)  # dN4/deta

    return shape_values, gradients


def _hex8_shape_functions(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate trilinear hexahedral shape functions."""
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    factors = 1.0 + points[:, None, :] * signs[None, :, :]
    shape_values = 0.125 * np.prod(factors, axis=2)

    gradients = np.zeros((points.shape[0], 8, 3), dtype=np.float64)
    gradients[:, :, 0] = 0.125 * signs[None, :, 0] * factors[:, :, 1] * factors[:, :, 2]
    gradients[:, :, 1] = 0.125 * signs[None, :, 1] * factors[:, :, 0] * factors[:, :, 2]
    gradients[:, :, 2] = 0.125 * signs[None, :, 2] * factors[:, :, 0] * factors[:, :, 1]

    return shape_values, gradients


def _tri3_shape_functions(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate linear triangle shape functions."""
    xi = points[:, 0]
    eta = points[:, 1]
    shape_values = np.column_stack([1.0 - xi - eta, xi, eta])
    gradients = np.repeat([[[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]], points.shape[0], axis=0)

    return shape_values, gradients


def _tet4_shape_functions(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate linear tetrahedral shape functions."""
    xi = points[:, 0]
    eta = points[:, 1]
    zeta = points[:, 2]
    shape_values = np.column_stack([1.0 - xi - eta - zeta, xi, eta, zeta])
    gradients = np.repeat(
        [[[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        points.shape[0],
        axis=0,
    )

    return shape_values, gradients


SHAPE_FUNCTION_BUILDERS = {
    "quad4": _quad4_shape_functions,
    "hex8": _hex8_shape_functions,
    "tri3": _tri3_shape_functions,
    "tet4": _tet4_shape_functions,
}
