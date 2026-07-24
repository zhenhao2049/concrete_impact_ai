"""Mandel transformations for symmetric three-dimensional tensors.

Author:
    Zhen Hao.
Created:
    2026-07-22.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

MANDEL_COMPONENT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
SQRT_TWO = np.sqrt(2.0)
ENGINEERING_TO_MANDEL_STRESS = np.diag(
    np.asarray([1.0, 1.0, 1.0, SQRT_TWO, SQRT_TWO, SQRT_TWO], dtype=np.float64)
)
ENGINEERING_TO_MANDEL_STRAIN = np.diag(
    np.asarray([1.0, 1.0, 1.0, 1.0 / SQRT_TWO, 1.0 / SQRT_TWO, 1.0 / SQRT_TWO])
)


def engineering_strain_to_mandel(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert engineering-shear strain vectors to Mandel coordinates."""
    return np.asarray(values, dtype=np.float64) @ ENGINEERING_TO_MANDEL_STRAIN.T


def mandel_strain_to_engineering(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Mandel strain vectors to engineering-shear coordinates."""
    return np.asarray(values, dtype=np.float64) @ ENGINEERING_TO_MANDEL_STRESS.T


def engineering_stress_to_mandel(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert engineering-Voigt stress vectors to Mandel coordinates."""
    return np.asarray(values, dtype=np.float64) @ ENGINEERING_TO_MANDEL_STRESS.T


def mandel_stress_to_engineering(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert Mandel stress vectors to engineering-Voigt coordinates."""
    return np.asarray(values, dtype=np.float64) @ ENGINEERING_TO_MANDEL_STRAIN.T


def engineering_stiffness_to_mandel(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert engineering-Voigt stiffness matrices to Mandel coordinates."""
    matrices = np.asarray(values, dtype=np.float64)
    return np.einsum(
        "ij,...jk,kl->...il",
        ENGINEERING_TO_MANDEL_STRESS,
        matrices,
        ENGINEERING_TO_MANDEL_STRESS,
    )


def mandel_stiffness_to_engineering(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Mandel stiffness matrices to engineering-Voigt coordinates."""
    matrices = np.asarray(values, dtype=np.float64)
    return np.einsum(
        "ij,...jk,kl->...il",
        ENGINEERING_TO_MANDEL_STRAIN,
        matrices,
        ENGINEERING_TO_MANDEL_STRAIN,
    )


def mandel_transform_operator(
    orthogonal_transform: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the Mandel action induced by one orthogonal spatial transform."""
    transform = np.asarray(orthogonal_transform, dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError("Mandel spatial transform must have shape (3, 3).")
    orthogonality_error = np.linalg.norm(transform @ transform.T - np.eye(3))
    if orthogonality_error > 1.0e-12:
        raise ValueError("Mandel spatial transform must be orthogonal.")

    operator = np.empty((6, 6), dtype=np.float64)
    for component in range(6):
        basis = np.zeros(6, dtype=np.float64)
        basis[component] = 1.0
        tensor = _mandel_vector_to_tensor(basis)
        operator[:, component] = _tensor_to_mandel_vector(
            transform @ tensor @ transform.T
        )
    return operator


def d4_mandel_operators() -> tuple[NDArray[np.float64], ...]:
    """Return the eight square-symmetry actions about the third axis."""
    reflection = np.diag(np.asarray([1.0, -1.0, 1.0], dtype=np.float64))
    transforms: list[NDArray[np.float64]] = []
    for quarter_turn in range(4):
        angle = 0.5 * np.pi * quarter_turn
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        transforms.extend((rotation, rotation @ reflection))
    return tuple(mandel_transform_operator(transform) for transform in transforms)


def project_mandel_stiffness_to_d4(
    stiffness: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project one Mandel stiffness matrix onto square-cell symmetry."""
    matrix = np.asarray(stiffness, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError("D4 stiffness projection requires shape (6, 6).")
    operators = d4_mandel_operators()
    projected = np.zeros_like(matrix)
    for operator in operators:
        projected += operator @ matrix @ operator.T
    return projected / len(operators)


def _mandel_vector_to_tensor(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expand one Mandel vector into a symmetric tensor."""
    tensor = np.asarray(
        [
            [vector[0], vector[5] / SQRT_TWO, vector[4] / SQRT_TWO],
            [vector[5] / SQRT_TWO, vector[1], vector[3] / SQRT_TWO],
            [vector[4] / SQRT_TWO, vector[3] / SQRT_TWO, vector[2]],
        ],
        dtype=np.float64,
    )
    return tensor


def _tensor_to_mandel_vector(tensor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compress one symmetric tensor into Mandel coordinates."""
    return np.asarray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            SQRT_TWO * tensor[1, 2],
            SQRT_TWO * tensor[0, 2],
            SQRT_TWO * tensor[0, 1],
        ],
        dtype=np.float64,
    )
