"""Nonlinear implicit Newmark effective residual helpers.

Contents:
    Newmark acceleration, velocity, effective residual, and effective tangent operators.
Author:
    Zhen Hao.
Created:
    2026-07-09.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def compute_newmark_acceleration(
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    previous_velocity: NDArray[np.float64],
    previous_acceleration: NDArray[np.float64],
    time_step: float,
    beta: float,
) -> NDArray[np.float64]:
    """Compute implicit Newmark acceleration from displacement."""
    return (
        (displacement - previous_displacement) / (beta * time_step**2)
        - previous_velocity / (beta * time_step)
        - (1.0 / (2.0 * beta) - 1.0) * previous_acceleration
    )


def compute_newmark_velocity(
    previous_velocity: NDArray[np.float64],
    previous_acceleration: NDArray[np.float64],
    acceleration: NDArray[np.float64],
    time_step: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Compute implicit Newmark velocity from acceleration."""
    return previous_velocity + time_step * (
        (1.0 - gamma) * previous_acceleration + gamma * acceleration
    )


def build_newmark_effective_residual(
    internal_force: NDArray[np.float64],
    external_force: NDArray[np.float64],
    mass: csr_matrix,
    damping: csr_matrix,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    previous_velocity: NDArray[np.float64],
    previous_acceleration: NDArray[np.float64],
    time_step: float,
    beta: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Build the nonlinear implicit Newmark residual."""
    acceleration = compute_newmark_acceleration(
        displacement,
        previous_displacement,
        previous_velocity,
        previous_acceleration,
        time_step,
        beta,
    )
    velocity = compute_newmark_velocity(
        previous_velocity,
        previous_acceleration,
        acceleration,
        time_step,
        gamma,
    )

    return internal_force + mass @ acceleration + damping @ velocity - external_force


def build_newmark_effective_tangent(
    material_tangent: csr_matrix,
    mass: csr_matrix,
    damping: csr_matrix,
    time_step: float,
    beta: float,
    gamma: float,
) -> csr_matrix:
    """Build the nonlinear implicit Newmark effective tangent."""
    return (
        material_tangent
        + (1.0 / (beta * time_step**2)) * mass
        + (gamma / (beta * time_step)) * damping
    )
