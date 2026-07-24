"""Discrete-defect algebra for two-stage velocity-Verlet INC.

Contents:
    Unique stage labels and one-step reconstruction on free degrees of freedom.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_velocity_verlet_defect_labels(
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    baseline_acceleration: NDArray[np.float64],
    mass_lumped: NDArray[np.float64],
    time_step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute unique start- and end-stage left residual-force labels."""
    start_source = (
        2.0
        / time_step**2
        * (displacement[1:] - displacement[:-1] - time_step * velocity[:-1])
        - baseline_acceleration[:-1]
    )
    end_source = (
        2.0 / time_step * (velocity[1:] - velocity[:-1])
        - baseline_acceleration[:-1]
        - start_source
        - baseline_acceleration[1:]
    )

    return -mass_lumped * start_source, -mass_lumped * end_source


def advance_velocity_verlet_with_residuals(
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    baseline_acceleration_start: NDArray[np.float64],
    baseline_acceleration_end: NDArray[np.float64],
    mass_lumped: NDArray[np.float64],
    residual_force_start: NDArray[np.float64],
    residual_force_end: NDArray[np.float64],
    time_step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance one algebraic two-stage step from supplied baseline endpoint forces."""
    source_start = -residual_force_start / mass_lumped
    source_end = -residual_force_end / mass_lumped
    next_displacement = (
        displacement
        + time_step * velocity
        + 0.5 * time_step**2 * (baseline_acceleration_start + source_start)
    )
    next_velocity = velocity + 0.5 * time_step * (
        baseline_acceleration_start
        + source_start
        + baseline_acceleration_end
        + source_end
    )

    return next_displacement, next_velocity
