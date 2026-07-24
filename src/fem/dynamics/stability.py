"""Dynamic stability checks.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fem.solvers.data import TimeIntegrationSettings

if TYPE_CHECKING:
    from fem.mesh.data import MeshInfo
    from fem.preprocess.data import PreprocessBundle


@dataclass(frozen=True)
class StabilityReport:
    """Store explicit dynamic stability data."""

    minimum_length: float
    maximum_wave_speed: float
    stable_time_step: float
    requested_time_step: float


def check_explicit_stability(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
) -> StabilityReport:
    """Check the explicit CFL time-step bound."""
    report = compute_explicit_stability_report(bundle, time_settings)

    if time_settings.time_step > report.stable_time_step:
        message = (
            "Explicit time step violates the CFL stability bound: "
            f"dt={time_settings.time_step:.6e}, dt_crit={report.stable_time_step:.6e}."
        )
        raise ValueError(message)

    return report


def compute_explicit_stability_report(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
) -> StabilityReport:
    """Compute the explicit CFL time-step report."""
    minimum_length = estimate_minimum_edge_length(bundle.mesh_info)
    maximum_wave_speed = compute_linear_elastic_pressure_wave_speed(bundle)
    stable_time_step = (
        time_settings.cfl_safety_factor * minimum_length / maximum_wave_speed
    )

    return StabilityReport(
        minimum_length=minimum_length,
        maximum_wave_speed=maximum_wave_speed,
        stable_time_step=stable_time_step,
        requested_time_step=time_settings.time_step,
    )


def estimate_minimum_edge_length(mesh_info: MeshInfo) -> float:
    """Estimate the minimum physical element edge length."""
    edge_nodes = EDGE_NODE_TABLE[mesh_info.element_type]
    minimum_length = np.inf

    for connectivity in mesh_info.elements:
        coordinates = mesh_info.nodes[connectivity, :]
        edge_lengths = _compute_element_edge_lengths(coordinates, edge_nodes)
        minimum_length = min(minimum_length, float(np.min(edge_lengths)))

    return float(minimum_length)


def compute_linear_elastic_pressure_wave_speed(bundle: PreprocessBundle) -> float:
    """Return the material-provided conservative longitudinal-wave-speed bound."""
    return float(bundle.material.maximum_wave_speed)


def _compute_element_edge_lengths(
    coordinates: NDArray[np.float64],
    edge_nodes: tuple[tuple[int, int], ...],
) -> NDArray[np.float64]:
    """Compute edge lengths for one element."""
    lengths = [
        np.linalg.norm(coordinates[node_j, :] - coordinates[node_i, :])
        for node_i, node_j in edge_nodes
    ]

    return np.asarray(lengths, dtype=np.float64)


EDGE_NODE_TABLE = {
    "quad4": ((0, 1), (0, 2), (1, 3), (2, 3)),
    "hex8": (
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
}
