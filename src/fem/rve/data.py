"""Representative-volume-element data interfaces.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from fem.assembly.nonlinear_solid import MaterialAssemblyCache
from fem.materials.data import MaterialResponseRequirements, MaterialState, MaterialUpdateSettings
from fem.solvers.data import NonlinearNewtonSettings


@dataclass(frozen=True)
class PeriodicConstraint:
    """Store exact periodic elimination operators for one structured box."""

    transformation: csr_matrix
    affine_matrix: NDArray[np.float64]
    equivalence_class_ids: NDArray[np.int64]
    representative_nodes: NDArray[np.int64]
    reference_class: int
    divisions: tuple[int, int, int]
    opposite_face_nodes: tuple[
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ]


@dataclass(frozen=True)
class RVERequest:
    """Store one strain-driven RVE update request."""

    macro_strain: NDArray[np.float64]
    time_step: float
    material_update_settings: MaterialUpdateSettings
    requirements: MaterialResponseRequirements = field(
        default_factory=lambda: MaterialResponseRequirements(tangent=True)
    )


@dataclass(frozen=True)
class RVEState:
    """Store the committed microstate and converged periodic fluctuation."""

    material_state: MaterialState
    macro_strain: NDArray[np.float64]
    fluctuation_dofs: NDArray[np.float64]


@dataclass(frozen=True)
class RVEResponse:
    """Store homogenized and microscopic outputs from one RVE update."""

    macro_stress: NDArray[np.float64]
    effective_tangent: NDArray[np.float64] | None
    state: RVEState
    displacement: NDArray[np.float64]
    micro_strains: NDArray[np.float64]
    micro_stresses: NDArray[np.float64]
    micro_diagnostics: dict[str, NDArray[np.float64]]
    free_energy_density: float
    dissipation_density: float
    phase_diagnostics: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RVEPathResult:
    """Store accepted responses along one complete macro-strain path."""

    times: NDArray[np.float64]
    responses: tuple[RVEResponse, ...]
    validation_diagnostics: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredHex8RVE:
    """Store one structured single-material Hex8 RVE definition."""

    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    divisions: tuple[int, int, int]
    material: Any
    constraint: PeriodicConstraint
    newton_settings: NonlinearNewtonSettings
    element_phase_ids: NDArray[np.int64] | None = None
    element_densities: NDArray[np.float64] | None = None
    phase_names: tuple[str, ...] = ()
    quadrature_order: int = 2
    hill_mandel_power_scale: float = 1.0


@dataclass(frozen=True)
class RVEOperatorCache:
    """Store immutable geometry and phase data reused by one RVE."""

    assembly: MaterialAssemblyCache
    quadrature_weights: NDArray[np.float64]
    volume: float
    point_phase_ids: NDArray[np.int64]


@dataclass(frozen=True)
class RVEExecutionSettings:
    """Select explicit serial or persistent process-pool RVE point execution."""

    backend: Literal["serial", "process_pool"] = "serial"
    workers: int = 1
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        """Reject invalid or unnecessary process counts."""
        if self.workers <= 0:
            raise ValueError("RVE execution workers must be positive.")
        if self.threads_per_worker <= 0:
            raise ValueError("RVE execution threads_per_worker must be positive.")
        if self.backend == "serial" and self.workers != 1:
            raise ValueError("Serial RVE execution requires exactly one worker.")
        logical_cpu_limit = max((os.cpu_count() or 1) // 2, 1)
        requested_cpus = self.workers * self.threads_per_worker
        if self.backend == "process_pool" and requested_cpus > logical_cpu_limit:
            raise ValueError(
                "RVE process CPU budget exceeds half of available CPUs: "
                f"workers={self.workers}, threads_per_worker={self.threads_per_worker}, "
                f"requested={requested_cpus}, limit={logical_cpu_limit}."
            )
