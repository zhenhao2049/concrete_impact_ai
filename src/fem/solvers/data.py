"""Solver-level data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import MaterialState

DisplacementFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


@dataclass(frozen=True)
class DirichletDofSet:
    """Store prescribed degrees of freedom and values."""

    dofs: NDArray[np.int64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class ArmijoSettings:
    """Store residual-merit Armijo line-search settings."""

    enabled: bool = False
    sufficient_decrease: float = 1.0e-4
    reduction_factor: float = 0.5
    max_backtracks: int = 20

    def __post_init__(self) -> None:
        """Validate the configured sufficient-decrease sequence."""
        if not 0.0 < self.sufficient_decrease < 1.0:
            raise ValueError("Armijo sufficient_decrease must lie in (0, 1).")
        if not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("Armijo reduction_factor must lie in (0, 1).")
        if self.max_backtracks < 0:
            raise ValueError("Armijo max_backtracks must be nonnegative.")


@dataclass(frozen=True)
class AdaptiveTimeStepSettings:
    """Store convergence-driven implicit time-step controls."""

    enabled: bool = False
    minimum_factor: float = 1.0 / 16.0
    maximum_factor: float = 2.0
    cutback_factor: float = 0.5
    growth_factor: float = 2.0
    easy_iteration_limit: int = 4
    easy_step_count: int = 3

    def __post_init__(self) -> None:
        """Validate convergence-driven time-step factors."""
        if not 0.0 < self.minimum_factor <= 1.0:
            raise ValueError("Adaptive minimum_factor must lie in (0, 1].")
        if self.maximum_factor < 1.0:
            raise ValueError("Adaptive maximum_factor must be at least one.")
        if not 0.0 < self.cutback_factor < 1.0:
            raise ValueError("Adaptive cutback_factor must lie in (0, 1).")
        if self.growth_factor <= 1.0:
            raise ValueError("Adaptive growth_factor must exceed one.")
        if self.easy_iteration_limit < 0 or self.easy_step_count <= 0:
            raise ValueError("Adaptive easy-step controls must be positive.")


@dataclass(frozen=True)
class NewtonSettings:
    """Store Newton nonlinear iteration settings."""

    max_iterations: int
    residual_tolerance: float
    increment_tolerance: float


@dataclass(frozen=True)
class NonlinearNewtonSettings:
    """Store scaled Newton convergence settings."""

    max_iterations: int
    residual_absolute_tolerance: float
    residual_relative_tolerance: float
    increment_absolute_tolerance: float
    increment_relative_tolerance: float
    armijo: ArmijoSettings = field(default_factory=ArmijoSettings)


@dataclass(frozen=True)
class LinearSolverSettings:
    """Store linear solver settings."""

    backend: str
    method: str


@dataclass(frozen=True)
class TimeIntegrationSettings:
    """Store time-integration settings."""

    scheme: str
    time_step: float
    num_steps: int
    beta: float
    gamma: float
    cfl_safety_factor: float
    adaptive: AdaptiveTimeStepSettings = field(default_factory=AdaptiveTimeStepSettings)


@dataclass(frozen=True)
class SurrogateSettings:
    """Store one validated solver-level surrogate selection."""

    enabled: bool
    kind: str | None = None
    model: str | None = None
    backend: str | None = None
    artifact_format: str | None = None
    model_path: Path | None = None
    metadata_path: Path | None = None
    device: str | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        """Reject partially defined enabled or disabled selections."""
        optional_values = (
            self.kind,
            self.model,
            self.backend,
            self.artifact_format,
            self.model_path,
            self.metadata_path,
            self.device,
            self.dtype,
        )
        if self.enabled and any(value is None for value in optional_values):
            raise ValueError("Enabled surrogate settings require every deployment field.")
        if not self.enabled and any(value is not None for value in optional_values):
            raise ValueError("Disabled surrogate settings must not define deployment fields.")


@dataclass(frozen=True)
class FEMRunDef:
    """Store finite element solver settings and mesh-dependent load data."""

    name: str
    analysis_type: str
    plane_state: str
    dirichlet: DirichletDofSet
    external_force: NDArray[np.float64]
    initial_displacement: NDArray[np.float64]
    initial_velocity: NDArray[np.float64]
    exact_displacement: DisplacementFunction
    linear_solver: LinearSolverSettings
    newton: NewtonSettings
    time: TimeIntegrationSettings
    surrogate: SurrogateSettings


@dataclass(frozen=True)
class StaticSolution:
    """Store one static finite element solution."""

    displacement: NDArray[np.float64]
    reaction: NDArray[np.float64]


@dataclass(frozen=True)
class DynamicSolution:
    """Store one dynamic finite element solution history."""

    times: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]


@dataclass(frozen=True)
class MaterialDynamicSolution:
    """Store nonlinear material-dynamics histories and diagnostics."""

    times: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    internal_force: NDArray[np.float64]
    external_force: NDArray[np.float64]
    stresses: NDArray[np.float64]
    equivalent_plastic_strain: NDArray[np.float64]
    kinetic_energy: NDArray[np.float64]
    free_energy: NDArray[np.float64]
    dissipated_energy: NDArray[np.float64]
    external_work: NDArray[np.float64]
    energy_residual: NDArray[np.float64]
    positive_incremental_work: NDArray[np.float64]
    negative_incremental_work: NDArray[np.float64]
    unloading_volume_fraction: NDArray[np.float64]
    viscoplastic_active_volume_fraction: NDArray[np.float64]
    minimum_yield_activation_margin: NDArray[np.float64]
    maximum_yield_activation_margin: NDArray[np.float64]
    newton_iterations: NDArray[np.int64]
    accepted_time_steps: NDArray[np.float64]
    rejected_step_counts: NDArray[np.int64]
    armijo_backtracks: NDArray[np.int64]
    rejected_reasons: tuple[tuple[str, ...], ...]
    material_diagnostics: dict[str, NDArray[np.float64]]
    integrator_diagnostics: dict[str, NDArray[np.float64]]
    final_state: MaterialState
    solve_time: float
    material_state_history: dict[str, NDArray[np.float64]] | None = None
