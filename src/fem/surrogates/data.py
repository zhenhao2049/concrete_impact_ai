"""Project-neutral surrogate protocols and metadata.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)

SurrogateKind = Literal["material_point", "residual_corrector", "response", "field"]
TimeRepresentation = Literal["continuous", "discrete"]


@dataclass(frozen=True)
class TensorFieldSpec:
    """Describe one ordered tensor field in a surrogate artifact."""

    name: str
    width: int
    unit: str


@dataclass(frozen=True)
class SurrogateCapabilities:
    """Declare optional outputs and time semantics supported by a surrogate."""

    provides_tangent: bool
    provides_free_energy: bool
    provides_dissipation: bool
    supports_autodiff_tangent: bool
    uses_strain_rate: bool
    time_representation: TimeRepresentation


@dataclass(frozen=True)
class SurrogateArtifactMetadata:
    """Store a versioned, backend-neutral surrogate artifact contract."""

    schema_version: str
    model_name: str
    model_family: str
    kind: SurrogateKind
    backend: str
    artifact_format: str
    dtype: str
    device: str
    kinematics: tuple[str, ...]
    mandel_convention: str
    input_fields: tuple[TensorFieldSpec, ...]
    output_fields: tuple[TensorFieldSpec, ...]
    state_fields: tuple[TensorFieldSpec, ...]
    capabilities: SurrogateCapabilities
    normalization: dict[str, dict[str, tuple[float, ...]]] = field(default_factory=dict)
    framework_version: str = ""


class MaterialPointSurrogate(Protocol):
    """Define a local constitutive surrogate without global FEM access."""

    name: str
    density: float
    metadata: SurrogateArtifactMetadata

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize local surrogate states."""
        ...

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Evaluate one local constitutive update."""
        ...


@dataclass(frozen=True)
class ResidualCorrectionRequest:
    """Store inputs for a solver-residual correction surrogate."""

    base_residual: NDArray[np.float64]
    state_features: NDArray[np.float64]
    control_parameters: NDArray[np.float64]


@dataclass(frozen=True)
class ResidualCorrectionResponse:
    """Store an additive residual correction and optional Jacobian correction."""

    residual_correction: NDArray[np.float64]
    jacobian_correction: NDArray[np.float64] | None
    diagnostics: dict[str, NDArray[np.float64]] = field(default_factory=dict)


class ResidualCorrectorSurrogate(Protocol):
    """Define an INC-style additive residual correction interface."""

    name: str
    metadata: SurrogateArtifactMetadata

    def evaluate(self, request: ResidualCorrectionRequest) -> ResidualCorrectionResponse:
        """Evaluate an additive correction to a declared solver residual."""
        ...


@dataclass(frozen=True)
class VelocityVerletCorrectorMetadata:
    """Store the runtime contract for a two-stage velocity-Verlet corrector."""

    schema_version: str
    model_name: str
    model_family: str
    time_step: float
    cfl_ratio: float
    output_semantics: str
    stage_order: tuple[str, str]
    dtype: str
    device: str
    control_parameter_order: tuple[str, ...]
    mesh_sha256: str
    lumped_mass_sha256: str
    material_config_sha256: str
    boundary_config_sha256: str
    free_dof_order_sha256: str


@dataclass(frozen=True)
class VelocityVerletCorrectionRequest:
    """Store one committed-state request for two-stage explicit correction."""

    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    baseline_acceleration: NDArray[np.float64]
    internal_force_start: NDArray[np.float64]
    external_force_start: NDArray[np.float64]
    external_force_end: NDArray[np.float64]
    free_dofs: NDArray[np.int64]
    control_parameters: NDArray[np.float64]
    time: float
    time_step: float


@dataclass(frozen=True)
class VelocityVerletCorrectionResponse:
    """Store free-DOF residual forces for both velocity-Verlet stages."""

    residual_force_start_free: NDArray[np.float64]
    residual_force_end_free: NDArray[np.float64]
    diagnostics: dict[str, NDArray[np.float64]] = field(default_factory=dict)


class VelocityVerletResidualCorrector(Protocol):
    """Define a two-stage residual corrector for explicit velocity-Verlet steps."""

    name: str
    metadata: VelocityVerletCorrectorMetadata

    def evaluate(
        self,
        request: VelocityVerletCorrectionRequest,
    ) -> VelocityVerletCorrectionResponse:
        """Evaluate start- and end-stage left residual forces."""
        ...
