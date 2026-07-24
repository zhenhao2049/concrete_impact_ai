"""Material point data containers and model protocol.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MaterialUpdateSettings:
    """Store local constitutive integration settings."""

    max_iterations: int
    yield_relative_tolerance: float
    residual_absolute_tolerance: float
    residual_relative_tolerance: float


@dataclass(frozen=True)
class MaterialState:
    """Store integration-point internal variables."""

    variables: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class MaterialResponseRequirements:
    """Declare optional constitutive outputs required by one caller."""

    tangent: bool = False
    free_energy: bool = False
    dissipation: bool = False


DEFAULT_RESPONSE_REQUIREMENTS = MaterialResponseRequirements()


@dataclass(frozen=True)
class MaterialPointRequest:
    """Store the input path data for one material-point update."""

    strains: NDArray[np.float64]
    strain_rates: NDArray[np.float64] | None
    time_step: float
    kinematics: str
    update_settings: MaterialUpdateSettings


@dataclass(frozen=True)
class MaterialPointResponse:
    """Store the output data from one material-point update."""

    stresses: NDArray[np.float64]
    state: MaterialState
    tangents: NDArray[np.float64] | None = None
    free_energy: NDArray[np.float64] | None = None
    dissipation: NDArray[np.float64] | None = None
    diagnostics: dict[str, NDArray[np.float64]] = field(default_factory=dict)


class MaterialModel(Protocol):
    """Define the local material update protocol."""

    @property
    def name(self) -> str:
        """Return the material model name."""
        ...

    @property
    def density(self) -> float:
        """Return the material mass density."""
        ...

    @property
    def maximum_wave_speed(self) -> float:
        """Return a conservative longitudinal-wave-speed bound."""
        ...

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize integration-point state variables."""
        ...

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update stresses and material state at integration points."""
        ...


def compute_isotropic_pressure_wave_speed(
    young_modulus: float,
    poisson_ratio: float,
    density: float,
) -> float:
    """Compute the isotropic three-dimensional longitudinal wave speed."""
    bulk_modulus = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    return float(np.sqrt((bulk_modulus + 4.0 * shear_modulus / 3.0) / density))
