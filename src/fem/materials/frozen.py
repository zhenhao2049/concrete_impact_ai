"""Frozen-history constitutive evaluation for RVE elastic localization tests.

Author:
    Zhen Hao.
Created:
    2026-07-22.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialModel,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)
from fem.materials.linear_elastic import (
    LinearElasticMaterial,
    build_elasticity_matrix,
    evaluate_linear_elastic_points,
)
from fem.materials.phased import PhasedMaterialModel, RVEPhase
from fem.materials.plasticity import J2ViscoplasticMaterial

STRESS_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)


@dataclass(frozen=True)
class FrozenHistoryMaterial:
    """Evaluate stress with immutable constitutive history variables."""

    base_material: LinearElasticMaterial | J2ViscoplasticMaterial

    @property
    def name(self) -> str:
        """Return the explicit frozen material name."""
        return f"frozen_history__{self.base_material.name}"

    @property
    def density(self) -> float:
        """Return the base material density."""
        return self.base_material.density

    @property
    def maximum_wave_speed(self) -> float:
        """Return the base elastic wave-speed bound."""
        return self.base_material.maximum_wave_speed

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize the exact base-material state layout."""
        return self.base_material.initialize_state(n_points)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Evaluate the registered frozen constitutive law."""
        evaluator: Any = FROZEN_HISTORY_EVALUATORS[type(self.base_material)]
        return evaluator(self.base_material, request, state, requirements)


def build_frozen_history_material(material: MaterialModel) -> MaterialModel:
    """Recursively freeze every supported phase in one material model."""
    if isinstance(material, PhasedMaterialModel):
        phases = tuple(
            RVEPhase(phase.name, build_frozen_history_material(phase.material))
            for phase in material.phases
        )
        return PhasedMaterialModel(
            name=f"frozen_history__{material.name}",
            phases=phases,
            point_phase_ids=material.point_phase_ids.copy(),
            effective_density=material.effective_density,
        )
    if type(material) not in FROZEN_HISTORY_EVALUATORS:
        raise TypeError(f"Unsupported frozen-history material: {type(material).__name__}.")
    supported = cast(LinearElasticMaterial | J2ViscoplasticMaterial, material)
    return FrozenHistoryMaterial(supported)


def _evaluate_frozen_linear_elastic(
    material: LinearElasticMaterial,
    request: MaterialPointRequest,
    state: MaterialState,
    requirements: MaterialResponseRequirements,
) -> MaterialPointResponse:
    """Evaluate a linear phase and append RVE path diagnostics."""
    internal_requirements = MaterialResponseRequirements(
        tangent=True,
        free_energy=requirements.free_energy,
        dissipation=requirements.dissipation,
    )
    response = evaluate_linear_elastic_points(
        request, material, state, internal_requirements
    )
    diagnostics = _increment_diagnostics(request, response.stresses, response.tangents)
    return MaterialPointResponse(
        stresses=response.stresses,
        state=_copy_state(state),
        tangents=response.tangents if requirements.tangent else None,
        free_energy=response.free_energy,
        dissipation=response.dissipation,
        diagnostics=diagnostics,
    )


def _evaluate_frozen_j2_viscoplastic(
    material: J2ViscoplasticMaterial,
    request: MaterialPointRequest,
    state: MaterialState,
    requirements: MaterialResponseRequirements,
) -> MaterialPointResponse:
    """Evaluate J2 stress with plastic strain and hardening state fixed."""
    if request.kinematics != "three_dimensional":
        raise ValueError("Frozen J2 evaluation requires three_dimensional kinematics.")
    if request.strains.ndim != 2 or request.strains.shape[1] != 6:
        raise ValueError("Frozen J2 strains must have shape (n_points, 6).")
    plastic_strain = state.variables["plastic_strain"]
    equivalent_plastic_strain = state.variables["equivalent_plastic_strain"]
    point_count = request.strains.shape[0]
    if plastic_strain.shape != request.strains.shape:
        raise ValueError("Frozen J2 plastic strain does not match the request shape.")
    if equivalent_plastic_strain.shape != (point_count,):
        raise ValueError("Frozen J2 equivalent plastic strain has an incompatible shape.")

    elastic_material = LinearElasticMaterial(
        material.name,
        material.density,
        material.young_modulus,
        material.poisson_ratio,
    )
    elasticity = build_elasticity_matrix(elastic_material, 3, "three_dimensional")
    elastic_strain = request.strains - plastic_strain
    stresses = elastic_strain @ elasticity.T
    diagnostic_tangents = np.repeat(elasticity[None, :, :], point_count, axis=0)
    tangents = diagnostic_tangents if requirements.tangent else None
    free_energy = None
    if requirements.free_energy:
        free_energy = 0.5 * np.einsum("qi,qi->q", stresses, elastic_strain)
        free_energy += 0.5 * material.hardening_modulus * equivalent_plastic_strain**2
    dissipation = None
    if requirements.dissipation:
        dissipation = np.zeros(point_count, dtype=np.float64)

    diagnostics = _increment_diagnostics(request, stresses, diagnostic_tangents)
    equivalent_stress = _equivalent_stress(stresses)
    yield_radius = material.yield_stress + (
        material.hardening_modulus * equivalent_plastic_strain
    )
    diagnostics.update(
        {
            "equivalent_stress": equivalent_stress,
            "equivalent_plastic_strain": equivalent_plastic_strain.copy(),
            "yield_activation_margin": equivalent_stress - yield_radius,
            "viscoplastic_active": np.zeros(point_count, dtype=np.float64),
        }
    )
    return MaterialPointResponse(
        stresses=stresses,
        state=_copy_state(state),
        tangents=tangents,
        free_energy=free_energy,
        dissipation=dissipation,
        diagnostics=diagnostics,
    )


def _increment_diagnostics(
    request: MaterialPointRequest,
    stresses: NDArray[np.float64],
    tangents: NDArray[np.float64] | None,
) -> dict[str, NDArray[np.float64]]:
    """Compute signed work and stress-change diagnostics for frozen updates."""
    if request.strain_rates is None or tangents is None:
        raise ValueError("Frozen RVE diagnostics require strain rates and tangents.")
    strain_increment = request.time_step * request.strain_rates
    stress_increment = np.einsum("qij,qj->qi", tangents, strain_increment)
    previous_stress = stresses - stress_increment
    return {
        "incremental_work_density": 0.5
        * np.einsum("qi,qi->q", previous_stress + stresses, strain_increment),
        "equivalent_stress_increment": (
            _equivalent_stress(stresses) - _equivalent_stress(previous_stress)
        ),
        "viscoplastic_active": np.zeros(stresses.shape[0], dtype=np.float64),
    }


def _equivalent_stress(stresses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the J2 equivalent stress for engineering-Voigt rows."""
    deviatoric = stresses.copy()
    mean_stress = np.mean(stresses[:, :3], axis=1)
    deviatoric[:, :3] -= mean_stress[:, None]
    return np.sqrt(
        1.5 * np.einsum("qi,i,qi->q", deviatoric, STRESS_WEIGHTS, deviatoric)
    )


def _copy_state(state: MaterialState) -> MaterialState:
    """Copy every state array without altering any stored value."""
    return MaterialState({name: values.copy() for name, values in state.variables.items()})


FROZEN_HISTORY_EVALUATORS = {
    LinearElasticMaterial: _evaluate_frozen_linear_elastic,
    J2ViscoplasticMaterial: _evaluate_frozen_j2_viscoplastic,
}
