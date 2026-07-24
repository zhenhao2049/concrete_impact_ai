"""Fixed-topology phased material adapter for heterogeneous RVEs.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from fem.materials.linear_elastic import LinearElasticMaterial


@dataclass(frozen=True)
class RVEPhase:
    """Define one named material phase."""

    name: str
    material: MaterialModel


@dataclass(frozen=True)
class PhasedMaterialModel:
    """Dispatch fixed integration-point subsets to phase material models."""

    name: str
    phases: tuple[RVEPhase, ...]
    point_phase_ids: NDArray[np.int64]
    effective_density: float

    @property
    def density(self) -> float:
        """Return the volume-averaged density supplied by the RVE builder."""
        return self.effective_density

    @property
    def maximum_wave_speed(self) -> float:
        """Return the maximum phase longitudinal-wave-speed bound."""
        return max(phase.material.maximum_wave_speed for phase in self.phases)

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize phase-local state arrays without fictitious variables."""
        if n_points != self.point_phase_ids.size:
            raise ValueError("Phased material point count does not match its phase map.")
        variables: dict[str, NDArray[np.float64]] = {}
        for phase_id, phase in enumerate(self.phases):
            phase_count = int(np.count_nonzero(self.point_phase_ids == phase_id))
            phase_state = phase.material.initialize_state(phase_count)
            for name, values in phase_state.variables.items():
                variables[encode_phase_state_key(phase.name, name)] = values

        return MaterialState(variables)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update every phase and scatter responses to the common point ordering."""
        point_count, component_count = request.strains.shape
        if point_count != self.point_phase_ids.size:
            raise ValueError("Phased material request does not match its fixed phase map.")
        stresses = np.zeros((point_count, component_count), dtype=np.float64)
        tangents = (
            np.zeros((point_count, component_count, component_count), dtype=np.float64)
            if requirements.tangent
            else None
        )
        free_energy = np.zeros(point_count, dtype=np.float64) if requirements.free_energy else None
        dissipation = np.zeros(point_count, dtype=np.float64) if requirements.dissipation else None
        next_variables: dict[str, NDArray[np.float64]] = {}
        phase_diagnostics: list[tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]] = []

        for phase_id, phase in enumerate(self.phases):
            point_ids = np.flatnonzero(self.point_phase_ids == phase_id)
            phase_state = _decode_phase_state(state, phase.name)
            phase_request = MaterialPointRequest(
                strains=request.strains[point_ids],
                strain_rates=(
                    None if request.strain_rates is None else request.strain_rates[point_ids]
                ),
                time_step=request.time_step,
                kinematics=request.kinematics,
                update_settings=request.update_settings,
            )
            response = phase.material.update(phase_request, phase_state, requirements)
            stresses[point_ids] = response.stresses
            if tangents is not None:
                if response.tangents is None:
                    raise ValueError("Phased material requested a tangent but received None.")
                tangents[point_ids] = response.tangents
            if free_energy is not None:
                if response.free_energy is None:
                    raise ValueError("Phased material requested free energy but received None.")
                free_energy[point_ids] = response.free_energy
            if dissipation is not None:
                if response.dissipation is None:
                    raise ValueError("Phased material requested dissipation but received None.")
                dissipation[point_ids] = response.dissipation
            for name, values in response.state.variables.items():
                next_variables[encode_phase_state_key(phase.name, name)] = values
            diagnostics = dict(response.diagnostics)
            if isinstance(phase.material, LinearElasticMaterial):
                _append_linear_increment_diagnostics(
                    diagnostics,
                    phase_request,
                    response,
                )
            phase_diagnostics.append((point_ids, diagnostics))

        diagnostic_names = sorted(
            set().union(*(diagnostics.keys() for _, diagnostics in phase_diagnostics))
        )
        combined_diagnostics = {
            name: np.full(point_count, np.nan, dtype=np.float64)
            for name in diagnostic_names
        }
        for point_ids, diagnostics in phase_diagnostics:
            for name, values in diagnostics.items():
                combined_diagnostics[name][point_ids] = values

        return MaterialPointResponse(
            stresses=stresses,
            state=MaterialState(next_variables),
            tangents=tangents,
            free_energy=free_energy,
            dissipation=dissipation,
            diagnostics=combined_diagnostics,
        )


def _decode_phase_state(state: MaterialState, phase_name: str) -> MaterialState:
    """Extract one phase's real state variables from a prefixed collection."""
    prefix = f"phase_{phase_name}__"
    return MaterialState(
        {
            name.removeprefix(prefix): values
            for name, values in state.variables.items()
            if name.startswith(prefix)
        }
    )


def encode_phase_state_key(phase_name: str, variable_name: str) -> str:
    """Build the stable phase namespace key with a double-underscore separator."""
    return f"phase_{phase_name}__{variable_name}"


def decode_phase_state_key(key: str) -> tuple[str, str]:
    """Decode one stable phase namespace key or reject an incompatible name."""
    prefix, variable_name = key.split("__", maxsplit=1)
    if not prefix.startswith("phase_"):
        raise ValueError(f"Invalid phased material state key: {key}.")
    return prefix.removeprefix("phase_"), variable_name


def _append_linear_increment_diagnostics(
    diagnostics: dict[str, NDArray[np.float64]],
    request: MaterialPointRequest,
    response: MaterialPointResponse,
) -> None:
    """Add work and stress-change diagnostics for an elastic phase."""
    if request.strain_rates is None or response.tangents is None:
        raise ValueError("Elastic RVE phase diagnostics require strain rates and tangents.")
    strain_increment = request.time_step * request.strain_rates
    stress_increment = np.einsum("qij,qj->qi", response.tangents, strain_increment)
    previous_stress = response.stresses - stress_increment
    diagnostics["incremental_work_density"] = 0.5 * np.einsum(
        "qi,qi->q",
        previous_stress + response.stresses,
        strain_increment,
    )
    diagnostics["equivalent_stress_increment"] = (
        _equivalent_stress(response.stresses) - _equivalent_stress(previous_stress)
    )
    diagnostics["viscoplastic_active"] = np.zeros(response.stresses.shape[0])


def _equivalent_stress(stresses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the three-dimensional J2 equivalent stress for each row."""
    deviatoric = stresses.copy()
    pressure = np.mean(stresses[:, :3], axis=1)
    deviatoric[:, :3] -= pressure[:, None]
    weights = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    return np.sqrt(1.5 * np.einsum("qi,i,qi->q", deviatoric, weights, deviatoric))
