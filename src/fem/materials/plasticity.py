"""Rate-independent plastic material models.

Contents:
    J2 materials, state updates, multiplier solves, diagnostics, and consistent tangents.
Author:
    Zhen Hao.
Created:
    2026-07-08.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
    compute_isotropic_pressure_wave_speed,
)
from fem.materials.errors import MaterialPointConvergenceError

VOIGT_DIM = 6
IDENTITY = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
STRESS_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
TENSOR_TO_ENGINEERING_STRAIN = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
ENGINEERING_STRAIN_TO_TENSOR = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
VOLUMETRIC_PROJECTION = np.outer(IDENTITY, IDENTITY) / 3.0
DEVIATORIC_PROJECTION = np.eye(VOIGT_DIM, dtype=np.float64) - VOLUMETRIC_PROJECTION
LOCAL_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class J2PlasticMaterial:
    """Store J2 plasticity parameters."""

    name: str
    density: float
    young_modulus: float
    poisson_ratio: float
    yield_stress: float
    hardening_modulus: float
    kinematic_fraction: float

    @property
    def maximum_wave_speed(self) -> float:
        """Return the elastic longitudinal wave-speed bound."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize J2 internal variables."""
        return MaterialState(
            variables={
                "plastic_strain": np.zeros((n_points, VOIGT_DIM), dtype=np.float64),
                "equivalent_plastic_strain": np.zeros(n_points, dtype=np.float64),
                "back_stress": np.zeros((n_points, VOIGT_DIM), dtype=np.float64),
            },
        )

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update J2 stress and history variables."""
        return _update_points(self, request, state, _update_j2_point, requirements)


@dataclass(frozen=True)
class J2ViscoplasticMaterial:
    """Store rate-dependent J2 viscoplasticity parameters."""

    name: str
    density: float
    young_modulus: float
    poisson_ratio: float
    yield_stress: float
    hardening_modulus: float
    time_scale: float
    reference_stress: float
    rate_exponent: float

    @property
    def maximum_wave_speed(self) -> float:
        """Return the elastic longitudinal wave-speed bound."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize J2 viscoplastic internal variables."""
        return MaterialState(
            variables={
                "plastic_strain": np.zeros((n_points, VOIGT_DIM), dtype=np.float64),
                "equivalent_plastic_strain": np.zeros(n_points, dtype=np.float64),
                "viscoplastic_multiplier": np.zeros(n_points, dtype=np.float64),
            },
        )

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update rate-dependent J2 stress and history variables."""
        return _update_points(self, request, state, _update_j2_viscoplastic_point, requirements)


@dataclass(frozen=True)
class DruckerPragerMaterial:
    """Store Drucker-Prager plasticity parameters."""

    name: str
    density: float
    young_modulus: float
    poisson_ratio: float
    friction_parameter: float
    cohesion: float
    flow_parameter: float
    cohesion_hardening: float
    kappa_rate: float

    @property
    def maximum_wave_speed(self) -> float:
        """Return the elastic longitudinal wave-speed bound."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize Drucker-Prager internal variables."""
        return _initialize_pressure_plastic_state(n_points, include_cap=False)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update Drucker-Prager stress and history variables."""
        return _update_points(self, request, state, _update_drucker_prager_point, requirements)


@dataclass(frozen=True)
class DruckerPragerCapMaterial:
    """Store Drucker-Prager/Cap plasticity parameters."""

    name: str
    density: float
    young_modulus: float
    poisson_ratio: float
    friction_parameter: float
    cohesion: float
    flow_parameter: float
    cohesion_hardening: float
    kappa_rate: float
    initial_cap_apex_pressure: float
    cap_transition_ratio: float
    cap_hardening: float
    max_local_iterations: int
    local_tolerance: float

    @property
    def maximum_wave_speed(self) -> float:
        """Return the elastic longitudinal wave-speed bound."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize Drucker-Prager/Cap internal variables."""
        state = _initialize_pressure_plastic_state(n_points, include_cap=True)
        state.variables["cap_apex_pressure"][:] = self.initial_cap_apex_pressure

        return state

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update Drucker-Prager/Cap stress and history variables."""
        return _update_points(self, request, state, _update_drucker_prager_cap_point, requirements)


@dataclass(frozen=True)
class DruckerPragerCapGeometry:
    """Store the C1 composite DP-Cap geometry in the p-q plane."""

    apex_pressure: float
    transition_pressure: float
    transition_q: float
    center_pressure: float
    axis_pressure: float
    axis_deviatoric: float


def build_j2_plastic_material(material_spec: dict[str, object]) -> J2PlasticMaterial:
    """Build a J2 plastic material from a dictionary."""
    return J2PlasticMaterial(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
        yield_stress=float(material_spec["yield_stress"]),
        hardening_modulus=float(material_spec["hardening_modulus"]),
        kinematic_fraction=float(material_spec["kinematic_fraction"]),
    )


def build_j2_viscoplastic_material(material_spec: dict[str, object]) -> J2ViscoplasticMaterial:
    """Build a rate-dependent J2 material from a dictionary."""
    material = J2ViscoplasticMaterial(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
        yield_stress=float(material_spec["yield_stress"]),
        hardening_modulus=float(material_spec["hardening_modulus"]),
        time_scale=float(material_spec["time_scale"]),
        reference_stress=float(material_spec["reference_stress"]),
        rate_exponent=float(material_spec["rate_exponent"]),
    )
    _validate_j2_viscoplastic_material(material)

    return material


def _validate_j2_viscoplastic_material(material: J2ViscoplasticMaterial) -> None:
    """Validate physical parameters required by the J2 evolution problem."""
    if material.young_modulus <= 0.0:
        raise ValueError("J2 viscoplasticity requires young_modulus > 0.")
    if not -1.0 < material.poisson_ratio < 0.5:
        raise ValueError("J2 viscoplasticity requires -1 < poisson_ratio < 0.5.")
    if material.yield_stress <= 0.0:
        raise ValueError("J2 viscoplasticity requires yield_stress > 0.")
    if material.hardening_modulus < 0.0:
        raise ValueError("J2 viscoplasticity requires hardening_modulus >= 0.")
    if material.time_scale <= 0.0:
        raise ValueError("J2 viscoplasticity requires time_scale > 0.")
    if material.reference_stress <= 0.0:
        raise ValueError("J2 viscoplasticity requires reference_stress > 0.")
    if material.rate_exponent < 1.0:
        raise ValueError("J2 viscoplasticity requires rate_exponent >= 1.")


def build_drucker_prager_material(
    material_spec: dict[str, object],
) -> DruckerPragerMaterial:
    """Build a Drucker-Prager material from a dictionary."""
    return DruckerPragerMaterial(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
        friction_parameter=float(material_spec["friction_parameter"]),
        cohesion=float(material_spec["cohesion"]),
        flow_parameter=float(material_spec["flow_parameter"]),
        cohesion_hardening=float(material_spec["cohesion_hardening"]),
        kappa_rate=float(material_spec["kappa_rate"]),
    )


def build_drucker_prager_cap_material(
    material_spec: dict[str, object],
) -> DruckerPragerCapMaterial:
    """Build a Drucker-Prager/Cap material from a dictionary."""
    return DruckerPragerCapMaterial(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
        friction_parameter=float(material_spec["friction_parameter"]),
        cohesion=float(material_spec["cohesion"]),
        flow_parameter=float(material_spec["flow_parameter"]),
        cohesion_hardening=float(material_spec["cohesion_hardening"]),
        kappa_rate=float(material_spec["kappa_rate"]),
        initial_cap_apex_pressure=float(material_spec["initial_cap_apex_pressure"]),
        cap_transition_ratio=float(material_spec["cap_transition_ratio"]),
        cap_hardening=float(material_spec["cap_hardening"]),
        max_local_iterations=int(material_spec["max_local_iterations"]),
        local_tolerance=float(material_spec["local_tolerance"]),
    )


def expand_plane_strain_to_3d(strains: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expand plane-strain vectors to six-component vectors."""
    expanded = np.zeros((strains.shape[0], VOIGT_DIM), dtype=np.float64)
    expanded[:, 0] = strains[:, 0]
    expanded[:, 1] = strains[:, 1]
    expanded[:, 5] = strains[:, 2]

    return expanded


def reduce_3d_stress_to_plane_strain(stresses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reduce six-component stresses to plane-strain assembly components."""
    reduced = np.zeros((stresses.shape[0], 3), dtype=np.float64)
    reduced[:, 0] = stresses[:, 0]
    reduced[:, 1] = stresses[:, 1]
    reduced[:, 2] = stresses[:, 5]

    return reduced


def compute_stress_invariants(
    stress: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    """Compute compression-positive pressure, q, and deviatoric stress."""
    pressure = -float(np.sum(stress[:3])) / 3.0
    deviatoric = _deviatoric(stress)
    q_value = _compute_j2_equivalent_stress(deviatoric)

    return pressure, q_value, deviatoric


def _initialize_pressure_plastic_state(n_points: int, include_cap: bool) -> MaterialState:
    """Initialize pressure-sensitive plastic internal variables."""
    variables = {
        "plastic_strain": np.zeros((n_points, VOIGT_DIM), dtype=np.float64),
        "kappa": np.zeros(n_points, dtype=np.float64),
        "plastic_volumetric_strain": np.zeros(n_points, dtype=np.float64),
        "plastic_multiplier": np.zeros(n_points, dtype=np.float64),
    }
    if include_cap:
        variables["cap_apex_pressure"] = np.zeros(n_points, dtype=np.float64)

    return MaterialState(variables=variables)


def _update_points(
    material: Any,
    request: MaterialPointRequest,
    state: MaterialState,
    update_point,
    requirements: MaterialResponseRequirements,
) -> MaterialPointResponse:
    """Update all material points in one batch."""
    if request.kinematics != "three_dimensional":
        raise ValueError("Plastic material updates currently require three_dimensional kinematics.")
    strains = _require_six_component_strains(request.strains)
    n_points = strains.shape[0]
    elasticity = _elasticity_matrix(material.young_modulus, material.poisson_ratio)
    next_variables = _copy_state_variables(state.variables)

    stresses = np.zeros((n_points, VOIGT_DIM), dtype=np.float64)
    tangents = np.zeros((n_points, VOIGT_DIM, VOIGT_DIM), dtype=np.float64)
    free_energy = np.zeros(n_points, dtype=np.float64)
    dissipation = np.zeros(n_points, dtype=np.float64)
    diagnostics: dict[str, NDArray[np.float64]] = {}

    for point_id in range(n_points):
        point_state = _slice_state(state, point_id)
        try:
            point_result = update_point(
                material,
                strains[point_id],
                point_state,
                elasticity,
                request.time_step,
                request.update_settings,
            )
        except MaterialPointConvergenceError as error:
            raise error.with_context(
                material_name=material.name,
                material_type=type(material).__name__,
                point_id=point_id,
                time_step=request.time_step,
                kinematics=request.kinematics,
                strain=strains[point_id].tolist(),
            ) from error
        stresses[point_id] = point_result["stress"]
        if requirements.tangent and "tangent" in point_result:
            tangents[point_id] = point_result["tangent"]
        elif requirements.tangent:
            try:
                tangents[point_id] = _numerical_tangent(
                    material,
                    strains[point_id],
                    point_state,
                    update_point,
                    request.time_step,
                    request.update_settings,
                )
            except MaterialPointConvergenceError as error:
                raise error.with_context(
                    material_name=material.name,
                    material_type=type(material).__name__,
                    point_id=point_id,
                    time_step=request.time_step,
                    kinematics=request.kinematics,
                    strain=strains[point_id].tolist(),
                    evaluation="numerical_tangent",
                ) from error
        free_energy[point_id] = point_result["free_energy"]
        dissipation[point_id] = point_result["dissipation"]
        if isinstance(material, J2ViscoplasticMaterial):
            _append_j2_increment_diagnostics(
                point_result["diagnostics"],
                strains[point_id],
                None if request.strain_rates is None else request.strain_rates[point_id],
                point_state,
                elasticity,
                request.time_step,
                point_result["stress"],
            )
        _assign_state(next_variables, point_id, point_result["state"].variables)
        _append_diagnostics(diagnostics, point_id, n_points, point_result["diagnostics"])

    updated_state = MaterialState(next_variables)

    return MaterialPointResponse(
        stresses=stresses,
        state=updated_state,
        tangents=tangents if requirements.tangent else None,
        free_energy=free_energy if requirements.free_energy else None,
        dissipation=dissipation if requirements.dissipation else None,
        diagnostics=diagnostics,
    )


def _update_j2_point(
    material: J2PlasticMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> dict[str, Any]:
    """Update one J2 material point."""
    del time_step
    del update_settings
    shear_modulus = _shear_modulus(material.young_modulus, material.poisson_ratio)
    beta = material.kinematic_fraction
    plastic_strain = state.variables["plastic_strain"][0]
    equivalent_plastic_strain = float(state.variables["equivalent_plastic_strain"][0])
    back_stress = state.variables["back_stress"][0]

    trial_stress = elasticity @ (strain - plastic_strain)
    trial_relative = _deviatoric(trial_stress) - _deviatoric(back_stress)
    trial_equivalent = _compute_j2_equivalent_stress(trial_relative)
    yield_radius = (
        material.yield_stress
        + (1.0 - beta) * material.hardening_modulus * equivalent_plastic_strain
    )
    yield_value = trial_equivalent - yield_radius

    if yield_value <= LOCAL_TOLERANCE:
        next_state = MaterialState(
            variables={
                "plastic_strain": plastic_strain.reshape(1, VOIGT_DIM),
                "equivalent_plastic_strain": np.asarray([equivalent_plastic_strain]),
                "back_stress": back_stress.reshape(1, VOIGT_DIM),
            },
        )
        return _point_payload(
            trial_stress,
            next_state,
            strain,
            elasticity,
            0.0,
            yield_value,
            0,
            "elastic",
        )

    flow_direction = 1.5 * trial_relative / trial_equivalent
    denominator = 3.0 * shear_modulus + material.hardening_modulus
    delta_gamma = yield_value / denominator
    stress = trial_stress - 2.0 * shear_modulus * delta_gamma * flow_direction
    plastic_increment = delta_gamma * TENSOR_TO_ENGINEERING_STRAIN @ flow_direction

    next_state = MaterialState(
        variables={
            "plastic_strain": (plastic_strain + plastic_increment).reshape(1, VOIGT_DIM),
            "equivalent_plastic_strain": np.asarray(
                [equivalent_plastic_strain + delta_gamma],
            ),
            "back_stress": (
                back_stress
                + (2.0 / 3.0)
                * beta
                * material.hardening_modulus
                * delta_gamma
                * flow_direction
            ).reshape(1, VOIGT_DIM),
        },
    )
    final_relative = _deviatoric(stress) - _deviatoric(next_state.variables["back_stress"][0])
    final_equivalent = _compute_j2_equivalent_stress(final_relative)
    final_yield = final_equivalent - (
        material.yield_stress
        + (1.0 - beta)
        * material.hardening_modulus
        * float(next_state.variables["equivalent_plastic_strain"][0])
    )
    plastic_work = _stress_strain_inner(stress, plastic_increment)
    payload = _point_payload(
        stress,
        next_state,
        strain,
        elasticity,
        plastic_work,
        final_yield,
        0,
        "plastic",
    )
    payload["diagnostics"]["plastic_multiplier"] = delta_gamma
    payload["diagnostics"]["equivalent_stress"] = final_equivalent

    return payload


def _update_j2_viscoplastic_point(
    material: J2ViscoplasticMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> dict[str, Any]:
    """Update one rate-dependent J2 material point."""
    shear_modulus = _shear_modulus(material.young_modulus, material.poisson_ratio)
    plastic_strain = state.variables["plastic_strain"][0]
    equivalent_plastic_strain = float(state.variables["equivalent_plastic_strain"][0])
    trial_stress = elasticity @ (strain - plastic_strain)
    trial_deviatoric = _deviatoric(trial_stress)
    trial_equivalent = _compute_j2_equivalent_stress(trial_deviatoric)
    yield_radius = material.yield_stress + material.hardening_modulus * equivalent_plastic_strain
    trial_yield = trial_equivalent - yield_radius
    yield_scale = max(material.yield_stress, material.reference_stress, trial_equivalent)
    yield_activation_threshold = update_settings.yield_relative_tolerance * yield_scale
    yield_activation_margin = (
        trial_yield - yield_activation_threshold
    ) / yield_scale
    if (
        not np.isfinite(update_settings.yield_relative_tolerance)
        or update_settings.yield_relative_tolerance < 0.0
        or not np.isfinite(yield_activation_threshold)
    ):
        raise MaterialPointConvergenceError(
            "J2 viscoplasticity received an invalid yield activation tolerance.",
            "invalid_yield_activation_tolerance",
            {
                "algorithm": "j2_viscoplastic_yield_activation",
                "yield_relative_tolerance": update_settings.yield_relative_tolerance,
                "yield_scale": yield_scale,
                "yield_activation_threshold": yield_activation_threshold,
                "trial_yield_value": trial_yield,
            },
        )

    if trial_yield <= yield_activation_threshold:
        next_variables = _copy_state_variables(state.variables)
        next_state = MaterialState(next_variables)
        payload = _point_payload(
            trial_stress,
            next_state,
            strain,
            elasticity,
            0.0,
            trial_yield,
            0,
            "elastic",
        )
        payload["diagnostics"]["plastic_multiplier"] = 0.0
        payload["diagnostics"]["viscoplastic_multiplier"] = 0.0
        payload["diagnostics"]["equivalent_stress"] = trial_equivalent
        payload["diagnostics"]["trial_yield_value"] = trial_yield
        payload["diagnostics"]["yield_scale"] = yield_scale
        payload["diagnostics"]["yield_activation_threshold"] = yield_activation_threshold
        payload["diagnostics"]["yield_activation_margin"] = yield_activation_margin
        payload["diagnostics"]["viscoplastic_active"] = 0.0
        payload["diagnostics"]["time_step"] = time_step
        payload["diagnostics"]["equivalent_plastic_strain"] = equivalent_plastic_strain
        payload["tangent"] = elasticity
        payload["free_energy"] += 0.5 * material.hardening_modulus * equivalent_plastic_strain**2
        payload["dissipation"] = 0.0
        payload["diagnostics"]["plastic_power"] = 0.0
        payload["diagnostics"]["hardening_storage_rate"] = 0.0
        payload["diagnostics"]["local_residual"] = 0.0
        payload["diagnostics"]["root_lower_bound"] = 0.0
        payload["diagnostics"]["root_upper_bound"] = 0.0

        return payload

    flow_direction = 1.5 * trial_deviatoric / trial_equivalent
    try:
        delta_gamma, local_iterations, local_residual = _solve_j2_viscoplastic_multiplier(
            material,
            trial_yield,
            shear_modulus,
            time_step,
            update_settings,
        )
    except MaterialPointConvergenceError as error:
        raise error.with_context(
            trial_equivalent_stress=trial_equivalent,
            yield_radius=yield_radius,
            equivalent_plastic_strain=equivalent_plastic_strain,
            trial_stress=trial_stress.tolist(),
        ) from error
    stress = trial_stress - 2.0 * shear_modulus * delta_gamma * flow_direction
    plastic_increment = delta_gamma * TENSOR_TO_ENGINEERING_STRAIN @ flow_direction

    next_variables = _copy_state_variables(state.variables)
    next_variables["plastic_strain"][0] += plastic_increment
    next_variables["equivalent_plastic_strain"][0] += delta_gamma
    next_variables["viscoplastic_multiplier"][0] += delta_gamma
    next_state = MaterialState(next_variables)

    final_deviatoric = _deviatoric(stress)
    final_equivalent = _compute_j2_equivalent_stress(final_deviatoric)
    final_yield = final_equivalent - (
        material.yield_stress
        + material.hardening_modulus * float(next_state.variables["equivalent_plastic_strain"][0])
    )
    plastic_work = _stress_strain_inner(stress, plastic_increment)
    payload = _point_payload(
        stress,
        next_state,
        strain,
        elasticity,
        plastic_work,
        final_yield,
        local_iterations,
        "viscoplastic",
    )
    payload["diagnostics"]["plastic_multiplier"] = delta_gamma
    payload["diagnostics"]["viscoplastic_multiplier"] = delta_gamma
    payload["diagnostics"]["equivalent_stress"] = final_equivalent
    payload["diagnostics"]["trial_yield_value"] = trial_yield
    payload["diagnostics"]["yield_scale"] = yield_scale
    payload["diagnostics"]["yield_activation_threshold"] = yield_activation_threshold
    payload["diagnostics"]["yield_activation_margin"] = yield_activation_margin
    payload["diagnostics"]["viscoplastic_active"] = 1.0
    payload["diagnostics"]["time_step"] = time_step
    payload["diagnostics"]["equivalent_plastic_strain"] = next_state.variables[
        "equivalent_plastic_strain"
    ][0]
    payload["diagnostics"]["viscoplastic_hv"] = _compute_j2_viscoplastic_hv(
        material,
        final_yield,
        shear_modulus,
        time_step,
    )
    payload["tangent"] = _j2_viscoplastic_consistent_tangent(
        material,
        flow_direction,
        trial_equivalent,
        final_yield,
        delta_gamma,
        shear_modulus,
        time_step,
    )
    next_q = float(next_state.variables["equivalent_plastic_strain"][0])
    plastic_power = plastic_work / time_step
    hardening_storage_rate = material.hardening_modulus * next_q * delta_gamma / time_step
    payload["free_energy"] += 0.5 * material.hardening_modulus * next_q**2
    payload["dissipation"] = plastic_power - hardening_storage_rate
    payload["diagnostics"]["plastic_power"] = plastic_power
    payload["diagnostics"]["hardening_storage_rate"] = hardening_storage_rate
    payload["diagnostics"]["local_residual"] = local_residual
    payload["diagnostics"]["root_lower_bound"] = 0.0
    payload["diagnostics"]["root_upper_bound"] = trial_yield / (
        3.0 * shear_modulus + material.hardening_modulus
    )

    return payload


def _append_j2_increment_diagnostics(
    diagnostics: dict[str, Any],
    strain: NDArray[np.float64],
    strain_rate: NDArray[np.float64] | None,
    state: MaterialState,
    elasticity: NDArray[np.float64],
    time_step: float,
    stress: NDArray[np.float64],
) -> None:
    """Append signed loading-path diagnostics to one J2 point result."""
    if strain_rate is None:
        return

    strain_increment = time_step * strain_rate
    previous_strain = strain - strain_increment
    previous_plastic_strain = state.variables["plastic_strain"][0]
    previous_stress = elasticity @ (previous_strain - previous_plastic_strain)
    previous_equivalent = _compute_j2_equivalent_stress(_deviatoric(previous_stress))
    current_equivalent = _compute_j2_equivalent_stress(_deviatoric(stress))

    diagnostics["incremental_work_density"] = 0.5 * _stress_strain_inner(
        previous_stress + stress,
        strain_increment,
    )
    diagnostics["equivalent_stress_increment"] = current_equivalent - previous_equivalent


def _solve_j2_viscoplastic_multiplier(
    material: J2ViscoplasticMaterial,
    trial_yield: float,
    shear_modulus: float,
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> tuple[float, int, float]:
    """Solve the scalar Perzyna viscoplastic multiplier."""
    hardening_slope = 3.0 * shear_modulus + material.hardening_modulus
    upper_bound = trial_yield / hardening_slope
    scale = upper_bound
    tolerance = max(
        update_settings.residual_absolute_tolerance,
        update_settings.residual_relative_tolerance * scale,
    )
    if material.rate_exponent == 1.0:
        denominator = material.time_scale * material.reference_stress / time_step + hardening_slope
        multiplier = trial_yield / denominator
        corrected_overstress = trial_yield - hardening_slope * multiplier
        residual = multiplier - time_step * corrected_overstress / (
            material.time_scale * material.reference_stress
        )
        derivative = 1.0 + time_step * hardening_slope / (
            material.time_scale * material.reference_stress
        )
        diagnostics = _build_j2_newton_diagnostics(
            material,
            trial_yield,
            shear_modulus,
            time_step,
            hardening_slope,
            upper_bound,
            tolerance,
            update_settings.max_iterations,
            0,
            multiplier,
            corrected_overstress,
            residual,
            derivative,
            multiplier,
        )
        if not all(
            np.isfinite(value)
            for value in (denominator, multiplier, corrected_overstress, residual, derivative)
        ):
            _raise_j2_newton_failure("non_finite_closed_form", diagnostics)
        if not 0.0 < multiplier < upper_bound or corrected_overstress <= 0.0:
            _raise_j2_newton_failure("closed_form_solution_out_of_bounds", diagnostics)

        return multiplier, 0, residual

    if update_settings.max_iterations <= 0:
        _raise_j2_newton_failure(
            "invalid_iteration_limit",
            {
                "algorithm": "j2_viscoplastic_local_newton",
                "max_iterations": update_settings.max_iterations,
            },
        )
    if not all(np.isfinite(value) for value in (upper_bound, tolerance)):
        _raise_j2_newton_failure(
            "non_finite_problem_scale",
            {
                "algorithm": "j2_viscoplastic_local_newton",
                "physical_upper_bound": upper_bound,
                "residual_tolerance": tolerance,
            },
        )

    multiplier = 0.5 * upper_bound
    for iteration in range(1, update_settings.max_iterations + 1):
        overstress = trial_yield - hardening_slope * multiplier
        normalized_overstress = overstress / material.reference_stress
        if not np.isfinite(normalized_overstress) or normalized_overstress <= 0.0:
            diagnostics = _build_j2_newton_diagnostics(
                material,
                trial_yield,
                shear_modulus,
                time_step,
                hardening_slope,
                upper_bound,
                tolerance,
                update_settings.max_iterations,
                iteration,
                multiplier,
                overstress,
                None,
                None,
                None,
            )
            _raise_j2_newton_failure("invalid_corrected_overstress", diagnostics)
        with np.errstate(over="ignore", invalid="ignore"):
            rate_value = float(
                np.power(normalized_overstress, material.rate_exponent)
                / material.time_scale
            )
        residual = multiplier - time_step * rate_value
        with np.errstate(over="ignore", invalid="ignore"):
            derivative = float(
                1.0
                + time_step
                * material.rate_exponent
                * hardening_slope
                * np.power(normalized_overstress, material.rate_exponent - 1.0)
                / (material.time_scale * material.reference_stress)
            )
        diagnostics = _build_j2_newton_diagnostics(
            material,
            trial_yield,
            shear_modulus,
            time_step,
            hardening_slope,
            upper_bound,
            tolerance,
            update_settings.max_iterations,
            iteration,
            multiplier,
            overstress,
            residual,
            derivative,
            None,
        )
        if not all(np.isfinite(value) for value in (rate_value, residual, derivative)):
            _raise_j2_newton_failure("non_finite_iteration_value", diagnostics)
        if derivative <= 0.0:
            _raise_j2_newton_failure("non_positive_derivative", diagnostics)
        if abs(residual) <= tolerance:
            if not 0.0 < multiplier < upper_bound or overstress <= 0.0:
                _raise_j2_newton_failure("converged_solution_out_of_bounds", diagnostics)

            return multiplier, iteration, residual

        newton_multiplier = multiplier - residual / derivative
        diagnostics["newton_candidate"] = newton_multiplier
        if not np.isfinite(newton_multiplier):
            _raise_j2_newton_failure("non_finite_newton_candidate", diagnostics)
        if not 0.0 < newton_multiplier < upper_bound:
            _raise_j2_newton_failure("newton_candidate_out_of_bounds", diagnostics)
        if iteration == update_settings.max_iterations:
            _raise_j2_newton_failure("maximum_iterations_exceeded", diagnostics)

        multiplier = newton_multiplier

    raise AssertionError("J2 local Newton loop terminated without a convergence decision.")


def _build_j2_newton_diagnostics(
    material: J2ViscoplasticMaterial,
    trial_yield: float,
    shear_modulus: float,
    time_step: float,
    hardening_slope: float,
    upper_bound: float,
    tolerance: float,
    max_iterations: int,
    iteration: int,
    multiplier: float,
    corrected_overstress: float,
    residual: float | None,
    derivative: float | None,
    newton_candidate: float | None,
) -> dict[str, Any]:
    """Build a serializable diagnostic snapshot for local J2 Newton."""
    return {
        "algorithm": "j2_viscoplastic_local_newton",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "time_step": time_step,
        "rate_exponent": material.rate_exponent,
        "time_scale": material.time_scale,
        "reference_stress": material.reference_stress,
        "yield_stress": material.yield_stress,
        "hardening_modulus": material.hardening_modulus,
        "shear_modulus": shear_modulus,
        "hardening_slope": hardening_slope,
        "trial_yield": trial_yield,
        "physical_lower_bound": 0.0,
        "physical_upper_bound": upper_bound,
        "plastic_multiplier": multiplier,
        "corrected_overstress": corrected_overstress,
        "residual": residual,
        "residual_tolerance": tolerance,
        "derivative": derivative,
        "newton_candidate": newton_candidate,
    }


def _raise_j2_newton_failure(reason: str, diagnostics: dict[str, Any]) -> None:
    """Raise one structured J2 local-Newton failure."""
    raise MaterialPointConvergenceError(
        f"J2 viscoplastic local Newton failed: {reason}.",
        reason,
        diagnostics,
    )


def _compute_j2_viscoplastic_hv(
    material: J2ViscoplasticMaterial,
    final_yield: float,
    shear_modulus: float,
    time_step: float,
) -> float:
    """Compute the scalar local modulus for implicit J2 viscoplastic flow."""
    hardening_slope = 3.0 * shear_modulus + material.hardening_modulus
    if material.rate_exponent == 1.0:
        return material.time_scale * material.reference_stress / time_step + hardening_slope

    if final_yield <= 0.0:
        raise RuntimeError("J2 viscoplastic tangent requires positive corrected overstress.")
    flow_derivative = (
        material.rate_exponent
        * (final_yield / material.reference_stress) ** (material.rate_exponent - 1.0)
        / (material.time_scale * material.reference_stress)
    )

    return 1.0 / (time_step * flow_derivative) + hardening_slope


def _j2_viscoplastic_consistent_tangent(
    material: J2ViscoplasticMaterial,
    flow_direction: NDArray[np.float64],
    trial_equivalent: float,
    final_yield: float,
    delta_gamma: float,
    shear_modulus: float,
    time_step: float,
) -> NDArray[np.float64]:
    """Build the analytic consistent tangent for implicit J2 viscoplasticity."""
    bulk_modulus = _bulk_modulus(material.young_modulus, material.poisson_ratio)
    local_modulus = _compute_j2_viscoplastic_hv(
        material,
        final_yield,
        shear_modulus,
        time_step,
    )
    volumetric_tangent = 3.0 * bulk_modulus * ENGINEERING_STRAIN_TO_TENSOR @ (
        VOLUMETRIC_PROJECTION
    )
    deviatoric_factor = (
        2.0 * shear_modulus
        - 6.0 * shear_modulus**2 * delta_gamma / trial_equivalent
    )
    deviatoric_tangent = deviatoric_factor * ENGINEERING_STRAIN_TO_TENSOR @ (
        DEVIATORIC_PROJECTION
    )
    radial_tangent = (
        4.0
        * shear_modulus**2
        * (delta_gamma / trial_equivalent - 1.0 / local_modulus)
        * np.outer(flow_direction, flow_direction)
    )

    return volumetric_tangent + deviatoric_tangent + radial_tangent


def _update_drucker_prager_point(
    material: DruckerPragerMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> dict[str, Any]:
    """Update one Drucker-Prager material point."""
    del time_step
    del update_settings
    return _update_drucker_prager_shear_point(material, strain, state, elasticity)


def _update_drucker_prager_cap_point(
    material: DruckerPragerCapMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> dict[str, Any]:
    """Update one Drucker-Prager/Cap material point."""
    del time_step
    del update_settings
    trial_stress = elasticity @ (strain - state.variables["plastic_strain"][0])
    pressure, q_value, _ = compute_stress_invariants(trial_stress)
    kappa = float(state.variables["kappa"][0])
    apex_pressure = float(state.variables["cap_apex_pressure"][0])
    composite_value, dp_value, cap_value = _compute_dp_cap_branch_values(
        material,
        pressure,
        q_value,
        kappa,
        apex_pressure,
    )

    if composite_value <= LOCAL_TOLERANCE:
        payload = _elastic_pressure_plastic_payload(
            material,
            strain,
            state,
            elasticity,
            trial_stress,
            composite_value,
            "elastic",
        )
        _append_dp_cap_diagnostics(
            payload,
            material,
            pressure,
            q_value,
            kappa,
            apex_pressure,
        )
        return payload

    geometry = _compute_dp_cap_geometry(material, apex_pressure)
    if pressure <= geometry.transition_pressure and dp_value > LOCAL_TOLERANCE:
        payload = _update_drucker_prager_shear_point(material, strain, state, elasticity)
        pressure_new, q_new, _ = compute_stress_invariants(payload["stress"])
        composite_after, _, _ = _compute_dp_cap_branch_values(
            material,
            pressure_new,
            q_new,
            float(payload["state"].variables["kappa"][0]),
            apex_pressure,
        )
        if pressure_new <= geometry.transition_pressure and composite_after <= 1.0e-8:
            _append_dp_cap_diagnostics(
                payload,
                material,
                pressure_new,
                q_new,
                float(payload["state"].variables["kappa"][0]),
                apex_pressure,
            )
            return payload

    return _update_cap_point(material, strain, state, elasticity)


def _update_drucker_prager_shear_point(
    material: DruckerPragerMaterial | DruckerPragerCapMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
) -> dict[str, Any]:
    """Update one pressure-sensitive material point on the DP shear surface."""
    bulk_modulus = _bulk_modulus(material.young_modulus, material.poisson_ratio)
    shear_modulus = _shear_modulus(material.young_modulus, material.poisson_ratio)
    plastic_strain = state.variables["plastic_strain"][0]
    trial_stress = elasticity @ (strain - plastic_strain)
    pressure_trial, q_trial, deviatoric_trial = compute_stress_invariants(trial_stress)
    kappa = float(state.variables["kappa"][0])
    yield_value = (
        q_trial
        - material.friction_parameter * pressure_trial
        - _cohesion(material, kappa)
    )

    if yield_value <= LOCAL_TOLERANCE:
        return _elastic_pressure_plastic_payload(
            material,
            strain,
            state,
            elasticity,
            trial_stress,
            yield_value,
            "elastic",
        )
    if q_trial <= LOCAL_TOLERANCE:
        raise ValueError(
            "Drucker-Prager plastic flow is undefined for zero deviatoric trial stress.",
        )

    denominator = (
        3.0 * shear_modulus
        + material.friction_parameter * bulk_modulus * material.flow_parameter
        + material.cohesion_hardening * material.kappa_rate
    )
    delta_gamma = yield_value / denominator
    pressure = pressure_trial + bulk_modulus * material.flow_parameter * delta_gamma
    q_value = q_trial - 3.0 * shear_modulus * delta_gamma
    stress = (q_value / q_trial) * deviatoric_trial - pressure * IDENTITY
    flow_direction = 1.5 * deviatoric_trial / q_trial + (material.flow_parameter / 3.0) * IDENTITY
    plastic_increment = delta_gamma * TENSOR_TO_ENGINEERING_STRAIN @ flow_direction

    next_variables = _copy_state_variables(state.variables)
    next_variables["plastic_strain"][0] += plastic_increment
    next_variables["kappa"][0] += material.kappa_rate * delta_gamma
    next_variables["plastic_volumetric_strain"][0] += material.flow_parameter * delta_gamma
    next_variables["plastic_multiplier"][0] += delta_gamma
    next_state = MaterialState(next_variables)
    final_pressure, final_q, _ = compute_stress_invariants(stress)
    final_yield = (
        final_q
        - material.friction_parameter * final_pressure
        - _cohesion(material, float(next_state.variables["kappa"][0]))
    )
    plastic_work = _stress_strain_inner(stress, plastic_increment)
    payload = _point_payload(
        stress,
        next_state,
        strain,
        elasticity,
        plastic_work,
        final_yield,
        0,
        "dp_shear",
    )
    payload["diagnostics"]["plastic_multiplier"] = delta_gamma
    payload["diagnostics"]["pressure"] = final_pressure
    payload["diagnostics"]["q_value"] = final_q
    payload["diagnostics"]["plastic_volumetric_strain"] = next_state.variables[
        "plastic_volumetric_strain"
    ][0]

    return payload


def _update_cap_point(
    material: DruckerPragerCapMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
) -> dict[str, Any]:
    """Update one pressure-sensitive material point on the cap surface."""
    plastic_strain = state.variables["plastic_strain"][0]
    trial_stress = elasticity @ (strain - plastic_strain)
    pressure_trial, q_trial, deviatoric_trial = compute_stress_invariants(trial_stress)
    apex_pressure_n = float(state.variables["cap_apex_pressure"][0])
    solution, iteration_count = _solve_cap_return(
        material,
        pressure_trial,
        q_trial,
        apex_pressure_n,
    )
    pressure, q_value, apex_pressure, delta_gamma = solution
    stress = _scale_deviatoric(deviatoric_trial, q_trial, q_value) - pressure * IDENTITY
    plastic_increment = np.linalg.solve(elasticity, trial_stress - stress)
    compression_plastic_increment = -float(np.sum(plastic_increment[:3]))

    next_variables = _copy_state_variables(state.variables)
    next_variables["plastic_strain"][0] += plastic_increment
    next_variables["plastic_volumetric_strain"][0] += compression_plastic_increment
    next_variables["plastic_multiplier"][0] += delta_gamma
    next_variables["cap_apex_pressure"][0] = apex_pressure
    next_state = MaterialState(next_variables)

    kappa = float(next_state.variables["kappa"][0])
    final_composite, _, _ = _compute_dp_cap_branch_values(
        material,
        pressure,
        q_value,
        kappa,
        apex_pressure,
    )
    plastic_work = _stress_strain_inner(stress, plastic_increment)
    payload = _point_payload(
        stress,
        next_state,
        strain,
        elasticity,
        plastic_work,
        final_composite,
        iteration_count,
        "cap",
    )
    payload["diagnostics"]["plastic_multiplier"] = delta_gamma
    payload["diagnostics"]["pressure"] = pressure
    payload["diagnostics"]["q_value"] = q_value
    payload["diagnostics"]["plastic_volumetric_strain"] = next_state.variables[
        "plastic_volumetric_strain"
    ][0]
    _append_dp_cap_diagnostics(payload, material, pressure, q_value, kappa, apex_pressure)

    return payload


def _solve_cap_return(
    material: DruckerPragerCapMaterial,
    pressure_trial: float,
    q_trial: float,
    apex_pressure_n: float,
) -> tuple[NDArray[np.float64], int]:
    """Solve the local cap-return residual."""
    if q_trial <= _hydrostatic_q_tolerance(material, pressure_trial, apex_pressure_n):
        return _solve_hydrostatic_cap_return(material, pressure_trial, apex_pressure_n), 1

    geometry = _compute_dp_cap_geometry(material, apex_pressure_n)
    pressure = min(
        max(pressure_trial, geometry.transition_pressure),
        geometry.apex_pressure - LOCAL_TOLERANCE * max(1.0, geometry.apex_pressure),
    )
    q_surface = _compute_cap_surface_q(material, pressure, apex_pressure_n)
    q_value = min(q_trial, q_surface)
    apex_pressure = apex_pressure_n
    pressure_derivative = _cap_pressure_derivative(material, pressure, apex_pressure)
    delta_gamma = max(
        (pressure_trial - pressure)
        / (
            _bulk_modulus(material.young_modulus, material.poisson_ratio)
            * max(pressure_derivative, LOCAL_TOLERANCE)
        ),
        0.0,
    )
    solution = np.asarray([pressure, q_value, apex_pressure, delta_gamma], dtype=np.float64)
    residual_tolerance = material.local_tolerance * max(
        1.0,
        abs(pressure_trial),
        abs(q_trial),
        abs(apex_pressure_n),
    )

    for iteration_id in range(1, material.max_local_iterations + 1):
        residual = _cap_residual(material, solution, pressure_trial, q_trial, apex_pressure_n)
        if np.linalg.norm(residual, ord=2) <= residual_tolerance:
            return solution, iteration_id
        jacobian = _finite_difference_jacobian(
            lambda values: _cap_residual(
                material,
                values,
                pressure_trial,
                q_trial,
                apex_pressure_n,
            ),
            solution,
        )
        increment = np.linalg.solve(jacobian, -residual)
        solution = solution + increment

    raise RuntimeError("Drucker-Prager/Cap local Newton iteration did not converge.")


def _solve_hydrostatic_cap_return(
    material: DruckerPragerCapMaterial,
    pressure_trial: float,
    apex_pressure_n: float,
) -> NDArray[np.float64]:
    """Solve the hydrostatic cap return in closed form."""
    bulk_modulus = _bulk_modulus(material.young_modulus, material.poisson_ratio)
    plastic_volumetric_increment = (pressure_trial - apex_pressure_n) / (
        bulk_modulus + material.cap_hardening
    )
    apex_pressure = apex_pressure_n + material.cap_hardening * plastic_volumetric_increment
    pressure = apex_pressure
    pressure_derivative = _cap_pressure_derivative(material, pressure, apex_pressure)
    delta_gamma = plastic_volumetric_increment / pressure_derivative

    return np.asarray([pressure, 0.0, apex_pressure, delta_gamma], dtype=np.float64)


def _hydrostatic_q_tolerance(
    material: DruckerPragerCapMaterial,
    pressure_trial: float,
    apex_pressure: float,
) -> float:
    """Compute a scale-aware hydrostatic q tolerance."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)
    stress_scale = max(1.0, abs(pressure_trial), geometry.axis_deviatoric)

    return material.local_tolerance * stress_scale


def _cap_residual(
    material: DruckerPragerCapMaterial,
    values: NDArray[np.float64],
    pressure_trial: float,
    q_trial: float,
    apex_pressure_n: float,
) -> NDArray[np.float64]:
    """Evaluate the cap-return local residual."""
    bulk_modulus = _bulk_modulus(material.young_modulus, material.poisson_ratio)
    shear_modulus = _shear_modulus(material.young_modulus, material.poisson_ratio)
    pressure, q_value, apex_pressure, delta_gamma = values
    pressure_derivative = _cap_pressure_derivative(material, pressure, apex_pressure)
    q_derivative = _cap_q_derivative(material, q_value, apex_pressure)

    return np.asarray(
        [
            pressure - pressure_trial + bulk_modulus * delta_gamma * pressure_derivative,
            q_value - q_trial + 3.0 * shear_modulus * delta_gamma * q_derivative,
            (
                apex_pressure
                - apex_pressure_n
                - material.cap_hardening * delta_gamma * pressure_derivative
            ),
            _cap_yield_value(material, pressure, q_value, apex_pressure),
        ],
        dtype=np.float64,
    )


def _point_payload(
    stress: NDArray[np.float64],
    state: MaterialState,
    strain: NDArray[np.float64],
    elasticity: NDArray[np.float64],
    plastic_work: float,
    yield_value: float,
    local_iterations: int,
    active_surface: str,
) -> dict[str, Any]:
    """Build one point update payload."""
    elastic_strain = strain - state.variables.get(
        "plastic_strain",
        np.zeros((1, VOIGT_DIM), dtype=np.float64),
    )[0]
    free_energy = 0.5 * _stress_strain_inner(elasticity @ elastic_strain, elastic_strain)

    return {
        "stress": stress,
        "state": state,
        "free_energy": free_energy,
        "dissipation": plastic_work,
        "diagnostics": {
            "yield_value": yield_value,
            "plastic_work": plastic_work,
            "local_iterations": float(local_iterations),
            "active_surface": _active_surface_code(active_surface),
        },
    }


def _elastic_pressure_plastic_payload(
    material: DruckerPragerMaterial | DruckerPragerCapMaterial,
    strain: NDArray[np.float64],
    state: MaterialState,
    elasticity: NDArray[np.float64],
    stress: NDArray[np.float64],
    yield_value: float,
    active_surface: str,
) -> dict[str, Any]:
    """Build an elastic payload for a pressure-sensitive material."""
    pressure, q_value, _ = compute_stress_invariants(stress)
    payload = _point_payload(stress, state, strain, elasticity, 0.0, yield_value, 0, active_surface)
    payload["diagnostics"]["pressure"] = pressure
    payload["diagnostics"]["q_value"] = q_value
    payload["diagnostics"]["plastic_multiplier"] = 0.0
    payload["diagnostics"]["plastic_volumetric_strain"] = state.variables[
        "plastic_volumetric_strain"
    ][0]
    if isinstance(material, DruckerPragerCapMaterial):
        apex_pressure = state.variables["cap_apex_pressure"][0]
        payload["diagnostics"]["cap_apex_pressure"] = apex_pressure
        payload["diagnostics"]["cap_pressure"] = apex_pressure
        payload["diagnostics"]["cap_yield_value"] = _cap_yield_value(
            material,
            pressure,
            q_value,
            apex_pressure,
        )

    return payload


def _numerical_tangent(
    material: Any,
    strain: NDArray[np.float64],
    state: MaterialState,
    update_point,
    time_step: float,
    update_settings: MaterialUpdateSettings,
) -> NDArray[np.float64]:
    """Compute the algorithmic tangent by differentiating the local update map."""
    elasticity = _elasticity_matrix(material.young_modulus, material.poisson_ratio)
    tangent = np.zeros((VOIGT_DIM, VOIGT_DIM), dtype=np.float64)

    for component_id in range(VOIGT_DIM):
        perturbation = np.zeros(VOIGT_DIM, dtype=np.float64)
        step = 1.0e-8 * max(1.0, abs(strain[component_id]))
        perturbation[component_id] = step
        stress_plus = update_point(
            material,
            strain + perturbation,
            state,
            elasticity,
            time_step,
            update_settings,
        )["stress"]
        stress_minus = update_point(
            material,
            strain - perturbation,
            state,
            elasticity,
            time_step,
            update_settings,
        )["stress"]
        tangent[:, component_id] = (stress_plus - stress_minus) / (2.0 * step)

    return tangent


def _require_six_component_strains(strains: NDArray[np.float64]) -> NDArray[np.float64]:
    """Require six-component strain vectors."""
    if strains.ndim != 2 or strains.shape[1] != VOIGT_DIM:
        raise ValueError("Plastic materials require strain arrays with shape (n_points, 6).")

    return strains


def _elasticity_matrix(young_modulus: float, poisson_ratio: float) -> NDArray[np.float64]:
    """Build the three-dimensional elastic stiffness matrix."""
    shear_modulus = _shear_modulus(young_modulus, poisson_ratio)
    lame_lambda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    matrix = np.zeros((VOIGT_DIM, VOIGT_DIM), dtype=np.float64)
    matrix[:3, :3] = lame_lambda
    matrix[0, 0] += 2.0 * shear_modulus
    matrix[1, 1] += 2.0 * shear_modulus
    matrix[2, 2] += 2.0 * shear_modulus
    matrix[3, 3] = shear_modulus
    matrix[4, 4] = shear_modulus
    matrix[5, 5] = shear_modulus

    return matrix


def _shear_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Compute the shear modulus."""
    return young_modulus / (2.0 * (1.0 + poisson_ratio))


def _bulk_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Compute the bulk modulus."""
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))


def _stress_inner(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Compute a tensor inner product for stress-like Voigt vectors."""
    return float(np.sum(STRESS_WEIGHTS * left * right))


def _compute_j2_equivalent_stress(deviatoric: NDArray[np.float64]) -> float:
    """Compute J2 equivalent stress without truncating an invalid invariant."""
    radicand = 1.5 * _stress_inner(deviatoric, deviatoric)
    if not np.isfinite(radicand):
        raise MaterialPointConvergenceError(
            "J2 equivalent-stress invariant is non-finite.",
            "non_finite_j2_invariant",
            {
                "algorithm": "j2_equivalent_stress",
                "radicand": radicand,
                "deviatoric_stress": deviatoric.tolist(),
            },
        )
    if radicand < 0.0:
        raise MaterialPointConvergenceError(
            "J2 equivalent-stress invariant is negative.",
            "negative_j2_invariant",
            {
                "algorithm": "j2_equivalent_stress",
                "radicand": radicand,
                "deviatoric_stress": deviatoric.tolist(),
            },
        )

    return float(np.sqrt(radicand))


def _stress_strain_inner(stress: NDArray[np.float64], strain: NDArray[np.float64]) -> float:
    """Compute the power-conjugate stress and engineering-strain product."""
    return float(stress @ strain)


def _deviatoric(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the deviatoric part of a stress-like vector."""
    return stress - (float(np.sum(stress[:3])) / 3.0) * IDENTITY


def _cohesion(material: DruckerPragerMaterial | DruckerPragerCapMaterial, kappa: float) -> float:
    """Compute Drucker-Prager cohesion hardening."""
    return material.cohesion + material.cohesion_hardening * kappa


def _compute_dp_cap_geometry(
    material: DruckerPragerCapMaterial,
    apex_pressure: float,
) -> DruckerPragerCapGeometry:
    """Compute the C1 composite DP-Cap geometry."""
    transition_pressure = material.cap_transition_ratio * apex_pressure
    transition_q = material.friction_parameter * transition_pressure + material.cohesion
    pressure_distance = apex_pressure - transition_pressure

    if material.friction_parameter <= LOCAL_TOLERANCE:
        center_pressure = transition_pressure
        axis_pressure = pressure_distance
        axis_deviatoric = transition_q
    else:
        slope_length = transition_q / material.friction_parameter
        center_offset = pressure_distance**2 / (2.0 * pressure_distance + slope_length)
        center_pressure = transition_pressure + center_offset
        axis_pressure = apex_pressure - center_pressure
        axis_deviatoric = np.sqrt(
            material.friction_parameter
            * axis_pressure**2
            * transition_q
            / center_offset,
        )

    if axis_pressure <= 0.0 or axis_deviatoric <= 0.0:
        raise ValueError("Invalid DP-Cap geometry from the supplied cap parameters.")

    return DruckerPragerCapGeometry(
        apex_pressure=apex_pressure,
        transition_pressure=transition_pressure,
        transition_q=transition_q,
        center_pressure=center_pressure,
        axis_pressure=axis_pressure,
        axis_deviatoric=axis_deviatoric,
    )


def _compute_dp_cap_branch_values(
    material: DruckerPragerCapMaterial,
    pressure: float,
    q_value: float,
    kappa: float,
    apex_pressure: float,
) -> tuple[float, float, float]:
    """Compute composite, DP-branch, and cap-branch yield values."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)
    dp_value = q_value - material.friction_parameter * pressure - _cohesion(material, kappa)
    cap_value = _cap_yield_value(material, pressure, q_value, apex_pressure)
    composite_value = dp_value if pressure <= geometry.transition_pressure else cap_value

    return composite_value, dp_value, cap_value


def _append_dp_cap_diagnostics(
    payload: dict[str, Any],
    material: DruckerPragerCapMaterial,
    pressure: float,
    q_value: float,
    kappa: float,
    apex_pressure: float,
) -> None:
    """Append composite DP-Cap diagnostics to one point payload."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)
    composite_value, dp_value, cap_value = _compute_dp_cap_branch_values(
        material,
        pressure,
        q_value,
        kappa,
        apex_pressure,
    )
    payload["diagnostics"]["composite_yield_value"] = composite_value
    payload["diagnostics"]["dp_branch_yield_value"] = dp_value
    payload["diagnostics"]["cap_branch_yield_value"] = cap_value
    payload["diagnostics"]["cap_yield_value"] = cap_value
    payload["diagnostics"]["cap_apex_pressure"] = apex_pressure
    payload["diagnostics"]["cap_pressure"] = apex_pressure
    payload["diagnostics"]["transition_pressure"] = geometry.transition_pressure
    payload["diagnostics"]["transition_q"] = geometry.transition_q
    payload["diagnostics"]["cap_center_pressure"] = geometry.center_pressure
    payload["diagnostics"]["cap_axis_pressure"] = geometry.axis_pressure
    payload["diagnostics"]["cap_axis_deviatoric"] = geometry.axis_deviatoric


def _cap_yield_value(
    material: DruckerPragerCapMaterial,
    pressure: float,
    q_value: float,
    apex_pressure: float,
) -> float:
    """Compute the cap yield value."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)

    return (
        ((pressure - geometry.center_pressure) / geometry.axis_pressure) ** 2
        + (q_value / geometry.axis_deviatoric) ** 2
        - 1.0
    )


def _compute_cap_surface_q(
    material: DruckerPragerCapMaterial,
    pressure: float,
    apex_pressure: float,
) -> float:
    """Compute the upper cap-surface q value at one pressure."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)
    normalized_pressure = (pressure - geometry.center_pressure) / geometry.axis_pressure
    normalized_q_square = max(1.0 - normalized_pressure**2, 0.0)

    return geometry.axis_deviatoric * np.sqrt(normalized_q_square)


def _cap_pressure_derivative(
    material: DruckerPragerCapMaterial,
    pressure: float,
    apex_pressure: float,
) -> float:
    """Compute dFcap/dp."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)

    return 2.0 * (pressure - geometry.center_pressure) / geometry.axis_pressure**2


def _cap_q_derivative(
    material: DruckerPragerCapMaterial,
    q_value: float,
    apex_pressure: float,
) -> float:
    """Compute dFcap/dq."""
    geometry = _compute_dp_cap_geometry(material, apex_pressure)

    return 2.0 * q_value / geometry.axis_deviatoric**2


def _scale_deviatoric(
    deviatoric_trial: NDArray[np.float64],
    q_trial: float,
    q_value: float,
) -> NDArray[np.float64]:
    """Scale a trial deviatoric stress to the corrected q value."""
    if q_trial <= LOCAL_TOLERANCE:
        return np.zeros(VOIGT_DIM, dtype=np.float64)

    return (q_value / q_trial) * deviatoric_trial


def _finite_difference_jacobian(function, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute a finite-difference Jacobian for a local residual."""
    base_size = values.shape[0]
    jacobian = np.zeros((base_size, base_size), dtype=np.float64)
    for component_id in range(base_size):
        perturbation = np.zeros(base_size, dtype=np.float64)
        step = 1.0e-7 * max(1.0, abs(values[component_id]))
        perturbation[component_id] = step
        jacobian[:, component_id] = (
            function(values + perturbation) - function(values - perturbation)
        ) / (2.0 * step)

    return jacobian


def _copy_state_variables(
    variables: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Copy state arrays."""
    return {name: values.copy() for name, values in variables.items()}


def _slice_state(state: MaterialState, point_id: int) -> MaterialState:
    """Slice one integration-point state."""
    return MaterialState(
        variables={
            name: values[point_id : point_id + 1].copy()
            for name, values in state.variables.items()
        },
    )


def _assign_state(
    variables: dict[str, NDArray[np.float64]],
    point_id: int,
    point_variables: dict[str, NDArray[np.float64]],
) -> None:
    """Assign one integration-point state into a batched state."""
    for name, values in point_variables.items():
        variables[name][point_id] = values[0]


def _pack_state_variables(variables: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Pack state variables into one diagnostic array."""
    if not variables:
        return np.zeros((0, 0), dtype=np.float64)

    parts = []
    for values in variables.values():
        parts.append(values.reshape(values.shape[0], -1))

    return np.concatenate(parts, axis=1)


def _append_diagnostics(
    diagnostics: dict[str, NDArray[np.float64]],
    point_id: int,
    n_points: int,
    point_diagnostics: dict[str, float],
) -> None:
    """Append one integration-point diagnostic record."""
    for name, value in point_diagnostics.items():
        if name not in diagnostics:
            diagnostics[name] = np.zeros(n_points, dtype=np.float64)
        diagnostics[name][point_id] = value


def _active_surface_code(active_surface: str) -> float:
    """Encode active-surface names for CSV diagnostics."""
    return {
        "elastic": 0.0,
        "plastic": 1.0,
        "dp_shear": 2.0,
        "cap": 3.0,
        "viscoplastic": 4.0,
    }[active_surface]
