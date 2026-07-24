"""Material-point plasticity benchmark runners.

Contents:
    Loading-path builders, analytical references, benchmark metrics, and outputs.
Author:
    Zhen Hao.
Created:
    2026-07-08.
"""

from __future__ import annotations

import copy
import json
import platform
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from concrete_impact.core.failure_records import record_material_point_failures
from fem.cases import RunResult
from fem.io.config import write_yaml_config
from fem.materials import (
    DruckerPragerCapMaterial,
    DruckerPragerMaterial,
    J2PlasticMaterial,
    J2ViscoplasticMaterial,
    MaterialModel,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
    build_material,
    compute_stress_invariants,
)
from fem.post.curves import write_response_csv, write_response_plot

VOIGT_DIM = 6

CurveBuilder = Callable[[dict[str, Any], MaterialModel], dict[str, NDArray[np.float64]]]


@record_material_point_failures
def run_plasticity_benchmark(config: dict[str, Any]) -> RunResult:
    """Run one material-point plasticity benchmark."""
    case_name = str(config["case"]["name"])
    output_root = Path(str(config["output"]["root"]))
    material = build_material(config["model"]["material"])
    curve_data = PLASTICITY_CURVE_BUILDERS[str(config["benchmark"]["type"])](config, material)

    output_paths = _build_output_paths(output_root)
    response_csv = write_response_csv(curve_data, output_paths["response_csv"])
    diagnostics_csv = write_response_csv(curve_data, output_paths["diagnostics_csv"])
    response_png = _write_confined_triaxial_plot(curve_data, output_paths["response_png"]) if (
        str(config["benchmark"]["type"]) == "dp_cap_confined_triaxial"
    ) else write_response_plot(
        curve_data,
        output_paths["response_png"],
        str(config["benchmark"]["x_name"]),
        _build_plot_pairs(config),
    )

    output_paths["response_csv"] = response_csv
    output_paths["diagnostics_csv"] = diagnostics_csv
    output_paths["response_png"] = response_png
    metrics = _compute_curve_metrics(
        curve_data,
        _build_plot_pairs(config),
        float(config["verification"]["absolute_tolerance"]),
    )
    passed = _passes_metrics(
        metrics,
        float(config["verification"]["relative_tolerance"]),
        float(config["verification"]["absolute_tolerance"]),
    )
    result = RunResult(name=case_name, metrics=metrics, output_paths=output_paths, passed=passed)
    _write_plasticity_records(config, result, output_root)

    return result


def build_j2_pure_shear_curve(
    config: dict[str, Any],
    material: MaterialModel,
) -> dict[str, NDArray[np.float64]]:
    """Build the J2 perfect-plastic pure-shear benchmark curve."""
    j2_material = material
    if not isinstance(j2_material, J2PlasticMaterial):
        raise TypeError("The J2 pure-shear benchmark requires J2PlasticMaterial.")

    shear_values = np.linspace(
        0.0,
        float(config["benchmark"]["max_shear_strain"]),
        int(config["benchmark"]["num_steps"]),
    )
    state = j2_material.initialize_state(1)
    stresses = np.zeros_like(shear_values)
    exact_stresses = np.zeros_like(shear_values)
    equivalent_plastic_strains = np.zeros_like(shear_values)
    yield_values = np.zeros_like(shear_values)
    tangent_values = np.zeros_like(shear_values)
    shear_modulus = _shear_modulus(j2_material.young_modulus, j2_material.poisson_ratio)
    yield_shear = j2_material.yield_stress / np.sqrt(3.0)

    for step_id, shear_strain in enumerate(shear_values):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[5] = shear_strain
        response = _update_single_point(j2_material, state, strain, config)
        state = _require_response_state(response)

        stresses[step_id] = response.stresses[0, 5]
        exact_stresses[step_id] = min(shear_modulus * shear_strain, yield_shear)
        equivalent_plastic_strains[step_id] = state.variables["equivalent_plastic_strain"][0]
        yield_values[step_id] = response.diagnostics["yield_value"][0]
        if response.tangents is None:
            raise ValueError("J2 pure-shear benchmark requires the material tangent.")
        tangent_values[step_id] = response.tangents[0, 5, 5]

    return {
        "step": np.arange(shear_values.size, dtype=np.float64),
        "shear_strain": shear_values,
        "shear_stress_numerical": stresses,
        "shear_stress_exact": exact_stresses,
        "equivalent_plastic_strain": equivalent_plastic_strains,
        "yield_value": yield_values,
        "shear_tangent_55": tangent_values,
    }


def build_j2_viscoplastic_material_point_curve(
    config: dict[str, Any],
    material: MaterialModel,
) -> dict[str, NDArray[np.float64]]:
    """Build the rate-dependent J2 pure-shear benchmark curve."""
    viscoplastic_material = material
    if not isinstance(viscoplastic_material, J2ViscoplasticMaterial):
        raise TypeError("The J2 viscoplastic benchmark requires J2ViscoplasticMaterial.")

    shear_values = np.linspace(
        0.0,
        float(config["benchmark"]["max_shear_strain"]),
        int(config["benchmark"]["num_steps"]),
    )
    state = viscoplastic_material.initialize_state(1)
    exact_state = {
        "plastic_shear_strain": 0.0,
        "equivalent_plastic_strain": 0.0,
    }
    stresses = np.zeros_like(shear_values)
    exact_stresses = np.zeros_like(shear_values)
    equivalent_plastic_strains = np.zeros_like(shear_values)
    exact_equivalent_plastic_strains = np.zeros_like(shear_values)
    plastic_multipliers = np.zeros_like(shear_values)
    exact_plastic_multipliers = np.zeros_like(shear_values)
    yield_values = np.zeros_like(shear_values)
    shear_modulus = _shear_modulus(
        viscoplastic_material.young_modulus,
        viscoplastic_material.poisson_ratio,
    )

    for step_id, shear_strain in enumerate(shear_values):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[5] = shear_strain
        response = _update_single_point(viscoplastic_material, state, strain, config)
        state = _require_response_state(response)
        exact = _evaluate_j2_viscoplastic_pure_shear_exact(
            viscoplastic_material,
            shear_modulus,
            shear_strain,
            float(config["analysis"]["time_step"]),
            exact_state,
        )

        stresses[step_id] = response.stresses[0, 5]
        exact_stresses[step_id] = exact["shear_stress"]
        equivalent_plastic_strains[step_id] = state.variables["equivalent_plastic_strain"][0]
        exact_equivalent_plastic_strains[step_id] = exact["equivalent_plastic_strain"]
        plastic_multipliers[step_id] = response.diagnostics["viscoplastic_multiplier"][0]
        exact_plastic_multipliers[step_id] = exact["plastic_multiplier"]
        yield_values[step_id] = response.diagnostics["yield_value"][0]

    return {
        "step": np.arange(shear_values.size, dtype=np.float64),
        "shear_strain": shear_values,
        "shear_stress_numerical": stresses,
        "shear_stress_exact": exact_stresses,
        "equivalent_plastic_strain_numerical": equivalent_plastic_strains,
        "equivalent_plastic_strain_exact": exact_equivalent_plastic_strains,
        "plastic_multiplier_numerical": plastic_multipliers,
        "plastic_multiplier_exact": exact_plastic_multipliers,
        "yield_value": yield_values,
    }


def build_dp_pure_shear_curve(
    config: dict[str, Any],
    material: MaterialModel,
) -> dict[str, NDArray[np.float64]]:
    """Build the Drucker-Prager pure-shear semi-analytic curve."""
    dp_material = material
    if not isinstance(dp_material, DruckerPragerMaterial):
        raise TypeError("The DP pure-shear benchmark requires DruckerPragerMaterial.")

    shear_values = np.linspace(
        0.0,
        float(config["benchmark"]["max_shear_strain"]),
        int(config["benchmark"]["num_steps"]),
    )
    state = dp_material.initialize_state(1)
    stresses = np.zeros_like(shear_values)
    exact_stresses = np.zeros_like(shear_values)
    pressures = np.zeros_like(shear_values)
    exact_pressures = np.zeros_like(shear_values)
    plastic_volumetric_strains = np.zeros_like(shear_values)
    exact_plastic_volumetric_strains = np.zeros_like(shear_values)
    yield_values = np.zeros_like(shear_values)

    shear_modulus = _shear_modulus(dp_material.young_modulus, dp_material.poisson_ratio)
    bulk_modulus = _bulk_modulus(dp_material.young_modulus, dp_material.poisson_ratio)

    for step_id, shear_strain in enumerate(shear_values):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[5] = shear_strain
        response = _update_single_point(dp_material, state, strain, config)
        state = _require_response_state(response)

        exact = _evaluate_dp_pure_shear_exact(
            dp_material,
            shear_modulus,
            bulk_modulus,
            shear_strain,
        )
        stresses[step_id] = response.stresses[0, 5]
        exact_stresses[step_id] = exact["shear_stress"]
        pressures[step_id] = response.diagnostics["pressure"][0]
        exact_pressures[step_id] = exact["pressure"]
        plastic_volumetric_strains[step_id] = response.diagnostics["plastic_volumetric_strain"][0]
        exact_plastic_volumetric_strains[step_id] = exact["plastic_volumetric_strain"]
        yield_values[step_id] = response.diagnostics["yield_value"][0]

    return {
        "step": np.arange(shear_values.size, dtype=np.float64),
        "shear_strain": shear_values,
        "shear_stress_numerical": stresses,
        "shear_stress_exact": exact_stresses,
        "pressure_numerical": pressures,
        "pressure_exact": exact_pressures,
        "plastic_volumetric_strain_numerical": plastic_volumetric_strains,
        "plastic_volumetric_strain_exact": exact_plastic_volumetric_strains,
        "yield_value": yield_values,
    }


def build_dp_cap_hydrostatic_curve(
    config: dict[str, Any],
    material: MaterialModel,
) -> dict[str, NDArray[np.float64]]:
    """Build the Drucker-Prager/Cap hydrostatic compression semi-analytic curve."""
    cap_material = material
    if not isinstance(cap_material, DruckerPragerCapMaterial):
        raise TypeError("The cap benchmark requires DruckerPragerCapMaterial.")

    compression_values = np.linspace(
        0.0,
        float(config["benchmark"]["max_compression_strain"]),
        int(config["benchmark"]["num_steps"]),
    )
    state = cap_material.initialize_state(1)
    pressures = np.zeros_like(compression_values)
    exact_pressures = np.zeros_like(compression_values)
    cap_pressures = np.zeros_like(compression_values)
    exact_cap_pressures = np.zeros_like(compression_values)
    plastic_volumetric_strains = np.zeros_like(compression_values)
    exact_plastic_volumetric_strains = np.zeros_like(compression_values)
    cap_yield_values = np.zeros_like(compression_values)
    exact_state = {
        "cap_apex_pressure": cap_material.initial_cap_apex_pressure,
        "plastic_volumetric_strain": 0.0,
    }

    bulk_modulus = _bulk_modulus(cap_material.young_modulus, cap_material.poisson_ratio)

    for step_id, compression_strain in enumerate(compression_values):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[:3] = -compression_strain / 3.0
        response = _update_single_point(cap_material, state, strain, config)
        state = _require_response_state(response)

        exact = _evaluate_cap_hydrostatic_exact(
            cap_material,
            bulk_modulus,
            compression_strain,
            exact_state,
        )
        pressures[step_id] = response.diagnostics["pressure"][0]
        exact_pressures[step_id] = exact["pressure"]
        cap_pressures[step_id] = response.diagnostics["cap_apex_pressure"][0]
        exact_cap_pressures[step_id] = exact["cap_apex_pressure"]
        plastic_volumetric_strains[step_id] = response.diagnostics["plastic_volumetric_strain"][0]
        exact_plastic_volumetric_strains[step_id] = exact["plastic_volumetric_strain"]
        cap_yield_values[step_id] = response.diagnostics["cap_yield_value"][0]

    return {
        "step": np.arange(compression_values.size, dtype=np.float64),
        "compression_strain": compression_values,
        "pressure_numerical": pressures,
        "pressure_exact": exact_pressures,
        "cap_pressure_numerical": cap_pressures,
        "cap_pressure_exact": exact_cap_pressures,
        "plastic_volumetric_strain_numerical": plastic_volumetric_strains,
        "plastic_volumetric_strain_exact": exact_plastic_volumetric_strains,
        "cap_yield_value": cap_yield_values,
    }


def build_dp_cap_confined_triaxial_curve(
    config: dict[str, Any],
    material: MaterialModel,
) -> dict[str, NDArray[np.float64]]:
    """Build fixed-confinement triaxial DP-Cap benchmark curves."""
    cap_material = material
    if not isinstance(cap_material, DruckerPragerCapMaterial):
        raise TypeError("The confined triaxial benchmark requires DruckerPragerCapMaterial.")

    confining_pressures = np.asarray(
        config["benchmark"]["confining_pressures"],
        dtype=np.float64,
    )
    axial_values = np.linspace(
        0.0,
        float(config["benchmark"]["max_axial_compression_strain"]),
        int(config["benchmark"]["num_steps"]),
    )

    records: dict[str, list[float]] = {
        "confining_pressure": [],
        "axial_compression_strain": [],
        "q_numerical": [],
        "q_exact": [],
        "pressure_numerical": [],
        "active_surface": [],
        "lateral_stress_error": [],
        "peak_q_exact": [],
    }
    for confining_pressure in confining_pressures:
        _append_confined_triaxial_path(
            records,
            cap_material,
            config,
            float(confining_pressure),
            axial_values,
        )

    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in records.items()
    }


PLASTICITY_CURVE_BUILDERS: dict[str, CurveBuilder] = {
    "j2_pure_shear": build_j2_pure_shear_curve,
    "j2_viscoplastic_material_point": build_j2_viscoplastic_material_point_curve,
    "dp_pure_shear": build_dp_pure_shear_curve,
    "dp_cap_hydrostatic": build_dp_cap_hydrostatic_curve,
    "dp_cap_confined_triaxial": build_dp_cap_confined_triaxial_curve,
}


def _update_single_point(
    material: MaterialModel,
    state: MaterialState,
    strain: NDArray[np.float64],
    config: dict[str, Any],
) -> MaterialPointResponse:
    """Update one committed material point at one strain value."""
    request = MaterialPointRequest(
        strains=strain.reshape(1, VOIGT_DIM),
        strain_rates=np.zeros((1, VOIGT_DIM), dtype=np.float64),
        time_step=float(config["analysis"]["time_step"]),
        kinematics="three_dimensional",
        update_settings=_build_material_update_settings(config),
    )

    return material.update(
        request,
        state,
        MaterialResponseRequirements(
            tangent=bool(config["benchmark"]["need_tangent"]),
            free_energy=True,
            dissipation=True,
        ),
    )


def _build_material_update_settings(config: dict[str, Any]) -> MaterialUpdateSettings:
    """Build local constitutive integration settings."""
    settings = config["analysis"]["material_update"]

    return MaterialUpdateSettings(
        max_iterations=int(settings["max_iterations"]),
        yield_relative_tolerance=float(settings["yield_relative_tolerance"]),
        residual_absolute_tolerance=float(settings["residual_absolute_tolerance"]),
        residual_relative_tolerance=float(settings["residual_relative_tolerance"]),
    )


def _require_response_state(response: MaterialPointResponse) -> MaterialState:
    """Return the updated material state from a response."""
    if response.state is None:
        raise ValueError("Plasticity benchmark requires material responses with updated state.")

    return response.state


def _append_confined_triaxial_path(
    records: dict[str, list[float]],
    material: DruckerPragerCapMaterial,
    config: dict[str, Any],
    confining_pressure: float,
    axial_values: NDArray[np.float64],
) -> None:
    """Append one fixed-confinement triaxial compression path."""
    bulk_modulus = _bulk_modulus(material.young_modulus, material.poisson_ratio)
    hydrostatic_strain = -confining_pressure / (3.0 * bulk_modulus)
    hydrostatic_vector = np.zeros(VOIGT_DIM, dtype=np.float64)
    hydrostatic_vector[:3] = hydrostatic_strain
    state = material.initialize_state(1)
    hydrostatic_response = _update_single_point(material, state, hydrostatic_vector, config)
    state = _require_response_state(hydrostatic_response)
    lateral_strain = hydrostatic_strain
    peak_q = _compute_confined_triaxial_peak_q(material, confining_pressure)

    for axial_compression_strain in axial_values:
        axial_strain = hydrostatic_strain - axial_compression_strain
        lateral_strain, response = _solve_lateral_strain_for_confinement(
            material,
            state,
            config,
            axial_strain,
            lateral_strain,
            confining_pressure,
        )
        state = _require_response_state(response)
        pressure, q_value, _ = compute_stress_invariants(response.stresses[0])
        exact_q = min(material.young_modulus * axial_compression_strain, peak_q)
        lateral_error = response.stresses[0, 1] + confining_pressure

        records["confining_pressure"].append(confining_pressure)
        records["axial_compression_strain"].append(axial_compression_strain)
        records["q_numerical"].append(q_value)
        records["q_exact"].append(exact_q)
        records["pressure_numerical"].append(pressure)
        records["active_surface"].append(response.diagnostics["active_surface"][0])
        records["lateral_stress_error"].append(lateral_error)
        records["peak_q_exact"].append(peak_q)


def _solve_lateral_strain_for_confinement(
    material: DruckerPragerCapMaterial,
    state: MaterialState,
    config: dict[str, Any],
    axial_strain: float,
    initial_lateral_strain: float,
    confining_pressure: float,
) -> tuple[float, MaterialPointResponse]:
    """Solve the lateral strain that keeps lateral stress fixed."""
    lateral_strain = initial_lateral_strain
    tolerance = float(config["benchmark"]["lateral_stress_tolerance"])
    max_iterations = int(config["benchmark"]["lateral_newton_max_iterations"])

    for _ in range(max_iterations):
        response = _update_single_point(
            material,
            state,
            _build_triaxial_strain(axial_strain, lateral_strain),
            config,
        )
        residual = response.stresses[0, 1] + confining_pressure
        if abs(residual) <= tolerance:
            return lateral_strain, response

        step = 1.0e-8 * max(1.0, abs(lateral_strain))
        response_plus = _update_single_point(
            material,
            state,
            _build_triaxial_strain(axial_strain, lateral_strain + step),
            config,
        )
        response_minus = _update_single_point(
            material,
            state,
            _build_triaxial_strain(axial_strain, lateral_strain - step),
            config,
        )
        derivative = (response_plus.stresses[0, 1] - response_minus.stresses[0, 1]) / (
            2.0 * step
        )
        if abs(derivative) <= 1.0:
            raise RuntimeError("Confined triaxial lateral Newton derivative is singular.")
        lateral_strain -= residual / derivative

    raise RuntimeError("Confined triaxial lateral Newton iteration did not converge.")


def _build_triaxial_strain(
    axial_strain: float,
    lateral_strain: float,
) -> NDArray[np.float64]:
    """Build an axisymmetric triaxial strain vector."""
    strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    strain[0] = axial_strain
    strain[1] = lateral_strain
    strain[2] = lateral_strain

    return strain


def _compute_confined_triaxial_peak_q(
    material: DruckerPragerCapMaterial,
    confining_pressure: float,
) -> float:
    """Compute the semi-analytic confined triaxial peak q."""
    denominator = 1.0 - material.friction_parameter / 3.0
    dp_peak_q = (
        material.friction_parameter * confining_pressure
        + material.cohesion
    ) / denominator
    dp_peak_pressure = confining_pressure + dp_peak_q / 3.0
    transition_pressure = (
        material.cap_transition_ratio * material.initial_cap_apex_pressure
    )
    if dp_peak_pressure <= transition_pressure:
        return dp_peak_q

    return _compute_cap_line_intersection_q(material, confining_pressure)


def _compute_cap_line_intersection_q(
    material: DruckerPragerCapMaterial,
    confining_pressure: float,
) -> float:
    """Solve the cap branch and triaxial stress-line intersection."""
    geometry = _compute_dp_cap_geometry_for_benchmark(
        material,
        material.initial_cap_apex_pressure,
    )
    pressure_offset = confining_pressure - geometry["center_pressure"]
    coefficient_a = 1.0 / (9.0 * geometry["axis_pressure"] ** 2) + (
        1.0 / geometry["axis_deviatoric"] ** 2
    )
    coefficient_b = 2.0 * pressure_offset / (3.0 * geometry["axis_pressure"] ** 2)
    coefficient_c = pressure_offset**2 / geometry["axis_pressure"] ** 2 - 1.0
    discriminant = coefficient_b**2 - 4.0 * coefficient_a * coefficient_c
    if discriminant < 0.0:
        raise RuntimeError("Confined triaxial cap reference has no real intersection.")
    roots = np.asarray(
        [
            (-coefficient_b - np.sqrt(discriminant)) / (2.0 * coefficient_a),
            (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a),
        ],
        dtype=np.float64,
    )
    positive_roots = roots[roots >= 0.0]
    if positive_roots.size == 0:
        raise RuntimeError("Confined triaxial cap reference has no positive q root.")

    return float(np.min(positive_roots))


def _compute_dp_cap_geometry_for_benchmark(
    material: DruckerPragerCapMaterial,
    apex_pressure: float,
) -> dict[str, float]:
    """Compute benchmark-side composite cap geometry."""
    transition_pressure = material.cap_transition_ratio * apex_pressure
    transition_q = material.friction_parameter * transition_pressure + material.cohesion
    pressure_distance = apex_pressure - transition_pressure

    if material.friction_parameter <= 1.0e-12:
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

    return {
        "transition_pressure": transition_pressure,
        "transition_q": transition_q,
        "center_pressure": center_pressure,
        "axis_pressure": axis_pressure,
        "axis_deviatoric": axis_deviatoric,
    }


def _evaluate_dp_pure_shear_exact(
    material: DruckerPragerMaterial,
    shear_modulus: float,
    bulk_modulus: float,
    shear_strain: float,
) -> dict[str, float]:
    """Evaluate the DP pure-shear semi-analytic solution."""
    q_trial = np.sqrt(3.0) * shear_modulus * shear_strain
    yield_trial = q_trial - material.cohesion

    if yield_trial <= 0.0:
        return {
            "shear_stress": shear_modulus * shear_strain,
            "pressure": 0.0,
            "plastic_volumetric_strain": 0.0,
        }

    denominator = (
        3.0 * shear_modulus
        + material.friction_parameter * bulk_modulus * material.flow_parameter
        + material.cohesion_hardening * material.kappa_rate
    )
    delta_gamma = yield_trial / denominator
    q_value = q_trial - 3.0 * shear_modulus * delta_gamma

    return {
        "shear_stress": q_value / np.sqrt(3.0),
        "pressure": bulk_modulus * material.flow_parameter * delta_gamma,
        "plastic_volumetric_strain": material.flow_parameter * delta_gamma,
    }


def _evaluate_j2_viscoplastic_pure_shear_exact(
    material: J2ViscoplasticMaterial,
    shear_modulus: float,
    shear_strain: float,
    time_step: float,
    exact_state: dict[str, float],
) -> dict[str, float]:
    """Evaluate one linear J2-Perzyna pure-shear recurrence."""
    plastic_shear_strain = exact_state["plastic_shear_strain"]
    equivalent_plastic_strain = exact_state["equivalent_plastic_strain"]
    trial_shear_stress = shear_modulus * (shear_strain - plastic_shear_strain)
    sign = 1.0 if trial_shear_stress >= 0.0 else -1.0
    trial_q = np.sqrt(3.0) * abs(trial_shear_stress)
    trial_yield = trial_q - (
        material.yield_stress + material.hardening_modulus * equivalent_plastic_strain
    )

    if trial_yield <= 0.0:
        plastic_multiplier = 0.0
        shear_stress = trial_shear_stress
    else:
        denominator = (
            material.time_scale * material.reference_stress / time_step
            + 3.0 * shear_modulus
            + material.hardening_modulus
        )
        plastic_multiplier = trial_yield / denominator
        shear_stress = trial_shear_stress - sign * np.sqrt(3.0) * shear_modulus * (
            plastic_multiplier
        )
        exact_state["plastic_shear_strain"] += sign * np.sqrt(3.0) * plastic_multiplier
        exact_state["equivalent_plastic_strain"] += plastic_multiplier

    return {
        "shear_stress": shear_stress,
        "plastic_multiplier": plastic_multiplier,
        "equivalent_plastic_strain": exact_state["equivalent_plastic_strain"],
    }


def _evaluate_cap_hydrostatic_exact(
    material: DruckerPragerCapMaterial,
    bulk_modulus: float,
    compression_strain: float,
    exact_state: dict[str, float],
) -> dict[str, float]:
    """Evaluate one hydrostatic cap step with the semi-analytic recurrence."""
    apex_pressure_n = exact_state["cap_apex_pressure"]
    plastic_volumetric_strain_n = exact_state["plastic_volumetric_strain"]
    pressure_trial = bulk_modulus * (compression_strain - plastic_volumetric_strain_n)

    if pressure_trial <= apex_pressure_n:
        pressure = pressure_trial
        apex_pressure = apex_pressure_n
        plastic_volumetric_strain = plastic_volumetric_strain_n
    else:
        plastic_increment = (pressure_trial - apex_pressure_n) / (
            bulk_modulus + material.cap_hardening
        )
        pressure = pressure_trial - bulk_modulus * plastic_increment
        apex_pressure = apex_pressure_n + material.cap_hardening * plastic_increment
        plastic_volumetric_strain = plastic_volumetric_strain_n + plastic_increment

    exact_state["cap_apex_pressure"] = apex_pressure
    exact_state["plastic_volumetric_strain"] = plastic_volumetric_strain

    return {
        "pressure": pressure,
        "cap_apex_pressure": apex_pressure,
        "plastic_volumetric_strain": plastic_volumetric_strain,
    }


def _build_plot_pairs(config: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Build numerical-exact curve pairs from the benchmark config."""
    return tuple(
        (str(pair["numerical"]), str(pair["exact"]))
        for pair in config["benchmark"]["plot_pairs"]
    )


def _compute_curve_metrics(
    data: dict[str, NDArray[np.float64]],
    curve_pairs: tuple[tuple[str, str], ...],
    absolute_tolerance: float,
) -> dict[str, float]:
    """Compute curve-wise relative and maximum absolute errors."""
    metrics: dict[str, float] = {}
    for numerical_name, exact_name in curve_pairs:
        numerical = data[numerical_name]
        exact = data[exact_name]
        difference = numerical - exact
        scale = max(float(np.linalg.norm(exact, ord=2)), absolute_tolerance)
        metrics[f"{numerical_name}_relative_l2_error"] = float(
            np.linalg.norm(difference, ord=2) / scale,
        )
        metrics[f"{numerical_name}_max_abs_error"] = float(
            np.max(np.abs(difference)),
        )
        if "confining_pressure" in data:
            for confining_pressure in np.unique(data["confining_pressure"]):
                mask = data["confining_pressure"] == confining_pressure
                group_difference = difference[mask]
                group_exact = exact[mask]
                group_scale = max(float(np.linalg.norm(group_exact, ord=2)), absolute_tolerance)
                pressure_label = f"{confining_pressure / 1.0e6:.0f}mpa"
                metrics[f"{numerical_name}_{pressure_label}_relative_l2_error"] = float(
                    np.linalg.norm(group_difference, ord=2) / group_scale,
                )
                metrics[f"{numerical_name}_{pressure_label}_max_abs_error"] = float(
                    np.max(np.abs(group_difference)),
                )
    if "lateral_stress_error" in data:
        metrics["lateral_stress_error_max_abs_error"] = float(
            np.max(np.abs(data["lateral_stress_error"])),
        )

    return metrics


def _passes_metrics(
    metrics: dict[str, float],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    """Evaluate pass/fail status from curve metrics."""
    return all(
        (
            value <= relative_tolerance
            if metric_name.endswith("relative_l2_error")
            else value <= absolute_tolerance
        )
        for metric_name, value in metrics.items()
    )


def _write_confined_triaxial_plot(
    data: dict[str, NDArray[np.float64]],
    output_path: str | Path,
) -> Path:
    """Write grouped confined-triaxial response plots."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0))
    confining_pressures = np.unique(data["confining_pressure"])
    peak_pressures = []
    peak_values = []

    for confining_pressure in confining_pressures:
        mask = data["confining_pressure"] == confining_pressure
        label = f"{confining_pressure / 1.0e6:.0f} MPa"
        axes[0].plot(
            data["axial_compression_strain"][mask],
            data["q_numerical"][mask],
            label=f"num {label}",
        )
        axes[0].plot(
            data["axial_compression_strain"][mask],
            data["q_exact"][mask],
            "--",
            label=f"exact {label}",
        )
        peak_pressures.append(confining_pressure)
        peak_values.append(data["peak_q_exact"][mask][0])

    axes[0].set_xlabel("axial_compression_strain")
    axes[0].set_ylabel("q")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(np.asarray(peak_pressures), np.asarray(peak_values), "o-")
    axes[1].set_xlabel("confining_pressure")
    axes[1].set_ylabel("peak_q_exact")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

    return path


def _build_output_paths(output_root: Path) -> dict[str, Path]:
    """Build standard output paths for plasticity benchmarks."""
    return {
        "response_csv": output_root / "response.csv",
        "diagnostics_csv": output_root / "diagnostics.csv",
        "response_png": output_root / "response.png",
        "original_config": output_root / "original_config.yaml",
        "resolved_config": output_root / "resolved_config.yaml",
        "metadata": output_root / "run_metadata.json",
        "metrics": output_root / "metrics.json",
    }


def _write_plasticity_records(
    config: dict[str, Any],
    result: RunResult,
    output_root: Path,
) -> None:
    """Write reproducibility records for a plasticity benchmark."""
    output_root.mkdir(parents=True, exist_ok=True)
    write_yaml_config(config, result.output_paths["original_config"])
    write_yaml_config(copy.deepcopy(config), result.output_paths["resolved_config"])
    _write_json(result.output_paths["metadata"], _build_metadata_record(config))
    _write_json(result.output_paths["metrics"], _build_metrics_record(result))


def _build_metadata_record(config: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable benchmark metadata record."""
    return {
        "case_name": str(config["case"]["name"]),
        "benchmark_type": str(config["benchmark"]["type"]),
        "analysis_type": "material_point_plasticity",
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "metadata": config["case"],
    }


def _build_metrics_record(result: RunResult) -> dict[str, Any]:
    """Build a JSON-serializable metric record."""
    return {
        "case_name": result.name,
        "passed": result.passed,
        "metrics": result.metrics,
        "output_paths": {
            name: str(path)
            for name, path in result.output_paths.items()
        },
    }


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _shear_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Compute shear modulus."""
    return young_modulus / (2.0 * (1.0 + poisson_ratio))


def _bulk_modulus(young_modulus: float, poisson_ratio: float) -> float:
    """Compute bulk modulus."""
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
