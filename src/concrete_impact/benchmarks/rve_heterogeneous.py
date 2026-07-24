"""Centered elastic-cylinder heterogeneous RVE acceptance benchmark.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fem.assembly.nonlinear_solid import build_hex8_material_assembly_cache
from fem.cases import RunResult
from fem.materials import LinearElasticMaterial, MaterialUpdateSettings
from fem.materials.plasticity import J2ViscoplasticMaterial
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.rve import (
    RVERequest,
    build_cylindrical_two_phase_rve,
    compute_linear_rve_dynamic_tangent,
    compute_rve_dynamic_screening,
    initialize_rve_state,
    solve_rve_microequilibrium,
)
from fem.solvers import ArmijoSettings, NonlinearNewtonSettings
from fem.tensors import engineering_stiffness_to_mandel


def run_cylindrical_inclusion_rve_benchmark(config: dict[str, Any]) -> RunResult:
    """Verify geometry, elastic bounds, and nonlinear phase consistency."""
    meshes = [
        build_cylindrical_ogrid_hex8(
            tuple(config["rve"]["lengths"]),
            float(config["rve"]["inclusion_volume_fraction"]),
            int(divisions),
            int(config["rve"]["matrix_radial_divisions"]),
            int(config["rve"]["axial_divisions"]),
        )
        for divisions in config["rve"]["circumferential_divisions"]
    ]
    metrics = {
        f"geometry_area_error_{divisions}": abs(
            mesh.discrete_inclusion_area - mesh.analytic_inclusion_area
        )
        / mesh.analytic_inclusion_area
        for divisions, mesh in zip(
            config["rve"]["circumferential_divisions"], meshes, strict=True
        )
    }
    geometry_errors = [
        metrics[f"geometry_area_error_{divisions}"]
        for divisions in config["rve"]["circumferential_divisions"]
    ]
    metrics["geometry_convergence_violation"] = max(
        [0.0]
        + [
            current - previous
            for previous, current in zip(geometry_errors[:-1], geometry_errors[1:], strict=True)
        ]
    )
    metrics["minimum_quadrature_weight"] = min(
        float(
            np.min(
                build_hex8_material_assembly_cache(mesh.nodes, mesh.elements, 2)
                .jacobian_weights
            )
        )
        for mesh in meshes
    )
    mesh = meshes[-1]
    newton = _newton_settings(config)
    update = _update_settings(config)
    strain = np.asarray(config["verification"]["elastic_macro_strain"], dtype=np.float64)
    matrix_elastic = _elastic_material("matrix_equal", config["materials"]["matrix_elastic"])
    equal_model = build_cylindrical_two_phase_rve(
        mesh,
        matrix_elastic,
        matrix_elastic,
        newton,
        float(config["verification"]["hill_mandel_power_scale"]),
    )
    equal = solve_rve_microequilibrium(
        equal_model,
        RVERequest(strain, 1.0, update),
        initialize_rve_state(equal_model),
    )
    elasticity = _elasticity_matrix(matrix_elastic)
    metrics["equal_material_stress_error"] = float(
        np.linalg.norm(equal.macro_stress - elasticity @ strain)
        / float(config["verification"]["stress_scale"])
    )
    metrics["equal_material_tangent_error"] = float(
        np.linalg.norm(equal.effective_tangent - elasticity) / np.linalg.norm(elasticity)
    )

    inclusion = _elastic_material("inclusion", config["materials"]["inclusion_elastic"])
    elastic_responses = []
    for level_mesh in meshes:
        level_model = build_cylindrical_two_phase_rve(
            level_mesh,
            matrix_elastic,
            inclusion,
            newton,
            float(config["verification"]["hill_mandel_power_scale"]),
        )
        elastic_responses.append(
            solve_rve_microequilibrium(
                level_model,
                RVERequest(strain, 1.0, update),
                initialize_rve_state(level_model),
            )
        )
    elastic = elastic_responses[-1]
    elastic_model = level_model
    if len(elastic_responses) > 1:
        previous_elastic = elastic_responses[-2]
        metrics["mechanical_stress_mesh_error"] = float(
            np.linalg.norm(elastic.macro_stress - previous_elastic.macro_stress)
            / float(config["verification"]["stress_scale"])
        )
        metrics["mechanical_tangent_mesh_error"] = float(
            np.linalg.norm(
                _require_tangent(elastic.effective_tangent)
                - _require_tangent(previous_elastic.effective_tangent)
            )
            / np.linalg.norm(_require_tangent(elastic.effective_tangent))
        )
    lower, upper = _voigt_reuss_bounds(
        matrix_elastic,
        inclusion,
        elastic.phase_diagnostics["inclusion"]["volume_fraction"],
    )
    effective_mandel = _to_mandel(_require_tangent(elastic.effective_tangent))
    metrics["elastic_lower_bound_violation"] = max(
        0.0,
        -float(np.linalg.eigvalsh(effective_mandel - lower).min()),
    )
    metrics["elastic_upper_bound_violation"] = max(
        0.0,
        -float(np.linalg.eigvalsh(upper - effective_mandel).min()),
    )
    metrics["elastic_tangent_asymmetry"] = float(
        np.linalg.norm(effective_mandel - effective_mandel.T) / np.linalg.norm(effective_mandel)
    )
    metrics["elastic_hill_mandel_error"] = float(elastic.diagnostics["hill_mandel_error"])
    metrics["elastic_periodic_class_reaction_error"] = float(
        elastic.diagnostics["periodic_class_reaction_error"]
    )
    loading_frequency = float(config["verification"]["maximum_loading_angular_frequency"])
    screening = compute_rve_dynamic_screening(elastic_model, update, loading_frequency)
    static_dynamic_tangent = compute_linear_rve_dynamic_tangent(elastic_model, update, 0.0)
    loading_dynamic_tangent = compute_linear_rve_dynamic_tangent(
        elastic_model, update, loading_frequency
    )
    metrics["dynamic_scale_ratio"] = screening.scale_ratio
    metrics["dynamic_frequency_ratio"] = screening.frequency_ratio
    metrics["dynamic_tangent_difference"] = float(
        np.linalg.norm(loading_dynamic_tangent - static_dynamic_tangent)
        / np.linalg.norm(static_dynamic_tangent)
    )
    metrics["quasistatic_applicability_passed"] = float(
        screening.scale_ratio <= float(config["verification"]["tolerances"]["dynamic_ratio"])
        and screening.frequency_ratio
        <= float(config["verification"]["tolerances"]["dynamic_ratio"])
        and metrics["dynamic_tangent_difference"]
        <= float(config["verification"]["tolerances"]["dynamic_tangent"])
    )

    matrix_vp = _viscoplastic_material(config["materials"]["matrix_viscoplastic"])
    nonlinear_model = build_cylindrical_two_phase_rve(
        mesh,
        matrix_vp,
        inclusion,
        newton,
        float(config["verification"]["hill_mandel_power_scale"]),
    )
    state = initialize_rve_state(nonlinear_model)
    previous_q = np.zeros_like(
        state.material_state.variables["phase_matrix__equivalent_plastic_strain"]
    )
    maximum_phase_average_error = 0.0
    minimum_q_increment = np.inf
    minimum_dissipation = np.inf
    maximum_q = 0.0
    maximum_dissipation = 0.0
    final_committed_state = state
    final_target_strain = np.zeros(6)
    final_response = None
    for path_strain in np.asarray(config["path"]["macro_strains"], dtype=np.float64):
        final_committed_state = state
        final_target_strain = path_strain
        response = solve_rve_microequilibrium(
            nonlinear_model,
            RVERequest(path_strain, float(config["path"]["time_step"]), update),
            state,
        )
        q_value = response.state.material_state.variables[
            "phase_matrix__equivalent_plastic_strain"
        ]
        minimum_q_increment = min(minimum_q_increment, float(np.min(q_value - previous_q)))
        minimum_dissipation = min(minimum_dissipation, response.dissipation_density)
        maximum_q = max(maximum_q, float(np.max(q_value)))
        maximum_dissipation = max(maximum_dissipation, response.dissipation_density)
        reconstructed = sum(
            phase["volume_fraction"] * phase["average_stress"]
            for phase in response.phase_diagnostics.values()
        )
        maximum_phase_average_error = max(
            maximum_phase_average_error,
            float(np.linalg.norm(reconstructed - response.macro_stress)),
        )
        state = response.state
        previous_q = q_value
        final_response = response
    if final_response is None:
        raise ValueError("Nonlinear heterogeneous benchmark requires a nonempty strain path.")
    direction = np.asarray([0.37, -0.21, -0.16, 0.29, -0.41, 0.58])
    direction /= np.linalg.norm(direction)
    perturbation = float(config["verification"]["tangent_perturbation"])
    plus = solve_rve_microequilibrium(
        nonlinear_model,
        RVERequest(
            final_target_strain + perturbation * direction,
            float(config["path"]["time_step"]),
            update,
        ),
        final_committed_state,
    )
    minus = solve_rve_microequilibrium(
        nonlinear_model,
        RVERequest(
            final_target_strain - perturbation * direction,
            float(config["path"]["time_step"]),
            update,
        ),
        final_committed_state,
    )
    directional_difference = (plus.macro_stress - minus.macro_stress) / (2.0 * perturbation)
    directional_tangent = _require_tangent(final_response.effective_tangent) @ direction
    directional_scale = max(
        float(np.linalg.norm(directional_tangent)),
        float(config["verification"]["stress_scale"]),
    )
    metrics["nonlinear_tangent_direction_error"] = float(
        np.linalg.norm(directional_difference - directional_tangent)
        / directional_scale
    )
    metrics["nonlinear_minimum_q_increment"] = minimum_q_increment
    metrics["nonlinear_minimum_dissipation"] = minimum_dissipation
    metrics["nonlinear_maximum_q"] = maximum_q
    metrics["nonlinear_maximum_dissipation"] = maximum_dissipation
    metrics["nonlinear_phase_average_error"] = maximum_phase_average_error
    metrics["inclusion_plastic_state_count"] = float(
        sum("phase_inclusion__" in name for name in state.material_state.variables)
    )

    tolerances = config["verification"]["tolerances"]
    passed = _passes(metrics, tolerances)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps({"passed": passed, "metrics": metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={"metrics": metrics_path},
        passed=passed,
    )


def _elastic_material(name: str, spec: dict[str, Any]) -> LinearElasticMaterial:
    """Build one benchmark elastic phase."""
    return LinearElasticMaterial(
        name,
        float(spec["density"]),
        float(spec["young_modulus"]),
        float(spec["poisson_ratio"]),
    )


def _viscoplastic_material(spec: dict[str, Any]) -> J2ViscoplasticMaterial:
    """Build the benchmark J2-VP matrix phase."""
    return J2ViscoplasticMaterial(
        "matrix_viscoplastic",
        float(spec["density"]),
        float(spec["young_modulus"]),
        float(spec["poisson_ratio"]),
        float(spec["yield_stress"]),
        float(spec["hardening_modulus"]),
        float(spec["time_scale"]),
        float(spec["reference_stress"]),
        float(spec["rate_exponent"]),
    )


def _elasticity_matrix(material: LinearElasticMaterial) -> np.ndarray:
    """Build the engineering-Voigt isotropic stiffness matrix."""
    shear = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    lame = material.young_modulus * material.poisson_ratio / (
        (1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)
    )
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = lame
    matrix[:3, :3] += 2.0 * shear * np.eye(3)
    matrix[3:, 3:] = shear * np.eye(3)
    return matrix


def _to_mandel(engineering_tangent: np.ndarray) -> np.ndarray:
    """Convert power-conjugate engineering-Voigt stiffness to Mandel form."""
    return engineering_stiffness_to_mandel(engineering_tangent)


def _require_tangent(tangent: np.ndarray | None) -> np.ndarray:
    """Require the effective tangent requested by the acceptance benchmark."""
    if tangent is None:
        raise ValueError("Cylindrical-inclusion benchmark requires an effective tangent.")
    return tangent


def _voigt_reuss_bounds(
    matrix: LinearElasticMaterial,
    inclusion: LinearElasticMaterial,
    inclusion_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two-phase elastic energy bounds in Mandel coordinates."""
    matrix_c = _to_mandel(_elasticity_matrix(matrix))
    inclusion_c = _to_mandel(_elasticity_matrix(inclusion))
    matrix_fraction = 1.0 - inclusion_fraction
    upper = matrix_fraction * matrix_c + inclusion_fraction * inclusion_c
    lower = np.linalg.inv(
        matrix_fraction * np.linalg.inv(matrix_c)
        + inclusion_fraction * np.linalg.inv(inclusion_c)
    )
    return lower, upper


def _newton_settings(config: dict[str, Any]) -> NonlinearNewtonSettings:
    """Build the benchmark RVE Newton settings."""
    values = config["rve"]["newton"]
    return NonlinearNewtonSettings(
        int(values["max_iterations"]),
        float(values["residual_absolute_tolerance"]),
        float(values["residual_relative_tolerance"]),
        float(values["increment_absolute_tolerance"]),
        float(values["increment_relative_tolerance"]),
        ArmijoSettings(enabled=True),
    )


def _update_settings(config: dict[str, Any]) -> MaterialUpdateSettings:
    """Build local matrix integration settings."""
    values = config["rve"]["material_update"]
    return MaterialUpdateSettings(
        int(values["max_iterations"]),
        float(values["yield_relative_tolerance"]),
        float(values["residual_absolute_tolerance"]),
        float(values["residual_relative_tolerance"]),
    )


def _passes(metrics: dict[str, float], tolerances: dict[str, float]) -> bool:
    """Evaluate explicit per-family acceptance thresholds."""
    for name, value in metrics.items():
        if "geometry" in name and value > float(tolerances["geometry_area"]):
            return False
        if "minimum_quadrature_weight" in name and value <= 0.0:
            return False
        if "stress_error" in name and value > float(tolerances["stress"]):
            return False
        if "tangent_error" in name and value > float(tolerances["tangent"]):
            return False
        if "mechanical_stress_mesh_error" in name and value > float(tolerances["mesh_stress"]):
            return False
        if "mechanical_tangent_mesh_error" in name and value > float(tolerances["mesh_tangent"]):
            return False
        if "nonlinear_tangent_direction_error" in name and value > float(
            tolerances["nonlinear_tangent_direction"]
        ):
            return False
        if "violation" in name and value > float(tolerances["bound"]):
            return False
        if "asymmetry" in name and value > float(tolerances["tangent"]):
            return False
        if "hill_mandel" in name and value > float(tolerances["hill_mandel"]):
            return False
        if "periodic_class_reaction_error" in name and value > float(
            tolerances["reaction"]
        ):
            return False
        if "q_increment" in name and value < -float(tolerances["state"]):
            return False
        if "dissipation" in name and value < -float(tolerances["energy"]):
            return False
        if "maximum_q" in name and value <= float(tolerances["plastic_activity"]):
            return False
        if "maximum_dissipation" in name and value <= float(tolerances["plastic_activity"]):
            return False
        if "phase_average_error" in name and value > float(tolerances["stress"]):
            return False
        if "plastic_state_count" in name and value != 0.0:
            return False
    return True
