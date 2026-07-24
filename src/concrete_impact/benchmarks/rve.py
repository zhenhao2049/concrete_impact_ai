"""Single-phase periodic RVE acceptance benchmark.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fem.cases import RunResult
from fem.materials import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
    build_material,
)
from fem.mesh.structured import build_structured_hex8_box
from fem.rve import (
    RVEPathResult,
    RVERequest,
    StructuredHex8RVE,
    build_structured_hex8_periodic_constraint,
    initialize_rve_state,
    solve_rve_microequilibrium,
    write_rve_path_hdf5,
)
from fem.solvers import ArmijoSettings, NonlinearNewtonSettings


def run_single_phase_rve_benchmark(config: dict[str, Any]) -> RunResult:
    """Run homogeneous periodic RVE equivalence and tangent checks."""
    material = build_material(config["rve"]["material"])
    update_settings = _build_material_update_settings(config)
    strains = np.asarray(config["path"]["macro_strains"], dtype=np.float64)
    time_step = float(config["path"]["time_step"])
    times = time_step * np.arange(1, strains.shape[0] + 1, dtype=np.float64)
    stress_scale = float(config["verification"]["stress_scale"])
    strain_scale = float(config["verification"]["strain_scale"])
    grid_results: dict[tuple[int, int, int], RVEPathResult] = {}
    metrics: dict[str, float] = {}

    for division_value in config["rve"]["mesh_divisions"]:
        divisions = (int(division_value),) * 3
        model = _build_rve_model(config, material, divisions)
        rve_state = initialize_rve_state(model)
        point_state = material.initialize_state(1)
        previous_strain = np.zeros(6, dtype=np.float64)
        responses = []
        maximum_stress_error = 0.0
        maximum_state_error = 0.0
        maximum_tangent_error = 0.0
        maximum_periodic_error = 0.0
        maximum_reaction_error = 0.0
        maximum_hill_mandel_error = 0.0

        for macro_strain in strains:
            committed_rve_state = rve_state
            response = solve_rve_microequilibrium(
                model,
                RVERequest(
                    macro_strain=macro_strain,
                    time_step=time_step,
                    material_update_settings=update_settings,
                ),
                committed_rve_state,
            )
            point_response = material.update(
                MaterialPointRequest(
                    strains=macro_strain.reshape(1, 6),
                    strain_rates=((macro_strain - previous_strain) / time_step).reshape(1, 6),
                    time_step=time_step,
                    kinematics="three_dimensional",
                    update_settings=update_settings,
                ),
                point_state,
                MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
            )
            if response.effective_tangent is None or point_response.tangents is None:
                raise ValueError("Single-phase RVE acceptance requires both algorithmic tangents.")
            maximum_stress_error = max(
                maximum_stress_error,
                float(np.linalg.norm(response.macro_stress - point_response.stresses[0]))
                / stress_scale,
            )
            maximum_tangent_error = max(
                maximum_tangent_error,
                float(np.linalg.norm(response.effective_tangent - point_response.tangents[0]))
                / float(np.linalg.norm(point_response.tangents[0])),
            )
            for name, point_values in point_response.state.variables.items():
                expected = np.repeat(
                    point_values,
                    response.state.material_state.variables[name].shape[0],
                    axis=0,
                )
                maximum_state_error = max(
                    maximum_state_error,
                    float(
                        np.max(
                            np.abs(response.state.material_state.variables[name] - expected)
                        )
                    )
                    / strain_scale,
                )
            maximum_periodic_error = max(
                maximum_periodic_error,
                float(response.diagnostics["periodic_fluctuation_error"]),
            )
            maximum_reaction_error = max(
                maximum_reaction_error,
                float(response.diagnostics["reaction_antiperiodicity_error"]),
            )
            maximum_hill_mandel_error = max(
                maximum_hill_mandel_error,
                float(response.diagnostics["hill_mandel_error"]),
            )
            responses.append(response)
            rve_state = response.state
            point_state = point_response.state
            previous_strain = macro_strain

        prefix = f"grid_{division_value}"
        metrics[f"{prefix}_stress_error"] = maximum_stress_error
        metrics[f"{prefix}_state_error"] = maximum_state_error
        metrics[f"{prefix}_tangent_error"] = maximum_tangent_error
        metrics[f"{prefix}_periodic_error"] = maximum_periodic_error
        metrics[f"{prefix}_reaction_error"] = maximum_reaction_error
        metrics[f"{prefix}_hill_mandel_error"] = maximum_hill_mandel_error
        grid_results[divisions] = RVEPathResult(times=times, responses=tuple(responses))

    largest_division = int(config["rve"]["mesh_divisions"][-1])
    largest_model = _build_rve_model(
        config,
        material,
        (largest_division,) * 3,
    )
    largest_result = grid_results[(largest_division,) * 3]
    metrics["effective_tangent_direction_error"] = _compute_directional_tangent_error(
        largest_model,
        update_settings,
        strains,
        time_step,
        float(config["verification"]["tangent_perturbation"]),
    )
    first_result = grid_results[(int(config["rve"]["mesh_divisions"][0]),) * 3]
    metrics["grid_stress_difference"] = max(
        float(np.linalg.norm(first.macro_stress - last.macro_stress)) / stress_scale
        for first, last in zip(first_result.responses, largest_result.responses, strict=True)
    )

    verification = config["verification"]
    passed = all(
        value <= float(verification[tolerance_name])
        for metric_name, value in metrics.items()
        for tolerance_name in (_metric_tolerance_name(metric_name),)
    )
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = write_rve_path_hdf5(
        output_root / str(config["output"]["dataset_name"]),
        largest_model,
        "acceptance_path",
        largest_result,
        {"case": config["case"], "rve": config["rve"], "path": config["path"]},
    )
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps({"passed": passed, "metrics": metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={"dataset": dataset_path, "metrics": metrics_path},
        passed=passed,
    )


def _build_rve_model(
    config: dict[str, Any],
    material: Any,
    divisions: tuple[int, int, int],
) -> StructuredHex8RVE:
    """Build one configured structured single-phase RVE."""
    length_values = config["rve"]["lengths"]
    lengths = (
        float(length_values[0]),
        float(length_values[1]),
        float(length_values[2]),
    )
    nodes, elements = build_structured_hex8_box(lengths, divisions)
    newton = config["rve"]["newton"]
    armijo = newton["armijo"]

    return StructuredHex8RVE(
        nodes=nodes,
        elements=elements,
        divisions=divisions,
        material=material,
        constraint=build_structured_hex8_periodic_constraint(nodes, divisions),
        newton_settings=NonlinearNewtonSettings(
            max_iterations=int(newton["max_iterations"]),
            residual_absolute_tolerance=float(newton["residual_absolute_tolerance"]),
            residual_relative_tolerance=float(newton["residual_relative_tolerance"]),
            increment_absolute_tolerance=float(newton["increment_absolute_tolerance"]),
            increment_relative_tolerance=float(newton["increment_relative_tolerance"]),
            armijo=ArmijoSettings(
                enabled=bool(armijo["enabled"]),
                sufficient_decrease=float(armijo["sufficient_decrease"]),
                reduction_factor=float(armijo["reduction_factor"]),
                max_backtracks=int(armijo["max_backtracks"]),
            ),
        ),
        quadrature_order=int(config["rve"]["quadrature_order"]),
        hill_mandel_power_scale=float(config["verification"]["hill_mandel_power_scale"]),
    )


def _build_material_update_settings(config: dict[str, Any]) -> MaterialUpdateSettings:
    """Build strict local J2 integration settings."""
    settings = config["rve"]["material_update"]

    return MaterialUpdateSettings(
        max_iterations=int(settings["max_iterations"]),
        yield_relative_tolerance=float(settings["yield_relative_tolerance"]),
        residual_absolute_tolerance=float(settings["residual_absolute_tolerance"]),
        residual_relative_tolerance=float(settings["residual_relative_tolerance"]),
    )


def _compute_directional_tangent_error(
    model: StructuredHex8RVE,
    update_settings: MaterialUpdateSettings,
    strains: np.ndarray,
    time_step: float,
    perturbation: float,
) -> float:
    """Check the final-step effective tangent from one committed state."""
    state = initialize_rve_state(model)
    for macro_strain in strains[:-1]:
        response = solve_rve_microequilibrium(
            model,
            RVERequest(macro_strain, time_step, update_settings),
            state,
        )
        state = response.state
    target = strains[-1]
    response = solve_rve_microequilibrium(
        model,
        RVERequest(target, time_step, update_settings),
        state,
    )
    if response.effective_tangent is None:
        raise ValueError("Directional RVE tangent verification requires an effective tangent.")
    direction = np.asarray([0.31, -0.22, 0.17, 0.41, -0.53, 0.59])
    direction /= np.linalg.norm(direction)
    plus = solve_rve_microequilibrium(
        model,
        RVERequest(target + perturbation * direction, time_step, update_settings),
        state,
    )
    minus = solve_rve_microequilibrium(
        model,
        RVERequest(target - perturbation * direction, time_step, update_settings),
        state,
    )
    numerical = (plus.macro_stress - minus.macro_stress) / (2.0 * perturbation)
    analytic = response.effective_tangent @ direction

    return float(np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic))


def _metric_tolerance_name(metric_name: str) -> str:
    """Map benchmark metric names to explicit configured tolerances."""
    if "direction" in metric_name:
        return "directional_tangent_tolerance"
    if "stress" in metric_name:
        return "stress_tolerance"
    if "state" in metric_name:
        return "state_tolerance"
    if "tangent" in metric_name:
        return "tangent_tolerance"
    if "periodic" in metric_name:
        return "periodic_tolerance"
    if "reaction" in metric_name:
        return "reaction_tolerance"
    if "hill_mandel" in metric_name:
        return "hill_mandel_tolerance"
    raise KeyError(f"No tolerance is defined for RVE metric {metric_name!r}.")
