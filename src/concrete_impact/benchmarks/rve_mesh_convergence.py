"""Nonlinear heterogeneous RVE mesh-convergence acceptance benchmark.

Contents:
    Field collection, mesh-level selection, extrema, quantiles, and mapping diagnostics.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from concrete_impact.benchmarks.fe2_consistency import fixed_scale_history_error
from concrete_impact.benchmarks.rve_heterogeneous import (
    _elastic_material,
    _newton_settings,
    _update_settings,
    _viscoplastic_material,
)
from concrete_impact.core.config import load_yaml_config
from concrete_impact.core.progress import JsonProgressRecorder
from fem.cases import RunResult
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.quadrature.rules import make_quadrature_rule
from fem.rve import (
    build_cylindrical_two_phase_rve,
    build_rve_operator_cache,
    solve_prescribed_rve_path,
)


def run_rve_mesh_convergence_benchmark(config: dict[str, Any]) -> RunResult:
    """Compare four O-grid levels over six fixed nonlinear macro-strain paths."""
    reference = load_yaml_config(config["reference_config"])
    levels = config["mesh_levels"]
    if len(levels) < 2 or not bool(levels[-1]["reference"]):
        raise ValueError("RVE mesh convergence requires a final explicit reference level.")
    update = _update_settings(reference)
    times = np.arange(int(config["path"]["time_points"]), dtype=np.float64) * float(
        config["path"]["time_step"]
    )
    models = []
    level_results = []
    output_root = Path(config["output"]["root"])
    progress_path = output_root / "progress.json"
    summary_log_path = output_root / "summary.log"
    completed_paths = 0
    total_paths = len(levels) * len(config["path"]["families"])
    for level in levels:
        mesh = build_cylindrical_ogrid_hex8(
            tuple(reference["rve"]["lengths"]),
            float(reference["rve"]["inclusion_volume_fraction"]),
            int(level["circumferential_divisions"]),
            int(level["matrix_radial_divisions"]),
            int(level["axial_divisions"]),
        )
        model = build_cylindrical_two_phase_rve(
            mesh,
            _viscoplastic_material(reference["materials"]["matrix_viscoplastic"]),
            _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
            _newton_settings(reference),
            float(reference["verification"]["hill_mandel_power_scale"]),
        )
        models.append(model)
        path_results = []
        for path in config["path"]["families"]:
            recorder = JsonProgressRecorder(
                progress_path,
                {"mesh_level": str(level["name"]), "path_family": str(path["name"])},
                summary_log_path,
                log_accepted_steps=False,
            )
            path_results.append(
                solve_prescribed_rve_path(
                    model,
                    times,
                    np.asarray(path["macro_strains"], dtype=np.float64),
                    update,
                    recorder,
                )
            )
            completed_paths += 1
            recorder.publish_stage("rve_mesh_convergence", completed_paths, total_paths)
        level_results.append(tuple(path_results))

    reference_fields, _ = _collect_fields(
        models[-1], level_results[-1], config, levels[-1]
    )
    thresholds = config["verification"]
    comparisons = []
    for level_id, (level, model, results) in enumerate(
        zip(levels, models, level_results, strict=True)
    ):
        fields, extrema = _collect_fields(model, results, config, level)
        tangent_error = _reference_tangent_direction_error(
            model,
            results[-1],
            np.asarray(config["path"]["families"][-1]["macro_strains"], dtype=np.float64),
            times,
            update,
            thresholds,
        )
        record = {
            "name": str(level["name"]),
            "element_count": int(model.elements.shape[0]),
            "macro_stress_error": fixed_scale_history_error(
                fields["macro_stress"],
                reference_fields["macro_stress"],
                float(thresholds["stress_scale"]),
            ),
            "effective_tangent_error": fixed_scale_history_error(
                fields["effective_tangent"],
                reference_fields["effective_tangent"],
                float(thresholds["tangent_scale"]),
            ),
            "dissipation_error": fixed_scale_history_error(
                fields["dissipation"],
                reference_fields["dissipation"],
                float(thresholds["dissipation_scale"]),
            ),
            "maximum_q_error": fixed_scale_history_error(
                fields["maximum_q"],
                reference_fields["maximum_q"],
                float(thresholds["q_scale"]),
            ),
            "maximum_micro_stress_error": fixed_scale_history_error(
                fields["maximum_micro_stress"],
                reference_fields["maximum_micro_stress"],
                float(thresholds["stress_scale"]),
            ),
            "phase_average_stress_error": fixed_scale_history_error(
                fields["phase_average_stress"],
                reference_fields["phase_average_stress"],
                float(thresholds["stress_scale"]),
            ),
            "active_fraction_error": fixed_scale_history_error(
                fields["active_fraction"],
                reference_fields["active_fraction"],
                1.0,
            ),
            "maximum_hill_mandel_error": float(np.max(fields["hill_mandel_error"])),
            "maximum_reaction_error": float(np.max(fields["reaction_error"])),
            "maximum_periodic_error": float(np.max(fields["periodic_error"])),
            "tangent_direction_error": tangent_error,
            "minimum_quadrature_weight": float(
                np.min(build_rve_operator_cache(model).quadrature_weights)
            ),
        }
        record["passed"] = _level_passed(record, thresholds)
        comparisons.append(record)
        for item in extrema:
            item["mesh_level"] = str(level["name"])
        if level_id == 0:
            all_extrema = extrema
        else:
            all_extrema.extend(extrema)
    selected_level_id, selection_quality = _select_level(comparisons, levels, config)
    reference_tangent_error = float(comparisons[-1]["tangent_direction_error"])
    passed = selected_level_id is not None
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "rve_mesh_convergence.json"
    report = {
        "passed": passed,
        "selected_level": (
            None if selected_level_id is None else levels[selected_level_id]["name"]
        ),
        "selection_quality": selection_quality,
        "reference_tangent_direction_error": reference_tangent_error,
        "coarser_level_passed": any(
            bool(record["passed"])
            for record, level in zip(comparisons, levels, strict=True)
            if not bool(level["reference"])
        ),
        "comparisons": comparisons,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    extrema_path = output_root / "rve_micro_extrema.json"
    extrema_path.write_text(
        json.dumps(
            {"reference_level": levels[-1]["name"], "records": all_extrema},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    selected_level = None if selected_level_id is None else levels[selected_level_id]
    selection_path = output_root / "rve_mesh_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "passed": passed,
                "selected_level": None if selected_level is None else selected_level["name"],
                "selected_mesh": (
                    None
                    if selected_level is None
                    else {
                        "circumferential_divisions": selected_level[
                            "circumferential_divisions"
                        ],
                        "matrix_radial_divisions": selected_level[
                            "matrix_radial_divisions"
                        ],
                        "axial_divisions": selected_level["axial_divisions"],
                    }
                ),
                "reference_level": levels[-1]["name"],
                "config_sha256": _mapping_sha256(config),
                "reason": (
                    selection_quality
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics = {}
    for index, record in enumerate(comparisons):
        for name, value in record.items():
            if name in {"name", "passed"}:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(f"RVE mesh metric must be numeric: {name}={value!r}.")
            metrics[f"level_{index}__{name}"] = float(value)
    metrics["reference_tangent_direction_error"] = reference_tangent_error
    metrics["selected_level_index"] = (
        -1.0 if selected_level_id is None else float(selected_level_id)
    )
    final_recorder = JsonProgressRecorder(
        progress_path,
        {"case_name": str(config["case"]["name"])},
        summary_log_path,
        log_accepted_steps=False,
    )
    final_recorder.publish_summary(
        "rve_mesh_convergence_complete",
        {
            "completed": total_paths,
            "total": total_paths,
            "passed": passed,
            "selected_level": (
                None if selected_level_id is None else str(levels[selected_level_id]["name"])
            ),
            "selection_quality": selection_quality,
        },
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={
            "mesh_convergence_report": report_path,
            "mesh_selection_report": selection_path,
            "micro_extrema_report": extrema_path,
        },
        passed=passed,
        failure_reason=None if passed else "no_rve_mesh_level_met_acceptance",
    )


def _collect_fields(model, path_results, config, level):
    """Collect macro and microscopic histories from all prescribed path families."""
    responses = [response for path in path_results for response in path.responses]
    cache = build_rve_operator_cache(model)
    coordinates = _quadrature_coordinates(model, cache)
    radius = _inclusion_radius(config)
    tangents = []
    maximum_q = []
    maximum_micro_stress = []
    extrema = []
    response_id = 0
    for family, path in zip(config["path"]["families"], path_results, strict=True):
        for path_step, response in enumerate(path.responses, start=1):
            equivalent = _equivalent_stress(response.micro_stresses)
            maximum_point = int(np.argmax(equivalent))
            extrema.append(
                _build_extremum_record(
                    model,
                    cache,
                    coordinates,
                    radius,
                    response,
                    equivalent,
                    maximum_point,
                    str(family["name"]),
                    path_step,
                )
            )
            response_id += 1
    if response_id != len(responses):
        raise ValueError(f"RVE extremum record count mismatch for {level['name']}.")
    for response in responses:
        if response.effective_tangent is None:
            raise ValueError("RVE mesh convergence requires effective tangents.")
        tangents.append(response.effective_tangent)
        q = response.state.material_state.variables[
            "phase_matrix__equivalent_plastic_strain"
        ]
        maximum_q.append(float(np.max(q)))
        maximum_micro_stress.append(float(np.max(_equivalent_stress(response.micro_stresses))))
    return {
        "macro_stress": np.stack([response.macro_stress for response in responses]),
        "effective_tangent": np.stack(tangents),
        "dissipation": np.asarray(
            [response.dissipation_density for response in responses]
        )[:, None],
        "maximum_q": np.asarray(maximum_q)[:, None],
        "maximum_micro_stress": np.asarray(maximum_micro_stress)[:, None],
        "active_fraction": np.asarray(
            [response.diagnostics["viscoplastic_active_volume_fraction"] for response in responses]
        )[:, None],
        "phase_average_stress": np.stack(
            [
                np.concatenate(
                    [
                        np.asarray(response.phase_diagnostics[name]["average_stress"])
                        for name in model.phase_names
                    ]
                )
                for response in responses
            ]
        ),
        "hill_mandel_error": np.asarray(
            [response.diagnostics["hill_mandel_error"] for response in responses]
        ),
        "reaction_error": np.asarray(
            [response.diagnostics["reaction_antiperiodicity_error"] for response in responses]
        ),
        "periodic_error": np.asarray(
            [response.diagnostics["periodic_fluctuation_error"] for response in responses]
        ),
    }, extrema


def _level_passed(record, thresholds):
    """Evaluate fixed macro, micro, and periodic RVE mesh criteria."""
    return bool(
        record["macro_stress_error"]
        <= _threshold(thresholds, "macro_stress_tolerance", "macro_tolerance")
        and record["effective_tangent_error"]
        <= _threshold(thresholds, "tangent_tolerance", "macro_tolerance")
        and record["dissipation_error"]
        <= _threshold(thresholds, "dissipation_tolerance", "macro_tolerance")
        and record["phase_average_stress_error"]
        <= _threshold(thresholds, "phase_average_stress_tolerance", "micro_tolerance")
        and record["maximum_q_error"] <= float(thresholds["micro_tolerance"])
        and record["active_fraction_error"] <= float(thresholds["micro_tolerance"])
        and record["maximum_hill_mandel_error"] <= float(thresholds["hill_mandel_tolerance"])
        and record["maximum_reaction_error"] <= float(thresholds["reaction_tolerance"])
        and record["maximum_periodic_error"] <= float(thresholds["periodic_tolerance"])
        and record["tangent_direction_error"]
        <= float(thresholds["tangent_direction_tolerance"])
        and record["minimum_quadrature_weight"] > 0.0
    )


def _threshold(mapping, primary, legacy):
    """Read one explicit criterion while retaining old benchmark compatibility."""
    return float(mapping[primary] if primary in mapping else mapping[legacy])


def _select_level(comparisons, levels, config):
    """Apply the configured deterministic fixed-cell mesh selection rule."""
    if config.get("selection_policy") != "fixed_cell_c32_c48_c64":
        selected = _select_lowest_nonreference_level(comparisons, levels)
        quality = (
            "lowest_cost_non_reference_level_passed"
            if selected is not None
            else "no_non_reference_level_met_all_acceptance_criteria"
        )
        return selected, quality
    if len(levels) != 3 or not bool(levels[2]["reference"]):
        raise ValueError("Fixed-cell selection requires ordered c32, c48, c64 levels.")
    trend_fields = ("macro_stress_error", "effective_tangent_error", "dissipation_error")
    nonincreasing = all(
        float(comparisons[1][name]) <= float(comparisons[0][name])
        for name in trend_fields
    )
    if bool(comparisons[0]["passed"]) and bool(comparisons[1]["passed"]) and nonincreasing:
        return 0, "c32_and_c48_passed_with_nonincreasing_macro_errors"
    if bool(comparisons[1]["passed"]):
        return 1, "c48_passed_fixed_cell_acceptance"
    if not bool(comparisons[2]["passed"]):
        return None, "c64_reference_failed_internal_acceptance"
    return 2, "reference_limited_preliminary_c64_data"


def _select_lowest_nonreference_level(comparisons, levels):
    """Select the least expensive passed candidate while excluding the reference."""
    candidates = [
        (int(record["element_count"]), level_id)
        for level_id, (record, level) in enumerate(
            zip(comparisons, levels, strict=True)
        )
        if bool(record["passed"]) and not bool(level["reference"])
    ]
    return None if not candidates else min(candidates)[1]


def _build_extremum_record(
    model,
    cache,
    coordinates,
    radius,
    response,
    equivalent,
    maximum_point,
    family,
    path_step,
):
    """Build one fully located microscopic stress-extremum diagnostic."""
    quadrature_count = cache.assembly.jacobian_weights.shape[1]
    phase_id = int(cache.point_phase_ids[maximum_point])
    phase_name = model.phase_names[phase_id]
    coordinate = coordinates[maximum_point]
    q = _scatter_matrix_state(
        response.state.material_state.variables[
            "phase_matrix__equivalent_plastic_strain"
        ],
        cache.point_phase_ids,
    )
    activation = response.micro_diagnostics["yield_activation_margin"]
    active = response.micro_diagnostics["viscoplastic_active"]
    phase_statistics = {}
    for current_phase_id, current_phase_name in enumerate(model.phase_names):
        mask = cache.point_phase_ids == current_phase_id
        weights = cache.quadrature_weights[mask]
        values = equivalent[mask]
        phase_statistics[current_phase_name] = {
            "l2_equivalent_stress": float(
                np.sqrt(np.sum(weights * values**2) / np.sum(weights))
            ),
            "percentile_95": _weighted_quantile(values, weights, 0.95),
            "percentile_99": _weighted_quantile(values, weights, 0.99),
            "percentile_99_9": _weighted_quantile(values, weights, 0.999),
        }
    return {
        "path_family": family,
        "path_step": path_step,
        "phase": phase_name,
        "element_id": maximum_point // quadrature_count,
        "quadrature_id": maximum_point % quadrature_count,
        "point_id": maximum_point,
        "coordinate": coordinate.tolist(),
        "signed_interface_distance": float(np.linalg.norm(coordinate[:2]) - radius),
        "equivalent_stress": float(equivalent[maximum_point]),
        "stress": response.micro_stresses[maximum_point].tolist(),
        "strain": response.micro_strains[maximum_point].tolist(),
        "equivalent_plastic_strain": float(q[maximum_point]),
        "yield_activation_margin": (
            None if not np.isfinite(activation[maximum_point]) else float(activation[maximum_point])
        ),
        "viscoplastic_active": (
            None if not np.isfinite(active[maximum_point]) else bool(active[maximum_point] > 0.5)
        ),
        "phase_statistics": phase_statistics,
    }


def _quadrature_coordinates(model, cache):
    """Evaluate physical quadrature coordinates in flattened assembly ordering."""
    points, _ = make_quadrature_rule("hex8", model.quadrature_order)
    values, _ = evaluate_lagrange_shape_functions("hex8", points)
    return np.concatenate([values @ model.nodes[element] for element in model.elements], axis=0)


def _inclusion_radius(config):
    """Compute the analytic centered-cylinder radius used by the O-grid builder."""
    reference = load_yaml_config(config["reference_config"])
    lx, ly, _ = (float(value) for value in reference["rve"]["lengths"])
    fraction = float(reference["rve"]["inclusion_volume_fraction"])
    return float(np.sqrt(fraction * lx * ly / np.pi))


def _scatter_matrix_state(values, point_phase_ids):
    """Scatter a matrix-only state array to the common quadrature-point ordering."""
    matrix_ids = np.flatnonzero(point_phase_ids == 0)
    flattened = np.asarray(values).reshape(values.shape[0], -1)
    if flattened.shape[1] != 1 or flattened.shape[0] != matrix_ids.size:
        raise ValueError("Matrix equivalent-plastic-strain state has an unexpected shape.")
    result = np.zeros(point_phase_ids.size, dtype=np.float64)
    result[matrix_ids] = flattened[:, 0]
    return result


def _weighted_quantile(values, weights, probability):
    """Compute one deterministic volume-weighted empirical quantile."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    point = int(np.searchsorted(cumulative, probability * cumulative[-1], side="left"))
    return float(sorted_values[point])


def _mapping_sha256(mapping):
    """Hash one resolved JSON-compatible mapping for cross-stage provenance."""
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reference_tangent_direction_error(model, path_result, strains, times, update, thresholds):
    """Check the final reference-grid tangent from its prior committed state."""
    direction = np.asarray([0.37, -0.21, -0.16, 0.29, -0.41, 0.58], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    perturbation = float(thresholds["tangent_perturbation"])
    previous_state = path_result.responses[-2].state
    target = strains[-1]
    time_step = float(times[-1] - times[-2])
    from fem.rve import RVERequest, solve_rve_microequilibrium

    plus = solve_rve_microequilibrium(
        model,
        RVERequest(target + perturbation * direction, time_step, update),
        previous_state,
    )
    minus = solve_rve_microequilibrium(
        model,
        RVERequest(target - perturbation * direction, time_step, update),
        previous_state,
    )
    tangent = path_result.responses[-1].effective_tangent
    if tangent is None:
        raise ValueError("Reference RVE tangent check requires an effective tangent.")
    finite_difference = (plus.macro_stress - minus.macro_stress) / (2.0 * perturbation)
    action = tangent @ direction
    scale = max(float(np.linalg.norm(action)), float(thresholds["stress_scale"]))
    return float(np.linalg.norm(finite_difference - action) / scale)


def _equivalent_stress(stresses):
    """Compute pointwise J2 equivalent stress."""
    pressure = np.mean(stresses[:, :3], axis=1)
    deviator = stresses[:, :3] - pressure[:, None]
    squared = np.sum(deviator**2, axis=1) + 2.0 * np.sum(stresses[:, 3:] ** 2, axis=1)
    return np.sqrt(1.5 * squared)
