"""High-fidelity explicit-implicit FE2 impact consistency benchmark.

Contents:
    Five-stage run construction, mesh resolution, histories, hashes, and physical checks.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np

from concrete_impact.benchmarks.fe2_consistency import (
    FE2PathScales,
    FE2PathTolerances,
    conservative_speed_ratio,
    evaluate_path_acceptance,
    exact_common_time_indices,
    fixed_scale_history_error,
    require_transverse_macro_resolution,
    select_accepted_snapshot_indices,
)
from concrete_impact.benchmarks.rve_plastic_impact import (
    solve_multiphase_rve_plastic_impact_scheme,
)
from concrete_impact.core.config import load_yaml_config
from concrete_impact.core.progress import JsonProgressRecorder
from fem.cases import RunResult
from fem.post.material_dynamic import extract_quadrature_strain_history
from fem.rve import solve_prescribed_rve_path


def run_fe2_path_consistency_benchmark(config: dict[str, Any]) -> RunResult:
    """Run fixed FE2 time/space refinements and select an accurate time integrator."""
    baseline_values = config["macro_mesh"]["baseline"]
    refined_values = config["macro_mesh"]["refined"]
    if len(baseline_values) != 3 or len(refined_values) != 3:
        raise ValueError("FE2 macro mesh divisions require exactly three components.")
    baseline_divisions = (
        int(baseline_values[0]),
        int(baseline_values[1]),
        int(baseline_values[2]),
    )
    refined_divisions = (
        int(refined_values[0]),
        int(refined_values[1]),
        int(refined_values[2]),
    )
    require_transverse_macro_resolution(baseline_divisions)
    require_transverse_macro_resolution(refined_divisions)
    if refined_divisions[0] <= baseline_divisions[0]:
        raise ValueError("Refined FE2 macro mesh must increase the axial element count.")

    base = load_yaml_config(config["base_config"])
    selection_hash = _resolve_rve_mesh(base, config["rve_mesh"])
    coarse_config = _build_level_config(
        base, baseline_divisions, config["time_levels"]["coarse"]
    )
    fine_config = _build_level_config(
        base, baseline_divisions, config["time_levels"]["fine"]
    )
    refined_config = _build_level_config(
        base, refined_divisions, config["time_levels"]["fine"]
    )
    output_root = Path(config["output"]["root"])
    config_hash = _mapping_sha256(config)
    run_specs = _build_five_stage_run_specs(
        coarse_config, fine_config, refined_config
    )
    runs = {}
    for stage_id, (stage_name, stage_config, scheme) in enumerate(run_specs, start=1):
        recorder = JsonProgressRecorder(
            output_root / "progress.json",
            {"benchmark_stage": stage_name, "stage_id": stage_id, "stage_count": 5},
        )
        run = solve_multiphase_rve_plastic_impact_scheme(
            stage_config,
            cast(Literal["explicit", "implicit"], scheme),
            recorder,
        )
        runs[stage_name] = run
        _write_stage_history(
            output_root / "correctness" / f"{stage_name}.h5",
            run,
            config_hash,
            selection_hash,
            stage_name,
        )
        recorder.publish_stage("fe2_correctness", stage_id, len(run_specs))

    scales = FE2PathScales(**config["verification"]["scales"])
    tolerances = FE2PathTolerances(**config["verification"]["path_tolerances"])
    event_tolerance = float(config["time_levels"]["fine"]["time_step"])
    comparisons = {
        "explicit_time": _compare_simulation_histories(
            runs["explicit_fine"].bundle,
            runs["explicit_fine"].solution,
            runs["explicit_coarse"].bundle,
            runs["explicit_coarse"].solution,
            scales,
        ),
        "implicit_time": _compare_simulation_histories(
            runs["implicit_fine"].bundle,
            runs["implicit_fine"].solution,
            runs["implicit_coarse"].bundle,
            runs["implicit_coarse"].solution,
            scales,
        ),
        "explicit_implicit_fine": _compare_simulation_histories(
            runs["explicit_fine"].bundle,
            runs["explicit_fine"].solution,
            runs["implicit_fine"].bundle,
            runs["implicit_fine"].solution,
            scales,
        ),
        "explicit_implicit_coarse": _compare_simulation_histories(
            runs["explicit_fine"].bundle,
            runs["explicit_fine"].solution,
            runs["implicit_coarse"].bundle,
            runs["implicit_coarse"].solution,
            scales,
        ),
        "explicit_space": _compare_spatial_histories(
            runs["explicit_fine"].bundle,
            runs["explicit_fine"].solution,
            runs["explicit_refined"].bundle,
            runs["explicit_refined"].solution,
            scales,
        ),
    }
    acceptance = {
        name: evaluate_path_acceptance(values, tolerances, event_tolerance)
        for name, values in comparisons.items()
    }
    numerical_passed = all(all(checks.values()) for checks in acceptance.values())
    physical_checks = _physical_checks(
        {
            "explicit": runs["explicit_fine"].solution,
            "implicit": runs["implicit_fine"].solution,
        },
        config["verification"]["physical"],
    )
    numerical_passed = numerical_passed and all(physical_checks.values())

    performance: dict[str, Any] = {
        "measured": False,
        "implicit_production_eligible": False,
        "selected_scheme": "explicit_fine",
        "reason": "numerical_acceptance_failed",
    }
    performance_enabled = bool(config["performance"]["enabled"])
    if numerical_passed and performance_enabled:
        implicit_name = (
            "implicit_coarse"
            if all(acceptance["explicit_implicit_coarse"].values())
            else "implicit_fine"
        )
        implicit_level = (
            config["time_levels"]["coarse"]
            if implicit_name == "implicit_coarse"
            else config["time_levels"]["fine"]
        )
        performance = _measure_performance(
            base,
            baseline_divisions,
            config["time_levels"]["fine"],
            implicit_level,
            config["performance"],
            implicit_name,
        )

    elif numerical_passed:
        performance = {
            "measured": False,
            "implicit_production_eligible": False,
            "selected_scheme": "not_evaluated_in_preflight",
            "reason": "performance_disabled_by_explicit_configuration",
        }

    metrics = _flatten_metrics(comparisons, acceptance, physical_checks, performance)
    passed = numerical_passed and (
        bool(performance["measured"]) if performance_enabled else True
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "fe2_path_consistency.json"
    report = {
        "passed": passed,
        "comparisons": comparisons,
        "acceptance": acceptance,
        "physical_checks": physical_checks,
        "performance": performance,
        "resolved_config": config,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    output_paths = {"consistency_report": report_path}
    if numerical_passed:
        output_paths["accepted_micro_snapshots"] = _write_micro_snapshots(
            output_root,
            runs["explicit_fine"],
            config["verification"]["physical"],
        )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths=output_paths,
        passed=passed,
        failure_reason=None if passed else "fe2_path_consistency_acceptance_failed",
    )


def _build_level_config(
    base: dict[str, Any],
    divisions: tuple[int, int, int],
    level: dict[str, Any],
) -> dict[str, Any]:
    """Build one explicit configuration without changing the source mapping."""
    result = copy.deepcopy(base)
    result["model"]["mesh"]["divisions"] = list(divisions)
    final_time = float(level["final_time"])
    explicit_dt = float(level["time_step"])
    implicit_dt = float(level["time_step"])
    explicit_steps = _exact_step_count(final_time, explicit_dt)
    implicit_steps = _exact_step_count(final_time, implicit_dt)
    result["analysis"]["explicit"].update(
        {"time_step": explicit_dt, "num_steps": explicit_steps}
    )
    result["analysis"]["implicit"].update(
        {"time_step": implicit_dt, "num_steps": implicit_steps}
    )
    result["output"].update(
        {"save_vtk": False, "save_quadrature_vtk": False, "save_history": False}
    )
    return result


def _build_five_stage_run_specs(coarse_config, fine_config, refined_config):
    """Build the immutable five-stage FE2 correctness task sequence."""
    return (
        ("explicit_coarse", coarse_config, "explicit"),
        ("explicit_fine", fine_config, "explicit"),
        ("implicit_coarse", coarse_config, "implicit"),
        ("implicit_fine", fine_config, "implicit"),
        ("explicit_refined", refined_config, "explicit"),
    )


def _resolve_rve_mesh(base: dict[str, Any], mesh_config: dict[str, Any]) -> str:
    """Apply an explicit preflight mesh or one strictly accepted selection report."""
    mode = str(mesh_config["mode"])
    if mode == "fixed_preflight":
        selected = mesh_config
        selection_hash = _mapping_sha256(mesh_config)
    elif mode == "selected":
        selection_path = Path(mesh_config["selection_report"])
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_config = load_yaml_config(mesh_config["selection_config"])
        expected_config_hash = _mapping_sha256(selection_config)
        if str(payload["config_sha256"]) != expected_config_hash:
            raise ValueError(
                "RVE mesh selection report does not match its current configuration: "
                f"report={selection_path}."
            )
        if not bool(payload["passed"]):
            raise ValueError(
                "High-fidelity FE2 requires a passed non-reference RVE mesh selection: "
                f"report={selection_path}, reason={payload['reason']}."
            )
        selected = payload["selected_mesh"]
        selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    else:
        raise ValueError(f"Unsupported FE2 RVE mesh resolution mode: {mode}.")
    base["rve"]["circumferential_divisions"] = int(
        selected["circumferential_divisions"]
    )
    base["rve"]["matrix_radial_divisions"] = int(
        selected["matrix_radial_divisions"]
    )
    base["rve"]["axial_divisions"] = int(selected["axial_divisions"])
    return selection_hash


def _write_stage_history(
    path: Path,
    run,
    config_hash: str,
    selection_hash: str,
    stage_name: str,
) -> None:
    """Write one accepted FE2 stage with strict provenance hashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with h5py.File(path, "r") as existing:
            observed = (
                str(existing.attrs["config_sha256"]),
                str(existing.attrs["rve_selection_sha256"]),
            )
        expected = (config_hash, selection_hash)
        if observed != expected:
            raise ValueError(
                "Existing FE2 stage provenance does not match the current calculation: "
                f"path={path}, observed={observed}, expected={expected}."
            )
    solution = run.solution
    strains = extract_quadrature_strain_history(run.bundle, solution.displacement)
    right_nodes = run.bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = run.bundle.mesh_info.dof_map[right_nodes, 0]
    q_key = "phase_matrix__equivalent_plastic_strain"
    with h5py.File(path, "w") as handle:
        handle.attrs["status"] = "complete"
        handle.attrs["stage_name"] = stage_name
        handle.attrs["config_sha256"] = config_hash
        handle.attrs["rve_selection_sha256"] = selection_hash
        handle.create_dataset("time", data=solution.times, compression="lzf")
        handle.create_dataset(
            "right_displacement",
            data=np.mean(solution.displacement[:, right_dofs], axis=1),
            compression="lzf",
        )
        handle.create_dataset(
            "right_velocity",
            data=np.mean(solution.velocity[:, right_dofs], axis=1),
            compression="lzf",
        )
        handle.create_dataset(
            "nodal_displacement", data=solution.displacement, compression="lzf"
        )
        handle.create_dataset("nodal_velocity", data=solution.velocity, compression="lzf")
        handle.create_dataset("macro_strain", data=strains, compression="lzf")
        handle.create_dataset("macro_stress", data=solution.stresses, compression="lzf")
        handle.create_dataset(
            "equivalent_plastic_strain",
            data=solution.material_diagnostics[q_key],
            compression="lzf",
        )
        handle.create_dataset(
            "dissipated_energy", data=solution.dissipated_energy, compression="lzf"
        )
        handle.create_dataset(
            "incremental_work_density",
            data=solution.material_diagnostics["incremental_work_density"],
            compression="lzf",
        )
        handle.create_dataset(
            "viscoplastic_active_volume_fraction",
            data=solution.viscoplastic_active_volume_fraction,
            compression="lzf",
        )
        handle.create_dataset(
            "energy_residual", data=solution.energy_residual, compression="lzf"
        )


def _mapping_sha256(mapping: dict[str, Any]) -> str:
    """Hash one resolved JSON-compatible benchmark mapping."""
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exact_step_count(final_time: float, time_step: float) -> int:
    """Require a time step that exactly partitions the configured final time."""
    quotient = final_time / time_step
    steps = int(round(quotient))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(quotient)) * 32.0
    if not np.isclose(quotient, steps, rtol=0.0, atol=tolerance):
        raise ValueError(
            "FE2 consistency final time must be an integer multiple of the time step: "
            f"final_time={final_time}, time_step={time_step}."
        )
    return steps


def _compare_simulation_histories(bundle_a, solution_a, bundle_b, solution_b, scales):
    """Compare accepted histories on their exact common time grid."""
    if bundle_a.mesh_info.elements.shape != bundle_b.mesh_info.elements.shape:
        raise ValueError("Time-refinement comparison requires the same macro mesh.")
    if solution_a.times.size >= solution_b.times.size:
        fine_bundle, fine_solution = bundle_a, solution_a
        coarse_bundle, coarse_solution = bundle_b, solution_b
    else:
        fine_bundle, fine_solution = bundle_b, solution_b
        coarse_bundle, coarse_solution = bundle_a, solution_a
    indices = exact_common_time_indices(fine_solution.times, coarse_solution.times)
    fine = _extract_path_fields(fine_bundle, fine_solution, indices)
    coarse = _extract_path_fields(
        coarse_bundle,
        coarse_solution,
        np.arange(coarse_solution.times.size, dtype=np.int64),
    )
    return _compare_field_mappings(fine, coarse, scales)


def _compare_spatial_histories(bundle_a, solution_a, bundle_b, solution_b, scales):
    """Compare volume-aggregate histories across different macro meshes."""
    indices = exact_common_time_indices(solution_a.times, solution_b.times)
    fields_a = _extract_aggregate_fields(bundle_a, solution_a, indices)
    fields_b = _extract_aggregate_fields(
        bundle_b,
        solution_b,
        np.arange(solution_b.times.size, dtype=np.int64),
    )
    return _compare_field_mappings(fields_a, fields_b, scales)


def _extract_path_fields(bundle, solution, indices):
    """Extract full accepted integration-point and boundary histories."""
    strains = extract_quadrature_strain_history(bundle, solution.displacement)
    q = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
    return {
        "times": solution.times[indices],
        "displacement": solution.displacement[indices],
        "velocity": solution.velocity[indices],
        "strain": strains[indices],
        "stress": solution.stresses[indices],
        "equivalent_plastic_strain": q[indices],
        "dissipation": solution.dissipated_energy[indices, None],
        "incremental_work_density": solution.material_diagnostics[
            "incremental_work_density"
        ][indices],
        "viscoplastic_active_volume_fraction": solution.viscoplastic_active_volume_fraction[
            indices, None
        ],
        "events": _event_times(solution, q),
    }


def _extract_aggregate_fields(bundle, solution, indices):
    """Extract mesh-independent response aggregates for spatial comparison."""
    fields = _extract_path_fields(bundle, solution, indices)
    right_nodes = bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = bundle.mesh_info.dof_map[right_nodes, 0]
    fields["displacement"] = np.mean(
        solution.displacement[:, right_dofs], axis=1
    )[indices, None]
    fields["velocity"] = np.mean(solution.velocity[:, right_dofs], axis=1)[
        indices, None
    ]
    fields["strain"] = np.mean(fields["strain"], axis=1)
    fields["stress"] = np.mean(fields["stress"], axis=1)
    fields["incremental_work_density"] = np.mean(
        fields["incremental_work_density"], axis=1
    )[:, None]
    fields["equivalent_plastic_strain"] = np.max(
        fields["equivalent_plastic_strain"], axis=1
    )[:, None]
    return fields


def _compare_field_mappings(a, b, scales):
    """Compute fixed-scale field and accepted-event errors."""
    event_errors = [abs(a["events"][name] - b["events"][name]) for name in a["events"]]
    return {
        "displacement_error": fixed_scale_history_error(
            a["displacement"], b["displacement"], scales.displacement
        ),
        "velocity_error": fixed_scale_history_error(
            a["velocity"], b["velocity"], scales.velocity
        ),
        "strain_error": fixed_scale_history_error(a["strain"], b["strain"], scales.strain),
        "stress_error": fixed_scale_history_error(a["stress"], b["stress"], scales.stress),
        "equivalent_plastic_strain_error": fixed_scale_history_error(
            a["equivalent_plastic_strain"],
            b["equivalent_plastic_strain"],
            scales.equivalent_plastic_strain,
        ),
        "dissipation_error": fixed_scale_history_error(
            a["dissipation"], b["dissipation"], scales.dissipation
        ),
        "incremental_work_density_error": fixed_scale_history_error(
            a["incremental_work_density"],
            b["incremental_work_density"],
            scales.incremental_work_density,
        ),
        "viscoplastic_active_volume_fraction_error": fixed_scale_history_error(
            a["viscoplastic_active_volume_fraction"],
            b["viscoplastic_active_volume_fraction"],
            scales.viscoplastic_active_volume_fraction,
        ),
        "maximum_event_time_error": float(max(event_errors)),
    }


def _event_times(solution, q):
    """Measure first yield, peak stress, unloading, and wave-arrival times."""
    equivalent = _mean_equivalent_stress(solution.stresses)
    q_maximum = np.max(q, axis=1)
    yield_ids = np.flatnonzero(q_maximum >= 1.0e-6)
    unloading_ids = np.flatnonzero(solution.negative_incremental_work < -1.0e-10)
    arrival_threshold = 0.1 * float(np.max(equivalent))
    arrival_ids = np.flatnonzero(equivalent >= arrival_threshold)
    if yield_ids.size == 0 or unloading_ids.size == 0 or arrival_ids.size == 0:
        raise ValueError("FE2 path lacks required yield, unloading, or wave-arrival event.")
    return {
        "first_yield": float(solution.times[yield_ids[0]]),
        "peak_stress": float(solution.times[int(np.argmax(equivalent))]),
        "first_unloading": float(solution.times[unloading_ids[0]]),
        "wave_arrival": float(solution.times[arrival_ids[0]]),
    }


def _mean_equivalent_stress(stresses):
    """Compute point-mean J2 equivalent-stress history."""
    pressure = np.mean(stresses[..., :3], axis=2)
    deviator = stresses[..., :3] - pressure[..., None]
    squared = np.sum(deviator**2, axis=2) + 2.0 * np.sum(stresses[..., 3:] ** 2, axis=2)
    return np.mean(np.sqrt(1.5 * squared), axis=1)


def _physical_checks(solutions, config):
    """Check irreversible variables and required loading-unloading behavior."""
    result = {}
    for name, solution in solutions.items():
        q = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
        equivalent = _mean_equivalent_stress(solution.stresses)
        result[f"{name}_plastic"] = bool(np.max(q) >= float(config["plastic_q_threshold"]))
        result[f"{name}_active"] = bool(
            np.max(solution.viscoplastic_active_volume_fraction) > 0.0
        )
        result[f"{name}_unloading"] = bool(
            np.min(solution.negative_incremental_work)
            <= -float(config["negative_work_threshold"])
        )
        result[f"{name}_stress_drop"] = bool(
            np.max(equivalent) - equivalent[-1] >= float(config["stress_drop_threshold"])
        )
        result[f"{name}_q_monotone"] = bool(np.min(np.diff(q, axis=0)) >= -1.0e-12)
        result[f"{name}_dissipation_nonnegative"] = bool(
            np.min(solution.material_diagnostics["dissipation_density"]) >= -1.0e-12
        )
        result[f"{name}_residual_displacement"] = bool(
            np.linalg.norm(solution.displacement[-1])
            >= float(config["residual_displacement_threshold"])
        )
        result[f"{name}_energy_residual"] = bool(
            np.max(np.abs(solution.energy_residual)) / float(config["energy_scale"])
            <= float(config["energy_tolerance"])
        )
    return result


def _measure_performance(
    base,
    divisions,
    explicit_level,
    implicit_level,
    performance_config,
    implicit_name,
):
    """Measure one path-consistent implicit candidate after an untimed warm-up."""
    repetitions = int(performance_config["repetitions"])
    if int(performance_config["warmup_repetitions"]) != 1 or repetitions != 3:
        raise ValueError("FE2 performance benchmark requires one warm-up and three repetitions.")
    timing_config = _build_level_config(
        base,
        divisions,
        explicit_level,
    )
    implicit_dt = float(implicit_level["time_step"])
    final_time = float(implicit_level["final_time"])
    timing_config["analysis"]["implicit"].update(
        {"time_step": implicit_dt, "num_steps": _exact_step_count(final_time, implicit_dt)}
    )
    solve_multiphase_rve_plastic_impact_scheme(timing_config, "explicit")
    solve_multiphase_rve_plastic_impact_scheme(timing_config, "implicit")
    explicit_samples = []
    implicit_samples = []
    for repetition in range(repetitions):
        order = (
            ("explicit", "implicit")
            if repetition % 2 == 0
            else ("implicit", "explicit")
        )
        samples = {}
        for scheme in order:
            run = solve_multiphase_rve_plastic_impact_scheme(timing_config, scheme)
            samples[scheme] = float(run.solution.solve_time)
        explicit_samples.append(samples["explicit"])
        implicit_samples.append(samples["implicit"])
    explicit_array = np.asarray(explicit_samples)
    implicit_array = np.asarray(implicit_samples)
    speed_ratio = conservative_speed_ratio(explicit_array, implicit_array)
    target = float(performance_config["conservative_speedup_target"])
    eligible = speed_ratio > target
    return {
        "measured": True,
        "implicit_candidate": implicit_name,
        "explicit_samples": explicit_samples,
        "implicit_samples": implicit_samples,
        "explicit_median": float(np.median(explicit_array)),
        "implicit_median": float(np.median(implicit_array)),
        "conservative_speed_ratio": float(speed_ratio),
        "implicit_production_eligible": eligible,
        "selected_scheme": "implicit" if eligible else "explicit_fine",
        "reason": (
            "implicit_path_consistent_and_stably_faster"
            if eligible
            else "implicit_path_consistent_without_stable_speedup"
        ),
    }


def _flatten_metrics(comparisons, acceptance, physical_checks, performance):
    """Flatten structured benchmark results into finite RunResult metrics."""
    metrics = {
        f"{comparison_name}__{metric_name}": float(value)
        for comparison_name, values in comparisons.items()
        for metric_name, value in values.items()
    }
    metrics.update(
        {
            f"{comparison_name}__{check_name}_passed": float(passed)
            for comparison_name, values in acceptance.items()
            for check_name, passed in values.items()
        }
    )
    metrics.update({name: float(value) for name, value in physical_checks.items()})
    metrics["performance_measured"] = float(performance["measured"])
    metrics["implicit_production_eligible"] = float(
        performance["implicit_production_eligible"]
    )
    if performance["measured"]:
        metrics["conservative_speed_ratio"] = float(
            performance["conservative_speed_ratio"]
        )
    return metrics


def _write_micro_snapshots(output_root: Path, simulation, physical_config) -> Path:
    """Replay three accepted macro-point paths and store four exact event states."""
    solution = simulation.solution
    bundle = simulation.bundle
    strains = extract_quadrature_strain_history(bundle, solution.displacement)
    coordinates = _quadrature_coordinates(bundle)
    q_history = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
    work_history = solution.material_diagnostics["incremental_work_density"]
    equivalent = _point_equivalent_stress(solution.stresses)
    x_min = float(np.min(coordinates[:, 0]))
    x_max = float(np.max(coordinates[:, 0]))
    region_bounds = {
        "near_fixed_end": (0.0, 0.25),
        "middle": (0.375, 0.625),
        "near_loading_end": (0.75, 1.0),
    }
    output_path = output_root / "accepted_micro_snapshots.h5"
    with h5py.File(output_path, "w") as handle:
        handle.attrs["status"] = "complete"
        handle.attrs["time_source"] = "accepted_explicit_fe2_states"
        for region_name, bounds in region_bounds.items():
            lower = x_min + bounds[0] * (x_max - x_min)
            upper = x_min + bounds[1] * (x_max - x_min)
            region_ids = np.flatnonzero(
                (coordinates[:, 0] >= lower) & (coordinates[:, 0] <= upper)
            )
            if region_ids.size == 0:
                raise ValueError(f"FE2 snapshot region contains no macro point: {region_name}.")
            peak_q = np.max(q_history[:, region_ids], axis=0)
            point_id = int(region_ids[int(np.argmax(peak_q))])
            events = select_accepted_snapshot_indices(
                q_history[:, point_id],
                equivalent[:, point_id],
                work_history[:, point_id],
                float(physical_config["plastic_q_threshold"]),
            )
            replay = solve_prescribed_rve_path(
                simulation.rve_model,
                solution.times,
                strains[:, point_id],
                simulation.material_update_settings,
            )
            region = handle.create_group(region_name)
            region.attrs["macro_point_id"] = point_id
            region.create_dataset("macro_point_coordinate", data=coordinates[point_id])
            for event_name, accepted_step in events.items():
                if accepted_step == 0:
                    raise ValueError(
                        "FE2 micro snapshot event occurred at the unadvanced initial state: "
                        f"{event_name}."
                    )
                response = replay.responses[accepted_step - 1]
                event = region.create_group(event_name)
                event.attrs["accepted_step"] = accepted_step
                event.attrs["time"] = float(solution.times[accepted_step])
                event.create_dataset("displacement", data=response.displacement)
                event.create_dataset("strain", data=response.micro_strains)
                event.create_dataset("stress", data=response.micro_stresses)
                q_key = "phase_matrix__equivalent_plastic_strain"
                event.create_dataset(
                    "equivalent_plastic_strain",
                    data=response.state.material_state.variables[q_key],
                )
                event.create_dataset(
                    "viscoplastic_active",
                    data=response.micro_diagnostics["viscoplastic_active"],
                )
                event.create_dataset(
                    "incremental_work_density",
                    data=response.micro_diagnostics["incremental_work_density"],
                )
                event.attrs["dissipation_density"] = response.dissipation_density
    return output_path


def _quadrature_coordinates(bundle) -> np.ndarray:
    """Evaluate fixed physical quadrature coordinates in assembly ordering."""
    values = bundle.shape_function_cache.shape_values
    coordinates = []
    for connectivity in bundle.mesh_info.elements:
        coordinates.extend(values @ bundle.mesh_info.nodes[connectivity])
    return np.asarray(coordinates, dtype=np.float64)


def _point_equivalent_stress(stresses: np.ndarray) -> np.ndarray:
    """Compute pointwise J2 equivalent stress histories."""
    pressure = np.mean(stresses[..., :3], axis=2)
    deviator = stresses[..., :3] - pressure[..., None]
    squared = np.sum(deviator**2, axis=2) + 2.0 * np.sum(stresses[..., 3:] ** 2, axis=2)
    return np.sqrt(1.5 * squared)
