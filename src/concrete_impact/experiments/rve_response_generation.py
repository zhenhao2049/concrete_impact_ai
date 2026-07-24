"""High-fidelity response workers for the four RVE-RNO data sources.

Contents:
    RVE shard writing, response workers, direct FE2 histories, and diagnostics.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from concrete_impact.benchmarks.dynamic_viscoplastic import (
    _build_left_dirichlet,
    _build_material_update_settings,
    _build_model_def,
    _build_newton_settings,
    _build_output_def,
    _build_pressure_load_function,
    _build_time_settings,
)
from concrete_impact.benchmarks.rve_heterogeneous import (
    _elastic_material,
    _newton_settings,
    _update_settings,
    _viscoplastic_material,
)
from concrete_impact.core.config import load_yaml_config
from concrete_impact.core.progress import JsonProgressRecorder
from concrete_impact.experiments.rve_data_plan import (
    AuditThresholds,
    DataSourcePlan,
    RVEDataTask,
)
from concrete_impact.experiments.rve_path_generation import generate_data_control_path
from fem.assembly.nonlinear_solid import build_material_assembly_cache
from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.materials import (
    J2ViscoplasticMaterial,
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
    build_material,
)
from fem.mesh.structured import build_cylindrical_ogrid_hex8, build_structured_hex8_box
from fem.preprocess import build_preprocess_data
from fem.rve import (
    RVECompactDataShardWriter,
    RVEDataShardWriter,
    RVEExecutionSettings,
    RVEHomogenizedMaterial,
    RVERequest,
    RVEResponse,
    StructuredHex8RVE,
    build_cylindrical_two_phase_rve,
    build_structured_hex8_periodic_constraint,
    initialize_rve_state,
    iterate_prescribed_rve_path,
    solve_prescribed_rve_path,
    solve_rve_microequilibrium,
)
from fem.solvers import ArmijoSettings, NonlinearNewtonSettings


def write_response_shard(
    output_directory: Path,
    shard_id: int,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
    audit: AuditThresholds,
    impact_control_path: Path | None,
    storage_mode: str = "full_history",
    compression: str = "lzf",
) -> Path:
    """Compute one source-homogeneous response shard without numerical retries."""
    total_start = perf_counter()
    if source.kind == "material_point":
        output_path, solve_time, write_time = _write_material_point_shard(
            output_directory, shard_id, source, tasks, audit
        )
        _write_shard_performance(
            output_path,
            source,
            tasks,
            perf_counter() - total_start,
            solve_time,
            write_time,
        )
        return output_path
    if source.kind == "single_phase_rve":
        model, update = _build_single_phase_rve()
        controls = [generate_data_control_path(source, task) for task in tasks]
    elif source.kind == "heterogeneous_rve":
        model, update = _build_heterogeneous_rve(source)
        controls = [generate_data_control_path(source, task) for task in tasks]
    elif source.kind == "impact_path":
        model, update = _build_heterogeneous_rve(source)
        controls = (
            _extract_impact_strain_paths(output_directory, shard_id, source, tasks)
            if impact_control_path is None
            else _read_impact_control_paths(impact_control_path, tasks)
        )
    else:
        raise ValueError(f"Unsupported RVE data source kind: {source.kind}.")
    output_path = output_directory / f"response_shard_{shard_id:06d}.h5"
    features = _fixed_parameter_features(source)
    if storage_mode == "compact_streaming":
        return _write_compact_rve_shard(
            output_path,
            source,
            tasks,
            controls,
            model,
            update,
            audit,
            features,
            compression,
        )
    if storage_mode != "full_history":
        raise ValueError(f"Unknown RVE storage mode: {storage_mode}.")
    writer = RVEDataShardWriter(
        output_path,
        model,
        {"dataset_kind": source.kind, "model_id": source.model},
    )
    try:
        solve_time = 0.0
        write_time = 0.0
        for task, control in zip(tasks, controls, strict=True):
            if control.control_kind != "macro_strain":
                raise ValueError(f"RVE response task requires macro strain: {task.task_id}.")
            load_case = next(
                case for case in source.load_cases if case.name == task.load_case_name
            )
            expected_response = _task_expected_response(load_case, task)
            solve_start = perf_counter()
            progress = JsonProgressRecorder(
                output_directory / f"progress_shard_{shard_id:06d}.json",
                {
                    "task_id": task.task_id,
                    "source_name": source.name,
                    "family": control.family,
                    "load_case": task.load_case_name,
                    "regime": expected_response["regime"],
                },
                output_directory / "summary.log",
            )
            result = solve_prescribed_rve_path(
                model,
                control.times,
                control.values,
                update,
                progress,
            )
            if _requires_tangent_check(source, task):
                result = replace(
                    result,
                    validation_diagnostics=_compute_rve_tangent_checks(
                        model,
                        result,
                        control.times,
                        control.values,
                        update,
                        audit,
                    ),
                )
            path_solve_time = perf_counter() - solve_start
            solve_time += path_solve_time
            write_start = perf_counter()
            writer.write_path(
                task.task_id,
                result,
                _path_metadata(source, task, control, load_case, features),
            )
            path_write_time = perf_counter() - write_start
            write_time += path_write_time
            progress.publish_summary(
                "rve_data_path_complete",
                {
                    "accepted_steps": int(control.times.size - 1),
                    "final_time": float(control.times[-1]),
                    "path_solve_seconds": path_solve_time,
                    "path_write_seconds": path_write_time,
                },
            )
    except Exception as error:
        writer.write_failure(
            tasks[0].task_id,
            {"error_type": type(error).__name__, "message": str(error)},
        )
        writer.close_failed()
        raise
    writer.close_complete()
    _write_shard_performance(
        output_path,
        source,
        tasks,
        perf_counter() - total_start,
        solve_time,
        write_time,
    )
    return output_path


def _write_compact_rve_shard(
    output_path: Path,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
    controls: list,
    model: StructuredHex8RVE,
    update: MaterialUpdateSettings,
    audit: AuditThresholds,
    features: tuple[dict[str, float], dict[str, float]],
    compression: str,
) -> Path:
    """Stream one fixed-cell RVE shard without retaining complete micro histories."""
    writer = RVECompactDataShardWriter(
        output_path,
        model,
        {"dataset_kind": source.kind, "model_id": source.model},
        compression,
    )
    solve_time = 0.0
    write_time = 0.0
    total_start = perf_counter()
    try:
        for task, control in zip(tasks, controls, strict=True):
            if control.control_kind != "macro_strain":
                raise ValueError(f"Compact RVE task requires macro strain: {task.task_id}.")
            load_case = next(
                case for case in source.load_cases if case.name == task.load_case_name
            )
            expected_response = _task_expected_response(load_case, task)
            metadata = _path_metadata(source, task, control, load_case, features)
            writer.start_path(task.task_id, metadata)
            progress = JsonProgressRecorder(
                output_path.parent / f"progress_shard_{task.shard_index:06d}.json",
                {
                    "task_id": task.task_id,
                    "source_name": source.name,
                    "family": control.family,
                    "load_case": task.load_case_name,
                    "regime": expected_response["regime"],
                },
                output_path.parent / "summary.log",
            )
            snapshots: dict[str, tuple[float, RVEResponse]] = {}
            validation_ids: list[int] = []
            validation_errors: list[float] = []
            peak_equivalent_stress = -np.inf
            most_negative_work = np.inf
            solve_start = perf_counter()
            path_write_time = 0.0
            accepted_steps = 0
            for accepted in iterate_prescribed_rve_path(
                model,
                control.times,
                control.values,
                update,
                progress,
            ):
                accepted_steps += 1
                response = accepted.response
                write_start = perf_counter()
                writer.append_step(accepted.time, accepted.time_step, response)
                path_write_time += perf_counter() - write_start
                equivalent = float(np.max(_equivalent_stress(response.micro_stresses[None, ...])))
                work = float(response.diagnostics["macro_work_increment"])
                q = response.state.material_state.variables[
                    "phase_matrix__equivalent_plastic_strain"
                ]
                if "first_yield" not in snapshots and float(np.max(q)) > audit.plastic_q_threshold:
                    snapshots["first_yield"] = (accepted.time, response)
                if equivalent > peak_equivalent_stress:
                    peak_equivalent_stress = equivalent
                    snapshots["peak_stress"] = (accepted.time, response)
                if work < most_negative_work:
                    most_negative_work = work
                    snapshots["maximum_negative_work"] = (accepted.time, response)
                snapshots["final"] = (accepted.time, response)
                if _requires_tangent_check(source, task) and accepted.step_id in {
                    1,
                    control.times.size - 1,
                }:
                    validation_ids.append(accepted.step_id - 1)
                    validation_errors.append(
                        _streaming_tangent_error(
                            model,
                            accepted.previous_state,
                            control.values[accepted.step_id],
                            accepted.time_step,
                            response,
                            update,
                            audit,
                        )
                    )
            write_start = perf_counter()
            if _requires_tangent_check(source, task):
                writer.write_validation(
                    {
                        "tangent_state_index": np.asarray(validation_ids, dtype=np.int64),
                        "tangent_direction_error": np.asarray(
                            validation_errors, dtype=np.float64
                        ),
                    }
                )
            if _requires_snapshots(source, task):
                writer.write_snapshots(snapshots)
            writer.finish_path()
            path_write_time += perf_counter() - write_start
            path_total_time = perf_counter() - solve_start
            path_solve_time = path_total_time - path_write_time
            solve_time += path_solve_time
            write_time += path_write_time
            progress.publish_summary(
                "rve_data_path_complete",
                {
                    "accepted_steps": accepted_steps,
                    "final_time": float(control.times[-1]),
                    "path_solve_seconds": path_solve_time,
                    "path_write_seconds": path_write_time,
                },
            )
    except Exception as error:
        writer.write_failure(
            tasks[0].task_id,
            {"error_type": type(error).__name__, "message": str(error)},
        )
        writer.close_failed()
        raise
    writer.close_complete()
    _write_shard_performance(
        output_path,
        source,
        tasks,
        perf_counter() - total_start,
        solve_time,
        write_time,
    )
    return output_path


def _task_expected_response(load_case, task: RVEDataTask) -> dict[str, object]:
    """Resolve path-level response expectations from the production stratum."""
    expected = load_case.expected_response.model_dump(mode="json")
    if task.expected_response_regime is not None:
        expected["regime"] = task.expected_response_regime
    if task.family in {"load_unload_reload", "reverse"}:
        expected["require_unloading"] = True
    if task.family == "reverse":
        expected["require_reverse_loading"] = True
    return expected


def _path_metadata(
    source: DataSourcePlan,
    task: RVEDataTask,
    control,
    load_case,
    features: tuple[dict[str, float], dict[str, float]],
) -> dict[str, object]:
    """Build complete immutable provenance for one response path."""
    return {
        "dataset_kind": source.kind,
        "source_name": source.name,
        "lineage_id": task.lineage_id,
        "macro_path_origin": task.macro_path_origin,
        "material_parameters": features[0],
        "microstructure_features": features[1],
        "seed": task.seed,
        "family": control.family,
        "load_case": task.load_case_name,
        "expected_response": _task_expected_response(load_case, task),
        "production_design": source.production_design is not None,
        "design_stratum": task.design_stratum,
        "amplitude_band": task.amplitude_band,
        "rate_band": task.rate_band,
        "tensor_direction": task.tensor_direction,
        "repeat_index": task.repeat_index,
        "target_amplitude": task.amplitude,
        "target_maximum_strain_rate": task.target_strain_rate,
        "accepted_increment_count": task.accepted_increment_count,
        "target_turning_angle": task.turning_angle,
        "preassigned_split": task.preassigned_split,
        "snapshot_required": task.snapshot_required,
        "tangent_check_required": task.tangent_check_required,
    }


def _requires_tangent_check(source: DataSourcePlan, task: RVEDataTask) -> bool:
    """Select configured production checks while preserving legacy first-path checks."""
    if source.production_design is not None:
        return task.tangent_check_required
    return task.path_index == 0


def _requires_snapshots(source: DataSourcePlan, task: RVEDataTask) -> bool:
    """Select configured production snapshots while preserving legacy pilot snapshots."""
    if source.production_design is not None:
        return task.snapshot_required
    return task.path_index < 5


def _streaming_tangent_error(
    model: StructuredHex8RVE,
    previous_state,
    target: np.ndarray,
    time_step: float,
    response,
    update: MaterialUpdateSettings,
    audit: AuditThresholds,
) -> float:
    """Check one accepted effective tangent from the same committed state."""
    direction = _tangent_check_direction()
    stresses = []
    for sign in (1.0, -1.0):
        trial = solve_rve_microequilibrium(
            model,
            RVERequest(
                target + sign * audit.tangent_perturbation * direction,
                time_step,
                update,
            ),
            previous_state,
        )
        stresses.append(trial.macro_stress)
    tangent = response.effective_tangent
    if tangent is None:
        raise ValueError("Compact RVE tangent validation requires an effective tangent.")
    finite_difference = (stresses[0] - stresses[1]) / (2.0 * audit.tangent_perturbation)
    action = tangent @ direction
    scale = max(float(np.linalg.norm(action)), audit.tangent_stress_scale)
    return float(np.linalg.norm(finite_difference - action) / scale)


def write_impact_control_paths(
    output_directory: Path,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
) -> Path:
    """Run each macro impact once and publish all extracted accepted strain paths."""
    if source.kind != "impact_path":
        raise ValueError("Only impact_path sources may write macro-extracted controls.")
    shard_id = min(task.shard_index for task in tasks)
    controls = _extract_impact_strain_paths(output_directory, shard_id, source, tasks)
    output_path = output_directory / f"impact_controls__{source.name}.h5"
    partial_path = output_path.with_suffix(".h5.partial")
    with h5py.File(partial_path, "w") as handle:
        handle.attrs["schema_version"] = "3.0"
        handle.attrs["status"] = "incomplete"
        handle.attrs["source_name"] = source.name
        for task, control in zip(tasks, controls, strict=True):
            group = handle.create_group(f"paths/{task.task_id}")
            group.attrs["family"] = control.family
            group.create_dataset("time", data=control.times, compression="lzf")
            group.create_dataset("strain", data=control.values, compression="lzf")
        handle.attrs["status"] = "complete"
    os.replace(partial_path, output_path)
    return output_path


def _read_impact_control_paths(path: Path, tasks: tuple[RVEDataTask, ...]):
    """Read explicitly precomputed accepted macro strain paths for RVE replay workers."""
    from concrete_impact.experiments.rve_path_generation import GeneratedControlPath

    controls = []
    with h5py.File(path, "r") as handle:
        if handle.attrs["status"] != "complete":
            raise ValueError(f"Impact control artifact is incomplete: {path}.")
        for task in tasks:
            group = handle[f"paths/{task.task_id}"]
            controls.append(
                GeneratedControlPath(
                    task.task_id,
                    "macro_strain",
                    group["time"][...],
                    group["strain"][...],
                    str(group.attrs["family"]),
                )
            )
    return controls


def _write_material_point_shard(
    output_directory: Path,
    shard_id: int,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
    audit: AuditThresholds,
) -> tuple[Path, float, float]:
    """Compute material-point labels and states in the common v3 path layout."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"response_shard_{shard_id:06d}.h5"
    partial_path = output_path.with_suffix(".h5.partial")
    material = _build_material_point_material(source)
    update = MaterialUpdateSettings(50, 1.0e-10, 1.0e-12, 1.0e-10)
    features = _fixed_parameter_features(source)
    solve_time = 0.0
    write_time = 0.0
    try:
        with h5py.File(partial_path, "w") as handle:
            handle.attrs["schema_version"] = "3.0"
            handle.attrs["status"] = "incomplete"
            handle.attrs["voigt_order"] = "xx,yy,zz,yz,xz,xy"
            handle.attrs["strain_convention"] = "engineering_shear"
            handle.attrs["model_id"] = source.model
            model_group = handle.create_group(f"models/{source.model}")
            model_group.attrs["material_type"] = type(material).__name__
            for task in tasks:
                control = generate_data_control_path(source, task)
                path_solve_time, path_write_time = _write_material_point_path(
                    handle,
                    task,
                    control.times,
                    control.values,
                    material,
                    update,
                    features,
                    source,
                    audit,
                )
                solve_time += path_solve_time
                write_time += path_write_time
            handle.attrs["status"] = "complete"
    except Exception as error:
        with h5py.File(partial_path, "a") as handle:
            handle.attrs["status"] = "failed"
            failure = handle.create_group(f"failures/{tasks[0].task_id}")
            failure.attrs["diagnostics_json"] = json.dumps(
                {"error_type": type(error).__name__, "message": str(error)},
                sort_keys=True,
            )
        os.replace(partial_path, output_path)
        raise
    os.replace(partial_path, output_path)
    return output_path, solve_time, write_time


def _write_material_point_path(
    handle: h5py.File,
    task: RVEDataTask,
    times: np.ndarray,
    strains: np.ndarray,
    material: J2ViscoplasticMaterial,
    update: MaterialUpdateSettings,
    features: tuple[dict[str, float], dict[str, float]],
    source: DataSourcePlan,
    audit: AuditThresholds,
) -> tuple[float, float]:
    """Integrate and write one complete material-point history."""
    state = material.initialize_state(1)
    responses = []
    committed_states = []
    solve_start = perf_counter()
    for step_id, time_step in enumerate(np.diff(times), start=1):
        committed_states.append(state)
        response = material.update(
            MaterialPointRequest(
                strains=strains[step_id : step_id + 1],
                strain_rates=((strains[step_id] - strains[step_id - 1]) / time_step)[None, :],
                time_step=float(time_step),
                kinematics="three_dimensional",
                update_settings=update,
            ),
            state,
            MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
        )
        responses.append(response)
        state = response.state
    solve_time = perf_counter() - solve_start
    write_start = perf_counter()
    group = handle.create_group(f"paths/{task.task_id}")
    load_case = next(case for case in source.load_cases if case.name == task.load_case_name)
    control = generate_data_control_path(source, task)
    group.attrs["metadata_json"] = json.dumps(
        _path_metadata(source, task, control, load_case, features),
        sort_keys=True,
    )
    feature_group = group.create_group("features")
    for name, mapping in zip(
        ("material_parameters", "microstructure_features"), features, strict=True
    ):
        names = tuple(sorted(mapping))
        feature_group.create_dataset(f"{name}_names", data=np.asarray(names, dtype="S"))
        feature_group.create_dataset(name, data=np.asarray([mapping[key] for key in names]))
    macro = group.create_group("macro")
    macro.create_dataset("time", data=times[1:])
    macro.create_dataset("time_step", data=np.diff(times))
    macro.create_dataset("strain", data=strains[1:], compression="lzf")
    macro.create_dataset(
        "stress",
        data=np.stack([item.stresses[0] for item in responses]),
        compression="lzf",
    )
    macro.create_dataset(
        "effective_tangent",
        data=np.stack([_require_array(item.tangents, "tangent")[0] for item in responses]),
        compression="lzf",
    )
    macro.create_dataset(
        "free_energy_density",
        data=np.asarray(
            [_require_array(item.free_energy, "free_energy")[0] for item in responses]
        ),
    )
    macro.create_dataset(
        "dissipation_density",
        data=np.asarray(
            [_require_array(item.dissipation, "dissipation")[0] for item in responses]
        ),
    )
    state_group = group.create_group("micro/state")
    for name in responses[0].state.variables:
        state_group.create_dataset(
            name,
            data=np.stack([item.state.variables[name][0] for item in responses]),
            compression="lzf",
        )
    diagnostic_group = group.create_group("micro/diagnostics")
    diagnostic_names = sorted(set().union(*(item.diagnostics.keys() for item in responses)))
    for name in diagnostic_names:
        point_count = responses[0].stresses.shape[0]
        values = np.full((len(responses), point_count), np.nan, dtype=np.float64)
        available = np.zeros((len(responses), point_count), dtype=np.bool_)
        for step_id, response in enumerate(responses):
            if name in response.diagnostics:
                values[step_id] = response.diagnostics[name]
                available[step_id] = np.isfinite(response.diagnostics[name])
        diagnostic_group.create_dataset(name, data=values, compression="lzf")
        diagnostic_group.create_dataset(
            f"{name}__available", data=available, compression="lzf"
        )
    validation_group = group.create_group("validation")
    if _requires_tangent_check(source, task):
        validation = _compute_material_tangent_checks(
            material,
            times,
            strains,
            tuple(committed_states),
            tuple(responses),
            update,
            audit,
        )
        for name, values in validation.items():
            validation_group.create_dataset(name, data=values)
    return solve_time, perf_counter() - write_start


def _build_material_point_material(source: DataSourcePlan) -> J2ViscoplasticMaterial:
    """Build the fixed default J2 material-point data source."""
    values = _fixed_parameter_features(source)[0]
    return J2ViscoplasticMaterial(
        source.model,
        1.0,
        values["young_modulus"],
        values["poisson_ratio"],
        values["yield_stress"],
        values["hardening_modulus"],
        values["time_scale"],
        values["yield_stress"],
        2.0,
    )


def _build_single_phase_rve() -> tuple[StructuredHex8RVE, MaterialUpdateSettings]:
    """Build the fixed homogeneous production-validation cell."""
    config = load_yaml_config("configs/benchmarks/rve/single_phase_j2_viscoplastic.yaml")
    material = build_material(config["rve"]["material"])
    nodes, elements = build_structured_hex8_box((1.0, 1.0, 1.0), (1, 1, 1))
    newton = config["rve"]["newton"]
    model = StructuredHex8RVE(
        nodes,
        elements,
        (1, 1, 1),
        material,
        build_structured_hex8_periodic_constraint(nodes, (1, 1, 1)),
        NonlinearNewtonSettings(
            int(newton["max_iterations"]),
            float(newton["residual_absolute_tolerance"]),
            float(newton["residual_relative_tolerance"]),
            float(newton["increment_absolute_tolerance"]),
            float(newton["increment_relative_tolerance"]),
            ArmijoSettings(enabled=True),
        ),
        hill_mandel_power_scale=float(config["verification"]["hill_mandel_power_scale"]),
    )
    update = config["rve"]["material_update"]
    return model, MaterialUpdateSettings(
        int(update["max_iterations"]),
        float(update["yield_relative_tolerance"]),
        float(update["residual_absolute_tolerance"]),
        float(update["residual_relative_tolerance"]),
    )


def _build_heterogeneous_rve(
    source: DataSourcePlan,
) -> tuple[StructuredHex8RVE, MaterialUpdateSettings]:
    """Build the fixed default cylinder cell while retaining path-local features."""
    config = load_yaml_config("configs/benchmarks/rve/cylindrical_inclusion.yaml")
    micro = _fixed_parameter_features(source)[1]
    divisions = int(micro.get("circumferential_divisions", 64.0))
    radial_divisions = int(
        micro.get("matrix_radial_divisions", config["rve"]["matrix_radial_divisions"])
    )
    axial_divisions = int(
        micro.get("axial_divisions", config["rve"]["axial_divisions"])
    )
    mesh = build_cylindrical_ogrid_hex8(
        tuple(config["rve"]["lengths"]),
        float(micro["inclusion_volume_fraction"]),
        divisions,
        radial_divisions,
        axial_divisions,
    )
    model = build_cylindrical_two_phase_rve(
        mesh,
        _viscoplastic_material(config["materials"]["matrix_viscoplastic"]),
        _elastic_material("inclusion", config["materials"]["inclusion_elastic"]),
        _newton_settings(config),
        float(config["verification"]["hill_mandel_power_scale"]),
    )
    return model, _update_settings(config)


def build_fixed_cell_heterogeneous_rve(
    source: DataSourcePlan,
) -> tuple[StructuredHex8RVE, MaterialUpdateSettings]:
    """Build the fixed-cell production RVE for project-level verification workflows."""
    return _build_heterogeneous_rve(source)


def _extract_impact_strain_paths(
    output_directory: Path,
    shard_id: int,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
):
    """Run each macro pulse once and extract deterministic axial-bin paths."""
    from concrete_impact.experiments.rve_path_generation import GeneratedControlPath

    controls: dict[str, GeneratedControlPath] = {}
    for load_case in source.load_cases:
        case_tasks = tuple(task for task in tasks if task.load_case_name == load_case.name)
        if not case_tasks:
            continue
        if source.macro_path_origin == "direct_j2_explicit":
            config = load_yaml_config(
                "configs/benchmarks/rve/multiphase_dynamic_embedding.yaml"
            )
            config["model"]["mesh"]["divisions"] = [16, 2, 2]
        else:
            config = load_yaml_config(
                "configs/benchmarks/rve/multiphase_plastic_impact.yaml"
            )
            config["model"]["mesh"]["divisions"] = [4, 2, 2]
        config["analysis"]["explicit"]["time_step"] = load_case.duration / (
            load_case.time_points - 1
        )
        config["analysis"]["explicit"]["num_steps"] = load_case.time_points - 1
        if "implicit" in config["analysis"]:
            config["analysis"]["implicit"]["time_step"] = load_case.duration / (
                load_case.time_points - 1
            )
            config["analysis"]["implicit"]["num_steps"] = load_case.time_points - 1
        config["loading"]["amplitude"] = load_case.peak_scale
        config["loading"]["duration"] = (
            load_case.duration * _require_pulse_fraction(load_case.pulse_duration_fraction)
        )
        task_root = output_directory / f"impact_macro_{shard_id:06d}_{load_case.name}"
        config["output"]["root"] = str(task_root)
        config["output"]["mesh_path"] = str(task_root / "mesh.msh")
        config["output"]["vtk_path"] = str(task_root / "mesh.vtu")
        model_def = _build_model_def(config["model"])
        bundle = build_preprocess_data(model_def, _build_output_def(config["output"]))
        initial = np.zeros(bundle.mesh_info.dof_map.size)
        macro_progress = JsonProgressRecorder(
            output_directory / "progress.json",
            {
                "calculation": "impact_control_extraction",
                "source_name": source.name,
                "load_case": load_case.name,
            },
        )
        if source.macro_path_origin == "direct_j2_explicit":
            solution = solve_material_explicit(
                bundle,
                _build_time_settings(config["analysis"]["explicit"]),
                initial,
                initial,
                _build_left_dirichlet(bundle),
                _build_pressure_load_function(bundle, config),
                "three_dimensional",
                _build_material_update_settings(config),
                macro_progress,
            )
        else:
            bundle, solution = _solve_direct_fe2_macro_history(
                config,
                source,
                macro_progress,
            )
        selected_ids = _select_impact_points(
            bundle,
            solution,
            load_case,
            len(case_tasks),
            _require_axial_region_count(load_case.axial_region_count),
        )
        cache = build_material_assembly_cache(bundle)
        quadrature_count = cache.b_matrices.shape[1]
        for task, point_id in zip(case_tasks, selected_ids, strict=True):
            element_id, quadrature_id = divmod(point_id, quadrature_count)
            connectivity = bundle.mesh_info.elements[element_id]
            element_dofs = bundle.mesh_info.dof_map[connectivity].reshape(-1)
            strains = solution.displacement[:, element_dofs] @ cache.b_matrices[
                element_id, quadrature_id
            ].T
            controls[task.task_id] = GeneratedControlPath(
                task.task_id,
                "macro_strain",
                solution.times,
                strains,
                "impact_extracted",
            )
    return [controls[task.task_id] for task in tasks]


def _select_impact_points(
    bundle,
    solution,
    load_case,
    selection_count: int,
    axial_region_count: int,
) -> tuple[int, ...]:
    """Select a fixed count of eligible points within each prescribed axial region."""
    coordinates = _quadrature_coordinates(bundle)
    q_history = _impact_q_history(solution)
    work_history = solution.material_diagnostics["incremental_work_density"]
    equivalent_stress = _equivalent_stress(solution.stresses)
    q_maximum = np.max(q_history, axis=0)
    work_minimum = np.min(work_history, axis=0)
    stress_drop = np.max(equivalent_stress, axis=0) - equivalent_stress[-1]
    if load_case.expected_response.regime == "elastic":
        eligible = q_maximum <= 1.0e-12
    else:
        eligible = q_maximum >= 1.0e-8
    if load_case.expected_response.require_unloading:
        eligible &= (work_minimum < -1.0e-12) & (stress_drop > 1.0e-8)
    x_min = float(np.min(bundle.mesh_info.nodes[:, 0]))
    x_max = float(np.max(bundle.mesh_info.nodes[:, 0]))
    edges = np.linspace(x_min, x_max, axial_region_count + 1)
    selections_per_region = selection_count // axial_region_count
    selected: list[int] = []
    for bin_id in range(axial_region_count):
        upper_condition = (
            coordinates[:, 0] <= edges[bin_id + 1]
            if bin_id == axial_region_count - 1
            else coordinates[:, 0] < edges[bin_id + 1]
        )
        candidates = np.flatnonzero(
            eligible & (coordinates[:, 0] >= edges[bin_id]) & upper_condition
        )
        if candidates.size < selections_per_region:
            diagnostics = {
                "load_case": load_case.name,
                "regime": load_case.expected_response.regime,
                "axial_bin": bin_id,
                "bin_bounds": [float(edges[bin_id]), float(edges[bin_id + 1])],
                "q_maximum": q_maximum.tolist(),
                "minimum_incremental_work": work_minimum.tolist(),
                "postpeak_stress_drop": stress_drop.tolist(),
                "required_selection_count": selections_per_region,
                "eligible_candidate_count": int(candidates.size),
            }
            raise ValueError(
                "Impact path selection has insufficient eligible points: "
                + json.dumps(diagnostics, sort_keys=True)
            )
        if load_case.expected_response.regime == "viscoplastic":
            order = np.argsort(q_maximum[candidates])[::-1]
        else:
            center = 0.5 * (edges[bin_id] + edges[bin_id + 1])
            order = np.argsort(abs(coordinates[candidates, 0] - center))
        selected.extend(int(value) for value in candidates[order[:selections_per_region]])
    return tuple(selected)


def _solve_direct_fe2_macro_history(config, source, progress_callback=None):
    """Solve one directly embedded FE2 macro history for calibration paths."""
    reference = load_yaml_config(config["rve"]["reference_config"])
    micro = _fixed_parameter_features(source)[1]
    mesh = build_cylindrical_ogrid_hex8(
        tuple(reference["rve"]["lengths"]),
        float(micro["inclusion_volume_fraction"]),
        int(micro["circumferential_divisions"]),
        int(config["rve"]["matrix_radial_divisions"]),
        int(config["rve"]["axial_divisions"]),
    )
    model = build_cylindrical_two_phase_rve(
        mesh,
        _viscoplastic_material(reference["materials"]["matrix_viscoplastic"]),
        _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
        _newton_settings(reference),
        float(reference["verification"]["hill_mandel_power_scale"]),
    )
    direct_bundle = build_preprocess_data(
        _build_model_def(config["model"]),
        _build_output_def(config["output"]),
    )
    material = RVEHomogenizedMaterial(
        model,
        "direct_fe2_calibration",
        RVEExecutionSettings("serial", 1),
    )
    bundle = replace(direct_bundle, material=material)
    initial = np.zeros(bundle.mesh_info.dof_map.size)
    try:
        if source.macro_path_origin == "direct_fe2_explicit":
            solution = solve_material_explicit(
                bundle,
                _build_time_settings(config["analysis"]["explicit"]),
                initial,
                initial,
                _build_left_dirichlet(bundle),
                _build_pressure_load_function(bundle, config),
                "three_dimensional",
                _build_material_update_settings(config),
                progress_callback,
            )
        elif source.macro_path_origin == "direct_fe2_implicit":
            solution = solve_material_implicit_newmark(
                bundle,
                _build_time_settings(config["analysis"]["implicit"]),
                initial,
                initial,
                _build_left_dirichlet(bundle),
                _build_pressure_load_function(bundle, config),
                "three_dimensional",
                _build_material_update_settings(config),
                _build_newton_settings(config),
                progress_callback,
            )
        else:
            raise ValueError(
                f"Unsupported direct FE2 macro path origin: {source.macro_path_origin}."
            )
    finally:
        material.close()
    return bundle, solution


def _impact_q_history(solution) -> np.ndarray:
    """Read the accepted equivalent-plastic-strain history for either macro material."""
    phase_key = "phase_matrix__equivalent_plastic_strain"
    if phase_key in solution.material_diagnostics:
        return solution.material_diagnostics[phase_key]
    return solution.equivalent_plastic_strain


def _quadrature_coordinates(bundle) -> np.ndarray:
    """Evaluate physical integration-point coordinates in assembly ordering."""
    shape_values = bundle.shape_function_cache.shape_values
    coordinates = []
    for connectivity in bundle.mesh_info.elements:
        element_coordinates = bundle.mesh_info.nodes[connectivity]
        coordinates.extend(shape_values @ element_coordinates)
    return np.asarray(coordinates, dtype=np.float64)


def _equivalent_stress(stresses: np.ndarray) -> np.ndarray:
    """Compute pointwise J2 equivalent stress histories."""
    pressure = np.mean(stresses[..., :3], axis=2)
    deviator = stresses[..., :3] - pressure[..., None]
    norm_squared = np.sum(deviator**2, axis=2) + 2.0 * np.sum(stresses[..., 3:] ** 2, axis=2)
    return np.sqrt(1.5 * norm_squared)


def _require_pulse_fraction(value: float | None) -> float:
    """Require the strictly validated impact pulse-duration fraction."""
    if value is None:
        raise ValueError("Impact load case requires pulse_duration_fraction.")
    return value


def _require_axial_region_count(value: int | None) -> int:
    """Require the strictly validated impact axial-region count."""
    if value is None:
        raise ValueError("Impact load case requires axial_region_count.")
    return value


def _fixed_parameter_features(
    source: DataSourcePlan,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve only explicitly fixed parameters in the first production version."""
    groups = []
    for mapping in (source.material_parameters, source.microstructure_parameters):
        values: dict[str, float] = {}
        for name, specification in mapping.items():
            if specification.kind != "fixed":
                raise ValueError(
                    f"First production generator requires fixed parameter {name}."
                )
            values[name] = specification.value
        groups.append(values)
    return groups[0], groups[1]


def _require_array(values: np.ndarray | None, name: str) -> np.ndarray:
    """Require an explicitly requested material response array."""
    if values is None:
        raise ValueError(f"Material-point data generation requires {name}.")
    return values


def _compute_rve_tangent_checks(
    model: StructuredHex8RVE,
    result,
    times: np.ndarray,
    strains: np.ndarray,
    update: MaterialUpdateSettings,
    audit: AuditThresholds,
) -> dict[str, np.ndarray]:
    """Check two accepted RVE algorithmic tangents by central directional differences."""
    if len(result.responses) < 2:
        raise ValueError("RVE tangent validation requires at least two accepted states.")
    direction = _tangent_check_direction()
    state_ids = np.asarray([0, len(result.responses) - 1], dtype=np.int64)
    errors = []
    for response_id in state_ids:
        previous_state = (
            initialize_rve_state(model)
            if response_id == 0
            else result.responses[response_id - 1].state
        )
        target = strains[response_id + 1]
        time_step = float(times[response_id + 1] - times[response_id])
        plus = solve_rve_microequilibrium(
            model,
            RVERequest(
                target + audit.tangent_perturbation * direction,
                time_step,
                update,
            ),
            previous_state,
        )
        minus = solve_rve_microequilibrium(
            model,
            RVERequest(
                target - audit.tangent_perturbation * direction,
                time_step,
                update,
            ),
            previous_state,
        )
        tangent = result.responses[response_id].effective_tangent
        if tangent is None:
            raise ValueError("RVE tangent validation requires an effective tangent.")
        finite_difference = (plus.macro_stress - minus.macro_stress) / (
            2.0 * audit.tangent_perturbation
        )
        tangent_action = tangent @ direction
        scale = max(float(np.linalg.norm(tangent_action)), audit.tangent_stress_scale)
        errors.append(float(np.linalg.norm(finite_difference - tangent_action) / scale))
    return {
        "tangent_state_index": state_ids,
        "tangent_direction_error": np.asarray(errors, dtype=np.float64),
    }


def _compute_material_tangent_checks(
    material: J2ViscoplasticMaterial,
    times: np.ndarray,
    strains: np.ndarray,
    committed_states: tuple,
    responses: tuple,
    update: MaterialUpdateSettings,
    audit: AuditThresholds,
) -> dict[str, np.ndarray]:
    """Check two accepted local algorithmic tangents from the same committed states."""
    if len(responses) < 2:
        raise ValueError("Material tangent validation requires at least two accepted states.")
    direction = _tangent_check_direction()
    state_ids = np.asarray([0, len(responses) - 1], dtype=np.int64)
    errors = []
    for response_id in state_ids:
        time_step = float(times[response_id + 1] - times[response_id])
        previous_strain = strains[response_id]
        target = strains[response_id + 1]
        perturbed_stresses = []
        for sign in (1.0, -1.0):
            perturbed = target + sign * audit.tangent_perturbation * direction
            response = material.update(
                MaterialPointRequest(
                    strains=perturbed[None, :],
                    strain_rates=((perturbed - previous_strain) / time_step)[None, :],
                    time_step=time_step,
                    kinematics="three_dimensional",
                    update_settings=update,
                ),
                committed_states[response_id],
                MaterialResponseRequirements(tangent=True),
            )
            perturbed_stresses.append(response.stresses[0])
        tangent = _require_array(responses[response_id].tangents, "tangent")[0]
        finite_difference = (perturbed_stresses[0] - perturbed_stresses[1]) / (
            2.0 * audit.tangent_perturbation
        )
        tangent_action = tangent @ direction
        scale = max(float(np.linalg.norm(tangent_action)), audit.tangent_stress_scale)
        errors.append(float(np.linalg.norm(finite_difference - tangent_action) / scale))
    return {
        "tangent_state_index": state_ids,
        "tangent_direction_error": np.asarray(errors, dtype=np.float64),
    }


def _tangent_check_direction() -> np.ndarray:
    """Return the fixed six-component direction used by every data source."""
    direction = np.asarray([0.37, -0.21, -0.16, 0.29, -0.41, 0.58], dtype=np.float64)
    return direction / np.linalg.norm(direction)


def _write_shard_performance(
    output_path: Path,
    source: DataSourcePlan,
    tasks: tuple[RVEDataTask, ...],
    total_time: float,
    solve_time: float,
    write_time: float,
) -> None:
    """Write per-shard throughput without mixing timing data into training arrays."""
    step_count = sum(
        next(case.time_points - 1 for case in source.load_cases if case.name == task.load_case_name)
        for task in tasks
    )
    payload = {
        "source_name": source.name,
        "source_kind": source.kind,
        "path_count": len(tasks),
        "accepted_increment_count": step_count,
        "total_time": total_time,
        "solver_time": solve_time,
        "hdf5_write_time": write_time,
        "other_time": total_time - solve_time - write_time,
        "paths_per_second": len(tasks) / total_time,
        "increments_per_second": step_count / total_time,
        "file_size_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(".performance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
