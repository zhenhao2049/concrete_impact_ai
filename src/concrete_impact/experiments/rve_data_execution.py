"""Manifest and process-launch orchestration for RVE-RNO data tasks.

Contents:
    Worker profiling, task manifests, process launch, response audits, and summaries.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import json
import multiprocessing
import resource
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, sleep
from typing import cast

import h5py
import numpy as np

from concrete_impact.core.progress import JsonProgressRecorder
from concrete_impact.experiments.rve_data_plan import (
    RVEDataPlan,
    RVEDataTask,
    build_rve_data_task_manifest,
    validate_parallel_resources,
)
from concrete_impact.experiments.rve_path_generation import generate_data_control_path


def write_rve_data_manifest(plan: RVEDataPlan, output_path: str | Path) -> Path:
    """Write the deterministic production task manifest without running solvers."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = build_rve_data_task_manifest(plan)
    payload = {
        "schema_version": plan.schema_version,
        "output_directory": str(plan.output_directory),
        "task_count": len(tasks),
        "tasks": [task.model_dump(mode="json") for task in tasks],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def generate_control_path_shards(
    plan: RVEDataPlan,
    available_cpus: int,
) -> tuple[Path, ...]:
    """Generate deterministic process-owned control-path shards for solver workers."""
    validate_parallel_resources(plan, available_cpus)
    tasks = build_rve_data_task_manifest(plan)
    tasks_by_shard: dict[int, list[RVEDataTask]] = {}
    for task in tasks:
        tasks_by_shard.setdefault(task.shard_index, []).append(task)
    source_map = {source.name: source for source in plan.sources}
    arguments = [
        (
            plan.output_directory,
            shard_id,
            tuple(task.model_dump(mode="json") for task in shard_tasks),
            {name: source.model_dump(mode="json") for name, source in source_map.items()},
        )
        for shard_id, shard_tasks in sorted(tasks_by_shard.items())
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=plan.parallel.workers,
        mp_context=context,
    ) as executor:
        paths = tuple(executor.map(_write_control_path_shard, arguments))
    return paths


def generate_response_shards(
    plan: RVEDataPlan,
    available_cpus: int,
) -> tuple[Path, ...]:
    """Launch process-owned high-fidelity response shards for all four sources."""
    validate_parallel_resources(plan, available_cpus)
    tasks = build_rve_data_task_manifest(plan)
    source_map = {source.name: source for source in plan.sources}
    from concrete_impact.experiments.rve_response_generation import (
        write_impact_control_paths,
    )

    impact_control_paths = {
        source.name: write_impact_control_paths(
            plan.output_directory,
            source,
            tuple(task for task in tasks if task.source_name == source.name),
        )
        for source in plan.sources
        if source.kind == "impact_path"
    }
    grouped: dict[tuple[str, int], list[RVEDataTask]] = {}
    for task in tasks:
        grouped.setdefault((task.source_name, task.shard_index), []).append(task)
    arguments = [
        (
            plan.output_directory,
            shard_id,
            source_map[source_name].model_dump(mode="json"),
            tuple(task.model_dump(mode="json") for task in shard_tasks),
            plan.audit.model_dump(mode="json"),
            (
                str(impact_control_paths[source_name])
                if source_name in impact_control_paths
                else None
            ),
            plan.storage_mode,
            plan.parallel.compression,
            plan.parallel.thread_count_per_worker,
        )
        for (source_name, shard_id), shard_tasks in sorted(grouped.items())
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=plan.parallel.workers,
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_calculate_response_shard, argument) for argument in arguments]
        completed_paths = []
        recorder = JsonProgressRecorder(
            plan.output_directory / "progress.json",
            {"calculation": "rve_response_generation"},
            plan.output_directory / "summary.log",
            log_accepted_steps=False,
        )
        for future in as_completed(futures):
            completed_paths.append(future.result())
            recorder.publish_stage(
                "response_shards",
                len(completed_paths),
                len(futures),
            )
        return tuple(sorted(completed_paths))


def profile_fixed_cell_path_workers(
    plan: RVEDataPlan,
    available_cpus: int,
    worker_counts: tuple[int, int] = (4, 5),
) -> Path:
    """Compare 4- and 5-process execution on four complete fixed-cell paths."""
    if plan.purpose != "fixed_cell_rve_training":
        raise ValueError("Path-worker profiling requires a fixed-cell RVE training plan.")
    root = plan.output_directory / "worker_precheck"
    records = []
    outputs = []
    for workers in worker_counts:
        load_cases = tuple(
            load_case.model_copy(update={"path_count": 2})
            for load_case in plan.sources[0].load_cases
        )
        source = plan.sources[0].model_copy(update={"load_cases": load_cases})
        parallel = plan.parallel.model_copy(update={"workers": workers})
        current = plan.model_copy(
            update={
                "output_directory": root / f"workers_{workers}",
                "sources": (source,),
                "parallel": parallel,
            }
        )
        validate_parallel_resources(current, available_cpus)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        stop_monitor = threading.Event()
        memory_record = {"peak_total_rss_kib": 0}
        monitor = threading.Thread(
            target=_monitor_process_tree_memory,
            args=(stop_monitor, memory_record),
            daemon=True,
        )
        monitor.start()
        start = perf_counter()
        try:
            shards = generate_response_shards(current, available_cpus)
            wall_time = perf_counter() - start
        finally:
            stop_monitor.set()
            monitor.join()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        audit_response_shards(current, shards)
        performance = [
            json.loads(path.with_suffix(".performance.json").read_text(encoding="utf-8"))
            for path in shards
        ]
        record = {
            "workers": workers,
            "threads_per_worker": current.parallel.thread_count_per_worker,
            "logical_cpu_budget": workers * current.parallel.thread_count_per_worker,
            "wall_time": wall_time,
            "child_cpu_time": (after.ru_utime + after.ru_stime)
            - (before.ru_utime + before.ru_stime),
            "peak_total_rss_kib": memory_record["peak_total_rss_kib"],
            "maximum_child_rss_kib": after.ru_maxrss,
            "summed_solver_time": sum(item["solver_time"] for item in performance),
            "summed_hdf5_write_time": sum(
                item["hdf5_write_time"] for item in performance
            ),
            "orchestration_critical_path_overhead": wall_time
            - max(item["total_time"] for item in performance),
            "file_size_bytes": sum(item["file_size_bytes"] for item in performance),
        }
        records.append(record)
        outputs.append(shards)
    equality = _compare_response_shard_sets(outputs[0], outputs[1])
    report = {
        "passed": equality["passed"],
        "path_count": 4,
        "records": records,
        "numerical_equality": equality,
        "selection_rule": "report_only_no_automatic_worker_change",
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / "path_worker_performance.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _monitor_process_tree_memory(
    stop_event: threading.Event,
    record: dict[str, int],
) -> None:
    """Sample parent-plus-descendant resident memory during one profile run."""
    import os

    while not stop_event.is_set():
        process_ids = _descendant_process_ids(os.getpid()) | {os.getpid()}
        total = sum(_process_rss_kib(process_id) for process_id in process_ids)
        record["peak_total_rss_kib"] = max(record["peak_total_rss_kib"], total)
        sleep(0.25)


def _descendant_process_ids(parent_id: int) -> set[int]:
    """Read Linux child-process topology rooted at the current project process."""
    descendants: set[int] = set()
    pending = [parent_id]
    while pending:
        current = pending.pop()
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children_text = children_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        children = tuple(int(value) for value in children_text.split())
        new_children = [child for child in children if child not in descendants]
        descendants.update(new_children)
        pending.extend(new_children)
    return descendants


def _process_rss_kib(process_id: int) -> int:
    """Read one Linux process resident-set size without modifying the process."""
    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    line = next(value for value in status.splitlines() if value.startswith("VmRSS:"))
    return int(line.split()[1])


def _compare_response_shard_sets(
    first: tuple[Path, ...], second: tuple[Path, ...]
) -> dict[str, object]:
    """Require identical accepted macro fields from two process counts."""
    arrays = []
    for paths in (first, second):
        mapping = {}
        for path in paths:
            with h5py.File(path, "r") as handle:
                for path_id, group in handle["paths"].items():
                    mapping[path_id] = np.concatenate(
                        (
                            group["macro/strain"][...].reshape(-1),
                            group["macro/stress"][...].reshape(-1),
                            group["macro/effective_tangent"][...].reshape(-1),
                            group["macro/dissipation_density"][...].reshape(-1),
                            group["macro/matrix_maximum_q"][...].reshape(-1),
                        )
                    )
        arrays.append(mapping)
    if set(arrays[0]) != set(arrays[1]):
        raise ValueError("Path-worker comparison produced different path identifiers.")
    maximum = max(
        float(np.max(np.abs(arrays[0][name] - arrays[1][name]))) for name in arrays[0]
    )
    return {"passed": maximum == 0.0, "maximum_absolute_difference": maximum}


def _calculate_response_shard(
    arguments: tuple[Path, int, dict, tuple[dict, ...], dict, str | None, str, str, int],
) -> Path:
    """Reconstruct validated payloads and execute one response shard."""
    from concrete_impact.experiments.rve_data_plan import AuditThresholds, DataSourcePlan
    from concrete_impact.experiments.rve_response_generation import write_response_shard

    (
        output_directory,
        shard_id,
        source_payload,
        task_payloads,
        audit_payload,
        impact_control_value,
        storage_mode,
        compression,
        thread_count,
    ) = arguments
    source = DataSourcePlan.model_validate(source_payload)
    tasks = tuple(RVEDataTask.model_validate(payload) for payload in task_payloads)
    audit = AuditThresholds.model_validate(audit_payload)
    impact_control_path = (
        None if impact_control_value is None else Path(impact_control_value)
    )
    from concrete_impact.core.thread_control import numerical_thread_limit

    with numerical_thread_limit(thread_count):
        return write_response_shard(
            output_directory,
            shard_id,
            source,
            tasks,
            audit,
            impact_control_path,
            storage_mode,
            compression,
        )


def audit_control_path_shards(paths: tuple[Path, ...]) -> dict[str, int]:
    """Validate deterministic control shards before launching expensive solvers."""
    task_count = 0
    strain_path_count = 0
    pressure_path_count = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs["status"] != "complete":
                raise ValueError(f"Control shard is incomplete: {path}.")
            for task_id, group in handle["tasks"].items():
                times = group["time"][...]
                controls = group["control"][...]
                if times.size < 2 or np.any(np.diff(times) <= 0.0):
                    raise ValueError(f"Control task has invalid time coordinates: {task_id}.")
                if not np.all(np.isfinite(controls)):
                    raise ValueError(f"Control task has non-finite values: {task_id}.")
                task_count += 1
                if group.attrs["control_kind"] == "macro_strain":
                    if controls.shape != (times.size, 6):
                        raise ValueError(f"Strain control shape is invalid: {task_id}.")
                    strain_path_count += 1
                elif group.attrs["control_kind"] == "pressure":
                    if controls.shape != (times.size, 1):
                        raise ValueError(f"Pressure control shape is invalid: {task_id}.")
                    pressure_path_count += 1
                else:
                    raise ValueError(f"Unknown control kind in task: {task_id}.")
    return {
        "task_count": task_count,
        "strain_path_count": strain_path_count,
        "pressure_path_count": pressure_path_count,
    }


def audit_response_shards(
    plan: RVEDataPlan,
    paths: tuple[Path, ...],
) -> dict[str, object]:
    """Audit physical branch coverage and complete-path HDF5 invariants."""
    expected_ids = {task.task_id for task in build_rve_data_task_manifest(plan)}
    return _audit_response_collection(plan, paths, expected_ids, "full")


def audit_response_subset(
    plan: RVEDataPlan,
    paths: tuple[Path, ...],
    expected_task_ids: tuple[str, ...],
) -> dict[str, object]:
    """Audit one explicit subset without weakening complete-production semantics."""
    expected_ids = set(expected_task_ids)
    if not expected_ids or len(expected_ids) != len(expected_task_ids):
        raise ValueError("Response subset task ids must be nonempty and unique.")
    plan_ids = {task.task_id for task in build_rve_data_task_manifest(plan)}
    unknown = sorted(expected_ids - plan_ids)
    if unknown:
        raise ValueError(f"Response subset contains tasks outside the frozen plan: {unknown}.")
    return _audit_response_collection(plan, paths, expected_ids, "pilot_subset")


def _audit_response_collection(
    plan: RVEDataPlan,
    paths: tuple[Path, ...],
    expected_ids: set[str],
    scope: str,
) -> dict[str, object]:
    """Audit one explicit response collection against an exact task-id set."""
    tasks = {task.task_id: task for task in build_rve_data_task_manifest(plan)}
    sources = {source.name: source for source in plan.sources}
    observed_ids: set[str] = set()
    records = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs["status"] != "complete":
                raise ValueError(f"Response shard is not complete: {path}.")
            for task_id, group in handle["paths"].items():
                if task_id in observed_ids:
                    raise ValueError(f"Duplicate response path id: {task_id}.")
                if task_id not in tasks:
                    raise ValueError(f"Response path id is outside the frozen plan: {task_id}.")
                observed_ids.add(task_id)
                metadata = json.loads(group.attrs["metadata_json"])
                task = tasks[task_id]
                record = _audit_response_path(
                    task_id,
                    group,
                    metadata,
                    sources[task.source_name],
                    plan,
                )
                records.append(record)
    if observed_ids != expected_ids:
        raise ValueError(
            "Response shard task ids do not match the deterministic manifest: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - expected_ids)}."
        )
    source_counts = {
        source.name: sum(task_id.startswith(f"{source.name}__") for task_id in observed_ids)
        for source in plan.sources
    }
    tangent_checks, maximum_tangent_error = _summarize_tangent_checks(
        records,
        plan.audit.tangent_direction_tolerance,
    )
    return {
        "passed": True,
        "scope": scope,
        "path_count": len(records),
        "source_path_counts": source_counts,
        "tangent_direction_checks": tangent_checks,
        "maximum_tangent_direction_error": maximum_tangent_error,
        "maximum_q": max(cast(float, item["maximum_q"]) for item in records),
        "minimum_q_increment": min(
            cast(float, item["minimum_q_increment"]) for item in records
        ),
        "minimum_dissipation_density": min(
            cast(float, item["minimum_dissipation_density"]) for item in records
        ),
        "paths": records,
    }


def _audit_response_path(
    task_id: str,
    group: h5py.Group,
    metadata: dict,
    source,
    plan: RVEDataPlan,
) -> dict[str, object]:
    """Audit one accepted material-point or RVE path without repairing data."""
    _require_declared_fields(task_id, group, source.fields)
    macro = group["macro"]
    times = macro["time"][...]
    _require_time_aligned_fields(task_id, group, source.fields, times.size)
    strains = macro["strain"][...]
    stresses = macro["stress"][...]
    dissipation = macro["dissipation_density"][...]
    if times.size == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"Response path time is not strictly increasing: {task_id}.")
    for name, values in (
        ("strain", strains),
        ("stress", stresses),
        ("dissipation_density", dissipation),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Response path contains non-finite {name}: {task_id}.")
    q = _read_equivalent_plastic_strain(group)
    q_increment = np.diff(q, axis=0)
    minimum_q_increment = 0.0 if q_increment.size == 0 else float(np.min(q_increment))
    minimum_dissipation = float(np.min(dissipation))
    regime = metadata["expected_response"]["regime"]
    maximum_q = float(np.max(q))
    thresholds = plan.audit
    if minimum_q_increment < -thresholds.q_tolerance:
        raise ValueError(f"Equivalent plastic strain decreased in response path: {task_id}.")
    if minimum_dissipation < -thresholds.dissipation_tolerance:
        raise ValueError(f"Negative dissipation detected in response path: {task_id}.")
    active = _read_diagnostic(group, "viscoplastic_active")
    maximum_active = float(np.max(active))
    cumulative_dissipation = float(np.sum(dissipation * macro["time_step"][...]))
    if regime == "elastic" and maximum_q > thresholds.q_tolerance:
        raise ValueError(f"Elastic pilot path entered viscoplastic flow: {task_id}, q={maximum_q}.")
    if regime == "elastic" and maximum_active > 0.0:
        raise ValueError(f"Elastic pilot path has active viscoplastic points: {task_id}.")
    if regime == "elastic" and cumulative_dissipation > thresholds.dissipation_tolerance:
        raise ValueError(f"Elastic pilot path has nonzero cumulative dissipation: {task_id}.")
    if regime == "viscoplastic" and maximum_q <= thresholds.plastic_q_threshold:
        raise ValueError(f"Viscoplastic pilot path remained elastic: {task_id}, q={maximum_q}.")
    if regime == "viscoplastic" and maximum_active <= 0.0:
        raise ValueError(f"Viscoplastic pilot path has no active flow point: {task_id}.")
    if (
        regime == "viscoplastic"
        and cumulative_dissipation
        <= thresholds.positive_cumulative_dissipation_threshold
    ):
        raise ValueError(f"Viscoplastic pilot path has no positive dissipation: {task_id}.")
    increments = np.diff(strains, axis=0)
    trapezoidal_stress = 0.5 * (stresses[1:] + stresses[:-1])
    work = np.sum(trapezoidal_stress * increments * np.asarray([1, 1, 1, 0.5, 0.5, 0.5]), axis=1)
    family = str(metadata["family"])
    unloading_required = bool(metadata["expected_response"]["require_unloading"])
    if family in {"load_unload_reload", "reverse"}:
        unloading_required = True
    equivalent_stress = _equivalent_stress(stresses)
    equivalent_stress_increment = np.diff(equivalent_stress)
    if unloading_required and not np.any(work < -thresholds.work_tolerance):
        raise ValueError(f"Pilot path lacks required negative incremental work: {task_id}.")
    if (
        unloading_required
        and not np.any(equivalent_stress_increment < -thresholds.stress_drop_threshold)
    ):
        raise ValueError(f"Pilot path lacks required equivalent-stress reduction: {task_id}.")
    reverse_required = bool(metadata["expected_response"]["require_reverse_loading"])
    if family == "reverse":
        reverse_required = True
    if reverse_required and not _has_reverse_increment(
        increments, thresholds.strain_increment_tolerance
    ):
        raise ValueError(f"Pilot path lacks required reverse strain increment: {task_id}.")
    if family == "load_unload_reload":
        negative_ids = np.flatnonzero(work < -thresholds.work_tolerance)
        if negative_ids.size == 0 or not np.any(
            work[negative_ids[0] + 1 :] > thresholds.work_tolerance
        ):
            raise ValueError(f"Pilot path lacks required reloading after unloading: {task_id}.")
    if family == "hold":
        _require_hold_relaxation(
            task_id,
            strains,
            stresses,
            q,
            active,
            regime,
            thresholds,
        )
    tangent_error = None
    tangent_path = "validation/tangent_direction_error"
    if tangent_path in group:
        tangent_values = group[tangent_path][...]
        if tangent_values.size < 2 or not np.all(np.isfinite(tangent_values)):
            raise ValueError(
                "Direction-tangent validation requires at least two finite states: "
                f"{task_id}."
            )
        tangent_error = float(np.max(tangent_values))
        if tangent_error > thresholds.tangent_direction_tolerance:
            raise ValueError(
                "Direction-tangent validation exceeded its fixed tolerance: "
                f"task_id={task_id}, error={tangent_error}, "
                f"tolerance={thresholds.tangent_direction_tolerance}."
            )
    return {
        "task_id": task_id,
        "source_name": source.name,
        "regime": regime,
        "family": family,
        "maximum_q": maximum_q,
        "minimum_q_increment": minimum_q_increment,
        "minimum_dissipation_density": minimum_dissipation,
        "minimum_incremental_work": float(np.min(work)) if work.size else 0.0,
        "maximum_viscoplastic_active": maximum_active,
        "cumulative_dissipation": cumulative_dissipation,
        "minimum_equivalent_stress_increment": (
            float(np.min(equivalent_stress_increment))
            if equivalent_stress_increment.size
            else 0.0
        ),
        "maximum_tangent_direction_error": tangent_error,
        "lineage_id": metadata["lineage_id"],
        "macro_path_origin": metadata["macro_path_origin"],
    }


def _summarize_tangent_checks(
    records: list[dict[str, object]],
    tolerance: float,
) -> tuple[dict[str, dict[str, float | bool]], float]:
    """Require one two-state tangent check for every source-response stratum."""
    required = {
        f"{record['source_name']}|{record['regime']}" for record in records
    }
    observed: dict[str, list[float]] = {key: [] for key in required}
    for record in records:
        value = record["maximum_tangent_direction_error"]
        if value is not None:
            key = f"{record['source_name']}|{record['regime']}"
            observed[key].append(cast(float, value))
    missing = sorted(key for key, values in observed.items() if not values)
    if missing:
        raise ValueError(
            "Direction-tangent validation is missing for response strata: "
            f"{missing}."
        )
    summary = {
        key: {
            "passed": max(values) <= tolerance,
            "maximum_error": max(values),
            "tolerance": tolerance,
        }
        for key, values in sorted(observed.items())
    }
    maximum = max(float(item["maximum_error"]) for item in summary.values())
    return summary, maximum


def _read_equivalent_plastic_strain(group: h5py.Group) -> np.ndarray:
    """Read the canonical single- or matrix-phase equivalent plastic strain."""
    if "micro/state" in group:
        state = group["micro/state"]
        for name in ("phase_matrix__equivalent_plastic_strain", "equivalent_plastic_strain"):
            if name in state:
                values = state[name][...]
                return values.reshape(values.shape[0], -1)
    if "macro/matrix_maximum_q" in group:
        return group["macro/matrix_maximum_q"][...][:, None]
    raise ValueError(f"Response path has no equivalent plastic strain: {group.name}.")


def _has_reverse_increment(increments: np.ndarray, tolerance: float = 1.0e-14) -> bool:
    """Detect two nonzero strain increments with negative tensor inner product."""
    weights = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    norms = np.sqrt(np.sum(increments**2 * weights, axis=1))
    nonzero = increments[norms > tolerance]
    if nonzero.shape[0] < 2:
        return False
    products = np.sum(nonzero[1:] * nonzero[:-1] * weights, axis=1)
    return bool(np.any(products < 0.0))


def _require_declared_fields(task_id: str, group: h5py.Group, fields) -> None:
    """Require every configured training field in the accepted path group."""
    direct = {
        "time": "macro/time",
        "time_step": "macro/time_step",
        "macro_strain": "macro/strain",
        "macro_stress": "macro/stress",
        "effective_tangent": "macro/effective_tangent",
        "dissipation_density": "macro/dissipation_density",
        "material_parameters": "features/material_parameters",
        "microstructure_features": "features/microstructure_features",
    }
    for name in (*fields.inputs, *fields.labels):
        path = direct[name]
        if path not in group:
            raise ValueError(f"Declared response field is missing: {task_id}/{path}.")
    for name in fields.state_fields:
        if f"micro/state/{name}" not in group:
            raise ValueError(f"Declared state field is missing: {task_id}/{name}.")
    for name in fields.diagnostics:
        if name == "phase_average_stress":
            available = "phases" in group and all(
                "average_stress" in phase for phase in group["phases"].values()
            )
        elif name == "phase_volume_fraction":
            available = "phases" in group and all(
                "volume_fraction" in phase for phase in group["phases"].values()
            )
        else:
            available = (
                f"micro/diagnostics/{name}" in group or f"solver/{name}" in group
            )
        if not available:
            raise ValueError(f"Declared diagnostic field is missing: {task_id}/{name}.")


def _require_time_aligned_fields(
    task_id: str,
    group: h5py.Group,
    fields,
    time_count: int,
) -> None:
    """Require declared histories to share the accepted macro time dimension."""
    required_shapes = {
        "time": (time_count,),
        "time_step": (time_count,),
        "macro_strain": (time_count, 6),
        "macro_stress": (time_count, 6),
        "effective_tangent": (time_count, 6, 6),
        "dissipation_density": (time_count,),
    }
    direct = {
        "time": "macro/time",
        "time_step": "macro/time_step",
        "macro_strain": "macro/strain",
        "macro_stress": "macro/stress",
        "effective_tangent": "macro/effective_tangent",
        "dissipation_density": "macro/dissipation_density",
    }
    for name in (*fields.inputs, *fields.labels):
        if name not in required_shapes:
            continue
        shape = group[direct[name]].shape
        if shape != required_shapes[name]:
            raise ValueError(
                f"Declared response field is not time aligned: {task_id}/{name}, "
                f"shape={shape}, expected={required_shapes[name]}."
            )
    for name in fields.state_fields:
        shape = group[f"micro/state/{name}"].shape
        if not shape or shape[0] != time_count:
            raise ValueError(
                f"Declared state field is not time aligned: {task_id}/{name}, shape={shape}."
            )
    for name in fields.diagnostics:
        for path in (f"micro/diagnostics/{name}", f"solver/{name}"):
            if path in group and group[path].shape[0] != time_count:
                raise ValueError(
                    f"Declared diagnostic is not time aligned: {task_id}/{name}, "
                    f"shape={group[path].shape}."
                )


def _read_diagnostic(group: h5py.Group, name: str) -> np.ndarray:
    """Read one required pointwise or homogenized diagnostic history."""
    aliases = {
        "viscoplastic_active": "viscoplastic_active_volume_fraction",
    }
    solver_name = aliases.get(name, name)
    for path in (f"micro/diagnostics/{name}", f"solver/{solver_name}"):
        if path in group:
            values = group[path][...]
            availability_path = f"{path}__available"
            if availability_path in group:
                available = group[availability_path][...].astype(bool)
                if not np.any(available):
                    raise ValueError(f"Diagnostic has no available values: {group.name}/{name}.")
                if not np.all(np.isfinite(values[available])):
                    raise ValueError(
                        f"Diagnostic contains non-finite available values: {group.name}/{name}."
                    )
                return np.where(available, values, 0.0)
            return values
    raise ValueError(f"Required diagnostic is missing: {group.name}/{name}.")


def _equivalent_stress(stresses: np.ndarray) -> np.ndarray:
    """Compute macroscopic J2 equivalent stress in engineering Voigt ordering."""
    pressure = np.mean(stresses[:, :3], axis=1)
    deviator = stresses[:, :3] - pressure[:, None]
    squared = np.sum(deviator**2, axis=1) + 2.0 * np.sum(stresses[:, 3:] ** 2, axis=1)
    return np.sqrt(1.5 * squared)


def _require_hold_relaxation(
    task_id: str,
    strains: np.ndarray,
    stresses: np.ndarray,
    q: np.ndarray,
    active: np.ndarray,
    regime: str,
    thresholds,
) -> None:
    """Check elastic constancy or viscoplastic relaxation on a fixed-strain interval."""
    weights = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    increments = np.diff(strains, axis=0)
    norms = np.sqrt(np.sum(increments**2 * weights, axis=1))
    hold_ids = np.flatnonzero(norms <= thresholds.hold_strain_tolerance)
    if hold_ids.size == 0:
        raise ValueError(f"Hold path has no fixed-strain interval: {task_id}.")
    equivalent_stress = _equivalent_stress(stresses)
    relaxation = equivalent_stress[hold_ids] - equivalent_stress[hold_ids + 1]
    if regime == "elastic" and np.any(
        np.abs(relaxation) > thresholds.hold_stress_relaxation_threshold
    ):
        raise ValueError(f"Elastic hold path changed stress at fixed strain: {task_id}.")
    if regime == "viscoplastic" and not np.any(
        relaxation >= thresholds.hold_stress_relaxation_threshold
    ):
        raise ValueError(f"Hold path has no measurable stress relaxation: {task_id}.")
    q_flat = np.max(q, axis=1)
    active_by_step = np.asarray(active).reshape(active.shape[0], -1).max(axis=1)
    active_hold_ids = hold_ids[active_by_step[hold_ids + 1] > 0.0]
    if active_hold_ids.size and not np.any(
        q_flat[active_hold_ids + 1] - q_flat[active_hold_ids] > thresholds.q_tolerance
    ):
        raise ValueError(f"Active hold path has no equivalent-plastic-strain growth: {task_id}.")


def _write_control_path_shard(
    arguments: tuple[Path, int, tuple[dict, ...], dict[str, dict]],
) -> Path:
    """Write one worker-owned shard of deterministic controls and task metadata."""
    from concrete_impact.experiments.rve_data_plan import DataSourcePlan

    output_directory, shard_id, task_payloads, source_payloads = arguments
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / f"control_shard_{shard_id:06d}.h5"
    partial_path = final_path.with_suffix(".h5.partial")
    sources = {
        name: DataSourcePlan.model_validate(payload)
        for name, payload in source_payloads.items()
    }
    with h5py.File(partial_path, "w") as handle:
        handle.attrs["schema_version"] = "3.0"
        handle.attrs["artifact_kind"] = "rve_control_paths"
        handle.attrs["status"] = "incomplete"
        for task_payload in task_payloads:
            task = RVEDataTask.model_validate(task_payload)
            path = generate_data_control_path(sources[task.source_name], task)
            group = handle.create_group(f"tasks/{task.task_id}")
            group.attrs["source_name"] = task.source_name
            group.attrs["source_kind"] = task.source_kind
            group.attrs["family"] = path.family
            group.attrs["seed"] = task.seed
            group.attrs["lineage_id"] = task.lineage_id
            group.attrs["macro_path_origin"] = task.macro_path_origin
            group.attrs["control_kind"] = path.control_kind
            group.create_dataset("time", data=path.times, compression="lzf")
            group.create_dataset("control", data=path.values, compression="lzf")
        handle.attrs["status"] = "complete"
    partial_path.replace(final_path)
    return final_path
