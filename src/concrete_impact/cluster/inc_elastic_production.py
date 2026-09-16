"""Manifest-worker-finalizer production for linear-elastic INC data.

Author:
    Zhen Hao.
Created:
    2026-09-15.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np

from concrete_impact.experiments.inc_elastic_generation import (
    ElasticINCAcceptanceError,
    build_elastic_inc_preflight,
    generate_elastic_inc_path,
    load_elastic_inc_system,
)
from concrete_impact.experiments.inc_elastic_plan import (
    ElasticINCPlan,
    ElasticINCTask,
    build_control_parameters,
    build_elastic_inc_tasks,
    load_elastic_inc_plan,
)


def prepare_elastic_inc_production(
    config_path: str | Path,
    bundle_count: int,
    parallel_paths: int,
) -> Path:
    """Prepare one deterministic elastic INC manifest and task bundles."""
    if bundle_count <= 0 or parallel_paths <= 0:
        raise ValueError("INC bundle count and path concurrency must be positive.")
    config = Path(config_path)
    plan = load_elastic_inc_plan(config)
    if plan.design == "p1_official_144" and plan.release_status != "approved":
        raise ValueError("P1 official production requires an approved trial review.")
    tasks = build_elastic_inc_tasks(plan)
    root = plan.output_directory
    manifest_path = root / "manifest.json"
    plan_payload = plan.model_dump(mode="json")
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest["plan"] != plan_payload:
            raise ValueError("Existing INC manifest differs from the requested plan.")
        if manifest["bundle_count"] != bundle_count:
            raise ValueError("Existing INC manifest uses a different bundle count.")
        if manifest["parallel_paths"] != parallel_paths:
            raise ValueError("Existing INC manifest uses different path concurrency.")
        return manifest_path
    if root.exists():
        raise FileExistsError(f"INC output root exists without a manifest: {root}.")

    (root / "bundles").mkdir(parents=True)
    (root / "tasks").mkdir()
    task_groups = tuple(tuple(tasks[index::bundle_count]) for index in range(bundle_count))
    if any(not group for group in task_groups):
        raise ValueError("INC bundle count exceeds the number of paths.")
    manifest = {
        "schema_version": "2.0",
        "created_utc": _utc_now(),
        "config_path": str(config),
        "output_root": str(root),
        "task_count": len(tasks),
        "bundle_count": bundle_count,
        "parallel_paths": parallel_paths,
        "plan": plan_payload,
        "tasks": {task.task_id: task.model_dump(mode="json") for task in tasks},
        "bundles": [f"bundles/bundle_{index:03d}.json" for index in range(bundle_count)],
    }
    _write_json_atomic(manifest_path, manifest)
    for bundle_id, group in enumerate(task_groups):
        _write_json_atomic(
            root / "bundles" / f"bundle_{bundle_id:03d}.json",
            {
                "schema_version": "2.0",
                "manifest_path": str(manifest_path),
                "bundle_id": bundle_id,
                "task_ids": [task.task_id for task in group],
            },
        )
    return manifest_path


def run_elastic_inc_preflight(manifest_path: str | Path) -> dict[str, Any]:
    """Build the fixed system for one prepared production root."""
    manifest = _read_json(Path(manifest_path))
    plan = ElasticINCPlan.model_validate(manifest["plan"])
    return build_elastic_inc_preflight(plan, manifest["output_root"])


def run_elastic_inc_bundle(
    manifest_path: str | Path,
    bundle_path: str | Path,
    allow_local_p1: bool = False,
) -> dict[str, Any]:
    """Run all pending paths from one independent Slurm bundle."""
    manifest_file = Path(manifest_path)
    bundle_file = Path(bundle_path)
    manifest = _read_json(manifest_file)
    bundle = _read_json(bundle_file)
    if Path(bundle["manifest_path"]) != manifest_file:
        raise ValueError("INC bundle references a different manifest.")
    task_ids = tuple(str(value) for value in bundle["task_ids"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("INC bundle contains duplicate task identifiers.")
    unknown = sorted(set(task_ids) - set(manifest["tasks"]))
    if unknown:
        raise ValueError(f"INC bundle contains unknown tasks: {unknown}.")
    arguments = tuple(
        {
            "plan": manifest["plan"],
            "task": manifest["tasks"][task_id],
            "output_root": manifest["output_root"],
        }
        for task_id in task_ids
    )
    _require_safe_execution_host(manifest, allow_local_p1)
    started = perf_counter()
    records = []
    with ProcessPoolExecutor(
        max_workers=min(int(manifest["parallel_paths"]), len(arguments)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = [pool.submit(_run_one_task, argument) for argument in arguments]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["task_id"])
    result = {
        "schema_version": "2.0",
        "bundle_id": int(bundle["bundle_id"]),
        "task_count": len(records),
        "success_count": sum(record["passed"] for record in records),
        "failure_count": sum(not record["passed"] for record in records),
        "wall_seconds": perf_counter() - started,
        "tasks": records,
    }
    _write_json_atomic(
        Path(manifest["output_root"])
        / "bundles"
        / f"bundle_{int(bundle['bundle_id']):03d}_result.json",
        result,
    )
    return result


def summarize_elastic_inc_status(manifest_path: str | Path) -> dict[str, Any]:
    """Return a compact production status for login-node monitoring."""
    manifest = _read_json(Path(manifest_path))
    root = Path(manifest["output_root"])
    counts = {"succeeded": 0, "failed": 0, "running": 0, "pending": 0}
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed_wall_seconds = 0.0
    for task_id in sorted(manifest["tasks"]):
        task_root = root / "tasks" / task_id
        status_path = task_root / "status.json"
        if not status_path.exists():
            counts["pending"] += 1
            continue
        status = _read_json(status_path)
        state = str(status["state"])
        if state == "succeeded":
            counts["succeeded"] += 1
            completed_wall_seconds += float(status["wall_seconds"])
        elif state == "failed":
            counts["failed"] += 1
            if len(failures) < 3:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": status["reason"],
                    }
                )
        else:
            counts["running"] += 1
            progress_path = task_root / "progress.json"
            if progress_path.exists() and len(active) < 3:
                progress = _read_json(progress_path)
                active.append(
                    {
                        "task_id": task_id,
                        "stage": progress["stage"],
                        "progress_percent": progress["progress_percent"],
                        "elapsed_seconds": progress["elapsed_seconds"],
                    }
                )
    return {
        "task_count": int(manifest["task_count"]),
        **counts,
        "completed_path_wall_seconds": completed_wall_seconds,
        "active_examples": active,
        "failure_examples": failures,
    }


def finalize_elastic_inc_production(manifest_path: str | Path) -> dict[str, Any]:
    """Validate every path and publish the sharded dataset manifest."""
    manifest = _read_json(Path(manifest_path))
    plan = ElasticINCPlan.model_validate(manifest["plan"])
    tasks = tuple(
        ElasticINCTask.model_validate(manifest["tasks"][task_id])
        for task_id in sorted(manifest["tasks"])
    )
    root = Path(manifest["output_root"])
    status = summarize_elastic_inc_status(manifest_path)
    if status["succeeded"] != manifest["task_count"] or any(
        status[name] != 0 for name in ("failed", "running", "pending")
    ):
        report = {
            "schema_version": "2.0",
            "passed": False,
            "reason": "INC production is incomplete.",
            "status": status,
        }
        _write_json_atomic(root / "data_acceptance.json", report)
        raise ValueError(f"INC production is incomplete: {status}.")

    response_paths = []
    split_counts = {"pilot": 0, "train": 0, "validation": 0, "test": 0}
    for task in tasks:
        response_path = root / "tasks" / task.task_id / "response.h5"
        _validate_response_file(response_path, plan, task)
        metrics = _read_json(root / "tasks" / task.task_id / "metrics.json")
        if metrics["passed"] is not True:
            raise ValueError(f"INC task metrics did not pass: {task.task_id}.")
        response_paths.append(str(response_path.relative_to(root)))
        split_counts[task.split] += 1
    expected_splits = {
        "p0_7": {"pilot": 0, "train": 4, "validation": 2, "test": 1},
        "p1_pilot_12": {"pilot": 12, "train": 0, "validation": 0, "test": 0},
        "p1_official_144": {"pilot": 0, "train": 96, "validation": 24, "test": 24},
    }[plan.design]
    if split_counts != expected_splits:
        raise ValueError(
            "INC split counts differ from the reviewed design: "
            f"expected={expected_splits}, received={split_counts}."
        )
    if plan.design == "p1_official_144":
        _validate_official_coverage(tasks, plan)
    shard_manifest = {
        "schema_version": "2.0",
        "system_path": "system.h5",
        "response_paths": response_paths,
        "split_counts": split_counts,
    }
    _write_json_atomic(root / "response_shards.json", shard_manifest)
    report = {
        "schema_version": "2.0",
        "passed": True,
        "design": plan.design,
        "task_count": len(tasks),
        "split_counts": split_counts,
        "status": status,
    }
    _write_json_atomic(root / "data_acceptance.json", report)
    if plan.design == "p1_pilot_12":
        report["pilot_analysis"] = analyze_elastic_inc_pilot(manifest_path)
    return report


def analyze_elastic_inc_pilot(manifest_path: str | Path) -> dict[str, Any]:
    """Compute compact two-dimensional response coverage for twelve trial paths."""
    manifest = _read_json(Path(manifest_path))
    plan = ElasticINCPlan.model_validate(manifest["plan"])
    if plan.design != "p1_pilot_12":
        raise ValueError("Pilot analysis requires the P1 twelve-path design.")
    root = Path(manifest["output_root"])
    system = load_elastic_inc_system(root / "system.h5")
    right_nodes = np.flatnonzero(np.isclose(system.nodes[:, 0], np.max(system.nodes[:, 0])))
    top_node = right_nodes[np.argmax(system.nodes[right_nodes, 1])]
    bottom_node = right_nodes[np.argmin(system.nodes[right_nodes, 1])]
    full_to_free = np.full(system.dof_map.size, -1, dtype=np.int64)
    full_to_free[system.free_dofs] = np.arange(system.free_dofs.size)
    right_x = full_to_free[system.dof_map[right_nodes, 0]]
    right_y = full_to_free[system.dof_map[right_nodes, 1]]
    top_x = int(full_to_free[system.dof_map[top_node, 0]])
    bottom_x = int(full_to_free[system.dof_map[bottom_node, 0]])
    height = float(np.ptp(system.nodes[:, 1]))
    path_reports = []
    feature_rows = []
    for task_id in sorted(manifest["tasks"]):
        response_path = root / "tasks" / task_id / "response.h5"
        metrics = _read_json(root / "tasks" / task_id / "metrics.json")
        with h5py.File(response_path, "r") as handle:
            displacement = handle["state/displacement"]
            sample_ids = np.linspace(
                0,
                displacement.shape[0] - 1,
                num=min(128, displacement.shape[0]),
                dtype=np.int64,
            )
            sample = displacement[sample_ids]
        axial = np.mean(sample[:, right_x], axis=1)
        transverse = np.mean(sample[:, right_y], axis=1)
        rotation = (sample[:, top_x] - sample[:, bottom_x]) / height
        shear = _mean_absolute_shear(sample, system, full_to_free)
        feature = sample.reshape(-1)
        feature_norm = np.linalg.norm(feature)
        if feature_norm == 0.0:
            raise ValueError(f"Pilot response feature is zero: {task_id}.")
        feature_rows.append(feature / feature_norm)
        path_reports.append(
            {
                "task_id": task_id,
                "maximum_axial_tip_displacement": float(np.max(np.abs(axial))),
                "maximum_transverse_tip_displacement": float(np.max(np.abs(transverse))),
                "maximum_tip_rotation": float(np.max(np.abs(rotation))),
                "maximum_mean_absolute_shear_strain": float(np.max(shear)),
                "wall_seconds": metrics["wall_seconds"],
                "peak_memory_mb": metrics["peak_memory_mb"],
                "response_file_mb": response_path.stat().st_size / 1024.0**2,
            }
        )
    singular_values = np.linalg.svd(np.stack(feature_rows), compute_uv=False)
    effective_rank = int(np.sum(singular_values / singular_values[0] >= 1.0e-6))
    report = {
        "schema_version": "2.0",
        "automatic_checks_passed": effective_rank >= 4,
        "review_status": "required",
        "effective_rank": effective_rank,
        "singular_values": singular_values.tolist(),
        "maximum_path_wall_seconds": max(item["wall_seconds"] for item in path_reports),
        "maximum_path_memory_mb": max(item["peak_memory_mb"] for item in path_reports),
        "total_response_size_mb": sum(item["response_file_mb"] for item in path_reports),
        "paths": path_reports,
    }
    _write_json_atomic(root / "pilot_analysis.json", report)
    return report


def _run_one_task(arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated path and preserve its failure diagnostics."""
    plan = ElasticINCPlan.model_validate(arguments["plan"])
    task = ElasticINCTask.model_validate(arguments["task"])
    root = Path(arguments["output_root"])
    task_root = root / "tasks" / task.task_id
    status_path = task_root / "status.json"
    response_path = task_root / "response.h5"
    if status_path.exists():
        status = _read_json(status_path)
        if status["state"] == "succeeded" and response_path.is_file():
            return {"task_id": task.task_id, "passed": True, "skipped": True}
        return {
            "task_id": task.task_id,
            "passed": False,
            "reason": f"Existing task state requires diagnosis: {status['state']}.",
        }
    task_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    _write_json_atomic(
        status_path,
        {
            "task_id": task.task_id,
            "state": "running",
            "started_utc": _utc_now(),
        },
    )
    try:
        metrics = generate_elastic_inc_path(plan, task, root)
    except Exception as error:
        diagnostics = error.diagnostics if isinstance(error, ElasticINCAcceptanceError) else {}
        failure = {
            "task_id": task.task_id,
            "state": "failed",
            "reason": str(error),
            "error_type": type(error).__name__,
            "diagnostics": diagnostics,
            "traceback": traceback.format_exc(),
            "wall_seconds": perf_counter() - started,
            "finished_utc": _utc_now(),
        }
        _write_json_atomic(task_root / "failure.json", failure)
        _write_json_atomic(status_path, failure)
        return {"task_id": task.task_id, "passed": False, "reason": str(error)}
    status = {
        "task_id": task.task_id,
        "state": "succeeded",
        "wall_seconds": metrics["wall_seconds"],
        "peak_memory_mb": metrics["peak_memory_mb"],
        "finished_utc": _utc_now(),
    }
    _write_json_atomic(status_path, status)
    return {"task_id": task.task_id, "passed": True, "skipped": False}


def _require_safe_execution_host(
    manifest: dict[str, Any],
    allow_local_p1: bool,
) -> None:
    """Restrict workstation execution to an explicitly authorized serial run."""
    if "SLURM_JOB_ID" in os.environ:
        return
    plan = ElasticINCPlan.model_validate(manifest["plan"])
    serial_p0 = plan.design == "p0_7" and int(manifest["parallel_paths"]) == 1
    serial_p1_pilot = (
        plan.design == "p1_pilot_12" and allow_local_p1 and int(manifest["parallel_paths"]) == 1
    )
    if not serial_p0 and not serial_p1_pilot:
        raise RuntimeError(
            "Workstation execution is restricted to serial P0 data generation or "
            "an explicitly authorized serial P1 pilot; official P1 paths must run "
            "inside a Slurm allocation."
        )
    memory_values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", maxsplit=1)
        memory_values[name] = int(value.strip().split()[0]) * 1024
    required_available = (12 if serial_p1_pilot else 8) * 1024**3
    if memory_values["MemAvailable"] < required_available:
        raise MemoryError(
            "Serial workstation INC generation does not have enough available memory "
            "for the configured workload."
        )


def _validate_response_file(
    path: Path,
    plan: ElasticINCPlan,
    task: ElasticINCTask,
) -> None:
    """Validate one completed response shard in bounded chunks."""
    with h5py.File(path, "r") as handle:
        if handle.attrs["schema_version"] != "2.0" or handle.attrs["status"] != "complete":
            raise ValueError(f"INC response is incomplete: {path}.")
        if handle.attrs["task_id"] != task.task_id or handle.attrs["split"] != task.split:
            raise ValueError(f"INC response metadata differs from its task: {path}.")
        expected = {
            "time": (plan.num_steps + 1,),
            "state/displacement": (plan.num_steps + 1, 80 if plan.design == "p0_7" else 400),
            "state/velocity": (plan.num_steps + 1, 80 if plan.design == "p0_7" else 400),
            "target/residual_force_start": (plan.num_steps, 80 if plan.design == "p0_7" else 400),
            "target/residual_force_end": (plan.num_steps, 80 if plan.design == "p0_7" else 400),
            "reference/mechanical_energy": (plan.num_steps + 1,),
        }
        for name, shape in expected.items():
            dataset = handle[name]
            if dataset.shape != shape:
                raise ValueError(
                    f"INC response field has wrong shape: field={name}, "
                    f"expected={shape}, received={dataset.shape}."
                )
            for start in range(0, dataset.shape[0], plan.chunk_steps):
                values = dataset[start : start + plan.chunk_steps]
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError(
                        f"INC response contains non-finite values: field={name}, start={start}."
                    )
        time = handle["time"][...]
        expected_time = np.arange(plan.num_steps + 1) * plan.time_step
        if not np.array_equal(time, expected_time):
            raise ValueError(f"INC response time grid differs from the plan: {path}.")
        controls = handle["control_parameters"][...]
        if not np.array_equal(controls, build_control_parameters(plan, task)):
            raise ValueError(f"INC response controls differ from the task: {path}.")


def _validate_official_coverage(
    tasks: tuple[ElasticINCTask, ...],
    plan: ElasticINCPlan,
) -> None:
    """Require balanced amplitudes and full category coverage in every split."""
    for amplitude in plan.pressure_amplitudes:
        count = sum(task.pressure_amplitude == amplitude for task in tasks)
        if count != 48:
            raise ValueError(
                "P1 official amplitude count differs from the reviewed design: "
                f"amplitude={amplitude}, count={count}."
            )
    for split in ("train", "validation", "test"):
        selected = tuple(task for task in tasks if task.split == split)
        spatial_count = len({task.spatial_coefficients for task in selected})
        pulse_count = len({task.pulse_kind for task in selected})
        if spatial_count != 8 or pulse_count != 3:
            raise ValueError(
                "P1 official split lacks spatial or temporal coverage: "
                f"split={split}, spatial={spatial_count}, pulse={pulse_count}."
            )


def _mean_absolute_shear(
    free_displacement: np.ndarray,
    system,
    full_to_free: np.ndarray,
) -> np.ndarray:
    """Compute a center-point engineering-shear indicator."""
    values = []
    for connectivity in system.elements:
        coordinates = system.nodes[connectivity]
        dofs = system.dof_map[connectivity]
        indices_x = full_to_free[dofs[:, 0]]
        indices_y = full_to_free[dofs[:, 1]]
        full_x = np.zeros((free_displacement.shape[0], connectivity.size))
        full_y = np.zeros_like(full_x)
        free_x = indices_x >= 0
        free_y = indices_y >= 0
        full_x[:, free_x] = free_displacement[:, indices_x[free_x]]
        full_y[:, free_y] = free_displacement[:, indices_y[free_y]]
        x_mid = 0.5 * (np.min(coordinates[:, 0]) + np.max(coordinates[:, 0]))
        y_mid = 0.5 * (np.min(coordinates[:, 1]) + np.max(coordinates[:, 1]))
        natural = np.column_stack(
            (
                2.0 * (coordinates[:, 0] - x_mid) / np.ptp(coordinates[:, 0]),
                2.0 * (coordinates[:, 1] - y_mid) / np.ptp(coordinates[:, 1]),
            )
        )
        dshape_dx = natural[:, 0] / (2.0 * np.ptp(coordinates[:, 0]))
        dshape_dy = natural[:, 1] / (2.0 * np.ptp(coordinates[:, 1]))
        values.append(full_x @ dshape_dy + full_y @ dshape_dx)
    return np.mean(np.abs(np.stack(values, axis=1)), axis=1)


def _utc_now() -> str:
    """Return one timezone-explicit UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
