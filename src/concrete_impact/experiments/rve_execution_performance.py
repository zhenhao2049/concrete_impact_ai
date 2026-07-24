"""Repeated serial and process-pool FE2 material performance experiment.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import numpy as np

from concrete_impact.benchmarks.rve_multiphase_dynamics import _build_heterogeneous_model
from concrete_impact.core.config import load_yaml_config
from fem.materials import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
)
from fem.rve import RVEExecutionSettings, RVEHomogenizedMaterial


def run_rve_execution_performance(config_path: str | Path) -> dict[str, object]:
    """Compare persistent serial and process FE2 batches after correctness checks."""
    config = load_yaml_config(config_path)
    reference = load_yaml_config(config["rve"]["reference_config"])
    model = _build_heterogeneous_model(config, reference)
    update = config["analysis"]["material_update"]
    update_settings = MaterialUpdateSettings(
        int(update["max_iterations"]),
        float(update["yield_relative_tolerance"]),
        float(update["residual_absolute_tolerance"]),
        float(update["residual_relative_tolerance"]),
    )
    rows = []
    profiles: dict[tuple[int, str], dict[str, float]] = {}
    for batch_size in config["performance"]["batch_sizes"]:
        materials = {
            backend: RVEHomogenizedMaterial(
                model,
                f"performance_{backend}",
                RVEExecutionSettings(
                    cast(Literal["serial", "process_pool"], backend),
                    (
                        1
                        if backend == "serial"
                        else int(config["performance"]["workers"])
                    ),
                ),
            )
            for backend in ("serial", "process_pool")
        }
        try:
            _run_batch_path(materials["serial"], int(batch_size), config, update_settings)
            _run_batch_path(materials["process_pool"], int(batch_size), config, update_settings)
            last_responses = {}
            repetitions = int(config["performance"]["repetitions"])
            for repetition in range(1, repetitions + 1):
                order = (
                    ("serial", "process_pool")
                    if repetition % 2 == 1
                    else ("process_pool", "serial")
                )
                for backend in order:
                    start = perf_counter()
                    response = _run_batch_path(
                        materials[backend], int(batch_size), config, update_settings
                    )
                    elapsed = perf_counter() - start
                    rows.append(
                        {
                            "batch_size": int(batch_size),
                            "repetition": repetition,
                            "backend": backend,
                            "elapsed_time": elapsed,
                        }
                    )
                    last_responses[backend] = response
            _assert_backend_equivalence(last_responses, config["verification"])
            for backend in ("serial", "process_pool"):
                profiler = cProfile.Profile()
                profiler.enable()
                _run_batch_path(materials[backend], int(batch_size), config, update_settings)
                profiler.disable()
                profiles[(int(batch_size), backend)] = _summarize_profile(profiler)
        finally:
            for material in materials.values():
                material.close()
    metrics = _summarize(rows, profiles, config)
    root = Path(config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    _write_rows(root / "rve_execution_performance.csv", rows)
    (root / "rve_execution_performance.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def _run_batch_path(material, batch_size: int, config: dict, update_settings):
    """Run one fixed accepted strain path from an initialized batch state."""
    state = material.initialize_state(batch_size)
    previous = np.zeros((batch_size, 6), dtype=np.float64)
    response = None
    for strain_row in config["performance"]["macro_strain_path"]:
        strain = np.repeat(np.asarray(strain_row, dtype=np.float64)[None, :], batch_size, axis=0)
        response = material.update(
            MaterialPointRequest(
                strains=strain,
                strain_rates=(strain - previous) / float(config["performance"]["time_step"]),
                time_step=float(config["performance"]["time_step"]),
                kinematics="three_dimensional",
                update_settings=update_settings,
            ),
            state,
            MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
        )
        state = response.state
        previous = strain
    if response is None:
        raise ValueError("RVE performance path must contain at least one strain state.")
    return response


def _assert_backend_equivalence(responses: dict, verification: dict) -> None:
    """Require identical numerical responses before reporting parallel timings."""
    serial = responses["serial"]
    process = responses["process_pool"]
    tolerance = float(verification["backend_equivalence_tolerance"])
    arrays = (
        ("stress", serial.stresses, process.stresses),
        ("tangent", serial.tangents, process.tangents),
        ("dissipation", serial.dissipation, process.dissipation),
    )
    for name, left, right in arrays:
        if left is None or right is None:
            raise ValueError(f"RVE performance comparison requires {name} output.")
        error = float(np.max(np.abs(left - right)))
        if error > tolerance:
            raise ValueError(f"RVE backend {name} mismatch: error={error}, tolerance={tolerance}.")
    for name in serial.state.variables:
        error = float(np.max(np.abs(serial.state.variables[name] - process.state.variables[name])))
        if error > tolerance:
            raise ValueError(
                f"RVE backend state mismatch: field={name}, error={error}, tolerance={tolerance}."
            )


def _summarize(
    rows: list[dict],
    profiles: dict[tuple[int, str], dict[str, float]],
    config: dict,
) -> dict[str, object]:
    """Compute median, MAD, and conservative speed ratios per batch size."""
    batches = {}
    for batch_size in config["performance"]["batch_sizes"]:
        times = {
            backend: np.asarray(
                [
                    row["elapsed_time"]
                    for row in rows
                    if row["batch_size"] == int(batch_size) and row["backend"] == backend
                ]
            )
            for backend in ("serial", "process_pool")
        }
        medians = {backend: float(np.median(values)) for backend, values in times.items()}
        mads = {
            backend: float(np.median(np.abs(values - np.median(values))))
            for backend, values in times.items()
        }
        speedup = medians["serial"] / medians["process_pool"]
        conservative = (medians["serial"] - mads["serial"]) / (
            medians["process_pool"] + mads["process_pool"]
        )
        batches[str(batch_size)] = {
            "serial_median_time": medians["serial"],
            "process_pool_median_time": medians["process_pool"],
            "serial_mad_time": mads["serial"],
            "process_pool_mad_time": mads["process_pool"],
            "process_pool_speedup": speedup,
            "conservative_speedup": conservative,
            "process_pool_stably_faster": conservative > 1.0,
            "engineering_speed_target_met": speedup >= 1.2,
        }
        for backend in ("serial", "process_pool"):
            for name in (
                "profile_micro_material_time",
                "profile_micro_assembly_time",
                "profile_micro_linear_time",
                "profile_rve_solver_time",
                "profile_parent_communication_time",
            ):
                batches[str(batch_size)][f"{backend}_{name}"] = float(
                    profiles[(int(batch_size), backend)][name]
                )
    return {
        "passed": True,
        "warmup_repetitions": 1,
        "timed_repetitions": int(config["performance"]["repetitions"]),
        "workers": int(config["performance"]["workers"]),
        "batches": batches,
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }


def _summarize_profile(profiler: cProfile.Profile) -> dict[str, float]:
    """Aggregate exclusive parent-process CPU time by FE2 subsystem."""
    categories = {
        "profile_micro_material_time": ("materials/plasticity.py", "materials/phased.py"),
        "profile_micro_assembly_time": ("assembly/nonlinear_solid.py",),
        "profile_micro_linear_time": ("solvers/linear.py",),
        "profile_rve_solver_time": ("rve/solver.py",),
        "profile_parent_communication_time": (
            "multiprocessing/reduction.py",
            "multiprocessing/queues.py",
            "concurrent/futures/process.py",
            "pickle.py",
        ),
    }
    totals = {name: 0.0 for name in categories}
    stats = pstats.Stats(profiler)
    raw_stats = stats.stats  # type: ignore[attr-defined]
    for (filename, _line, _function), values in raw_stats.items():
        total_time = float(values[2])
        for category, fragments in categories.items():
            if any(fragment in filename for fragment in fragments):
                totals[category] += total_time
                break
    return totals


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Write raw repeated FE2 timing observations."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run the configured FE2 backend performance experiment."""
    parser = argparse.ArgumentParser(description="Profile serial and process FE2 batches.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    run_rve_execution_performance(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
