"""Repeated FEM and INC timing for the elastic multimode delivery cases.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from concrete_impact.experiments.inc_elastic_generation import ElasticINCSystem
from concrete_impact.nn.config_inc_elastic import INCElasticMultimodeConfig
from concrete_impact.nn.datasets.inc_elastic_multimode import (
    INCElasticNormalization,
    build_inc_elastic_features,
)
from concrete_impact.nn.evaluation.inc_elastic_multimode import predict_inc_elastic_residuals
from concrete_impact.nn.models.inc_elastic_multimode import INCElasticMultimodeMLP

DELIVERY_CASE_LABELS = ("工况一", "工况二", "工况三")


def benchmark_inc_elastic_delivery(
    config: INCElasticMultimodeConfig,
    model: INCElasticMultimodeMLP,
    system: ElasticINCSystem,
    normalization: INCElasticNormalization,
    angular_frequencies: NDArray[np.float64],
    final_time: float,
    output: Path,
) -> dict[str, Any]:
    """Benchmark fine64 FEM, coarse FEM, CPU INC, and CUDA INC."""
    output.mkdir(parents=True, exist_ok=False)
    cold_start_seconds = _measure_cuda_cold_start(config, system)
    records = []
    coarse_step_count = int(round(final_time / system.time_step))
    cpu_model = copy.deepcopy(model).to(device="cpu", dtype=torch.float32)
    cpu_model.eval()
    model.eval()
    with threadpool_limits(limits=1):
        for path_id in config.data.test_paths:
            parameters = _load_path_parameters(config, path_id)
            fine_function = partial(
                _integrate_final_state,
                system,
                parameters["pressure_amplitude"],
                parameters["pulse_duration"],
                parameters["spatial_coefficients"],
                coarse_step_count,
                config.benchmark.fine_ratio,
                None,
                None,
            )
            coarse_function = partial(
                _integrate_final_state,
                system,
                parameters["pressure_amplitude"],
                parameters["pulse_duration"],
                parameters["spatial_coefficients"],
                coarse_step_count,
                1,
                None,
                None,
            )
            cpu_inc_function = partial(
                _run_timed_inc,
                config,
                cpu_model,
                system,
                normalization,
                angular_frequencies,
                final_time,
                parameters,
            )
            cuda_inc_function = partial(
                _run_timed_inc,
                config,
                model,
                system,
                normalization,
                angular_frequencies,
                final_time,
                parameters,
            )
            fine_state = fine_function()
            fine_reference_error = _final_state_error(
                fine_state,
                (
                    parameters["reference_displacement_final"],
                    parameters["reference_velocity_final"],
                ),
                system.mass,
            )
            if fine_reference_error > 1.0e-10:
                raise ValueError(
                    "Timed fine64 integration differs from the stored reference: "
                    f"path={path_id}, error={fine_reference_error:.12e}."
                )
            cpu_state = cpu_inc_function()
            cuda_state = cuda_inc_function()
            cpu_cuda_difference = _final_state_error(cuda_state, cpu_state, system.mass)
            if cpu_cuda_difference > 1.0e-5:
                raise ValueError(
                    "CPU and CUDA INC final states differ beyond tolerance: "
                    f"path={path_id}, error={cpu_cuda_difference:.12e}."
                )
            fine_times = _measure(
                fine_function,
                config.benchmark.fine_warmup - 1,
                config.benchmark.fine_repeats,
                synchronize_cuda=False,
            )
            coarse_times = _measure(
                coarse_function,
                config.benchmark.coarse_warmup,
                config.benchmark.coarse_repeats,
                synchronize_cuda=False,
            )
            cpu_times = _measure(
                cpu_inc_function,
                config.benchmark.inc_warmup - 1,
                config.benchmark.inc_repeats,
                synchronize_cuda=False,
            )
            cuda_times = _measure(
                cuda_inc_function,
                config.benchmark.inc_warmup - 1,
                config.benchmark.inc_repeats,
                synchronize_cuda=True,
            )
            fine_statistics = _timing_statistics(fine_times)
            coarse_statistics = _timing_statistics(coarse_times)
            cpu_statistics = _timing_statistics(cpu_times)
            cuda_statistics = _timing_statistics(cuda_times)
            fine_median = fine_statistics["median_seconds"]
            coarse_median = coarse_statistics["median_seconds"]
            cpu_median = cpu_statistics["median_seconds"]
            cuda_median = cuda_statistics["median_seconds"]
            records.append(
                {
                    "path_id": path_id,
                    "fine64_fem": fine_statistics,
                    "coarse_fem": coarse_statistics,
                    "inc_cpu": cpu_statistics,
                    "inc_cuda": cuda_statistics,
                    "speedup_cpu": fine_median / cpu_median,
                    "speedup_cuda": fine_median / cuda_median,
                    "cuda_overhead_ratio_to_coarse": cuda_median / coarse_median,
                    "fine64_reference_final_state_error": fine_reference_error,
                    "cpu_cuda_final_state_relative_difference": cpu_cuda_difference,
                }
            )
    summary = {
        "schema_version": "1.0",
        "timing_scope": {
            "steady_state_excludes": ["HDF5_input", "model_loading"],
            "inc_includes": [
                "feature_construction",
                "neural_inference",
                "device_transfer",
                "float64_coarse_integration",
            ],
            "cpu_threads": 1,
        },
        "cuda_cold_start_seconds": cold_start_seconds,
        "records": records,
        "minimum_cpu_speedup": min(record["speedup_cpu"] for record in records),
        "minimum_cuda_speedup": min(record["speedup_cuda"] for record in records),
    }
    (output / "benchmark_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_speedups(records, output / "speedup_comparison.png")
    return summary


def _measure_cuda_cold_start(
    config: INCElasticMultimodeConfig,
    system: ElasticINCSystem,
) -> float:
    """Measure model loading plus the first complete CUDA-assisted INC solve."""
    from concrete_impact.nn.training.inc_elastic_multimode import load_inc_elastic_artifact

    torch.cuda.synchronize()
    started = perf_counter()
    model, _, normalization, frequencies, final_time = load_inc_elastic_artifact(
        config,
        checkpoint_name="response_model.pt",
    )
    parameters = _load_path_parameters(config, config.data.test_paths[0])
    _run_timed_inc(
        config,
        model,
        system,
        normalization,
        frequencies,
        final_time,
        parameters,
    )
    torch.cuda.synchronize()
    return perf_counter() - started


def _run_timed_inc(
    config: INCElasticMultimodeConfig,
    model: INCElasticMultimodeMLP,
    system: ElasticINCSystem,
    normalization: INCElasticNormalization,
    angular_frequencies: NDArray[np.float64],
    final_time: float,
    parameters: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Construct features, infer residuals, and integrate one coarse path."""
    step_count = int(round(final_time / system.time_step))
    time = np.arange(step_count + 1, dtype=np.float64) * system.time_step
    features = build_inc_elastic_features(
        time[:-1],
        system.time_step,
        parameters["pulse_duration"],
        parameters["spatial_coefficients"],
        angular_frequencies,
        normalization,
        final_time,
        config.features.include_duration_modal_phase,
    )
    start, end = predict_inc_elastic_residuals(
        model,
        features,
        normalization,
        parameters["pressure_amplitude"],
        config.execution.prediction_batch_size,
    )
    return _integrate_final_state(
        system,
        parameters["pressure_amplitude"],
        parameters["pulse_duration"],
        parameters["spatial_coefficients"],
        step_count,
        1,
        start,
        end,
    )


def _integrate_final_state(
    system: ElasticINCSystem,
    pressure_amplitude: float,
    pulse_duration: float,
    spatial_coefficients: NDArray[np.float64],
    coarse_step_count: int,
    time_refinement: int,
    residual_start: NDArray[np.float64] | None,
    residual_end: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate to the final time without retaining trajectory histories."""
    step_count = coarse_step_count * time_refinement
    time_step = system.time_step / time_refinement
    unit_force = spatial_coefficients @ system.load_basis
    displacement = np.zeros(system.free_dofs.size, dtype=np.float64)
    velocity = np.zeros_like(displacement)
    for step in range(step_count):
        time_start = step * time_step
        time_end = (step + 1) * time_step
        if time_refinement == 1 and residual_start is not None and residual_end is not None:
            start_correction = residual_start[step]
            end_correction = residual_end[step]
        else:
            start_correction = 0.0
            end_correction = 0.0
        force_start = (
            pressure_amplitude * _half_sine_scalar(time_start, pulse_duration) * unit_force
        )
        total_start = force_start - system.stiffness @ displacement - start_correction
        acceleration_start = total_start / system.mass
        next_displacement = (
            displacement + time_step * velocity + 0.5 * time_step**2 * acceleration_start
        )
        force_end = pressure_amplitude * _half_sine_scalar(time_end, pulse_duration) * unit_force
        total_end = force_end - system.stiffness @ next_displacement - end_correction
        acceleration_end = total_end / system.mass
        next_velocity = velocity + 0.5 * time_step * (acceleration_start + acceleration_end)
        displacement = next_displacement
        velocity = next_velocity
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("Elastic INC timing integration produced non-finite states.")
    return displacement, velocity


def _load_path_parameters(
    config: INCElasticMultimodeConfig, path_id: str
) -> dict[str, Any]:
    """Load one test path's scalar loading parameters without trajectory data."""
    path = config.data.root / "tasks" / path_id / "response.h5"
    with h5py.File(path, "r") as handle:
        return {
            "pressure_amplitude": float(handle.attrs["pressure_amplitude"]),
            "pulse_duration": float(handle.attrs["pulse_duration"]),
            "spatial_coefficients": handle["spatial_coefficients"][...],
            "reference_displacement_final": handle["state/displacement"][-1],
            "reference_velocity_final": handle["state/velocity"][-1],
        }


def _final_state_error(
    values: tuple[NDArray[np.float64], NDArray[np.float64]],
    reference: tuple[NDArray[np.float64], NDArray[np.float64]],
    mass: NDArray[np.float64],
) -> float:
    """Return the combined mass-weighted final displacement and velocity error."""
    numerator = np.sum(mass * (values[0] - reference[0]) ** 2)
    numerator += np.sum(mass * (values[1] - reference[1]) ** 2)
    denominator = np.sum(mass * reference[0] ** 2)
    denominator += np.sum(mass * reference[1] ** 2)
    return float(np.sqrt(numerator / denominator))


def _measure(
    function: Callable[[], Any],
    warmup: int,
    repeats: int,
    synchronize_cuda: bool,
) -> list[float]:
    """Measure repeated wall time after a fixed number of warmup calls."""
    for _ in range(warmup):
        function()
    measurements = []
    for _ in range(repeats):
        if synchronize_cuda:
            torch.cuda.synchronize()
        started = perf_counter()
        function()
        if synchronize_cuda:
            torch.cuda.synchronize()
        measurements.append(perf_counter() - started)
    return measurements


def _timing_statistics(values: list[float]) -> dict[str, Any]:
    """Return raw repeats, median, and median absolute deviation."""
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "repeats_seconds": values,
        "median_seconds": median,
        "mad_seconds": float(np.median(np.abs(array - median))),
    }


def _half_sine_scalar(time: float, duration: float) -> float:
    """Evaluate one compactly supported half-sine value."""
    normalized = time / duration
    if normalized < 0.0 or normalized > 1.0:
        return 0.0
    return float(np.sin(np.pi * normalized))


def _plot_speedups(records: list[dict[str, Any]], output_path: Path) -> None:
    """Plot CPU and CUDA INC speedups against the fine64 FEM reference."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = list(DELIVERY_CASE_LABELS[: len(records)])
    x_values = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    axis.bar(
        x_values - width / 2,
        [record["speedup_cpu"] for record in records],
        width,
        label="CPU上的INC",
    )
    axis.bar(
        x_values + width / 2,
        [record["speedup_cuda"] for record in records],
        width,
        label="GPU上的INC",
    )
    axis.set_xticks(x_values, labels)
    axis.set_ylabel("相对于细时间步有限元的加速比")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
