"""Complete-path evaluation and plotting for the elastic multimode INC.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import correlate, correlation_lags

from concrete_impact.experiments.inc_elastic_generation import ElasticINCSystem
from concrete_impact.nn.config_inc_elastic import INCElasticMultimodeConfig
from concrete_impact.nn.datasets.inc_elastic_multimode import (
    INCElasticNormalization,
    build_inc_elastic_features,
)
from concrete_impact.nn.models.inc_elastic_multimode import INCElasticMultimodeMLP

KINEMATIC_DELIVERY_ERROR_RATIO_LIMIT = 1.0
DELIVERY_CASE_LABELS = ("工况一", "工况二", "工况三")


def predict_inc_elastic_residuals(
    model: INCElasticMultimodeMLP,
    features: NDArray[np.float32],
    normalization: INCElasticNormalization,
    pressure_amplitude: float,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict physical two-stage residual forces in bounded batches."""
    device = next(model.parameters()).device
    starts = []
    ends = []
    model.eval()
    with torch.no_grad():
        for first in range(0, features.shape[0], batch_size):
            batch = torch.from_numpy(features[first : first + batch_size]).to(device=device)
            start, end = model(batch)
            starts.append(start.detach().cpu())
            ends.append(end.detach().cpu())
    normalized_start = torch.cat(starts).numpy().astype(np.float64)
    normalized_end = torch.cat(ends).numpy().astype(np.float64)
    residual_start = (
        normalized_start
        * normalization.residual_start_scale[None, :]
        * pressure_amplitude
    )
    residual_end = (
        normalized_end * normalization.residual_end_scale[None, :] * pressure_amplitude
    )
    if not np.all(np.isfinite(residual_start)) or not np.all(np.isfinite(residual_end)):
        raise FloatingPointError("Elastic INC prediction contains non-finite residual forces.")
    return residual_start, residual_end


def evaluate_inc_elastic_paths(
    config: INCElasticMultimodeConfig,
    model: INCElasticMultimodeMLP,
    system: ElasticINCSystem,
    normalization: INCElasticNormalization,
    angular_frequencies: NDArray[np.float64],
    final_time: float,
    path_ids: tuple[str, ...],
    keep_trajectories: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, NDArray[np.float64]]]]:
    """Evaluate baseline, oracle, and trained INC on complete independent paths."""
    records = []
    saved_trajectories: dict[str, dict[str, NDArray[np.float64]]] = {}
    for path_id in path_ids:
        response_path = config.data.root / "tasks" / path_id / "response.h5"
        with h5py.File(response_path, "r") as handle:
            if handle.attrs["schema_version"] != "2.0" or handle.attrs["status"] != "complete":
                raise ValueError(f"Elastic INC response is not complete schema 2.0: {path_id}.")
            expected_split = (
                "validation" if path_id in config.data.validation_paths else "test"
            )
            if config.data.source_split_policy == "training_pool_resplit":
                expected_split = "train"
            if str(handle.attrs["split"]) != expected_split:
                raise ValueError(f"Elastic INC response has the wrong split: {path_id}.")
            time = handle["time"][...]
            reference_u = handle["state/displacement"][...]
            reference_v = handle["state/velocity"][...]
            reference_energy = handle["reference/mechanical_energy"][...]
            target_start = handle["target/residual_force_start"][...]
            target_end = handle["target/residual_force_end"][...]
            spatial = handle["spatial_coefficients"][...]
            amplitude = float(handle.attrs["pressure_amplitude"])
            duration = float(handle.attrs["pulse_duration"])
            pulse_kind = str(handle.attrs["pulse_kind"])
        if pulse_kind != "half_sine":
            raise ValueError(f"Elastic multimode INC received a non-half-sine path: {path_id}.")

        features = build_inc_elastic_features(
            time[:-1],
            system.time_step,
            duration,
            spatial,
            angular_frequencies,
            normalization,
            final_time,
            config.features.include_duration_modal_phase,
        )
        predicted_start, predicted_end = predict_inc_elastic_residuals(
            model,
            features,
            normalization,
            amplitude,
            config.execution.prediction_batch_size,
        )
        zero_start = np.zeros_like(predicted_start)
        zero_end = np.zeros_like(predicted_end)
        baseline = integrate_inc_elastic_path(
            system, time, amplitude, duration, spatial, zero_start, zero_end
        )
        oracle = integrate_inc_elastic_path(
            system, time, amplitude, duration, spatial, target_start, target_end
        )
        corrected = integrate_inc_elastic_path(
            system,
            time,
            amplitude,
            duration,
            spatial,
            predicted_start,
            predicted_end,
        )
        record, channels = _compute_path_metrics(
            config,
            path_id,
            system,
            time,
            reference_u,
            reference_v,
            reference_energy,
            target_start,
            target_end,
            predicted_start,
            predicted_end,
            baseline,
            oracle,
            corrected,
        )
        records.append(record)
        if keep_trajectories:
            saved_trajectories[path_id] = {"time": time, **channels}
    return records, saved_trajectories


def integrate_inc_elastic_path(
    system: ElasticINCSystem,
    time: NDArray[np.float64],
    pressure_amplitude: float,
    pulse_duration: float,
    spatial_coefficients: NDArray[np.float64],
    residual_start: NDArray[np.float64],
    residual_end: NDArray[np.float64],
) -> dict[str, NDArray[np.float64] | float]:
    """Integrate one coarse path in float64 with supplied stage residuals."""
    step_count = time.size - 1
    free_count = system.free_dofs.size
    if residual_start.shape != (step_count, free_count):
        raise ValueError("Elastic INC start residual shape differs from the coarse path.")
    if residual_end.shape != (step_count, free_count):
        raise ValueError("Elastic INC end residual shape differs from the coarse path.")
    unit_force = np.asarray(spatial_coefficients, dtype=np.float64) @ system.load_basis
    pulse = _half_sine_values(time, pulse_duration)
    load_values = pressure_amplitude * pulse
    displacement = np.zeros((time.size, free_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    energy = np.zeros(time.size, dtype=np.float64)
    impulse_max = 0.0
    time_step = system.time_step

    for step in range(step_count):
        force_start = load_values[step] * unit_force
        force_end = load_values[step + 1] * unit_force
        total_start = (
            force_start - system.stiffness @ displacement[step] - residual_start[step]
        )
        acceleration_start = total_start / system.mass
        displacement[step + 1] = (
            displacement[step]
            + time_step * velocity[step]
            + 0.5 * time_step**2 * acceleration_start
        )
        total_end = (
            force_end
            - system.stiffness @ displacement[step + 1]
            - residual_end[step]
        )
        acceleration_end = total_end / system.mass
        velocity[step + 1] = velocity[step] + 0.5 * time_step * (
            acceleration_start + acceleration_end
        )
        balance = (
            system.mass * (velocity[step + 1] - velocity[step])
            - 0.5 * time_step * (total_start + total_end)
        )
        impulse_max = max(impulse_max, float(np.linalg.norm(balance)))
        energy[step + 1] = _mechanical_energy(
            system, displacement[step + 1], velocity[step + 1]
        )
    if not all(np.all(np.isfinite(values)) for values in (displacement, velocity, energy)):
        raise FloatingPointError("Elastic INC complete path contains non-finite states.")
    return {
        "displacement": displacement,
        "velocity": velocity,
        "energy": energy,
        "impulse_max": impulse_max,
    }


def validation_score(record: dict[str, Any], config: INCElasticMultimodeConfig) -> float:
    """Return a continuous primary-accuracy score for checkpoint selection."""
    limits = config.acceptance
    ratios = [
        record["label_mass_dual_nrmse"] / limits.label_mass_dual_nrmse,
        record["displacement_error_ratio"] / limits.displacement_error_ratio,
        record["velocity_error_ratio"] / limits.velocity_error_ratio,
        record["energy_error_ratio"] / limits.energy_error_ratio,
    ]
    return float(max(ratios))


def kinematic_validation_score(
    record: dict[str, Any],
    config: INCElasticMultimodeConfig,
) -> float:
    """Return the worst normalized displacement or velocity acceptance margin."""
    return float(
        max(
            record["displacement_error_ratio"] / config.acceptance.displacement_error_ratio,
            record["velocity_error_ratio"] / config.acceptance.velocity_error_ratio,
        )
    )


def write_inc_elastic_evaluation(
    config: INCElasticMultimodeConfig,
    records: list[dict[str, Any]],
    trajectories: dict[str, dict[str, NDArray[np.float64]]],
    output: Path,
) -> dict[str, Any]:
    """Write independent test metrics, compact trajectories, and figures."""
    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        record["kinematic_delivery_passed"] = _passes_kinematic_delivery(record)
    summary = {
        "schema_version": "1.0",
        "passed": all(record["kinematic_delivery_passed"] for record in records),
        "delivery_acceptance_passed": all(
            record["kinematic_delivery_passed"] for record in records
        ),
        "full_acceptance_passed": all(record["passed"] for record in records),
        "delivery_acceptance": {
            "criterion": "displacement_and_velocity_strictly_better_than_coarse_baseline",
            "displacement_error_ratio_exclusive_upper_bound": (
                KINEMATIC_DELIVERY_ERROR_RATIO_LIMIT
            ),
            "velocity_error_ratio_exclusive_upper_bound": (
                KINEMATIC_DELIVERY_ERROR_RATIO_LIMIT
            ),
        },
        "path_count": len(records),
        "test_paths": list(config.data.test_paths),
        "worst_displacement_error_ratio": max(
            record["displacement_error_ratio"] for record in records
        ),
        "worst_velocity_error_ratio": max(record["velocity_error_ratio"] for record in records),
        "worst_energy_error_ratio": max(record["energy_error_ratio"] for record in records),
        "worst_label_mass_dual_nrmse": max(
            record["label_mass_dual_nrmse"] for record in records
        ),
    }
    (output / "path_metrics.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path_id, values in trajectories.items():
        np.savez_compressed(
            output / f"{path_id}_channels.npz",
            **values,  # type: ignore[arg-type]
        )
    _plot_response_trajectories(trajectories, output / "response_comparison.png")
    _plot_accuracy_ratios(records, output / "accuracy_ratios.png")
    return summary


def _passes_kinematic_delivery(record: dict[str, Any]) -> bool:
    """Require both global state errors to improve on the coarse-step baseline."""
    ratios = np.asarray(
        [record["displacement_error_ratio"], record["velocity_error_ratio"]],
        dtype=np.float64,
    )
    return bool(
        np.all(np.isfinite(ratios))
        and np.all(ratios < KINEMATIC_DELIVERY_ERROR_RATIO_LIMIT)
    )


def plot_inc_elastic_training_history(metrics_path: Path, output_path: Path) -> None:
    """Plot label, one-step, energy, and full-validation convergence."""
    import matplotlib.pyplot as plt

    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    epochs = np.asarray([record["epoch"] for record in records])
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    fields = (
        ("train_label_loss", "Label loss"),
        ("train_step_loss", "One-step state loss"),
        ("train_energy_loss", "Energy loss"),
    )
    for axis, (field, title) in zip(axes.flat[:3], fields, strict=True):
        axis.semilogy(epochs, [record[field] for record in records], label="train")
        validation_field = field.replace("train_", "validation_")
        axis.semilogy(
            epochs,
            [record[validation_field] for record in records],
            label="validation",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
    full_records = [record for record in records if record["full_validation_score"] is not None]
    axes[1, 1].semilogy(
        [record["epoch"] for record in full_records],
        [record["full_validation_score"] for record in full_records],
        marker="o",
    )
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Worst normalized validation score")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_inc_elastic_kinematic_selection(
    response_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot the complete-validation score used to select the response checkpoint."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    candidates = response_summary["candidate_scores"]
    epochs = [record["epoch"] for record in candidates]
    scores = [record["score"] for record in candidates]
    best_epoch = response_summary["best_epoch"]
    best_score = response_summary["best_kinematic_validation_score"]
    figure, axis = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    axis.semilogy(epochs, scores, marker="o", linewidth=1.4)
    axis.scatter((best_epoch,), (best_score,), color="#d95f02", s=70, zorder=3)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.annotate(
        f"选定第{best_epoch}轮：{best_score:.3f}",
        xy=(best_epoch, best_score),
        xytext=(best_epoch + 10, best_score * 1.8),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
    )
    axis.set_xlabel("训练轮数")
    axis.set_ylabel(r"最不利评分 $\max(R_u/0.70,R_v/0.70)$")
    axis.set_title("完整验证轨迹上的模型选择")
    axis.grid(True, which="both", alpha=0.3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _compute_path_metrics(
    config: INCElasticMultimodeConfig,
    path_id: str,
    system: ElasticINCSystem,
    time: NDArray[np.float64],
    reference_u: NDArray[np.float64],
    reference_v: NDArray[np.float64],
    reference_energy: NDArray[np.float64],
    target_start: NDArray[np.float64],
    target_end: NDArray[np.float64],
    predicted_start: NDArray[np.float64],
    predicted_end: NDArray[np.float64],
    baseline: dict[str, NDArray[np.float64] | float],
    oracle: dict[str, NDArray[np.float64] | float],
    corrected: dict[str, NDArray[np.float64] | float],
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]]]:
    """Compute complete-path accuracy, phase, energy, and balance metrics."""
    baseline_u = np.asarray(baseline["displacement"])
    baseline_v = np.asarray(baseline["velocity"])
    oracle_u = np.asarray(oracle["displacement"])
    oracle_v = np.asarray(oracle["velocity"])
    corrected_u = np.asarray(corrected["displacement"])
    corrected_v = np.asarray(corrected["velocity"])
    baseline_u_error = _relative_mass_error(baseline_u, reference_u, system.mass)
    baseline_v_error = _relative_mass_error(baseline_v, reference_v, system.mass)
    corrected_u_error = _relative_mass_error(corrected_u, reference_u, system.mass)
    corrected_v_error = _relative_mass_error(corrected_v, reference_v, system.mass)
    oracle_error = max(
        _relative_mass_error(oracle_u, reference_u, system.mass),
        _relative_mass_error(oracle_v, reference_v, system.mass),
    )
    label_error = _label_mass_dual_error(
        predicted_start,
        predicted_end,
        target_start,
        target_end,
        system.mass,
    )
    energy_scale = float(np.max(np.abs(reference_energy)))
    baseline_energy_error = float(
        np.max(np.abs(np.asarray(baseline["energy"]) - reference_energy)) / energy_scale
    )
    corrected_energy_error = float(
        np.max(np.abs(np.asarray(corrected["energy"]) - reference_energy)) / energy_scale
    )
    momentum_scale = float(np.max(np.linalg.norm(reference_v * system.mass[None, :], axis=1)))
    impulse_error = float(corrected["impulse_max"]) / momentum_scale
    reference_channels = _response_channels(path_id, system, reference_u)
    baseline_channels = _response_channels(path_id, system, baseline_u)
    corrected_channels = _response_channels(path_id, system, corrected_u)
    channel_metrics = {
        name: _channel_metrics(
            time,
            reference_channels[name],
            baseline_channels[name],
            corrected_channels[name],
            config.acceptance.arrival_threshold_ratio,
        )
        for name in reference_channels
    }
    record: dict[str, Any] = {
        "path_id": path_id,
        "oracle_reconstruction_error": oracle_error,
        "label_mass_dual_nrmse": label_error,
        "baseline_displacement_error": baseline_u_error,
        "inc_displacement_error": corrected_u_error,
        "displacement_error_ratio": corrected_u_error / baseline_u_error,
        "baseline_velocity_error": baseline_v_error,
        "inc_velocity_error": corrected_v_error,
        "velocity_error_ratio": corrected_v_error / baseline_v_error,
        "baseline_energy_error": baseline_energy_error,
        "inc_energy_error": corrected_energy_error,
        "energy_error_ratio": corrected_energy_error / baseline_energy_error,
        "normalized_impulse_balance_error": impulse_error,
        "channels": channel_metrics,
    }
    record["validation_score"] = validation_score(record, config)
    record["passed"] = _passes_acceptance(record, config)
    channel_trajectories = {}
    for name in reference_channels:
        channel_trajectories[f"reference_{name}"] = reference_channels[name]
        channel_trajectories[f"baseline_{name}"] = baseline_channels[name]
        channel_trajectories[f"inc_{name}"] = corrected_channels[name]
    return record, channel_trajectories


def _passes_acceptance(record: dict[str, Any], config: INCElasticMultimodeConfig) -> bool:
    """Apply every locked complete-path acceptance requirement."""
    limits = config.acceptance
    channel_passed = all(
        values["inc_arrival_error"] <= values["baseline_arrival_error"]
        and values["inc_peak_error"] <= values["baseline_peak_error"]
        and values["inc_phase_error_steps"] <= values["baseline_phase_error_steps"]
        for values in record["channels"].values()
    )
    scalar_values = np.asarray(
        [
            record["oracle_reconstruction_error"],
            record["label_mass_dual_nrmse"],
            record["displacement_error_ratio"],
            record["velocity_error_ratio"],
            record["energy_error_ratio"],
            record["normalized_impulse_balance_error"],
        ]
    )
    return bool(
        np.all(np.isfinite(scalar_values))
        and record["oracle_reconstruction_error"] <= limits.oracle_reconstruction_error
        and record["label_mass_dual_nrmse"] <= limits.label_mass_dual_nrmse
        and record["displacement_error_ratio"] <= limits.displacement_error_ratio
        and record["velocity_error_ratio"] <= limits.velocity_error_ratio
        and record["energy_error_ratio"] <= limits.energy_error_ratio
        and record["normalized_impulse_balance_error"]
        <= limits.normalized_impulse_balance_error
        and channel_passed
    )


def _response_channels(
    path_id: str,
    system: ElasticINCSystem,
    displacement: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Extract section-mean axial, transverse, and tip-rotation responses."""
    free_position = np.full(system.dof_map.max() + 1, -1, dtype=np.int64)
    free_position[system.free_dofs] = np.arange(system.free_dofs.size)
    x_values = system.nodes[:, 0]
    right_nodes = np.flatnonzero(np.isclose(x_values, np.max(x_values)))
    axial = displacement[:, free_position[system.dof_map[right_nodes, 0]]].mean(axis=1)
    transverse = displacement[:, free_position[system.dof_map[right_nodes, 1]]].mean(axis=1)
    if path_id.startswith("official_s1_"):
        return {"right_axial_displacement": axial}
    if path_id.startswith("official_s2_"):
        return {"right_transverse_displacement": transverse}
    if any(path_id.startswith(f"official_s{index}_") for index in range(3, 9)):
        unique_x = np.unique(np.round(x_values, decimals=12))
        previous_x = unique_x[-2]
        previous_nodes = np.flatnonzero(np.isclose(x_values, previous_x))
        previous_transverse = displacement[
            :, free_position[system.dof_map[previous_nodes, 1]]
        ].mean(axis=1)
        rotation = (transverse - previous_transverse) / (unique_x[-1] - previous_x)
        return {
            "right_axial_displacement": axial,
            "right_transverse_displacement": transverse,
            "tip_rotation": rotation,
        }
    raise ValueError(f"Elastic INC response channels are undefined for {path_id}.")


def _channel_metrics(
    time: NDArray[np.float64],
    reference: NDArray[np.float64],
    baseline: NDArray[np.float64],
    corrected: NDArray[np.float64],
    threshold_ratio: float,
) -> dict[str, float | int]:
    """Compute wave-arrival, peak-amplitude, and phase errors."""
    threshold = threshold_ratio * float(np.max(np.abs(reference)))
    reference_arrival = _arrival_time(time, reference, threshold)
    baseline_arrival = _arrival_time(time, baseline, threshold)
    corrected_arrival = _arrival_time(time, corrected, threshold)
    return {
        "baseline_arrival_error": abs(baseline_arrival - reference_arrival),
        "inc_arrival_error": abs(corrected_arrival - reference_arrival),
        "baseline_peak_error": _peak_error(baseline, reference),
        "inc_peak_error": _peak_error(corrected, reference),
        "baseline_phase_error_steps": _phase_error_steps(baseline, reference),
        "inc_phase_error_steps": _phase_error_steps(corrected, reference),
    }


def _arrival_time(
    time: NDArray[np.float64], values: NDArray[np.float64], threshold: float
) -> float:
    """Return the first threshold crossing or the final observation time."""
    indices = np.flatnonzero(np.abs(values) >= threshold)
    return float(time[indices[0]]) if indices.size else float(time[-1])


def _peak_error(values: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Return relative absolute-peak error."""
    reference_peak = float(np.max(np.abs(reference)))
    return abs(float(np.max(np.abs(values))) - reference_peak) / reference_peak


def _phase_error_steps(values: NDArray[np.float64], reference: NDArray[np.float64]) -> int:
    """Return the absolute FFT-correlation lag in coarse steps."""
    centered_values = values - np.mean(values)
    centered_reference = reference - np.mean(reference)
    correlation = correlate(centered_values, centered_reference, mode="full", method="fft")
    lags = correlation_lags(values.size, reference.size, mode="full")
    return int(abs(lags[int(np.argmax(correlation))]))


def _relative_mass_error(
    values: NDArray[np.float64],
    reference: NDArray[np.float64],
    mass: NDArray[np.float64],
) -> float:
    """Return the mass-weighted relative trajectory error."""
    numerator = np.sum((values - reference) ** 2 * mass[None, :])
    denominator = np.sum(reference**2 * mass[None, :])
    return float(np.sqrt(numerator / denominator))


def _label_mass_dual_error(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    reference_start: NDArray[np.float64],
    reference_end: NDArray[np.float64],
    mass: NDArray[np.float64],
) -> float:
    """Return the two-stage mass-dual normalized residual error."""
    numerator = np.sum((start - reference_start) ** 2 / mass[None, :])
    numerator += np.sum((end - reference_end) ** 2 / mass[None, :])
    denominator = np.sum(reference_start**2 / mass[None, :])
    denominator += np.sum(reference_end**2 / mass[None, :])
    return float(np.sqrt(numerator / denominator))


def _mechanical_energy(
    system: ElasticINCSystem,
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> float:
    """Return lumped kinetic plus elastic strain energy."""
    kinetic = 0.5 * np.sum(system.mass * velocity**2)
    elastic = 0.5 * displacement @ (system.stiffness @ displacement)
    return float(kinetic + elastic)


def _half_sine_values(times: NDArray[np.float64], duration: float) -> NDArray[np.float64]:
    """Evaluate the compactly supported half-sine pulse."""
    normalized = times / duration
    inside = (normalized >= 0.0) & (normalized <= 1.0)
    values = np.zeros_like(normalized)
    values[inside] = np.sin(np.pi * normalized[inside])
    return values


def _plot_response_trajectories(
    trajectories: dict[str, dict[str, NDArray[np.float64]]], output_path: Path
) -> None:
    """Plot the selected longitudinal, transverse, and mixed response channels."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    channel_order = (
        "right_axial_displacement",
        "right_transverse_displacement",
        "tip_rotation",
    )
    channel_titles = {
        "right_axial_displacement": "右端轴向位移",
        "right_transverse_displacement": "右端横向位移",
        "tip_rotation": "端部转角",
    }
    channel_units = {
        "right_axial_displacement": "位移/m",
        "right_transverse_displacement": "位移/m",
        "tip_rotation": "转角/rad",
    }
    rows = len(trajectories)
    columns = max((len(values) - 1) // 3 for values in trajectories.values())
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 3.1 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row, values in enumerate(trajectories.values()):
        channel_names = sorted(
            (
                name.removeprefix("reference_")
                for name in values
                if name.startswith("reference_")
            ),
            key=channel_order.index,
        )
        for column, channel in enumerate(channel_names):
            axis = axes[row, column]
            axis.plot(
                values["time"],
                values[f"reference_{channel}"],
                color="black",
                linewidth=1.6,
                label="细时间步有限元参考",
            )
            axis.plot(
                values["time"],
                values[f"baseline_{channel}"],
                color="#d95f02",
                linestyle="--",
                linewidth=1.1,
                label="粗时间步有限元",
            )
            axis.plot(
                values["time"],
                values[f"inc_{channel}"],
                color="#1b9e77",
                linestyle="-.",
                linewidth=1.1,
                label="INC校正结果",
            )
            axis.set_title(f"{DELIVERY_CASE_LABELS[row]}：{channel_titles[channel]}")
            axis.set_xlabel("时间/s")
            axis.set_ylabel(channel_units[channel])
            axis.grid(True, alpha=0.3)
            axis.legend()
        for column in range(len(channel_names), columns):
            axes[row, column].set_visible(False)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_accuracy_ratios(records: list[dict[str, Any]], output_path: Path) -> None:
    """Plot displacement, velocity, and energy error ratios by test path."""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = list(DELIVERY_CASE_LABELS[: len(records)])
    x_values = np.arange(len(labels))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].bar(
        x_values - width / 2,
        [record["displacement_error_ratio"] for record in records],
        width,
        label="位移",
    )
    axes[0].bar(
        x_values + width / 2,
        [record["velocity_error_ratio"] for record in records],
        width,
        label="速度",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x_values, labels)
    axes[0].set_ylabel("INC误差/粗时间步有限元误差")
    axes[0].set_title("位移和速度轨迹误差比")
    axes[0].set_ylim(0.0, 1.08)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(
        x_values,
        [record["energy_error_ratio"] for record in records],
        0.5,
        color="#7570b3",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x_values, labels)
    axes[1].set_ylabel("INC误差/粗时间步有限元误差")
    axes[1].set_title("机械能轨迹误差比")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
