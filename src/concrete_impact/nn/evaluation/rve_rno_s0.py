"""Post-training S0 diagnostics for one accepted RVE-RNO artifact.

Contents:
    Native-resolution errors, paired coarsening, state-rate statistics, and figures.
Author:
    Zhen Hao.
Created:
    2026-07-19.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray

from concrete_impact.core.progress import write_json_atomic
from concrete_impact.core.response_manifest import load_response_shard_paths
from concrete_impact.nn.config import RVERNOArtifactMetadata, RVERNOModelConfig
from concrete_impact.nn.datasets.rve import (
    HDF5RNOPathDataset,
    PreloadedRNOPathDataset,
    RNOConditioningSpec,
    RNOPathBatch,
    RNOPathSample,
    collate_rno_paths,
)
from concrete_impact.nn.deployment import load_rve_rno_state_dict
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class RVERNOS0DiagnosticsConfig:
    """Store one immutable post-training diagnostic request."""

    run_directory: Path
    response_shard_manifest: Path
    split_manifest: Path
    output_directory: Path
    device: Literal["cpu", "cuda"]
    batch_size: int
    dpi: int


@dataclass(frozen=True)
class PathPrediction:
    """Store native-resolution constitutive histories needed by S0 diagnostics."""

    stress: NDArray[np.float64]
    dissipation: NDArray[np.float64]
    thermodynamic_violation: NDArray[np.float64]
    dimensionless_state_rate: NDArray[np.float64]


def generate_rve_rno_s0_diagnostics(
    config: RVERNOS0DiagnosticsConfig,
) -> dict[str, Any]:
    """Generate S0 diagnostics without changing or retraining the selected model."""
    started = perf_counter()
    if config.output_directory.exists():
        raise FileExistsError(f"RVE-RNO S0 output exists: {config.output_directory}.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RVE-RNO S0 diagnostics requested CUDA, but CUDA is unavailable.")
    if config.batch_size <= 0 or config.dpi <= 0:
        raise ValueError("RVE-RNO S0 batch size and figure DPI must be positive.")

    model, metadata = _load_model(config)
    samples, path_metadata = _load_samples(config)
    stress_scale = np.asarray(
        metadata.normalization["macro_stress"]["rms_scale"], dtype=np.float64
    )
    dissipation_scale = float(
        metadata.normalization["dissipation_density"]["rms_scale"][0]
    )
    selected_samples = tuple(
        sample
        for sample in samples
        if path_metadata[sample.path_id]["split"] in {"validation", "test"}
    )
    predictions = _evaluate_native_paths(
        model,
        selected_samples,
        config.batch_size,
    )
    path_records = _build_path_records(
        selected_samples,
        predictions,
        path_metadata,
        stress_scale,
        dissipation_scale,
    )
    performance = _summarize_stress_and_dissipation(
        selected_samples,
        predictions,
        path_metadata,
        stress_scale,
        dissipation_scale,
    )
    native_resolution = _summarize_native_resolution(
        path_records,
        selected_samples,
        predictions,
        stress_scale,
    )
    paired_coarsening = _evaluate_paired_coarsening(
        model,
        selected_samples,
        predictions,
        path_metadata,
        stress_scale,
        config.batch_size,
    )
    state_rate = _summarize_state_rates(
        selected_samples,
        predictions,
        path_metadata,
    )

    config.output_directory.mkdir(parents=True)
    diagnostics = {
        "schema_version": "1.0",
        "completed": True,
        "scope": "S0_post_training_diagnostics",
        "model_modified": False,
        "model_retrained": False,
        "checkpoint": {
            "path": str(config.run_directory / "best_model.pt"),
            "sha256": _sha256_file(config.run_directory / "best_model.pt"),
            "best_epoch": _load_best_epoch(config.run_directory),
            "integrator": metadata.integrator,
            "evolution": metadata.evolution,
        },
        "evaluated_splits": ["validation", "test"],
        "path_counts": {
            split: sum(
                path_metadata[sample.path_id]["split"] == split
                for sample in selected_samples
            )
            for split in ("validation", "test")
        },
        "definitions": {
            "whole_history_relative_stress_error": (
                "sqrt(sum_t,c(predicted-reference)^2 / "
                "max(sum_t,c(reference^2), N*sum_c(training_stress_rms_scale_c^2)))"
            ),
            "native_resolution": (
                "Observed validation/test paths grouped by their original accepted "
                "increment count; path difficulty remains a possible confounder."
            ),
            "paired_coarsening": (
                "The same reference path is rerolled at a coarser time grid; the fine "
                "rollout is restricted to identical endpoints before comparison."
            ),
            "state_rate_grouping": (
                "Each state inherits the path-level expected response regime from the "
                "frozen split manifest."
            ),
        },
        "native_resolution": native_resolution,
        "paired_coarsening": paired_coarsening,
        "stress_and_dissipation_performance": performance,
        "path_metrics": path_records,
        "dimensionless_state_rate": state_rate,
        "runtime_seconds": perf_counter() - started,
    }
    diagnostics_path = config.output_directory / "s0_diagnostics.json"
    write_json_atomic(diagnostics_path, diagnostics)
    figure_paths = _render_figures(
        native_resolution,
        paired_coarsening,
        selected_samples,
        predictions,
        path_records,
        state_rate,
        config.output_directory,
        config.dpi,
    )
    manifest = {
        "schema_version": "1.0",
        "scope": "S0_post_training_diagnostics",
        "sources": [
            _file_record(config.run_directory / "best_model.pt"),
            _file_record(config.run_directory / "artifact_metadata.json"),
            _file_record(config.run_directory / "training_summary.json"),
            _file_record(config.response_shard_manifest),
            _file_record(config.split_manifest),
        ],
        "outputs": [
            _file_record(diagnostics_path),
            *(_file_record(path) for path in figure_paths),
        ],
    }
    write_json_atomic(config.output_directory / "diagnostic_manifest.json", manifest)
    return diagnostics


def whole_history_relative_stress_error(
    reference: NDArray[np.float64],
    predicted: NDArray[np.float64],
    stress_scale: NDArray[np.float64],
) -> float:
    """Compute the scale-floored relative error of one full stress history."""
    difference = predicted - reference
    denominator = max(
        float(np.sum(reference**2)),
        float(reference.shape[0] * np.sum(stress_scale**2)),
    )
    return float(np.sqrt(float(np.sum(difference**2)) / denominator))


def coarsen_rno_path(sample: RNOPathSample, factor: int) -> RNOPathSample:
    """Coarsen one path by endpoint sampling and exact blockwise time-step sums."""
    length = sample.time.size
    if factor <= 1 or length % factor != 0:
        raise ValueError(
            f"RVE-RNO path length {length} is not divisible by coarsening factor {factor}."
        )
    endpoints = np.arange(factor - 1, length, factor)
    time_step = sample.time_step.reshape(-1, factor).sum(axis=1)
    return RNOPathSample(
        path_id=sample.path_id,
        time=sample.time[endpoints],
        time_step=time_step,
        macro_strain=sample.macro_strain[endpoints],
        macro_stress=sample.macro_stress[endpoints],
        material_parameters=sample.material_parameters,
        microstructure_features=sample.microstructure_features,
        dissipation_density=sample.dissipation_density[endpoints],
    )


def _load_model(
    config: RVERNOS0DiagnosticsConfig,
) -> tuple[EnergyDissipationRVERNO, RVERNOArtifactMetadata]:
    """Load the selected checkpoint on the requested diagnostic device."""
    metadata_path = config.run_directory / "artifact_metadata.json"
    metadata = RVERNOArtifactMetadata.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )
    model_config = RVERNOModelConfig(
        family=metadata.model_family,
        time_representation=metadata.time_representation,
        integrator=metadata.integrator,
        evolution=metadata.evolution,
        reference_time_scale=metadata.reference_time_scale,
        latent_dimension=metadata.latent_dimension,
        hidden_size=metadata.hidden_size,
        hidden_layers=metadata.hidden_layers,
        activation=metadata.activation,
        strain_input_scale=metadata.strain_input_scale,
        dtype=metadata.dtype,
        device=config.device,
    )
    return (
        load_rve_rno_state_dict(
            model_config,
            config.run_directory / "best_model.pt",
            metadata_path,
        ),
        metadata,
    )


def _load_samples(
    config: RVERNOS0DiagnosticsConfig,
) -> tuple[tuple[RNOPathSample, ...], dict[str, dict[str, str]]]:
    """Load all accepted paths and their immutable frozen-split labels."""
    dataset = PreloadedRNOPathDataset(
        HDF5RNOPathDataset(
            load_response_shard_paths(config.response_shard_manifest),
            RNOConditioningSpec((), ()),
        )
    )
    samples = tuple(dataset[index] for index in range(len(dataset)))
    split_payload = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    records = {
        str(item["path_id"]): {
            "split": str(item["split"]),
            "family": str(item["family"]),
            "regime": str(item["regime"]),
        }
        for item in split_payload["paths"]
    }
    if {sample.path_id for sample in samples} != set(records):
        raise ValueError("RVE-RNO S0 paths differ from the frozen split manifest.")
    return samples, records


def _evaluate_native_paths(
    model: EnergyDissipationRVERNO,
    samples: tuple[RNOPathSample, ...],
    batch_size: int,
) -> dict[str, PathPrediction]:
    """Roll out selected validation and test paths at their native resolution."""
    predictions: dict[str, PathPrediction] = {}
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        arrays = _rollout(model, collate_rno_paths(chunk))
        for index, sample in enumerate(chunk):
            length = sample.time.size
            predictions[sample.path_id] = PathPrediction(
                stress=arrays["stress"][index, :length],
                dissipation=arrays["dissipation"][index, :length],
                thermodynamic_violation=arrays["thermodynamic_violation"][index, :length],
                dimensionless_state_rate=arrays["dimensionless_state_rate"][index, :length],
            )
    return predictions


def _rollout(
    model: EnergyDissipationRVERNO,
    batch: RNOPathBatch,
) -> dict[str, NDArray[np.float64]]:
    """Roll out one padded batch and retain six state-rate components."""
    strain = batch.network_inputs[..., :6].to(device=model.device, dtype=model.dtype)
    time_step = batch.time_step.to(device=model.device, dtype=model.dtype)
    mask = batch.valid_mask.to(device=model.device)
    latent = model.initial_state_batch(strain.shape[0])
    anchors = model.prepare_energy_anchors()
    stress_history: list[torch.Tensor] = []
    dissipation_history: list[torch.Tensor] = []
    violation_history: list[torch.Tensor] = []
    rate_history: list[torch.Tensor] = []
    for step in range(strain.shape[1]):
        active = mask[:, step]
        previous = latent
        update = model.update_batch(
            strain[:, step],
            previous,
            torch.where(active, time_step[:, step], torch.ones_like(time_step[:, step])),
            anchors,
        )
        latent = torch.where(active.unsqueeze(-1), update.latent_state, previous)
        stress_history.append(torch.where(active.unsqueeze(-1), update.stress, 0.0))
        dissipation_history.append(torch.where(active, update.dissipation, 0.0))
        violation_history.append(
            torch.where(active, update.thermodynamic_violation, 0.0)
        )
        rate_history.append(
            torch.where(active.unsqueeze(-1), update.dimensionless_state_rate, 0.0)
        )
        latent = latent.detach().requires_grad_(True)
    return {
        "stress": np.asarray(
            torch.stack(stress_history, dim=1).detach().cpu().numpy(),
            dtype=np.float64,
        ),
        "dissipation": np.asarray(
            torch.stack(dissipation_history, dim=1).detach().cpu().numpy(),
            dtype=np.float64,
        ),
        "thermodynamic_violation": np.asarray(
            torch.stack(violation_history, dim=1).detach().cpu().numpy(),
            dtype=np.float64,
        ),
        "dimensionless_state_rate": np.asarray(
            torch.stack(rate_history, dim=1).detach().cpu().numpy(),
            dtype=np.float64,
        ),
    }


def _build_path_records(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    path_metadata: dict[str, dict[str, str]],
    stress_scale: NDArray[np.float64],
    dissipation_scale: float,
) -> list[dict[str, Any]]:
    """Build one user-readable stress-error record per complete path."""
    records = []
    for sample in samples:
        predicted = predictions[sample.path_id].stress
        predicted_dissipation = predictions[sample.path_id].dissipation
        dissipation_error = _dissipation_relative_error(
            sample.dissipation_density,
            predicted_dissipation,
            dissipation_scale,
        )
        reference_cumulative = float(
            np.sum(sample.dissipation_density * sample.time_step)
        )
        predicted_cumulative = float(
            np.sum(predicted_dissipation * sample.time_step)
        )
        cumulative_floor = dissipation_scale * float(np.sum(sample.time_step))
        records.append(
            {
                "path_id": sample.path_id,
                "split": path_metadata[sample.path_id]["split"],
                "family": path_metadata[sample.path_id]["family"],
                "regime": path_metadata[sample.path_id]["regime"],
                "increment_count": int(sample.time.size),
                "whole_history_relative_stress_error_percent": 100.0
                * whole_history_relative_stress_error(
                    sample.macro_stress,
                    predicted,
                    stress_scale,
                ),
                "dissipation_history_relative_error_percent": 100.0
                * dissipation_error,
                "cumulative_dissipation_relative_error_percent": 100.0
                * abs(predicted_cumulative - reference_cumulative)
                / max(abs(reference_cumulative), cumulative_floor),
                "negative_dissipation_state_count": int(
                    np.count_nonzero(predicted_dissipation < 0.0)
                ),
                "maximum_thermodynamic_violation": float(
                    np.max(predictions[sample.path_id].thermodynamic_violation)
                ),
            }
        )
    return records


def _summarize_stress_and_dissipation(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    path_metadata: dict[str, dict[str, str]],
    stress_scale: NDArray[np.float64],
    dissipation_scale: float,
) -> dict[str, Any]:
    """Summarize comparable stress and monitored dissipation metrics by physical group."""
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_samples = tuple(
            sample for sample in samples if path_metadata[sample.path_id]["split"] == split
        )
        result[split] = {
            "all": _constitutive_group_summary(
                split_samples,
                predictions,
                stress_scale,
                dissipation_scale,
            ),
            "by_family": {
                family: _constitutive_group_summary(
                    tuple(
                        sample
                        for sample in split_samples
                        if path_metadata[sample.path_id]["family"] == family
                    ),
                    predictions,
                    stress_scale,
                    dissipation_scale,
                )
                for family in sorted(
                    {path_metadata[sample.path_id]["family"] for sample in split_samples}
                )
            },
            "by_regime": {
                regime: _constitutive_group_summary(
                    tuple(
                        sample
                        for sample in split_samples
                        if path_metadata[sample.path_id]["regime"] == regime
                    ),
                    predictions,
                    stress_scale,
                    dissipation_scale,
                )
                for regime in sorted(
                    {path_metadata[sample.path_id]["regime"] for sample in split_samples}
                )
            },
        }
    return result


def _constitutive_group_summary(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    stress_scale: NDArray[np.float64],
    dissipation_scale: float,
) -> dict[str, Any]:
    """Aggregate pathwise and combined-history metrics for one nonempty group."""
    reference_stress = np.concatenate(
        [sample.macro_stress for sample in samples], axis=0
    )
    predicted_stress = np.concatenate(
        [predictions[sample.path_id].stress for sample in samples], axis=0
    )
    reference_dissipation = np.concatenate(
        [sample.dissipation_density for sample in samples], axis=0
    )
    predicted_dissipation = np.concatenate(
        [predictions[sample.path_id].dissipation for sample in samples], axis=0
    )
    path_stress_errors = 100.0 * np.asarray(
        [
            whole_history_relative_stress_error(
                sample.macro_stress,
                predictions[sample.path_id].stress,
                stress_scale,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    path_dissipation_errors = 100.0 * np.asarray(
        [
            _dissipation_relative_error(
                sample.dissipation_density,
                predictions[sample.path_id].dissipation,
                dissipation_scale,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    return {
        "path_count": len(samples),
        "state_count": int(reference_dissipation.size),
        "path_stress_error_percent": _summarize_values(path_stress_errors),
        "combined_stress_history_error_percent": 100.0
        * whole_history_relative_stress_error(
            reference_stress,
            predicted_stress,
            stress_scale,
        ),
        "component_stress_history_error_percent": _component_stress_errors(
            reference_stress,
            predicted_stress,
            stress_scale,
        ),
        "path_dissipation_error_percent": _summarize_values(
            path_dissipation_errors
        ),
        "combined_dissipation_history_error_percent": 100.0
        * _dissipation_relative_error(
            reference_dissipation,
            predicted_dissipation,
            dissipation_scale,
        ),
        "negative_dissipation_state_count": int(
            np.count_nonzero(predicted_dissipation < 0.0)
        ),
        "maximum_thermodynamic_violation": float(
            max(
                np.max(predictions[sample.path_id].thermodynamic_violation)
                for sample in samples
            )
        ),
    }


def _dissipation_relative_error(
    reference: NDArray[np.float64],
    predicted: NDArray[np.float64],
    scale: float,
) -> float:
    """Compute the scale-floored relative error of one dissipation history."""
    denominator = max(
        float(np.sum(reference**2)),
        float(reference.size * scale**2),
    )
    return float(
        np.sqrt(float(np.sum((predicted - reference) ** 2)) / denominator)
    )


def _summarize_native_resolution(
    path_records: list[dict[str, Any]],
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    stress_scale: NDArray[np.float64],
) -> dict[str, Any]:
    """Summarize original validation/test paths by accepted increment count."""
    sample_by_id = {sample.path_id: sample for sample in samples}
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_result = {}
        for increment_count in (64, 96, 128, 192):
            group = [
                record
                for record in path_records
                if record["split"] == split
                and record["increment_count"] == increment_count
            ]
            errors = np.asarray(
                [record["whole_history_relative_stress_error_percent"] for record in group],
                dtype=np.float64,
            )
            group_samples = [sample_by_id[str(record["path_id"])] for record in group]
            reference = np.concatenate(
                [sample.macro_stress for sample in group_samples], axis=0
            )
            predicted = np.concatenate(
                [predictions[sample.path_id].stress for sample in group_samples], axis=0
            )
            split_result[str(increment_count)] = {
                "path_count": len(group),
                "path_error_percent": _summarize_values(errors),
                "combined_history_error_percent": 100.0
                * whole_history_relative_stress_error(reference, predicted, stress_scale),
                "component_history_error_percent": _component_stress_errors(
                    reference,
                    predicted,
                    stress_scale,
                ),
            }
        result[split] = split_result
    return result


def _evaluate_paired_coarsening(
    model: EnergyDissipationRVERNO,
    samples: tuple[RNOPathSample, ...],
    native_predictions: dict[str, PathPrediction],
    path_metadata: dict[str, dict[str, str]],
    stress_scale: NDArray[np.float64],
    batch_size: int,
) -> dict[str, Any]:
    """Compare fine and coarsened rollouts at identical RVE reference endpoints."""
    mappings = ((192, 2), (192, 3), (128, 2))
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_result = {}
        for native_length, factor in mappings:
            source_samples = tuple(
                sample
                for sample in samples
                if path_metadata[sample.path_id]["split"] == split
                and sample.time.size == native_length
            )
            coarse_samples = tuple(
                coarsen_rno_path(sample, factor) for sample in source_samples
            )
            coarse_predictions = _evaluate_native_paths(model, coarse_samples, batch_size)
            fine_errors = []
            coarse_errors = []
            fine_histories = []
            coarse_histories = []
            references = []
            endpoints = np.arange(factor - 1, native_length, factor)
            for sample, coarse_sample in zip(source_samples, coarse_samples, strict=True):
                reference = coarse_sample.macro_stress
                fine = native_predictions[sample.path_id].stress[endpoints]
                coarse = coarse_predictions[sample.path_id].stress
                fine_errors.append(
                    whole_history_relative_stress_error(reference, fine, stress_scale)
                )
                coarse_errors.append(
                    whole_history_relative_stress_error(reference, coarse, stress_scale)
                )
                references.append(reference)
                fine_histories.append(fine)
                coarse_histories.append(coarse)
            fine_array = 100.0 * np.asarray(fine_errors, dtype=np.float64)
            coarse_array = 100.0 * np.asarray(coarse_errors, dtype=np.float64)
            reference_all = np.concatenate(references, axis=0)
            fine_all = np.concatenate(fine_histories, axis=0)
            coarse_all = np.concatenate(coarse_histories, axis=0)
            target_length = native_length // factor
            split_result[f"{native_length}_to_{target_length}"] = {
                "path_count": len(source_samples),
                "fine_endpoint_path_error_percent": _summarize_values(fine_array),
                "coarse_path_error_percent": _summarize_values(coarse_array),
                "coarse_minus_fine_path_error_percentage_points": _summarize_values(
                    coarse_array - fine_array
                ),
                "fine_endpoint_combined_history_error_percent": 100.0
                * whole_history_relative_stress_error(
                    reference_all,
                    fine_all,
                    stress_scale,
                ),
                "coarse_combined_history_error_percent": 100.0
                * whole_history_relative_stress_error(
                    reference_all,
                    coarse_all,
                    stress_scale,
                ),
                "fine_endpoint_component_history_error_percent": (
                    _component_stress_errors(reference_all, fine_all, stress_scale)
                ),
                "coarse_component_history_error_percent": _component_stress_errors(
                    reference_all,
                    coarse_all,
                    stress_scale,
                ),
            }
        result[split] = split_result
    return result


def _summarize_state_rates(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    path_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Summarize six direct-rate outputs by split, regime, and path family."""
    result: dict[str, Any] = {}
    for split in ("validation", "test"):
        split_samples = tuple(
            sample
            for sample in samples
            if path_metadata[sample.path_id]["split"] == split
        )
        regimes = sorted({path_metadata[sample.path_id]["regime"] for sample in split_samples})
        families = sorted({path_metadata[sample.path_id]["family"] for sample in split_samples})
        result[split] = {
            "all": _state_rate_group_summary(split_samples, predictions),
            "by_regime": {
                regime: _state_rate_group_summary(
                    tuple(
                        sample
                        for sample in split_samples
                        if path_metadata[sample.path_id]["regime"] == regime
                    ),
                    predictions,
                )
                for regime in regimes
            },
            "by_family": {
                family: _state_rate_group_summary(
                    tuple(
                        sample
                        for sample in split_samples
                        if path_metadata[sample.path_id]["family"] == family
                    ),
                    predictions,
                )
                for family in families
            },
        }
    return result


def _state_rate_group_summary(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
) -> dict[str, Any]:
    """Aggregate state-rate components over every state in a path group."""
    values = np.concatenate(
        [predictions[sample.path_id].dimensionless_state_rate for sample in samples],
        axis=0,
    )
    norms = np.linalg.vector_norm(values, axis=1)
    return {
        "path_count": len(samples),
        "state_count": int(values.shape[0]),
        "norm": _summarize_values(norms),
        "components": {
            f"z_rate_{component + 1}": _summarize_values(values[:, component])
            for component in range(values.shape[1])
        },
    }


def _component_stress_errors(
    reference: NDArray[np.float64],
    predicted: NDArray[np.float64],
    stress_scale: NDArray[np.float64],
) -> dict[str, float]:
    """Compute scale-floored relative errors for six stress components."""
    names = ("xx", "yy", "zz", "yz", "xz", "xy")
    result = {}
    for index, name in enumerate(names):
        numerator = float(np.sum((predicted[:, index] - reference[:, index]) ** 2))
        denominator = max(
            float(np.sum(reference[:, index] ** 2)),
            float(reference.shape[0] * stress_scale[index] ** 2),
        )
        result[name] = 100.0 * float(np.sqrt(numerator / denominator))
    return result


def _summarize_values(values: NDArray[np.float64]) -> dict[str, float]:
    """Return deterministic distribution statistics for one finite vector."""
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("RVE-RNO S0 summaries require a nonempty finite vector.")
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _render_figures(
    native_resolution: dict[str, Any],
    paired_coarsening: dict[str, Any],
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    path_records: list[dict[str, Any]],
    state_rate: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> tuple[Path, ...]:
    """Render stress-resolution and direct-rate discrimination figures."""
    resolution_paths = _render_resolution_figure(
        native_resolution,
        paired_coarsening,
        output_directory,
        dpi,
    )
    state_rate_paths = _render_state_rate_figure(state_rate, output_directory, dpi)
    path_evolution_paths = _render_worst_test_path_figure(
        samples,
        predictions,
        path_records,
        output_directory,
        dpi,
    )
    return (*resolution_paths, *state_rate_paths, *path_evolution_paths)


def _render_worst_test_path_figure(
    samples: tuple[RNOPathSample, ...],
    predictions: dict[str, PathPrediction],
    path_records: list[dict[str, Any]],
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Plot six stress components and dissipation for the worst frozen-test path."""
    worst = max(
        (record for record in path_records if record["split"] == "test"),
        key=lambda record: float(record["whole_history_relative_stress_error_percent"]),
    )
    sample = next(sample for sample in samples if sample.path_id == worst["path_id"])
    prediction = predictions[sample.path_id]
    abscissa = np.linspace(0.0, 1.0, sample.time.size)
    names = ("xx", "yy", "zz", "yz", "xz", "xy")
    figure, axes = plt.subplots(4, 2, figsize=(12.0, 12.0), constrained_layout=True)
    flat_axes = axes.ravel()
    for component, name in enumerate(names):
        axis = flat_axes[component]
        axis.plot(abscissa, sample.macro_stress[:, component], label="RVE", linewidth=1.6)
        axis.plot(
            abscissa,
            prediction.stress[:, component],
            label="RNO",
            linestyle="--",
            linewidth=1.3,
        )
        axis.set_title(f"Stress {name}")
        axis.set_xlabel("Normalized path coordinate")
        axis.set_ylabel("Stress")
        axis.grid(alpha=0.25)
    flat_axes[6].plot(abscissa, sample.dissipation_density, label="RVE", linewidth=1.6)
    flat_axes[6].plot(
        abscissa,
        prediction.dissipation,
        label="RNO",
        linestyle="--",
        linewidth=1.3,
    )
    flat_axes[6].set_title("Dissipation density (monitor only)")
    flat_axes[6].set_xlabel("Normalized path coordinate")
    flat_axes[6].set_ylabel("Dissipation density")
    flat_axes[6].grid(alpha=0.25)
    flat_axes[7].axis("off")
    flat_axes[0].legend()
    figure.suptitle(
        "Worst test path: "
        f"{sample.path_id} | "
        f"stress error={worst['whole_history_relative_stress_error_percent']:.2f}%"
    )
    png = output_directory / "worst_test_path_evolution.png"
    svg = output_directory / "worst_test_path_evolution.svg"
    figure.savefig(png, dpi=dpi)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _render_resolution_figure(
    native_resolution: dict[str, Any],
    paired_coarsening: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Plot native groups and paired coarsening on one compact figure."""
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    lengths = np.asarray((64, 96, 128, 192), dtype=np.float64)
    for split, color in (("validation", "#1f77b4"), ("test", "#d62728")):
        median = [
            native_resolution[split][str(int(length))]["path_error_percent"]["median"]
            for length in lengths
        ]
        p95 = [
            native_resolution[split][str(int(length))]["path_error_percent"]["p95"]
            for length in lengths
        ]
        axes[0].plot(lengths, median, marker="o", color=color, label=f"{split} median")
        axes[0].plot(
            lengths,
            p95,
            marker="s",
            linestyle="--",
            color=color,
            label=f"{split} p95",
        )
    axes[0].set_xlabel("Accepted increment count")
    axes[0].set_ylabel("Whole-history stress error (%)")
    axes[0].set_title("(a) Native-resolution groups")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = ("192→96", "192→64", "128→64")
    keys = ("192_to_96", "192_to_64", "128_to_64")
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.2
    for split_index, split in enumerate(("validation", "test")):
        fine = [
            paired_coarsening[split][key][
                "fine_endpoint_combined_history_error_percent"
            ]
            for key in keys
        ]
        coarse = [
            paired_coarsening[split][key]["coarse_combined_history_error_percent"]
            for key in keys
        ]
        base = x + (split_index * 2 - 1.5) * width
        axes[1].bar(base, fine, width, label=f"{split} fine endpoints")
        axes[1].bar(base + width, coarse, width, label=f"{split} coarse rollout")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Combined stress-history error (%)")
    axes[1].set_title("(b) Paired time-grid coarsening")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)
    png = output_directory / "stress_time_resolution_diagnostics.png"
    svg = output_directory / "stress_time_resolution_diagnostics.svg"
    figure.savefig(png, dpi=dpi)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _render_state_rate_figure(
    state_rate: dict[str, Any],
    output_directory: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Plot componentwise state-rate quantiles for every response regime."""
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    component_x = np.arange(1, 7, dtype=np.float64)
    regime_order = ("elastic", "transition", "viscoplastic")
    colors = ("#2ca02c", "#ff7f0e", "#9467bd")
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        available = state_rate[split]["by_regime"]
        for regime_index, (regime, color) in enumerate(
            zip(regime_order, colors, strict=True)
        ):
            components = available[regime]["components"]
            median = np.asarray(
                [components[f"z_rate_{index}"]["median"] for index in range(1, 7)]
            )
            p05 = np.asarray(
                [components[f"z_rate_{index}"]["p05"] for index in range(1, 7)]
            )
            p95 = np.asarray(
                [components[f"z_rate_{index}"]["p95"] for index in range(1, 7)]
            )
            offset = (regime_index - 1) * 0.12
            axis.errorbar(
                component_x + offset,
                median,
                yerr=np.vstack((median - p05, p95 - median)),
                fmt="o",
                capsize=3,
                color=color,
                label=regime,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(component_x, [f"rate {index}" for index in range(1, 7)])
        axis.set_xlabel("Dimensionless internal-variable rate component")
        axis.set_ylabel("Component value (median, p05–p95)")
        axis.set_title(f"{split.capitalize()} split")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    png = output_directory / "state_rate_component_diagnostics.png"
    svg = output_directory / "state_rate_component_diagnostics.svg"
    figure.savefig(png, dpi=dpi)
    figure.savefig(svg)
    plt.close(figure)
    return png, svg


def _load_best_epoch(run_directory: Path) -> int:
    """Read the validation-selected checkpoint epoch."""
    payload = json.loads(
        (run_directory / "training_summary.json").read_text(encoding="utf-8")
    )
    return int(payload["best_epoch"])


def _file_record(path: Path) -> dict[str, str | int]:
    """Build one path, size, and SHA-256 provenance record."""
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    """Compute one binary file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
