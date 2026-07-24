"""Evaluate trained RVE-RNO accuracy, tangents, and local-update throughput.

Contents:
    Strict configuration, split rollouts, path metrics, tangent checks, and timing.
Author:
    Zhen Hao.
Created:
    2026-07-17.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from concrete_impact.core.progress import write_json_atomic
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
from fem.io.config import load_yaml_config


class StrictEvaluationModel(BaseModel):
    """Forbid undeclared RVE-RNO evaluation fields."""

    model_config = ConfigDict(extra="forbid")


class RVERNOTangentEvaluationConfig(StrictEvaluationModel):
    """Store deterministic frozen-test tangent evaluation settings."""

    path_count: PositiveInt = 10
    states_per_path: PositiveInt = 4
    perturbation: PositiveFloat = 1.0e-5
    relative_norm_floor: PositiveFloat = 1.0e-12


class RVERNOPerformanceEvaluationConfig(StrictEvaluationModel):
    """Store repeated local material-update timing settings."""

    batch_sizes: tuple[PositiveInt, ...] = (1, 16, 64, 256, 1024)
    warmup_updates: PositiveInt = 20
    timing_groups: PositiveInt = 10
    updates_per_group: PositiveInt = 100

    @model_validator(mode="after")
    def require_declared_batch_sizes(self) -> RVERNOPerformanceEvaluationConfig:
        """Require unique increasing material-point batch sizes."""
        if tuple(sorted(set(self.batch_sizes))) != self.batch_sizes:
            raise ValueError("RVE-RNO timing batch sizes must be unique and increasing.")
        return self


class RVERNOSlurmBaselineConfig(StrictEvaluationModel):
    """Store the archived Slurm c48 timing provenance."""

    expected_path_count: PositiveInt
    threads_per_path: PositiveInt
    launcher: Path
    sbatch: Path


class RVERNOReferenceThresholds(StrictEvaluationModel):
    """Store formal Direct-RNO thresholds used only as Pilot references."""

    test_stress_rmse: PositiveFloat
    family_stress_error: PositiveFloat
    maximum_path_stress_error: PositiveFloat
    normalized_dissipation_error: PositiveFloat
    negative_dissipation_tolerance: float = Field(ge=0.0)
    tangent_median_error: PositiveFloat
    tangent_p95_error: PositiveFloat
    macro_relative_error: PositiveFloat


class RVERNOFigureEvaluationConfig(StrictEvaluationModel):
    """Store deterministic scientific-figure output settings."""

    output_directory: Path
    dpi: PositiveInt = 300
    representative_path_count: PositiveInt = 6


class RVERNOEvaluationConfig(StrictEvaluationModel):
    """Store one complete trained RVE-RNO evaluation request."""

    schema_version: Literal["1.0"]
    certification_scope: Literal["pilot"]
    run_directory: Path
    response_shard_manifest: Path
    split_manifest: Path
    subset_manifest: Path
    subset_summary: Path
    output_directory: Path
    reference_thresholds_config: Path
    device: Literal["cpu", "cuda"]
    evaluation_batch_size: PositiveInt
    tangent: RVERNOTangentEvaluationConfig = Field(
        default_factory=RVERNOTangentEvaluationConfig
    )
    performance: RVERNOPerformanceEvaluationConfig = Field(
        default_factory=RVERNOPerformanceEvaluationConfig
    )
    slurm_baseline: RVERNOSlurmBaselineConfig
    figures: RVERNOFigureEvaluationConfig

    @model_validator(mode="after")
    def require_independent_outputs(self) -> RVERNOEvaluationConfig:
        """Require evaluation and figure directories to be new independent outputs."""
        if self.output_directory == self.figures.output_directory:
            raise ValueError("Evaluation data and figure directories must be distinct.")
        return self


@dataclass(frozen=True)
class EvaluationPathSource:
    """Store one immutable HDF5 path and its grouping metadata."""

    path_id: str
    file_path: Path
    group_name: str
    split: str
    metadata: dict[str, Any]


def load_rve_rno_evaluation_config(path: str | Path) -> RVERNOEvaluationConfig:
    """Load one strict trained RVE-RNO evaluation configuration."""
    return RVERNOEvaluationConfig.model_validate(load_yaml_config(path))


def evaluate_rve_rno_artifact(
    config: RVERNOEvaluationConfig,
    config_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one selected checkpoint and publish traceable numerical outputs."""
    if config.output_directory.exists():
        raise FileExistsError(f"RVE-RNO evaluation output exists: {config.output_directory}.")
    if config.figures.output_directory.exists():
        raise FileExistsError(
            f"RVE-RNO evaluation figure output exists: {config.figures.output_directory}."
        )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RVE-RNO evaluation requested CUDA, but CUDA is unavailable.")
    training_summary = json.loads(
        (config.run_directory / "training_summary.json").read_text(encoding="utf-8")
    )
    if training_summary["selection_split"] != "validation":
        raise ValueError("RVE-RNO artifact was not selected by validation loss.")
    if int(training_summary["frozen_test_evaluation_count"]) != 1:
        raise ValueError("RVE-RNO training did not preserve one frozen-test evaluation.")

    model, metadata = _load_evaluation_model(config)
    samples, sources = _load_evaluation_paths(config)
    thresholds = _load_reference_thresholds(config.reference_thresholds_config)
    config.output_directory.mkdir(parents=True)
    prediction_path = config.output_directory / "predictions.h5"
    path_records = _evaluate_all_splits(
        model,
        metadata,
        samples,
        sources,
        prediction_path,
        config.evaluation_batch_size,
        thresholds.negative_dissipation_tolerance,
    )
    _write_path_metrics(config.output_directory / "path_metrics.csv", path_records)
    group_metrics = summarize_group_metrics(path_records)
    write_json_atomic(config.output_directory / "group_metrics.json", group_metrics)
    global_metrics = summarize_global_split_metrics(path_records)

    tangent_metrics = evaluate_tangents(
        model,
        sources,
        prediction_path,
        config.tangent,
    )
    write_json_atomic(config.output_directory / "tangent_metrics.json", tangent_metrics)
    inference = benchmark_local_material_updates(model, config.performance)
    c48 = summarize_c48_performance(config.subset_manifest, config.slurm_baseline)
    inference["c48_baseline"] = c48
    inference["cross_platform_local_material_update_throughput_ratio"] = {
        str(record["batch_size"]): c48["median_seconds_per_increment"]
        / record["median_seconds_per_state"]
        for record in inference["batch_results"]
    }
    write_json_atomic(
        config.output_directory / "inference_performance.json", inference
    )

    quality_reference = compare_formal_quality_reference(
        path_records,
        global_metrics,
        tangent_metrics["summary"],
        thresholds,
    )
    observed_families = sorted({str(record["family"]) for record in path_records})
    formal_families = (
        "hold",
        "load_unload_reload",
        "monotonic",
        "nonproportional",
        "reverse",
    )
    missing_families = sorted(set(formal_families) - set(observed_families))

    summary = {
        "schema_version": "1.0",
        "passed": True,
        "evaluation_completed": True,
        "certification_scope": config.certification_scope,
        "formal_artifact_certified": False,
        "formal_certification_blockers": [
            f"missing_path_families:{','.join(missing_families)}",
            "macro_rno_fem_embedding_not_evaluated",
            "pilot_subset_not_formal_2000_path_dataset",
        ],
        "selection_split": "validation",
        "frozen_test_used_for_model_selection": False,
        "path_count": len(path_records),
        "split_counts": dict(
            sorted(
                {
                    split: sum(record["split"] == split for record in path_records)
                    for split in ("train", "validation", "test")
                }.items()
            )
        ),
        "group_metrics": group_metrics,
        "global_split_metrics": global_metrics,
        "tangent_summary": tangent_metrics["summary"],
        "formal_quality_reference": quality_reference,
        "observed_path_families": observed_families,
        "missing_formal_path_families": missing_families,
        "artifact_sha256": _sha256_file(config.run_directory / "best_model.pt"),
        "artifact_metadata_sha256": _sha256_file(
            config.run_directory / "artifact_metadata.json"
        ),
        "response_shard_manifest_sha256": _sha256_file(
            config.response_shard_manifest
        ),
        "split_manifest_sha256": _sha256_file(config.split_manifest),
        "subset_manifest_sha256": _sha256_file(config.subset_manifest),
        "evaluation_config_sha256": _sha256_file(Path(config_path)),
        "reference_thresholds_config_sha256": _sha256_file(
            config.reference_thresholds_config
        ),
    }
    write_json_atomic(config.output_directory / "evaluation_summary.json", summary)
    from concrete_impact.reporting.rve_rno_evaluation import (
        render_rve_rno_evaluation_figures,
    )

    figure_manifest = render_rve_rno_evaluation_figures(
        run_directory=config.run_directory,
        evaluation_directory=config.output_directory,
        subset_summary_path=config.subset_summary,
        evaluation_config_path=Path(config_path),
        output_directory=config.figures.output_directory,
        dpi=config.figures.dpi,
        representative_path_count=config.figures.representative_path_count,
    )
    summary["figure_manifest"] = str(figure_manifest)
    summary["figure_manifest_sha256"] = _sha256_file(figure_manifest)
    write_json_atomic(config.output_directory / "evaluation_summary.json", summary)
    report_metrics = build_report_metrics(
        config,
        training_summary,
        summary,
        inference,
        figure_manifest,
    )
    write_json_atomic(config.output_directory / "report_metrics.json", report_metrics)
    return summary


def compute_path_metrics(
    reference_stress: NDArray[np.float64],
    predicted_stress: NDArray[np.float64],
    reference_dissipation: NDArray[np.float64],
    predicted_dissipation: NDArray[np.float64],
    time_step: NDArray[np.float64],
    stress_scale: NDArray[np.float64],
    dissipation_scale: float,
    latent_norm: NDArray[np.float64],
    dimensionless_rate_norm: NDArray[np.float64],
    latent_increment_norm: NDArray[np.float64],
    negative_dissipation_tolerance: float,
) -> dict[str, float | int]:
    """Compute whole-history stress, dissipation, and latent-state diagnostics."""
    arrays = (
        reference_stress,
        predicted_stress,
        reference_dissipation,
        predicted_dissipation,
        time_step,
        stress_scale,
        latent_norm,
        dimensionless_rate_norm,
        latent_increment_norm,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise FloatingPointError("RVE-RNO path metrics received non-finite values.")
    if reference_stress.shape != predicted_stress.shape:
        raise ValueError("RVE-RNO reference and predicted stress shapes differ.")
    if reference_dissipation.shape != predicted_dissipation.shape:
        raise ValueError("RVE-RNO reference and predicted dissipation shapes differ.")
    length = reference_stress.shape[0]
    if reference_stress.shape != (length, 6):
        raise ValueError("RVE-RNO stress history must have shape [state, 6].")
    if reference_dissipation.shape != (length,) or time_step.shape != (length,):
        raise ValueError("RVE-RNO scalar histories must match the stress history length.")
    if stress_scale.shape != (6,) or np.any(stress_scale <= 0.0):
        raise ValueError("RVE-RNO stress scale must contain six positive values.")
    if not np.isfinite(dissipation_scale) or dissipation_scale <= 0.0:
        raise ValueError("RVE-RNO dissipation scale must be finite and positive.")
    if np.any(time_step <= 0.0):
        raise ValueError("RVE-RNO evaluation time steps must be positive.")
    if not np.isfinite(negative_dissipation_tolerance) or negative_dissipation_tolerance < 0.0:
        raise ValueError("RVE-RNO negative-dissipation tolerance must be finite and nonnegative.")
    for norm_history in (latent_norm, dimensionless_rate_norm, latent_increment_norm):
        if norm_history.shape != (length,):
            raise ValueError("RVE-RNO latent diagnostics must match the path length.")
    stress_difference = predicted_stress - reference_stress
    stress_numerator = float(np.sum(stress_difference**2))
    stress_denominator = max(
        float(np.sum(reference_stress**2)),
        float(length * np.sum(stress_scale**2)),
    )
    dissipation_difference = predicted_dissipation - reference_dissipation
    dissipation_denominator = max(
        float(np.sum(reference_dissipation**2)),
        float(length * dissipation_scale**2),
    )
    reference_cumulative = float(np.sum(reference_dissipation * time_step))
    predicted_cumulative = float(np.sum(predicted_dissipation * time_step))
    cumulative_floor = float(dissipation_scale * np.sum(time_step))
    violation = np.maximum(-predicted_dissipation, 0.0)
    return {
        "stress_normalized_rmse": float(np.sqrt(stress_numerator / stress_denominator)),
        "stress_rmse": float(np.sqrt(np.mean(stress_difference**2))),
        "stress_squared_error_sum": stress_numerator,
        "stress_normalization_denominator": stress_denominator,
        "dissipation_normalized_rmse": float(
            np.sqrt(float(np.sum(dissipation_difference**2)) / dissipation_denominator)
        ),
        "dissipation_squared_error_sum": float(np.sum(dissipation_difference**2)),
        "dissipation_normalization_denominator": dissipation_denominator,
        "cumulative_dissipation_relative_error": abs(
            predicted_cumulative - reference_cumulative
        )
        / max(abs(reference_cumulative), cumulative_floor),
        "reference_cumulative_dissipation": reference_cumulative,
        "predicted_cumulative_dissipation": predicted_cumulative,
        "negative_dissipation_state_count": int(
            np.count_nonzero(predicted_dissipation < -negative_dissipation_tolerance)
        ),
        "maximum_thermodynamic_violation": float(np.max(violation)),
        **_norm_quantiles("latent_norm", latent_norm),
        **_norm_quantiles("dimensionless_state_rate_norm", dimensionless_rate_norm),
        **_norm_quantiles("latent_increment_norm", latent_increment_norm),
    }


def summarize_group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize path errors by split and immutable physical design groups."""
    if not records:
        raise ValueError("RVE-RNO group metrics require at least one path record.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        keys = (
            f"split:{record['split']}",
            f"family:{record['family']}",
            f"split:{record['split']}|family:{record['family']}",
            f"amplitude_band:{record['amplitude_band']}",
            f"rate_band:{record['rate_band']}",
            f"tensor_direction:{record['tensor_direction']}",
            f"regime:{record['regime']}",
            f"split:{record['split']}|regime:{record['regime']}",
        )
        for key in keys:
            groups.setdefault(key, []).append(record)
    return {
        "schema_version": "1.0",
        "groups": {
            key: _summarize_metric_records(group)
            for key, group in sorted(groups.items())
        },
    }


def summarize_global_split_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate path energy numerators and denominators within each split."""
    summaries = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            raise ValueError(f"RVE-RNO global metrics require split: {split}.")
        stress_numerator = sum(
            float(record["stress_squared_error_sum"]) for record in split_records
        )
        stress_denominator = sum(
            float(record["stress_normalization_denominator"])
            for record in split_records
        )
        dissipation_numerator = sum(
            float(record["dissipation_squared_error_sum"])
            for record in split_records
        )
        dissipation_denominator = sum(
            float(record["dissipation_normalization_denominator"])
            for record in split_records
        )
        summaries[split] = {
            "path_count": len(split_records),
            "stress_normalized_rmse": float(
                np.sqrt(stress_numerator / stress_denominator)
            ),
            "dissipation_normalized_rmse": float(
                np.sqrt(dissipation_numerator / dissipation_denominator)
            ),
            "maximum_path_stress_error": max(
                float(record["stress_normalized_rmse"])
                for record in split_records
            ),
            "negative_dissipation_state_count": sum(
                int(record["negative_dissipation_state_count"])
                for record in split_records
            ),
            "maximum_thermodynamic_violation": max(
                float(record["maximum_thermodynamic_violation"])
                for record in split_records
            ),
        }
    return {"schema_version": "1.0", "splits": summaries}


def compare_formal_quality_reference(
    records: list[dict[str, Any]],
    global_metrics: dict[str, Any],
    tangent_summary: dict[str, Any],
    thresholds: RVERNOReferenceThresholds,
) -> dict[str, Any]:
    """Compare Pilot metrics with formal thresholds without certifying the artifact."""
    test_records = [record for record in records if record["split"] == "test"]
    family_errors = {}
    for family in sorted({str(record["family"]) for record in test_records}):
        family_records = [record for record in test_records if record["family"] == family]
        numerator = sum(
            float(record["stress_squared_error_sum"]) for record in family_records
        )
        denominator = sum(
            float(record["stress_normalization_denominator"])
            for record in family_records
        )
        value = float(np.sqrt(numerator / denominator))
        family_errors[family] = {
            "value": value,
            "threshold": thresholds.family_stress_error,
            "passed": value <= thresholds.family_stress_error,
        }

    test = global_metrics["splits"]["test"]
    tangent = tangent_summary["autodiff_vs_rve"]
    checks: dict[str, dict[str, Any]] = {
        "test_stress_rmse": _threshold_check(
            test["stress_normalized_rmse"], thresholds.test_stress_rmse
        ),
        "family_stress_error": {
            "groups": family_errors,
            "passed": all(record["passed"] for record in family_errors.values()),
        },
        "maximum_path_stress_error": _threshold_check(
            test["maximum_path_stress_error"],
            thresholds.maximum_path_stress_error,
        ),
        "normalized_dissipation_error": _threshold_check(
            test["dissipation_normalized_rmse"],
            thresholds.normalized_dissipation_error,
        ),
        "negative_dissipation_state_count": {
            "value": test["negative_dissipation_state_count"],
            "tolerance": thresholds.negative_dissipation_tolerance,
            "passed": test["negative_dissipation_state_count"] == 0,
        },
        "tangent_median_error": _threshold_check(
            tangent["median"], thresholds.tangent_median_error
        ),
        "tangent_p95_error": _threshold_check(
            tangent["p95"], thresholds.tangent_p95_error
        ),
    }
    return {
        "scope": "formal_thresholds_used_as_pilot_reference_only",
        "checks": checks,
        "all_available_metrics_passed": all(
            bool(record["passed"]) for record in checks.values()
        ),
    }


def _threshold_check(value: float, threshold: float) -> dict[str, float | bool]:
    """Compare one finite scalar with an upper threshold."""
    if not np.isfinite(value):
        raise FloatingPointError("RVE-RNO threshold comparison received a non-finite value.")
    return {"value": float(value), "threshold": float(threshold), "passed": value <= threshold}


def _load_reference_thresholds(path: Path) -> RVERNOReferenceThresholds:
    """Load only the formal threshold block from the declared acceptance config."""
    payload = load_yaml_config(path)
    return RVERNOReferenceThresholds.model_validate(payload["thresholds"])


def build_report_metrics(
    config: RVERNOEvaluationConfig,
    training_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    inference_performance: dict[str, Any],
    figure_manifest_path: Path,
) -> dict[str, Any]:
    """Build the sole structured numerical source for the final report."""
    run_status_path = config.run_directory.parent / "run_status.json"
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    if run_status["status"] != "completed" or int(run_status["return_code"]) != 0:
        raise ValueError("RVE-RNO report metrics require a completed training run.")
    subset_summary = json.loads(config.subset_summary.read_text(encoding="utf-8"))
    resolved_config = load_yaml_config(config.run_directory / "resolved_config.yaml")
    figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    figures = {
        record["name"]: {
            item["path"].rsplit(".", 1)[-1]: item
            for item in record["outputs"]
        }
        for record in figure_manifest["figures"]
    }
    wall_seconds = float(run_status["finished_unix_time"]) - float(
        run_status["started_unix_time"]
    )
    if not np.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise FloatingPointError("RVE-RNO training wall time must be finite and positive.")
    return {
        "schema_version": "1.0",
        "certification_scope": config.certification_scope,
        "formal_artifact_certified": False,
        "dataset": {
            "path_count": int(evaluation_summary["path_count"]),
            "split_counts": evaluation_summary["split_counts"],
            "family_counts": subset_summary["family_counts"],
            "tensor_direction_counts": subset_summary["tensor_direction_counts"],
            "accepted_increment_counts": subset_summary["accepted_increment_counts"],
        },
        "training": {
            "epochs": int(resolved_config["epochs"]),
            "best_epoch": int(training_summary["best_epoch"]),
            "best_validation_loss": float(training_summary["best_validation_loss"]),
            "wall_seconds": wall_seconds,
            "wall_minutes": wall_seconds / 60.0,
            "peak_cuda_memory_bytes": int(
                training_summary["test_metrics"]["peak_cuda_memory_allocated_bytes"]
            ),
            "selection_split": training_summary["selection_split"],
            "frozen_test_evaluation_count": int(
                training_summary["frozen_test_evaluation_count"]
            ),
        },
        "evaluation": {
            "global_split_metrics": evaluation_summary["global_split_metrics"],
            "test_path_distribution": evaluation_summary["group_metrics"]["groups"][
                "split:test"
            ],
            "formal_quality_reference": evaluation_summary[
                "formal_quality_reference"
            ],
            "tangent_summary": evaluation_summary["tangent_summary"],
        },
        "performance": inference_performance,
        "figures": figures,
        "limitations": evaluation_summary["formal_certification_blockers"],
        "provenance": {
            "training_run_status": {
                "path": str(run_status_path),
                "sha256": _sha256_file(run_status_path),
            },
            "artifact": {
                "path": str(config.run_directory / "best_model.pt"),
                "sha256": evaluation_summary["artifact_sha256"],
            },
            "evaluation_summary": {
                "path": str(config.output_directory / "evaluation_summary.json"),
                "sha256": _sha256_file(
                    config.output_directory / "evaluation_summary.json"
                ),
            },
            "figure_manifest": {
                "path": str(figure_manifest_path),
                "sha256": _sha256_file(figure_manifest_path),
            },
            "reference_thresholds_config": {
                "path": str(config.reference_thresholds_config),
                "sha256": _sha256_file(config.reference_thresholds_config),
            },
        },
    }


def deterministic_tangent_state_selection(
    test_sources: tuple[EvaluationPathSource, ...],
    path_count: int,
    states_per_path: int,
    path_lengths: dict[str, int],
) -> dict[str, list[int]]:
    """Select frozen-test paths by hash and equally spaced state indices."""
    ordered = sorted(
        test_sources,
        key=lambda source: (hashlib.sha256(source.path_id.encode()).hexdigest(), source.path_id),
    )
    if len(ordered) < path_count:
        raise ValueError("Frozen test set contains fewer paths than tangent evaluation requests.")
    selected: dict[str, list[int]] = {}
    for source in ordered[:path_count]:
        length = path_lengths[source.path_id]
        if length < states_per_path:
            raise ValueError(f"Tangent path has too few states: {source.path_id}.")
        indices = np.rint(np.linspace(0, length - 1, states_per_path)).astype(int)
        if np.unique(indices).size != states_per_path:
            raise ValueError(f"Tangent state selection is not unique: {source.path_id}.")
        selected[source.path_id] = indices.tolist()
    return selected


def summarize_timing_samples(
    batch_size: int,
    updates_per_group: int,
    samples: NDArray[np.float64],
) -> dict[str, float | int | list[float]]:
    """Summarize repeated timing groups with median and median absolute deviation."""
    if samples.size == 0 or not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
        raise FloatingPointError("RVE-RNO timing samples must be finite and positive.")
    seconds_per_state = samples / (batch_size * updates_per_group)
    median = float(np.median(seconds_per_state))
    mad = float(np.median(np.abs(seconds_per_state - median)))
    return {
        "batch_size": batch_size,
        "timing_group_count": int(samples.size),
        "updates_per_group": updates_per_group,
        "group_seconds": samples.tolist(),
        "median_seconds_per_state": median,
        "median_absolute_deviation_seconds_per_state": mad,
        "median_states_per_second": 1.0 / median,
    }


def benchmark_local_material_updates(
    model: EnergyDissipationRVERNO,
    settings: RVERNOPerformanceEvaluationConfig,
) -> dict[str, Any]:
    """Benchmark batched local updates with synchronization only at group boundaries."""
    results = []
    for batch_size in settings.batch_sizes:
        strain = torch.linspace(
            -1.0e-2,
            1.0e-2,
            batch_size * 6,
            dtype=model.dtype,
            device=model.device,
        ).reshape(batch_size, 6)
        latent = torch.zeros(
            (batch_size, model.config.latent_dimension),
            dtype=model.dtype,
            device=model.device,
        )
        time_step = torch.full(
            (batch_size,),
            model.config.reference_time_scale / 100.0,
            dtype=model.dtype,
            device=model.device,
        )
        anchors = model.prepare_energy_anchors()
        for _ in range(settings.warmup_updates):
            update = model.update_batch(strain, latent, time_step, anchors)
            latent = update.latent_state.detach()
        samples = []
        for _ in range(settings.timing_groups):
            _synchronize_device(model.device)
            start = perf_counter()
            for _ in range(settings.updates_per_group):
                update = model.update_batch(strain, latent, time_step, anchors)
                latent = update.latent_state.detach()
            _synchronize_device(model.device)
            samples.append(perf_counter() - start)
        if not bool(torch.isfinite(update.stress).all().detach().cpu()):
            raise FloatingPointError("RVE-RNO timing produced non-finite stress.")
        results.append(
            summarize_timing_samples(
                batch_size,
                settings.updates_per_group,
                np.asarray(samples, dtype=np.float64),
            )
        )
    return {
        "schema_version": "1.0",
        "benchmark_scope": "local_material_update_only",
        "device": str(model.device),
        "hardware": {
            "platform": platform.platform(),
            "cpu_model": _read_local_cpu_model(),
            "gpu_model": (
                torch.cuda.get_device_name(model.device)
                if model.device.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "dtype": str(model.dtype),
        "warmup_updates": settings.warmup_updates,
        "batch_results": results,
    }


def summarize_c48_performance(
    subset_manifest_path: Path,
    baseline: RVERNOSlurmBaselineConfig,
) -> dict[str, Any]:
    """Summarize archived c48 solver seconds per accepted increment."""
    subset = json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    if not subset["tasks"]:
        raise ValueError("Archived c48 performance requires at least one task.")
    if len(subset["tasks"]) != baseline.expected_path_count:
        raise ValueError(
            "Archived c48 performance path count differs from the declared baseline."
        )
    values = []
    performance_hashes = {}
    status_hashes = {}
    hostnames = set()
    job_ids = set()
    for task in subset["tasks"]:
        status_path = Path(task["status_path"])
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["state"] != "succeeded":
            raise ValueError(f"Archived c48 task did not succeed: {status_path}.")
        performance_path = status_path.parent / "performance.json"
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        solver_time = float(performance["solver_time"])
        increments = int(performance["accepted_increment_count"])
        if not np.isfinite(solver_time) or solver_time <= 0.0 or increments <= 0:
            raise FloatingPointError(
                f"Invalid archived c48 performance record: {performance_path}."
            )
        values.append(solver_time / increments)
        performance_hashes[str(performance_path)] = _sha256_file(performance_path)
        status_hashes[str(status_path)] = _sha256_file(status_path)
        hostnames.add(str(status["hostname"]))
        job_ids.add(str(status["slurm_job_id"]))
    array = np.asarray(values, dtype=np.float64)
    return {
        "scope": "archived_slurm_cpu_c48_local_increment",
        "path_count": int(array.size),
        "threads_per_path": baseline.threads_per_path,
        "hostnames": sorted(hostnames),
        "slurm_job_ids": sorted(job_ids),
        "slurm_cpu_model_recorded": False,
        "median_seconds_per_increment": float(np.median(array)),
        "p95_seconds_per_increment": float(np.quantile(array, 0.95)),
        "launcher": {
            "path": str(baseline.launcher),
            "sha256": _sha256_file(baseline.launcher),
        },
        "sbatch": {
            "path": str(baseline.sbatch),
            "sha256": _sha256_file(baseline.sbatch),
        },
        "performance_file_sha256": performance_hashes,
        "status_file_sha256": status_hashes,
    }


def evaluate_tangents(
    model: EnergyDissipationRVERNO,
    sources: dict[str, EvaluationPathSource],
    prediction_path: Path,
    settings: RVERNOTangentEvaluationConfig,
) -> dict[str, Any]:
    """Compare automatic-differentiation tangents with RVE and central differences."""
    test_sources = tuple(source for source in sources.values() if source.split == "test")
    with h5py.File(prediction_path, "r") as predictions:
        lengths = {
            path_id: int(predictions[f"paths/{source.group_name}/time"].shape[0])
            for path_id, source in sources.items()
        }
        selection = deterministic_tangent_state_selection(
            test_sources,
            settings.path_count,
            settings.states_per_path,
            lengths,
        )
        records: list[dict[str, str | int | float]] = []
        rve_errors: list[float] = []
        finite_difference_errors: list[float] = []
        anchors = model.prepare_energy_anchors()
        for path_id, state_indices in selection.items():
            source = sources[path_id]
            predicted = predictions[f"paths/{source.group_name}"]
            with h5py.File(source.file_path, "r") as reference:
                reference_group = reference[f"paths/{source.group_name}"]
                for state_index in state_indices:
                    strain = predicted["macro_strain"][state_index]
                    previous_latent = predicted["previous_latent_state"][state_index]
                    time_step = predicted["time_step"][state_index]
                    rve_tangent = reference_group["macro/effective_tangent"][state_index]
                    ad_tangent, fd_tangent = _compute_model_tangents(
                        model,
                        strain,
                        previous_latent,
                        time_step,
                        settings.perturbation,
                        anchors,
                    )
                    rve_error = _relative_frobenius_error(
                        ad_tangent, rve_tangent, settings.relative_norm_floor
                    )
                    finite_difference_error = _relative_frobenius_error(
                        ad_tangent,
                        fd_tangent,
                        settings.relative_norm_floor,
                    )
                    records.append(
                        {
                            "path_id": path_id,
                            "state_index": state_index,
                            "autodiff_vs_rve_relative_error": rve_error,
                            "autodiff_vs_central_difference_relative_error": (
                                finite_difference_error
                            ),
                        }
                    )
                    rve_errors.append(rve_error)
                    finite_difference_errors.append(finite_difference_error)
    return {
        "schema_version": "1.0",
        "selection_rule": "sha256_path_id_then_equal_state_spacing",
        "selected_states": selection,
        "records": records,
        "summary": {
            "autodiff_vs_rve": _summarize_values(rve_errors),
            "autodiff_vs_central_difference": _summarize_values(
                finite_difference_errors
            ),
        },
    }


def _load_evaluation_model(
    config: RVERNOEvaluationConfig,
) -> tuple[EnergyDissipationRVERNO, RVERNOArtifactMetadata]:
    """Load the selected artifact on the explicitly requested evaluation device."""
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
    model = load_rve_rno_state_dict(
        model_config,
        config.run_directory / "best_model.pt",
        metadata_path,
    )
    return model, metadata


def _load_evaluation_paths(
    config: RVERNOEvaluationConfig,
) -> tuple[tuple[RNOPathSample, ...], dict[str, EvaluationPathSource]]:
    """Load complete paths and require exact frozen split ownership."""
    response_manifest = json.loads(
        config.response_shard_manifest.read_text(encoding="utf-8")
    )
    shard_paths = tuple(Path(value) for value in response_manifest["response_shards"])
    dataset = PreloadedRNOPathDataset(
        HDF5RNOPathDataset(shard_paths, RNOConditioningSpec((), ()))
    )
    samples = tuple(dataset[index] for index in range(len(dataset)))
    split_payload = json.loads(config.split_manifest.read_text(encoding="utf-8"))
    split_by_id = {str(item["path_id"]): str(item["split"]) for item in split_payload["paths"]}
    sample_ids = {sample.path_id for sample in samples}
    if sample_ids != set(split_by_id):
        raise ValueError("RVE-RNO evaluation paths differ from the frozen split manifest.")
    split_counts = {split: sum(value == split for value in split_by_id.values()) for split in (
        "train", "validation", "test"
    )}
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"RVE-RNO evaluation requires three nonempty splits: {split_counts}.")

    sources = {}
    for file_path in shard_paths:
        with h5py.File(file_path, "r") as handle:
            for group_name, group in handle["paths"].items():
                path_id = f"{file_path.name}:{group_name}"
                metadata = json.loads(group.attrs["metadata_json"])
                sources[path_id] = EvaluationPathSource(
                    path_id=path_id,
                    file_path=file_path,
                    group_name=group_name,
                    split=split_by_id[path_id],
                    metadata=metadata,
                )
    if set(sources) != sample_ids:
        raise ValueError("RVE-RNO evaluation metadata index differs from dataset paths.")
    return samples, sources


def _evaluate_all_splits(
    model: EnergyDissipationRVERNO,
    metadata: RVERNOArtifactMetadata,
    samples: tuple[RNOPathSample, ...],
    sources: dict[str, EvaluationPathSource],
    prediction_path: Path,
    batch_size: int,
    negative_dissipation_tolerance: float,
) -> list[dict[str, Any]]:
    """Roll out every split and write predictions with per-path metrics."""
    stress_scale = np.asarray(
        metadata.normalization["macro_stress"]["rms_scale"], dtype=np.float64
    )
    dissipation_scale = float(
        metadata.normalization["dissipation_density"]["rms_scale"][0]
    )
    records = []
    with h5py.File(prediction_path, "x") as output:
        output.attrs["schema_version"] = "1.0"
        output.attrs["status"] = "writing"
        paths_group = output.create_group("paths")
        for split in ("train", "validation", "test"):
            split_samples = tuple(
                sample
                for sample in samples
                if sources[sample.path_id].split == split
            )
            for start in range(0, len(split_samples), batch_size):
                chunk = split_samples[start : start + batch_size]
                batch = collate_rno_paths(chunk)
                prediction = _rollout_evaluation_batch(model, batch)
                for index, sample in enumerate(chunk):
                    length = sample.time.size
                    source = sources[sample.path_id]
                    arrays = {
                        name: values[index, :length]
                        for name, values in prediction.items()
                    }
                    metrics = compute_path_metrics(
                        sample.macro_stress,
                        arrays["predicted_stress"],
                        sample.dissipation_density,
                        arrays["predicted_dissipation"],
                        sample.time_step,
                        stress_scale,
                        dissipation_scale,
                        arrays["latent_norm"],
                        arrays["dimensionless_state_rate_norm"],
                        arrays["latent_increment_norm"],
                        negative_dissipation_tolerance,
                    )
                    record = {
                        "path_id": sample.path_id,
                        "task_id": source.group_name,
                        "split": split,
                        "family": source.metadata["family"],
                        "regime": source.metadata["expected_response"]["regime"],
                        "amplitude_band": source.metadata.get("amplitude_band"),
                        "rate_band": source.metadata.get("rate_band"),
                        "tensor_direction": source.metadata.get("tensor_direction"),
                        **metrics,
                    }
                    records.append(record)
                    group = paths_group.create_group(source.group_name)
                    group.attrs["path_id"] = sample.path_id
                    group.attrs["split"] = split
                    group.attrs["metadata_json"] = json.dumps(
                        source.metadata, sort_keys=True
                    )
                    group.create_dataset("time", data=sample.time, compression="lzf")
                    group.create_dataset("time_step", data=sample.time_step, compression="lzf")
                    group.create_dataset(
                        "macro_strain", data=sample.macro_strain, compression="lzf"
                    )
                    group.create_dataset(
                        "reference_stress", data=sample.macro_stress, compression="lzf"
                    )
                    group.create_dataset(
                        "reference_dissipation",
                        data=sample.dissipation_density,
                        compression="lzf",
                    )
                    for name, values in arrays.items():
                        group.create_dataset(name, data=values, compression="lzf")
        output.attrs["status"] = "complete"
    return records


def _rollout_evaluation_batch(
    model: EnergyDissipationRVERNO,
    batch: RNOPathBatch,
) -> dict[str, NDArray[np.float64]]:
    """Roll out one padded batch while recording full latent-state histories."""
    strain = batch.network_inputs[..., :6].to(device=model.device, dtype=model.dtype)
    time_step = batch.time_step.to(device=model.device, dtype=model.dtype)
    mask = batch.valid_mask.to(device=model.device)
    latent = model.initial_state_batch(strain.shape[0])
    anchors = model.prepare_energy_anchors()
    chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "predicted_stress",
            "predicted_dissipation",
            "thermodynamic_violation",
            "previous_latent_state",
            "latent_state",
            "latent_norm",
            "dimensionless_state_rate_norm",
            "latent_increment_norm",
        )
    }
    for step in range(strain.shape[1]):
        active = mask[:, step]
        previous = latent
        update = model.update_batch(
            strain[:, step],
            previous,
            torch.where(active, time_step[:, step], torch.ones_like(time_step[:, step])),
            anchors,
        )
        increment = update.latent_state - previous
        latent = torch.where(active.unsqueeze(-1), update.latent_state, previous)
        chunks["predicted_stress"].append(
            torch.where(active.unsqueeze(-1), update.stress, 0.0)
        )
        chunks["predicted_dissipation"].append(
            torch.where(active, update.dissipation, 0.0)
        )
        chunks["thermodynamic_violation"].append(
            torch.where(active, update.thermodynamic_violation, 0.0)
        )
        chunks["previous_latent_state"].append(
            torch.where(active.unsqueeze(-1), previous, 0.0)
        )
        chunks["latent_state"].append(torch.where(active.unsqueeze(-1), latent, 0.0))
        chunks["latent_norm"].append(
            torch.where(active, torch.linalg.vector_norm(latent, dim=-1), 0.0)
        )
        chunks["dimensionless_state_rate_norm"].append(
            torch.where(
                active,
                torch.linalg.vector_norm(update.dimensionless_state_rate, dim=-1),
                0.0,
            )
        )
        chunks["latent_increment_norm"].append(
            torch.where(active, torch.linalg.vector_norm(increment, dim=-1), 0.0)
        )
        latent = latent.detach().requires_grad_(True)
    return {
        name: torch.stack(values, dim=1).detach().cpu().numpy()
        for name, values in chunks.items()
    }


def _compute_model_tangents(
    model: EnergyDissipationRVERNO,
    strain: NDArray[np.float64],
    previous_latent: NDArray[np.float64],
    time_step: float,
    perturbation: float,
    anchors: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute automatic-differentiation and central-difference stress tangents."""
    strain_tensor = torch.as_tensor(strain, dtype=model.dtype, device=model.device).reshape(1, 6)
    latent_tensor = torch.as_tensor(
        previous_latent, dtype=model.dtype, device=model.device
    ).reshape(1, -1)
    dt_tensor = torch.as_tensor([time_step], dtype=model.dtype, device=model.device)
    update = model.update_batch(
        strain_tensor, latent_tensor, dt_tensor, anchors, compute_tangent=True
    )
    if update.tangent is None:
        raise RuntimeError("RVE-RNO automatic-differentiation tangent was not returned.")
    columns = []
    for component in range(6):
        direction = torch.zeros_like(strain_tensor)
        direction[0, component] = perturbation
        plus = model.update_batch(strain_tensor + direction, latent_tensor, dt_tensor, anchors)
        minus = model.update_batch(strain_tensor - direction, latent_tensor, dt_tensor, anchors)
        columns.append((plus.stress[0] - minus.stress[0]) / (2.0 * perturbation))
    finite_difference = torch.stack(columns, dim=1)
    return (
        update.tangent[0].detach().cpu().numpy(),
        finite_difference.detach().cpu().numpy(),
    )


def _relative_frobenius_error(
    values: NDArray[np.float64],
    reference: NDArray[np.float64],
    floor: float,
) -> float:
    """Compute a finite relative Frobenius error with one declared norm floor."""
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference)):
        raise FloatingPointError("RVE-RNO tangent comparison received non-finite values.")
    difference_norm = float(np.linalg.norm(values - reference))
    reference_norm = float(np.linalg.norm(reference))
    return difference_norm / max(reference_norm, floor)


def _norm_quantiles(name: str, values: NDArray[np.float64]) -> dict[str, float]:
    """Compute median, p95, p99, and maximum for one state norm history."""
    return {
        f"{name}_median": float(np.median(values)),
        f"{name}_p95": float(np.quantile(values, 0.95)),
        f"{name}_p99": float(np.quantile(values, 0.99)),
        f"{name}_maximum": float(np.max(values)),
    }


def _summarize_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy and thermodynamic metrics for one path group."""
    names = (
        "stress_normalized_rmse",
        "dissipation_normalized_rmse",
        "cumulative_dissipation_relative_error",
    )
    return {
        "path_count": len(records),
        **{
            name: _summarize_values([float(record[name]) for record in records])
            for name in names
        },
        "negative_dissipation_state_count": sum(
            int(record["negative_dissipation_state_count"]) for record in records
        ),
        "maximum_thermodynamic_violation": max(
            float(record["maximum_thermodynamic_violation"]) for record in records
        ),
    }


def _summarize_values(values: list[float]) -> dict[str, float]:
    """Compute finite median, p95, and maximum statistics."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise FloatingPointError("RVE-RNO metric summary requires finite nonempty values.")
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _write_path_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic per-path metrics as one CSV table."""
    if not records:
        raise ValueError("RVE-RNO path-metric table cannot be empty.")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA only at an explicit timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read_local_cpu_model() -> str:
    """Read the unique Linux CPU model used during local timing."""
    values = {
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        if line.startswith("model name")
    }
    if len(values) != 1:
        raise ValueError("Local timing requires one unambiguous CPU model name.")
    return values.pop()


def _sha256_file(path: Path) -> str:
    """Compute one complete file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
