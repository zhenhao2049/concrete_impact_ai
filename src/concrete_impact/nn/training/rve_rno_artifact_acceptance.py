"""Formal acceptance for a frozen c48 Direct-RNO artifact.

Contents:
    Frozen-test rollout, family errors, dissipation, tangent checks, and macro gate.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from concrete_impact.core.response_manifest import load_response_shard_paths
from concrete_impact.nn.config import load_rve_rno_training_config
from concrete_impact.nn.deployment import load_rve_rno_state_dict
from fem.io.config import load_yaml_config


class StrictArtifactAcceptanceModel(BaseModel):
    """Forbid undeclared formal-acceptance settings."""

    model_config = ConfigDict(extra="forbid")


class DirectRNOArtifactThresholds(StrictArtifactAcceptanceModel):
    """Store all frozen numerical and macroscopic acceptance limits."""

    test_stress_rmse: PositiveFloat
    family_stress_error: PositiveFloat
    maximum_path_stress_error: PositiveFloat
    normalized_dissipation_error: PositiveFloat
    negative_dissipation_tolerance: float = Field(ge=0.0)
    tangent_median_error: PositiveFloat
    tangent_p95_error: PositiveFloat
    macro_relative_error: PositiveFloat


class DirectRNOArtifactAcceptanceConfig(StrictArtifactAcceptanceModel):
    """Define one strict post-training artifact acceptance request."""

    schema_version: str
    training_config: Path
    checkpoint: Path
    artifact_metadata: Path
    training_data_acceptance: Path
    macro_embedding_report: Path
    output_path: Path
    tangent_paths: PositiveInt
    tangent_states_per_path: PositiveInt
    finite_difference_perturbation: PositiveFloat
    thresholds: DirectRNOArtifactThresholds


def load_direct_rno_artifact_acceptance_config(
    path: str | Path,
) -> DirectRNOArtifactAcceptanceConfig:
    """Load one strict Direct-RNO artifact-acceptance configuration."""
    return DirectRNOArtifactAcceptanceConfig.model_validate(load_yaml_config(path))


def validate_direct_rno_artifact(
    config: DirectRNOArtifactAcceptanceConfig,
) -> dict[str, Any]:
    """Evaluate frozen test paths and require every formal artifact gate."""
    training = load_rve_rno_training_config(config.training_config)
    acceptance = json.loads(config.training_data_acceptance.read_text(encoding="utf-8"))
    if not bool(acceptance["passed"]):
        raise ValueError("Direct-RNO artifact cannot use unaccepted training data.")
    model = load_rve_rno_state_dict(
        training.model,
        config.checkpoint,
        config.artifact_metadata,
    )
    model.eval()
    path_lookup = _response_path_lookup(training.response_shard_manifest)
    test_records = tuple(
        item for item in acceptance["split_manifest"]["paths"] if item["split"] == "test"
    )
    stress_scale = np.asarray(
        acceptance["normalization"]["macro_stress"]["rms_scale"], dtype=np.float64
    )
    dissipation_scale = float(acceptance["normalization"]["dissipation_density"]["rms_scale"][0])
    path_errors: list[float] = []
    family_errors: dict[str, list[float]] = {}
    dissipation_numerator = 0.0
    dissipation_count = 0
    negative_dissipation_count = 0
    tangent_rve_errors: list[float] = []
    tangent_fd_errors: list[float] = []
    for record_index, record in enumerate(test_records):
        file_path, path_name = path_lookup[str(record["path_id"])]
        with h5py.File(file_path, "r") as handle:
            group = handle[f"paths/{path_name}"]
            strain = group["macro/strain"][...]
            time_step = group["macro/time_step"][...]
            target_stress = group["macro/stress"][...]
            target_dissipation = group["macro/dissipation_density"][...]
            target_tangent = group["macro/effective_tangent"][...]
        predicted = _rollout_artifact_path(
            model,
            strain,
            time_step,
            target_tangent,
            config,
            collect_tangents=record_index < config.tangent_paths,
        )
        stress_difference = predicted["stress"] - target_stress
        numerator = float(np.sum(stress_difference**2))
        denominator = max(
            float(np.sum(target_stress**2)),
            strain.shape[0] * float(np.sum(stress_scale**2)),
        )
        path_error = float(np.sqrt(numerator / denominator))
        path_errors.append(path_error)
        family_errors.setdefault(str(record["family"]), []).append(path_error)
        dissipation_difference = (predicted["dissipation"] - target_dissipation) / dissipation_scale
        dissipation_numerator += float(np.sum(dissipation_difference**2))
        dissipation_count += int(dissipation_difference.size)
        negative_dissipation_count += int(
            np.count_nonzero(
                predicted["dissipation"] < -config.thresholds.negative_dissipation_tolerance
            )
        )
        tangent_rve_errors.extend(predicted["tangent_rve_errors"])
        tangent_fd_errors.extend(predicted["tangent_fd_errors"])
    test_stress_rmse = float(np.sqrt(np.mean(np.square(path_errors))))
    family_maximum = {
        family: float(np.max(values)) for family, values in sorted(family_errors.items())
    }
    normalized_dissipation_error = float(np.sqrt(dissipation_numerator / dissipation_count))
    tangent_rve = _quantile_summary(tangent_rve_errors)
    tangent_fd = _quantile_summary(tangent_fd_errors)
    macro = json.loads(config.macro_embedding_report.read_text(encoding="utf-8"))
    macro_metrics = {
        name: float(macro[name])
        for name in (
            "displacement_error",
            "wave_arrival_error",
            "energy_error",
            "dissipation_error",
        )
    }
    checks: dict[str, dict[str, Any]] = {
        "test_stress_rmse": {
            "value": test_stress_rmse,
            "limit": config.thresholds.test_stress_rmse,
            "passed": test_stress_rmse <= config.thresholds.test_stress_rmse,
        },
        "family_stress_error": {
            "values": family_maximum,
            "limit": config.thresholds.family_stress_error,
            "passed": all(
                value <= config.thresholds.family_stress_error for value in family_maximum.values()
            ),
        },
        "maximum_path_stress_error": {
            "value": float(np.max(path_errors)),
            "limit": config.thresholds.maximum_path_stress_error,
            "passed": max(path_errors) <= config.thresholds.maximum_path_stress_error,
        },
        "normalized_dissipation_error": {
            "value": normalized_dissipation_error,
            "limit": config.thresholds.normalized_dissipation_error,
            "passed": normalized_dissipation_error
            <= config.thresholds.normalized_dissipation_error,
        },
        "negative_dissipation": {
            "count": negative_dissipation_count,
            "tolerance": config.thresholds.negative_dissipation_tolerance,
            "passed": negative_dissipation_count == 0,
        },
        "rve_tangent_error": {
            **tangent_rve,
            "median_limit": config.thresholds.tangent_median_error,
            "p95_limit": config.thresholds.tangent_p95_error,
            "passed": tangent_rve["median"] <= config.thresholds.tangent_median_error
            and tangent_rve["p95"] <= config.thresholds.tangent_p95_error,
        },
        "finite_difference_tangent_error": {
            **tangent_fd,
            "median_limit": config.thresholds.tangent_median_error,
            "p95_limit": config.thresholds.tangent_p95_error,
            "passed": tangent_fd["median"] <= config.thresholds.tangent_median_error
            and tangent_fd["p95"] <= config.thresholds.tangent_p95_error,
        },
        "macro_embedding": {
            "values": macro_metrics,
            "limit": config.thresholds.macro_relative_error,
            "passed": bool(macro["passed"])
            and all(
                value <= config.thresholds.macro_relative_error for value in macro_metrics.values()
            ),
        },
    }
    report = {
        "schema_version": config.schema_version,
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "frozen_test_path_count": len(test_records),
        "tangent_path_count": min(len(test_records), config.tangent_paths),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if report["passed"]:
        files = {
            "checkpoint": config.checkpoint,
            "metadata": config.artifact_metadata,
            "rve_data_acceptance_report": config.training_data_acceptance,
            "artifact_acceptance_report": config.output_path,
            "training_config": config.training_config,
            "macro_embedding_report": config.macro_embedding_report,
        }
        accepted_manifest = {
            "schema_version": "1.0",
            "passed": True,
            "stress_update": training.model.integrator,
            "evolution": training.model.evolution,
            "model": training.model.model_dump(mode="json"),
            "bindings": {
                "model_structure_sha256": _sha256_payload(training.model.model_dump(mode="json")),
                "data_split_sha256": str(acceptance["split_manifest_sha256"]),
                "macro_embedding_report_sha256": _sha256_file(config.macro_embedding_report),
            },
            "files": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in files.items()
            },
        }
        config.output_path.with_name("accepted_artifact_manifest.json").write_text(
            json.dumps(accepted_manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    return report


def _rollout_artifact_path(
    model,
    strain: np.ndarray,
    time_step: np.ndarray,
    target_tangent: np.ndarray,
    config: DirectRNOArtifactAcceptanceConfig,
    collect_tangents: bool,
) -> dict[str, Any]:
    """Roll out one path and compare selected AD, RVE, and centered-FD tangents."""
    dtype = model.dtype
    device = model.device
    latent = model.initial_state_batch(1)
    stress: list[np.ndarray] = []
    dissipation: list[float] = []
    tangent_rve_errors: list[float] = []
    tangent_fd_errors: list[float] = []
    selected = set(
        np.linspace(
            0,
            strain.shape[0] - 1,
            config.tangent_states_per_path,
            dtype=np.int64,
        ).tolist()
        if collect_tangents
        else ()
    )
    for step_id in range(strain.shape[0]):
        strain_tensor = torch.as_tensor(strain[step_id : step_id + 1], dtype=dtype, device=device)
        dt_tensor = torch.as_tensor(time_step[step_id : step_id + 1], dtype=dtype, device=device)
        update = model.update_batch(
            strain_tensor,
            latent,
            dt_tensor,
            model.prepare_energy_anchors(),
            compute_tangent=step_id in selected,
        )
        stress.append(update.stress[0].detach().cpu().numpy())
        dissipation.append(float(update.dissipation[0].detach().cpu()))
        if step_id in selected:
            if update.tangent is None:
                raise ValueError("Direct-RNO tangent collection returned no AD tangent.")
            ad_tangent = update.tangent[0].detach().cpu().numpy()
            rve_tangent = target_tangent[step_id]
            fd_tangent = _centered_model_tangent(
                model,
                strain[step_id],
                latent.detach(),
                float(time_step[step_id]),
                config.finite_difference_perturbation,
            )
            tangent_rve_errors.append(_matrix_relative_error(ad_tangent, rve_tangent))
            tangent_fd_errors.append(_matrix_relative_error(ad_tangent, fd_tangent))
        latent = update.latent_state.detach().requires_grad_(True)
    return {
        "stress": np.stack(stress),
        "dissipation": np.asarray(dissipation),
        "tangent_rve_errors": tangent_rve_errors,
        "tangent_fd_errors": tangent_fd_errors,
    }


def _centered_model_tangent(
    model,
    strain: np.ndarray,
    latent: torch.Tensor,
    time_step: float,
    perturbation: float,
) -> np.ndarray:
    """Compute all six centered stress JVP columns at fixed committed state."""
    columns = []
    for component in range(6):
        direction = np.zeros(6, dtype=np.float64)
        direction[component] = perturbation
        stresses = []
        for sign in (1.0, -1.0):
            values = torch.as_tensor(
                (strain + sign * direction)[None, :],
                dtype=model.dtype,
                device=model.device,
            )
            response = model.update_batch(
                values,
                latent.detach().requires_grad_(True),
                torch.as_tensor([time_step], dtype=model.dtype, device=model.device),
                model.prepare_energy_anchors(),
            )
            stresses.append(response.stress[0].detach().cpu().numpy())
        columns.append((stresses[0] - stresses[1]) / (2.0 * perturbation))
    return np.stack(columns, axis=1)


def _matrix_relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    """Compute a strict relative Frobenius error."""
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        raise ValueError("Direct-RNO tangent reference has zero norm.")
    return float(np.linalg.norm(values - reference) / denominator)


def _quantile_summary(values: list[float]) -> dict[str, float | int]:
    """Summarize a required nonempty tangent error population."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Direct-RNO tangent acceptance requires finite samples.")
    return {
        "sample_count": int(array.size),
        "median": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _response_path_lookup(manifest_path: Path) -> dict[str, tuple[Path, str]]:
    """Map accepted path ids to exact HDF5 groups."""
    result = {}
    for path in load_response_shard_paths(manifest_path):
        with h5py.File(path, "r") as handle:
            for path_name in handle["paths"]:
                result[f"{path.name}:{path_name}"] = (path, path_name)
    return result


def _sha256_file(path: Path) -> str:
    """Compute one complete file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: object) -> str:
    """Compute one canonical JSON payload digest."""
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(serialized).hexdigest()
