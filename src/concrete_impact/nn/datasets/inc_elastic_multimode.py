"""Data preparation for the fixed-system elastic multimode INC.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import eigh

from concrete_impact.experiments.inc_elastic_generation import (
    ElasticINCSystem,
    load_elastic_inc_system,
)
from concrete_impact.nn.config_inc_elastic import INCElasticMultimodeConfig

INC_ELASTIC_CACHE_SCHEMA = "2.0"


@dataclass(frozen=True)
class INCElasticNormalization:
    """Store training-only scales for features, forces, states, and energy."""

    duration_mean: float
    duration_scale: float
    residual_start_scale: NDArray[np.float64]
    residual_end_scale: NDArray[np.float64]
    displacement_scale: float
    velocity_scale: float
    energy_scale: float

    def to_mapping(self) -> dict[str, Any]:
        """Convert the normalization to JSON-compatible values."""
        return {
            "duration_mean": self.duration_mean,
            "duration_scale": self.duration_scale,
            "residual_start_scale": self.residual_start_scale.tolist(),
            "residual_end_scale": self.residual_end_scale.tolist(),
            "displacement_scale": self.displacement_scale,
            "velocity_scale": self.velocity_scale,
            "energy_scale": self.energy_scale,
        }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> INCElasticNormalization:
        """Restore strict floating-point normalization values."""
        return cls(
            duration_mean=float(values["duration_mean"]),
            duration_scale=float(values["duration_scale"]),
            residual_start_scale=np.asarray(values["residual_start_scale"], dtype=np.float64),
            residual_end_scale=np.asarray(values["residual_end_scale"], dtype=np.float64),
            displacement_scale=float(values["displacement_scale"]),
            velocity_scale=float(values["velocity_scale"]),
            energy_scale=float(values["energy_scale"]),
        )


@dataclass(frozen=True)
class INCElasticPreparedData:
    """Store the fixed system, modal phases, and training normalization."""

    system: ElasticINCSystem
    angular_frequencies: NDArray[np.float64]
    normalization: INCElasticNormalization
    final_time: float


def elastic_inc_input_width(modal_count: int, include_duration_modal_phase: bool = False) -> int:
    """Return the reviewed time, loading, and modal-phase feature width."""
    phase_blocks = 4 if include_duration_modal_phase else 2
    return 8 + phase_blocks * modal_count


def elastic_inc_feature_layout(
    modal_count: int,
    include_duration_modal_phase: bool = False,
) -> dict[str, Any]:
    """Describe the exact feature packing order without listing every mode."""
    layout = {
        "scalar_order": [
            "normalized_time",
            "normalized_duration",
            "load_coefficient_1",
            "load_coefficient_2",
            "load_coefficient_3",
            "load_coefficient_4",
            "pulse_value_start",
            "pulse_value_end",
        ],
        "modal_sine": {"start": 8, "count": modal_count},
        "modal_cosine": {"start": 8 + modal_count, "count": modal_count},
    }
    if include_duration_modal_phase:
        layout["duration_modal_sine"] = {"start": 8 + 2 * modal_count, "count": modal_count}
        layout["duration_modal_cosine"] = {
            "start": 8 + 3 * modal_count,
            "count": modal_count,
        }
    return layout


def validate_inc_elastic_sources(
    config: INCElasticMultimodeConfig,
    response_splits: tuple[str, ...] = ("train", "validation"),
) -> None:
    """Validate accepted data, fixed-system dimensions, and selected response files."""
    acceptance = _read_json(config.data.acceptance_path)
    if acceptance["passed"] is not True or acceptance["status"]["succeeded"] != 144:
        raise ValueError("The official elastic INC source data did not pass acceptance.")
    if acceptance["status"]["failed"] != 0:
        raise ValueError("The official elastic INC source data contains failed paths.")

    manifest = _read_json(config.data.manifest_path)
    if manifest["task_count"] != 144 or manifest["plan"]["design"] != "p1_official_144":
        raise ValueError("Elastic INC manifest is not the accepted 144-path design.")
    system = load_elastic_inc_system(config.data.system_path)
    if system.free_dofs.size != config.model.free_dof_count:
        raise ValueError("Elastic INC free-DOF count differs from the model configuration.")
    if system.mass.shape != (config.model.free_dof_count,):
        raise ValueError("Elastic INC lumped mass has an incompatible shape.")

    split_paths = {
        "train": config.data.train_paths,
        "validation": config.data.validation_paths,
        "test": config.data.test_paths,
    }
    for split, path_ids in split_paths.items():
        expected_source_split = (
            split if config.data.source_split_policy == "locked_manifest" else "train"
        )
        for path_id in path_ids:
            task = manifest["tasks"][path_id]
            if task["split"] != expected_source_split:
                raise ValueError(
                    f"INC path {path_id} does not belong to source split={expected_source_split}."
                )
            if task["pulse_kind"] != "half_sine":
                raise ValueError(f"INC path {path_id} is not a half-sine path.")
            if split not in response_splits:
                continue
            response_path = _response_path(config, path_id)
            with h5py.File(response_path, "r") as handle:
                if handle.attrs["schema_version"] != "2.0":
                    raise ValueError(f"Unsupported elastic INC response: {response_path}.")
                if handle.attrs["status"] != "complete":
                    raise ValueError(f"Incomplete elastic INC response: {response_path}.")
                if str(handle.attrs["split"]) != expected_source_split:
                    raise ValueError(f"Response split differs from manifest: {path_id}.")
                if handle["target/residual_force_start"].shape[1] != system.free_dofs.size:
                    raise ValueError(f"Response free-DOF width differs from system: {path_id}.")


def prepare_inc_elastic_training_cache(
    config: INCElasticMultimodeConfig,
) -> INCElasticPreparedData:
    """Prepare or strictly reuse train and validation tensors without opening test paths."""
    validate_inc_elastic_sources(config)
    system = load_elastic_inc_system(config.data.system_path)
    frequencies = compute_inc_elastic_angular_frequencies(system)
    normalization, final_time = compute_inc_elastic_normalization(config, system)
    prepared = INCElasticPreparedData(system, frequencies, normalization, final_time)
    config.data.cache_directory.mkdir(parents=True, exist_ok=True)
    _prepare_split_cache(config, prepared, "train", config.data.train_paths)
    _prepare_split_cache(config, prepared, "validation", config.data.validation_paths)
    return prepared


def compute_inc_elastic_angular_frequencies(
    system: ElasticINCSystem,
) -> NDArray[np.float64]:
    """Compute every constrained generalized eigenfrequency in radians per second."""
    eigenvalues = eigh(
        system.stiffness.toarray(),
        np.diag(system.mass),
        eigvals_only=True,
        check_finite=True,
    )
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Elastic INC fixed system is not positive definite.")
    frequencies = np.sqrt(eigenvalues)
    if not np.all(np.isfinite(frequencies)):
        raise FloatingPointError("Elastic INC modal frequencies contain non-finite values.")
    return np.asarray(frequencies, dtype=np.float64)


def compute_inc_elastic_normalization(
    config: INCElasticMultimodeConfig,
    system: ElasticINCSystem,
) -> tuple[INCElasticNormalization, float]:
    """Compute all normalization values from training paths only."""
    mass = system.mass
    mass_sum = float(np.sum(mass))
    stride = config.data.sample_stride
    sample_count = 0
    start_dual_sum = 0.0
    end_dual_sum = 0.0
    displacement_sum = 0.0
    velocity_sum = 0.0
    energy_sum = 0.0
    durations = []
    final_times = []

    for path_id in config.data.train_paths:
        with h5py.File(_response_path(config, path_id), "r") as handle:
            amplitude = float(handle.attrs["pressure_amplitude"])
            duration = float(handle.attrs["pulse_duration"])
            indices = np.arange(0, handle["time"].shape[0] - 1, stride, dtype=np.int64)
            residual_start = handle["target/residual_force_start"][indices] / amplitude
            residual_end = handle["target/residual_force_end"][indices] / amplitude
            reference_displacement = handle["state/displacement"][indices + 1] / amplitude
            reference_velocity = handle["state/velocity"][indices + 1] / amplitude
            reference_energy = handle["reference/mechanical_energy"][indices + 1] / amplitude**2

            zero_residual_displacement_error = (
                0.5 * system.time_step**2 * residual_start / mass[None, :]
            )
            stiffness_error = np.asarray(
                system.stiffness @ zero_residual_displacement_error.T
            ).T
            zero_residual_velocity_error = 0.5 * system.time_step * (
                residual_start / mass[None, :]
                - stiffness_error / mass[None, :]
                + residual_end / mass[None, :]
            )
            zero_residual_displacement = (
                reference_displacement + zero_residual_displacement_error
            )
            zero_residual_velocity = reference_velocity + zero_residual_velocity_error
            stiffness_displacement = np.asarray(
                system.stiffness @ zero_residual_displacement.T
            ).T
            zero_residual_energy = 0.5 * (
                np.sum(mass[None, :] * zero_residual_velocity**2, axis=1)
                + np.sum(zero_residual_displacement * stiffness_displacement, axis=1)
            )

            start_dual_sum += float(np.sum(residual_start**2 / mass[None, :]))
            end_dual_sum += float(np.sum(residual_end**2 / mass[None, :]))
            displacement_sum += float(
                np.sum(zero_residual_displacement_error**2 * mass[None, :])
            )
            velocity_sum += float(np.sum(zero_residual_velocity_error**2 * mass[None, :]))
            energy_sum += float(np.sum((zero_residual_energy - reference_energy) ** 2))
            sample_count += indices.size
            durations.append(duration)
            final_times.append(float(handle["time"][-1]))

    duration_mean = float(np.mean(durations))
    duration_scale = float(np.sqrt(np.mean((np.asarray(durations) - duration_mean) ** 2)))
    dof_count = system.free_dofs.size
    start_scalar = np.sqrt(start_dual_sum / (sample_count * dof_count))
    end_scalar = np.sqrt(end_dual_sum / (sample_count * dof_count))
    normalization = INCElasticNormalization(
        duration_mean=duration_mean,
        duration_scale=duration_scale,
        residual_start_scale=np.sqrt(mass) * start_scalar,
        residual_end_scale=np.sqrt(mass) * end_scalar,
        displacement_scale=float(np.sqrt(displacement_sum / (sample_count * mass_sum))),
        velocity_scale=float(np.sqrt(velocity_sum / (sample_count * mass_sum))),
        energy_scale=float(np.sqrt(energy_sum / sample_count)),
    )
    scalar_values = np.asarray(
        [
            normalization.duration_scale,
            start_scalar,
            end_scalar,
            normalization.displacement_scale,
            normalization.velocity_scale,
            normalization.energy_scale,
        ]
    )
    if np.any(scalar_values <= 0.0) or not np.all(np.isfinite(scalar_values)):
        raise ValueError("Elastic INC training normalization is degenerate.")
    if len(set(final_times)) != 1:
        raise ValueError("Elastic INC selected paths do not share one final time.")
    return normalization, final_times[0]


def build_inc_elastic_features(
    times: NDArray[np.float64],
    time_step: float,
    pulse_duration: float,
    spatial_coefficients: NDArray[np.float64],
    angular_frequencies: NDArray[np.float64],
    normalization: INCElasticNormalization,
    final_time: float,
    include_duration_modal_phase: bool = False,
) -> NDArray[np.float32]:
    """Pack state-independent time, loading, and modal phase features."""
    time_values = np.asarray(times, dtype=np.float64)
    pulse_start = _half_sine_values(time_values, pulse_duration)
    pulse_end = _half_sine_values(time_values + time_step, pulse_duration)
    modal_arguments = time_values[:, None] * angular_frequencies[None, :]
    scalar = np.column_stack(
        (
            time_values / final_time,
            np.full(
                time_values.size,
                (pulse_duration - normalization.duration_mean) / normalization.duration_scale,
            ),
            np.broadcast_to(spatial_coefficients, (time_values.size, 4)),
            pulse_start,
            pulse_end,
        )
    )
    feature_blocks = [scalar, np.sin(modal_arguments), np.cos(modal_arguments)]
    if include_duration_modal_phase:
        duration_arguments = pulse_duration * angular_frequencies
        feature_blocks.extend(
            (
                np.broadcast_to(np.sin(duration_arguments), modal_arguments.shape),
                np.broadcast_to(np.cos(duration_arguments), modal_arguments.shape),
            )
        )
    features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("Elastic INC features contain non-finite values.")
    return features


def load_inc_elastic_cache(
    config: INCElasticMultimodeConfig,
    split: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load one prepared tensor cache after exact metadata comparison."""
    tensor_path, metadata_path = _cache_paths(config, split)
    metadata = _read_json(metadata_path)
    expected = _cache_metadata(
        config,
        split,
        getattr(config.data, f"{split}_paths"),
        INCElasticNormalization.from_mapping(metadata["normalization"]),
        np.asarray(metadata["angular_frequencies"], dtype=np.float64),
        float(metadata["final_time"]),
    )
    if metadata != expected:
        raise ValueError(f"Elastic INC {split} cache metadata does not match current inputs.")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if set(tensors) != {
        "features",
        "target_start",
        "target_end",
        "reference_displacement_next",
        "reference_velocity_next",
        "reference_energy_next",
    }:
        raise ValueError(f"Elastic INC {split} cache has unexpected tensor fields.")
    return tensors, metadata


def _prepare_split_cache(
    config: INCElasticMultimodeConfig,
    prepared: INCElasticPreparedData,
    split: str,
    path_ids: tuple[str, ...],
) -> None:
    """Build one immutable split tensor cache."""
    tensor_path, metadata_path = _cache_paths(config, split)
    expected = _cache_metadata(
        config,
        split,
        path_ids,
        prepared.normalization,
        prepared.angular_frequencies,
        prepared.final_time,
    )
    if tensor_path.exists() or metadata_path.exists():
        if not tensor_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f"Incomplete elastic INC {split} cache requires diagnosis.")
        if _read_json(metadata_path) != expected:
            raise ValueError(f"Existing elastic INC {split} cache does not match current inputs.")
        return

    stride = config.data.sample_stride
    sample_counts = []
    for path_id in path_ids:
        with h5py.File(_response_path(config, path_id), "r") as handle:
            sample_counts.append(len(range(0, handle["time"].shape[0] - 1, stride)))
    sample_count = sum(sample_counts)
    input_width = elastic_inc_input_width(
        config.features.modal_count,
        config.features.include_duration_modal_phase,
    )
    free_count = config.model.free_dof_count
    tensors = {
        "features": torch.empty((sample_count, input_width), dtype=torch.float32),
        "target_start": torch.empty((sample_count, free_count), dtype=torch.float32),
        "target_end": torch.empty((sample_count, free_count), dtype=torch.float32),
        "reference_displacement_next": torch.empty(
            (sample_count, free_count), dtype=torch.float64
        ),
        "reference_velocity_next": torch.empty((sample_count, free_count), dtype=torch.float64),
        "reference_energy_next": torch.empty((sample_count,), dtype=torch.float64),
    }

    offset = 0
    for path_id, count in zip(path_ids, sample_counts, strict=True):
        with h5py.File(_response_path(config, path_id), "r") as handle:
            amplitude = float(handle.attrs["pressure_amplitude"])
            duration = float(handle.attrs["pulse_duration"])
            indices = np.arange(0, handle["time"].shape[0] - 1, stride, dtype=np.int64)
            selection = slice(offset, offset + count)
            features = build_inc_elastic_features(
                handle["time"][indices],
                prepared.system.time_step,
                duration,
                handle["spatial_coefficients"][...],
                prepared.angular_frequencies,
                prepared.normalization,
                prepared.final_time,
                config.features.include_duration_modal_phase,
            )
            target_start = (
                handle["target/residual_force_start"][indices]
                / amplitude
                / prepared.normalization.residual_start_scale[None, :]
            )
            target_end = (
                handle["target/residual_force_end"][indices]
                / amplitude
                / prepared.normalization.residual_end_scale[None, :]
            )
            tensors["features"][selection] = torch.from_numpy(features)
            tensors["target_start"][selection] = torch.from_numpy(
                target_start.astype(np.float32)
            )
            tensors["target_end"][selection] = torch.from_numpy(
                target_end.astype(np.float32)
            )
            tensors["reference_displacement_next"][selection] = torch.from_numpy(
                handle["state/displacement"][indices + 1] / amplitude
            )
            tensors["reference_velocity_next"][selection] = torch.from_numpy(
                handle["state/velocity"][indices + 1] / amplitude
            )
            tensors["reference_energy_next"][selection] = torch.from_numpy(
                handle["reference/mechanical_energy"][indices + 1] / amplitude**2
            )
            offset += count

    partial_tensor = tensor_path.with_suffix(".pt.partial")
    partial_metadata = metadata_path.with_suffix(".json.partial")
    torch.save(tensors, partial_tensor)
    partial_metadata.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial_tensor.replace(tensor_path)
    partial_metadata.replace(metadata_path)


def _cache_metadata(
    config: INCElasticMultimodeConfig,
    split: str,
    path_ids: tuple[str, ...],
    normalization: INCElasticNormalization,
    angular_frequencies: NDArray[np.float64],
    final_time: float,
) -> dict[str, Any]:
    """Build exact cache provenance from explicit source metadata."""
    return {
        "schema_version": INC_ELASTIC_CACHE_SCHEMA,
        "split": split,
        "path_ids": list(path_ids),
        "sample_stride": config.data.sample_stride,
        "input_width": elastic_inc_input_width(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
        "free_dof_count": config.model.free_dof_count,
        "feature_layout": elastic_inc_feature_layout(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
        "tensor_dtypes": {
            "features": "float32",
            "target_start": "float32",
            "target_end": "float32",
            "reference_displacement_next": "float64",
            "reference_velocity_next": "float64",
            "reference_energy_next": "float64",
        },
        "normalization": normalization.to_mapping(),
        "angular_frequencies": angular_frequencies.tolist(),
        "final_time": final_time,
        "sources": [_source_descriptor(_response_path(config, path_id)) for path_id in path_ids],
    }


def _source_descriptor(path: Path) -> dict[str, Any]:
    """Describe one source by path, size, and modification time."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_paths(config: INCElasticMultimodeConfig, split: str) -> tuple[Path, Path]:
    """Return immutable tensor and metadata cache paths."""
    return (
        config.data.cache_directory / f"{split}.pt",
        config.data.cache_directory / f"{split}.json",
    )


def _response_path(config: INCElasticMultimodeConfig, path_id: str) -> Path:
    """Return one official sharded response path."""
    return config.data.root / "tasks" / path_id / "response.h5"


def _half_sine_values(times: NDArray[np.float64], duration: float) -> NDArray[np.float64]:
    """Evaluate the compactly supported half-sine pulse."""
    normalized = times / duration
    inside = (normalized >= 0.0) & (normalized <= 1.0)
    values = np.zeros_like(normalized)
    values[inside] = np.sin(np.pi * normalized[inside])
    return values


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON mapping."""
    return json.loads(path.read_text(encoding="utf-8"))
