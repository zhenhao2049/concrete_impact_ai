"""HDF5 path data and normalization for fixed-system velocity-Verlet INC.

Contents:
    Complete path records, strict HDF5 loading, feature packing, and training statistics.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

INC_DATA_SCHEMA_VERSION = "1.0"
INC_INPUT_FIELD_ORDER = (
    "displacement",
    "velocity",
    "baseline_acceleration",
    "internal_force_start",
    "external_force_start",
    "external_force_end",
    "time",
    "control_parameters",
)


@dataclass(frozen=True)
class VelocityVerletINCPath:
    """Store one complete accepted INC reference path on coarse time nodes."""

    path_id: str
    split: str
    time: NDArray[np.float64]
    control_parameters: NDArray[np.float64]
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    baseline_acceleration: NDArray[np.float64]
    internal_force: NDArray[np.float64]
    external_force: NDArray[np.float64]
    residual_force_start: NDArray[np.float64]
    residual_force_end: NDArray[np.float64]
    mechanical_energy: NDArray[np.float64]


@dataclass(frozen=True)
class VelocityVerletINCSystemData:
    """Store fixed free-DOF operators and artifact compatibility fields."""

    free_dofs: NDArray[np.int64]
    mass_lumped_free: NDArray[np.float64]
    stiffness_free: NDArray[np.float64]
    control_parameter_order: tuple[str, ...]
    time_step: float
    cfl_ratio: float
    mesh_sha256: str
    lumped_mass_sha256: str
    material_config_sha256: str
    boundary_config_sha256: str
    free_dof_order_sha256: str


@dataclass(frozen=True)
class VelocityVerletINCNormalization:
    """Store scalar input statistics and mass-consistent output scales."""

    statistics: dict[str, dict[str, NDArray[np.float64]]]

    def to_json_mapping(self) -> dict[str, dict[str, list[float]]]:
        """Convert every statistic into strict JSON-compatible lists."""
        return {
            field: {name: values.tolist() for name, values in field_statistics.items()}
            for field, field_statistics in self.statistics.items()
        }


def load_velocity_verlet_inc_dataset(
    path: str | Path,
) -> tuple[VelocityVerletINCSystemData, tuple[VelocityVerletINCPath, ...]]:
    """Load one complete fixed-system INC HDF5 dataset."""
    dataset_path = Path(path)
    with h5py.File(dataset_path, "r") as handle:
        if handle.attrs["schema_version"] != INC_DATA_SCHEMA_VERSION:
            raise ValueError(f"Unsupported INC dataset schema: {dataset_path}.")
        if handle.attrs["status"] != "complete":
            raise ValueError(f"INC dataset is not complete: {dataset_path}.")
        system_group = handle["system"]
        system = VelocityVerletINCSystemData(
            free_dofs=system_group["free_dofs"][...],
            mass_lumped_free=system_group["mass_lumped_free"][...],
            stiffness_free=system_group["stiffness_free"][...],
            control_parameter_order=tuple(
                value.decode() for value in system_group["control_parameter_order"][...]
            ),
            time_step=float(system_group.attrs["time_step"]),
            cfl_ratio=float(system_group.attrs["cfl_ratio"]),
            mesh_sha256=str(system_group.attrs["mesh_sha256"]),
            lumped_mass_sha256=str(system_group.attrs["lumped_mass_sha256"]),
            material_config_sha256=str(system_group.attrs["material_config_sha256"]),
            boundary_config_sha256=str(system_group.attrs["boundary_config_sha256"]),
            free_dof_order_sha256=str(system_group.attrs["free_dof_order_sha256"]),
        )
        paths = tuple(
            _read_path(path_id, handle[f"paths/{path_id}"])
            for path_id in sorted(handle["paths"])
        )

    _validate_loaded_dataset(system, paths)
    return system, paths


def select_inc_split(
    paths: tuple[VelocityVerletINCPath, ...],
    split: str,
) -> tuple[VelocityVerletINCPath, ...]:
    """Select complete paths from exactly one declared split."""
    selected = tuple(path for path in paths if path.split == split)
    if not selected:
        raise ValueError(f"INC dataset has no paths for split={split}.")
    return selected


def compute_inc_normalization(
    training_paths: tuple[VelocityVerletINCPath, ...],
    mass_lumped_free: NDArray[np.float64],
) -> VelocityVerletINCNormalization:
    """Compute training-only scalar input and mass-consistent output scales."""
    field_values = {
        "displacement": np.concatenate(
            [path.displacement[:-1].reshape(-1) for path in training_paths]
        ),
        "velocity": np.concatenate([path.velocity[:-1].reshape(-1) for path in training_paths]),
        "baseline_acceleration": np.concatenate(
            [path.baseline_acceleration[:-1].reshape(-1) for path in training_paths]
        ),
        "internal_force_start": np.concatenate(
            [path.internal_force[:-1].reshape(-1) for path in training_paths]
        ),
        "external_force_start": np.concatenate(
            [path.external_force[:-1].reshape(-1) for path in training_paths]
        ),
        "external_force_end": np.concatenate(
            [path.external_force[1:].reshape(-1) for path in training_paths]
        ),
        "time": np.concatenate([path.time[:-1] for path in training_paths]),
    }
    statistics: dict[str, dict[str, NDArray[np.float64]]] = {}
    for name, values in field_values.items():
        mean = np.asarray([np.mean(values)], dtype=np.float64)
        scale = np.asarray([np.sqrt(np.mean((values - mean[0]) ** 2))], dtype=np.float64)
        if scale[0] == 0.0:
            raise ValueError(f"INC input field has zero training scale: {name}.")
        statistics[name] = {"mean": mean, "scale": scale}
    controls = np.vstack([path.control_parameters for path in training_paths])
    control_mean = np.mean(controls, axis=0)
    control_scale = np.sqrt(np.mean((controls - control_mean) ** 2, axis=0))
    if np.any(control_scale == 0.0):
        raise ValueError("INC control parameters must all vary in the training split.")
    statistics["control_parameters"] = {
        "mean": control_mean,
        "scale": control_scale,
    }
    residual_start = np.concatenate(
        [path.residual_force_start for path in training_paths], axis=0
    )
    residual_end = np.concatenate([path.residual_force_end for path in training_paths], axis=0)
    start_dual_scale = np.sqrt(
        np.mean(np.sum(residual_start**2 / mass_lumped_free[None, :], axis=1))
    )
    end_dual_scale = np.sqrt(
        np.mean(np.sum(residual_end**2 / mass_lumped_free[None, :], axis=1))
    )
    if start_dual_scale == 0.0 or end_dual_scale == 0.0:
        raise ValueError("INC residual-force labels require nonzero stage scales.")
    statistics["residual_force_start"] = {
        "mean": np.zeros(mass_lumped_free.size, dtype=np.float64),
        "scale": np.sqrt(mass_lumped_free) * start_dual_scale,
    }
    statistics["residual_force_end"] = {
        "mean": np.zeros(mass_lumped_free.size, dtype=np.float64),
        "scale": np.sqrt(mass_lumped_free) * end_dual_scale,
    }

    return VelocityVerletINCNormalization(statistics)


def pack_inc_step_features(
    path: VelocityVerletINCPath,
    step_id: int,
    normalization: VelocityVerletINCNormalization,
) -> NDArray[np.float64]:
    """Pack one teacher-forced step in the fixed semantic field order."""
    values = {
        "displacement": path.displacement[step_id],
        "velocity": path.velocity[step_id],
        "baseline_acceleration": path.baseline_acceleration[step_id],
        "internal_force_start": path.internal_force[step_id],
        "external_force_start": path.external_force[step_id],
        "external_force_end": path.external_force[step_id + 1],
        "time": np.asarray([path.time[step_id]], dtype=np.float64),
        "control_parameters": path.control_parameters,
    }
    normalized = [
        normalize_inc_field(name, values[name], normalization) for name in INC_INPUT_FIELD_ORDER
    ]
    return np.concatenate(normalized)


def normalize_inc_field(
    name: str,
    values: NDArray[np.float64],
    normalization: VelocityVerletINCNormalization,
) -> NDArray[np.float64]:
    """Normalize one semantic field using training-only statistics."""
    statistics = normalization.statistics[name]
    return (np.asarray(values, dtype=np.float64) - statistics["mean"]) / statistics["scale"]


def denormalize_inc_field(
    name: str,
    values: NDArray[np.float64],
    normalization: VelocityVerletINCNormalization,
) -> NDArray[np.float64]:
    """Restore one semantic field from normalized coordinates."""
    statistics = normalization.statistics[name]
    return np.asarray(values, dtype=np.float64) * statistics["scale"] + statistics["mean"]


def _read_path(path_id: str, group: h5py.Group) -> VelocityVerletINCPath:
    """Read one complete path group."""
    return VelocityVerletINCPath(
        path_id=path_id,
        split=str(group.attrs["split"]),
        time=group["time"][...],
        control_parameters=group["control_parameters"][...],
        displacement=group["state/displacement"][...],
        velocity=group["state/velocity"][...],
        baseline_acceleration=group["baseline/acceleration"][...],
        internal_force=group["baseline/internal_force"][...],
        external_force=group["baseline/external_force"][...],
        residual_force_start=group["target/residual_force_start"][...],
        residual_force_end=group["target/residual_force_end"][...],
        mechanical_energy=group["reference/mechanical_energy"][...],
    )


def _validate_loaded_dataset(
    system: VelocityVerletINCSystemData,
    paths: tuple[VelocityVerletINCPath, ...],
) -> None:
    """Validate fixed widths, time nodes, and complete path split coverage."""
    if not paths:
        raise ValueError("INC dataset contains no complete paths.")
    free_count = system.free_dofs.size
    if system.mass_lumped_free.shape != (free_count,):
        raise ValueError("INC dataset mass vector does not match free DOF order.")
    if system.stiffness_free.shape != (free_count, free_count):
        raise ValueError("INC dataset stiffness matrix does not match free DOF order.")
    for path in paths:
        step_count = path.time.size - 1
        state_shape = (step_count + 1, free_count)
        target_shape = (step_count, free_count)
        state_fields = (
            path.displacement,
            path.velocity,
            path.baseline_acceleration,
            path.internal_force,
            path.external_force,
        )
        if any(values.shape != state_shape for values in state_fields):
            raise ValueError(f"INC path state shape mismatch: {path.path_id}.")
        if path.residual_force_start.shape != target_shape:
            raise ValueError(f"INC start-label shape mismatch: {path.path_id}.")
        if path.residual_force_end.shape != target_shape:
            raise ValueError(f"INC end-label shape mismatch: {path.path_id}.")
        if path.mechanical_energy.shape != (step_count + 1,):
            raise ValueError(f"INC energy history shape mismatch: {path.path_id}.")
        if not np.allclose(np.diff(path.time), system.time_step, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"INC path time nodes mismatch the fixed time step: {path.path_id}.")
        arrays = (*state_fields, path.residual_force_start, path.residual_force_end)
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise FloatingPointError(f"INC path contains non-finite values: {path.path_id}.")
    declared_splits = {path.split for path in paths}
    if declared_splits != {"train", "validation", "test"}:
        raise ValueError(f"INC dataset split coverage is incomplete: {declared_splits}.")
