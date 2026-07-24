"""Versioned HDF5 storage for RVE path data.

Contents:
    Complete-path HDF5 writing and append-only compact response shards.
Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from fem.rve.data import RVEPathResult, RVEResponse, StructuredHex8RVE

RVE_DATA_SCHEMA_VERSION = "3.0"
RVE_COMPACT_DATA_SCHEMA_VERSION = "4.0"
SUPPORTED_RVE_DATA_SCHEMA_VERSIONS = (
    RVE_DATA_SCHEMA_VERSION,
    RVE_COMPACT_DATA_SCHEMA_VERSION,
)


def write_rve_path_hdf5(
    output_path: str | Path,
    model: StructuredHex8RVE,
    path_name: str,
    result: RVEPathResult,
    metadata: dict[str, Any],
) -> Path:
    """Write one accepted RVE path with complete microscopic histories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    responses = result.responses
    if len(responses) != result.times.size:
        raise ValueError("RVE path times and responses must have identical lengths.")

    writer = RVEDataShardWriter(path, model, metadata)
    writer.write_path(path_name, result, metadata)
    writer.close_complete()

    return path


class RVEDataShardWriter:
    """Write one process-owned HDF5 v3 shard and publish it atomically."""

    def __init__(
        self,
        output_path: str | Path,
        model: StructuredHex8RVE,
        metadata: dict[str, Any],
    ) -> None:
        """Create an incomplete shard with one immutable model definition."""
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_suffix(self.output_path.suffix + ".partial")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs["schema_version"] = RVE_DATA_SCHEMA_VERSION
        self.handle.attrs["status"] = "incomplete"
        self.handle.attrs["voigt_order"] = "xx,yy,zz,yz,xz,xy"
        self.handle.attrs["strain_convention"] = "engineering_shear"
        self.handle.attrs["stress_unit"] = "project_material_unit"
        self.handle.attrs["strain_unit"] = "1"
        self.handle.attrs["dataset_kind"] = metadata.get("dataset_kind", "rve_validation")
        self.handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self.handle.attrs["model_id"] = str(metadata.get("model_id", "default"))
        _write_mesh_group(self.handle, model, str(self.handle.attrs["model_id"]))

    def write_path(
        self,
        path_name: str,
        result: RVEPathResult,
        metadata: dict[str, Any],
    ) -> None:
        """Append one accepted path with path-local conditioning features."""
        if f"paths/{path_name}" in self.handle:
            raise ValueError(f"Duplicate RVE path id in shard: {path_name}.")
        _write_path_group(self.handle, path_name, result)
        path_group = self.handle[f"paths/{path_name}"]
        path_group.attrs["model_id"] = self.handle.attrs["model_id"]
        path_group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        _write_feature_group(path_group, metadata)

    def close_complete(self) -> None:
        """Validate a nonempty shard, close it, and atomically publish it."""
        if "paths" not in self.handle or len(self.handle["paths"]) == 0:
            raise ValueError("RVE data shard must contain at least one accepted path.")
        self.handle.attrs["status"] = "complete"
        self.handle.flush()
        self.handle.close()
        os.replace(self.partial_path, self.output_path)

    def write_failure(self, task_id: str, diagnostics: dict[str, Any]) -> None:
        """Record one structured numerical failure without retrying the task."""
        group = self.handle.create_group(f"failures/{task_id}")
        group.attrs["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)

    def close_failed(self) -> None:
        """Publish a shard containing explicit failure diagnostics."""
        if "failures" not in self.handle or len(self.handle["failures"]) == 0:
            raise ValueError("Failed RVE shard requires at least one failure record.")
        self.handle.attrs["status"] = "failed"
        self.handle.flush()
        self.handle.close()
        os.replace(self.partial_path, self.output_path)


class RVECompactDataShardWriter:
    """Stream accepted macroscopic RVE fields into one HDF5 v4 shard."""

    def __init__(
        self,
        output_path: str | Path,
        model: StructuredHex8RVE,
        metadata: dict[str, Any],
        compression: str,
    ) -> None:
        """Create an incomplete compact shard and immutable mesh record."""
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_suffix(self.output_path.suffix + ".partial")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "w")
        self.compression = compression
        self.active_path: h5py.Group | None = None
        self.handle.attrs["schema_version"] = RVE_COMPACT_DATA_SCHEMA_VERSION
        self.handle.attrs["status"] = "incomplete"
        self.handle.attrs["voigt_order"] = "xx,yy,zz,yz,xz,xy"
        self.handle.attrs["strain_convention"] = "engineering_shear"
        self.handle.attrs["stress_unit"] = "project_material_unit"
        self.handle.attrs["strain_unit"] = "1"
        self.handle.attrs["dataset_kind"] = metadata.get("dataset_kind", "rve_training")
        self.handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        self.handle.attrs["model_id"] = str(metadata.get("model_id", "default"))
        _write_mesh_group(self.handle, model, str(self.handle.attrs["model_id"]))

    def start_path(self, path_name: str, metadata: dict[str, Any]) -> None:
        """Create one path and its zero-length accepted-step datasets."""
        if self.active_path is not None:
            raise RuntimeError("Compact RVE writer already has an active path.")
        group = self.handle.create_group(f"paths/{path_name}")
        group.attrs["model_id"] = self.handle.attrs["model_id"]
        group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        _write_feature_group(group, metadata)
        macro = group.create_group("macro")
        for name, width in (
            ("time", ()),
            ("time_step", ()),
            ("strain", (6,)),
            ("stress", (6,)),
            ("effective_tangent", (6, 6)),
            ("free_energy_density", ()),
            ("dissipation_density", ()),
            ("matrix_average_q", ()),
            ("matrix_maximum_q", ()),
        ):
            _create_stream_dataset(macro, name, width, self.compression)
        solver = group.create_group("solver")
        for name in (
            "micro_residual_norm",
            "newton_iterations",
            "armijo_backtracks",
            "incremental_work_density",
            "positive_work_density",
            "negative_work_density",
            "viscoplastic_active_volume_fraction",
            "unloading_volume_fraction",
            "hill_mandel_residual",
            "hill_mandel_error",
            "periodic_fluctuation_error",
            "reaction_antiperiodicity_error",
            "minimum_yield_activation_margin",
            "maximum_yield_activation_margin",
            "yield_activation_margin",
            "yield_activation_margin_available",
        ):
            _create_stream_dataset(solver, name, (), self.compression)
        phases = group.create_group("phases")
        for phase_name in model_phase_names(self.handle, str(self.handle.attrs["model_id"])):
            phase = phases.create_group(phase_name)
            _create_stream_dataset(phase, "volume_fraction", (), self.compression)
            _create_stream_dataset(phase, "average_stress", (6,), self.compression)
            _create_stream_dataset(phase, "free_energy_density", (), self.compression)
            _create_stream_dataset(phase, "dissipation_density", (), self.compression)
        self.active_path = group

    def append_step(
        self,
        time: float,
        time_step: float,
        response: RVEResponse,
    ) -> None:
        """Append exactly one converged and accepted RVE response."""
        if self.active_path is None:
            raise RuntimeError("Compact RVE writer has no active path.")
        tangent = _require_effective_tangent(response.effective_tangent)
        q = response.state.material_state.variables[
            "phase_matrix__equivalent_plastic_strain"
        ]
        macro = self.active_path["macro"]
        values = {
            "time": time,
            "time_step": time_step,
            "strain": response.state.macro_strain,
            "stress": response.macro_stress,
            "effective_tangent": tangent,
            "free_energy_density": response.free_energy_density,
            "dissipation_density": response.dissipation_density,
            "matrix_average_q": float(np.mean(q)),
            "matrix_maximum_q": float(np.max(q)),
        }
        for name, value in values.items():
            _append_stream_value(macro[name], value)
        solver = self.active_path["solver"]
        for name in solver:
            diagnostic_name = (
                "macro_work_increment" if name == "incremental_work_density" else name
            )
            if name == "yield_activation_margin":
                diagnostic_name = "minimum_yield_activation_margin"
            value = response.diagnostics[diagnostic_name]
            _append_stream_value(solver[name], value)
        phases = self.active_path["phases"]
        for phase_name, phase_group in phases.items():
            diagnostic = response.phase_diagnostics[phase_name]
            for name in phase_group:
                _append_stream_value(phase_group[name], diagnostic[name])

    def write_snapshots(self, snapshots: dict[str, tuple[float, RVEResponse]]) -> None:
        """Write selected complete microscopic states for one audit path."""
        if self.active_path is None:
            raise RuntimeError("Compact RVE writer has no active path.")
        root = self.active_path.create_group("micro_snapshots")
        for name, (time, response) in snapshots.items():
            group = root.create_group(name)
            group.attrs["time"] = time
            group.create_dataset(
                "displacement", data=response.displacement, compression=self.compression
            )
            group.create_dataset(
                "strain", data=response.micro_strains, compression=self.compression
            )
            group.create_dataset(
                "stress", data=response.micro_stresses, compression=self.compression
            )
            state = group.create_group("state")
            for field_name, values in response.state.material_state.variables.items():
                state.create_dataset(field_name, data=values, compression=self.compression)
            diagnostics = group.create_group("diagnostics")
            for field_name in (
                "viscoplastic_active",
                "incremental_work_density",
                "dissipation_density",
            ):
                if field_name in response.micro_diagnostics:
                    diagnostics.create_dataset(
                        field_name,
                        data=response.micro_diagnostics[field_name],
                        compression=self.compression,
                    )

    def write_validation(self, diagnostics: dict[str, NDArray[np.float64]]) -> None:
        """Write independent validation arrays for the active accepted path."""
        if self.active_path is None:
            raise RuntimeError("Compact RVE writer has no active path.")
        group = self.active_path.create_group("validation")
        for name, values in diagnostics.items():
            group.create_dataset(name, data=values)

    def finish_path(self) -> None:
        """Validate one nonempty accepted path and release its stream handle."""
        if self.active_path is None:
            raise RuntimeError("Compact RVE writer has no active path.")
        times = self.active_path["macro/time"][...]
        if times.size == 0 or (times.size > 1 and np.any(np.diff(times) <= 0.0)):
            raise ValueError("Compact RVE path requires strictly increasing accepted times.")
        self.active_path = None
        self.handle.flush()

    def write_failure(self, task_id: str, diagnostics: dict[str, Any]) -> None:
        """Record a numerical failure without writing trial states as accepted data."""
        group = self.handle.require_group(f"failures/{task_id}")
        group.attrs["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True)

    def close_complete(self) -> None:
        """Atomically publish a nonempty compact shard."""
        if self.active_path is not None:
            raise RuntimeError("Cannot close compact shard while a path is active.")
        if "paths" not in self.handle or len(self.handle["paths"]) == 0:
            raise ValueError("Compact RVE shard must contain at least one path.")
        self.handle.attrs["status"] = "complete"
        self.handle.flush()
        self.handle.close()
        os.replace(self.partial_path, self.output_path)

    def close_failed(self) -> None:
        """Publish only the explicit failure record for a failed shard."""
        self.handle.attrs["status"] = "failed"
        self.handle.flush()
        self.handle.close()
        os.replace(self.partial_path, self.output_path)


def _create_stream_dataset(
    group: h5py.Group,
    name: str,
    trailing_shape: tuple[int, ...],
    compression: str,
) -> None:
    """Create one losslessly compressed extensible float64 dataset."""
    group.create_dataset(
        name,
        shape=(0, *trailing_shape),
        maxshape=(None, *trailing_shape),
        chunks=(1, *trailing_shape),
        compression=compression,
        dtype=np.float64,
    )


def _append_stream_value(dataset: h5py.Dataset, value: Any) -> None:
    """Append one accepted value to an extensible dataset."""
    index = dataset.shape[0]
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def model_phase_names(handle: h5py.File, model_id: str) -> tuple[str, ...]:
    """Read ordered phase names from the immutable model group."""
    return tuple(
        value.decode() for value in handle[f"models/{model_id}/mesh/phase_names"][...]
    )


def _write_feature_group(handle: h5py.Group, metadata: dict[str, Any]) -> None:
    """Write ordered scalar material and microstructure conditioning features."""
    feature_group = handle.create_group("features")
    for feature_kind in ("material_parameters", "microstructure_features"):
        mapping = metadata.get(feature_kind, {})
        names = tuple(sorted(mapping))
        feature_group.create_dataset(f"{feature_kind}_names", data=np.asarray(names, dtype="S"))
        feature_group.create_dataset(
            feature_kind,
            data=np.asarray([mapping[name] for name in names], dtype=np.float64),
        )


def read_rve_path_arrays(
    input_path: str | Path,
    path_name: str,
) -> dict[str, NDArray[np.float64]]:
    """Read the principal macro and micro arrays from one RVE path."""
    with h5py.File(input_path, "r") as handle:
        if handle.attrs["schema_version"] != RVE_DATA_SCHEMA_VERSION:
            raise ValueError("Unsupported RVE data schema version.")
        path_group = handle[f"paths/{path_name}"]

        return {
            "time": path_group["macro/time"][...],
            "time_step": path_group["macro/time_step"][...],
            "macro_strain": path_group["macro/strain"][...],
            "macro_stress": path_group["macro/stress"][...],
            "effective_tangent": path_group["macro/effective_tangent"][...],
            "micro_strain": path_group["micro/strain"][...],
            "micro_stress": path_group["micro/stress"][...],
        }


def validate_rve_data_shard(input_path: str | Path) -> dict[str, int]:
    """Validate one completed HDF5 v3 shard and return its structural counts."""
    with h5py.File(input_path, "r") as handle:
        if handle.attrs["schema_version"] not in SUPPORTED_RVE_DATA_SCHEMA_VERSIONS:
            raise ValueError("Unsupported RVE data schema version.")
        if handle.attrs["status"] != "complete":
            raise ValueError("RVE data shard is not complete.")
        if "models" not in handle or "paths" not in handle:
            raise ValueError("RVE data shard requires models and paths groups.")
        for path_name, path_group in handle["paths"].items():
            if "features" not in path_group or "macro" not in path_group:
                raise ValueError(f"Incomplete RVE data path: {path_name}.")
            time = path_group["macro/time"][...]
            if time.size == 0 or np.any(np.diff(time) <= 0.0):
                raise ValueError(f"RVE data path time must be strictly increasing: {path_name}.")
            for dataset in _iter_datasets(path_group):
                values = dataset[...]
                availability_name = f"{dataset.name}__available"
                if availability_name in handle:
                    available = handle[availability_name][...]
                    values = values[available]
                if dataset.name.endswith("__available"):
                    continue
                if dataset.name.endswith((
                    "minimum_yield_activation_margin",
                    "maximum_yield_activation_margin",
                )):
                    availability = path_group[
                        "solver/yield_activation_margin_available"
                    ][...].astype(bool)
                    values = values[availability]
                if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"RVE data path contains non-finite values: {path_name}/{dataset.name}."
                    )
        return {
            "model_count": len(handle["models"]),
            "path_count": len(handle["paths"]),
        }


def _iter_datasets(group: h5py.Group) -> list[h5py.Dataset]:
    """Collect every dataset below one HDF5 group."""
    datasets: list[h5py.Dataset] = []
    group.visititems(
        lambda _name, item: datasets.append(item) if isinstance(item, h5py.Dataset) else None
    )
    return datasets


def _write_mesh_group(
    handle: h5py.File,
    model: StructuredHex8RVE,
    model_id: str,
) -> None:
    """Write mesh and periodic-topology datasets."""
    mesh = handle.create_group(f"models/{model_id}/mesh")
    mesh.create_dataset("nodes", data=model.nodes)
    mesh.create_dataset("elements", data=model.elements)
    phase_ids = (
        np.zeros(model.elements.shape[0], dtype=np.int64)
        if model.element_phase_ids is None
        else model.element_phase_ids
    )
    phase_names = model.phase_names or ("domain",)
    mesh.create_dataset("element_phase_id", data=phase_ids)
    mesh.create_dataset("phase_names", data=np.asarray(phase_names, dtype="S"))
    mesh.create_dataset(
        "divisions",
        data=np.asarray(model.divisions, dtype=np.int64),
    )
    mesh.create_dataset(
        "periodic_equivalence_class",
        data=model.constraint.equivalence_class_ids,
    )
    mesh.create_dataset(
        "periodic_representative_nodes",
        data=model.constraint.representative_nodes,
    )
    cache_weights = _quadrature_weights(model)
    mesh.create_dataset("quadrature_weights", data=cache_weights)


def _write_path_group(
    handle: h5py.File,
    path_name: str,
    result: RVEPathResult,
) -> None:
    """Write one complete accepted RVE history."""
    responses = result.responses
    group = handle.create_group(f"paths/{path_name}")
    macro = group.create_group("macro")
    macro.create_dataset("time", data=result.times)
    time_steps = np.zeros(result.times.size, dtype=np.float64)
    time_steps[0] = result.times[0]
    time_steps[1:] = np.diff(result.times)
    macro.create_dataset("time_step", data=time_steps)
    macro.create_dataset(
        "strain",
        data=np.stack([response.state.macro_strain for response in responses]),
        compression="lzf",
    )
    macro.create_dataset(
        "stress",
        data=np.stack([response.macro_stress for response in responses]),
        compression="lzf",
    )
    tangents = [_require_effective_tangent(response.effective_tangent) for response in responses]
    macro.create_dataset(
        "effective_tangent",
        data=np.stack(tangents),
        compression="lzf",
    )
    macro.create_dataset(
        "free_energy_density",
        data=np.asarray([response.free_energy_density for response in responses]),
    )
    macro.create_dataset(
        "dissipation_density",
        data=np.asarray([response.dissipation_density for response in responses]),
    )

    micro = group.create_group("micro")
    micro.create_dataset(
        "strain",
        data=np.stack([response.micro_strains for response in responses]),
        compression="lzf",
        chunks=True,
    )
    micro.create_dataset(
        "stress",
        data=np.stack([response.micro_stresses for response in responses]),
        compression="lzf",
        chunks=True,
    )
    state_group = micro.create_group("state")
    for name in responses[0].state.material_state.variables:
        state_group.create_dataset(
            name,
            data=np.stack(
                [response.state.material_state.variables[name] for response in responses]
            ),
            compression="lzf",
            chunks=True,
        )
    diagnostic_group = micro.create_group("diagnostics")
    diagnostic_names = sorted(
        set().union(*(response.micro_diagnostics.keys() for response in responses))
    )
    point_count = responses[0].micro_stresses.shape[0]
    for name in diagnostic_names:
        values = np.full((len(responses), point_count), np.nan, dtype=np.float64)
        available = np.zeros((len(responses), point_count), dtype=np.bool_)
        for step_id, response in enumerate(responses):
            if name in response.micro_diagnostics:
                values[step_id] = response.micro_diagnostics[name]
                available[step_id] = np.isfinite(response.micro_diagnostics[name])
        diagnostic_group.create_dataset(
            name,
            data=values,
            compression="lzf",
            chunks=True,
        )
        diagnostic_group.create_dataset(
            f"{name}__available",
            data=available,
            compression="lzf",
            chunks=True,
        )

    solver = group.create_group("solver")
    solver.create_dataset(
        "accepted_time_step",
        data=time_steps,
    )
    solver.create_dataset(
        "rejected_step_count",
        data=np.zeros(result.times.size, dtype=np.int64),
    )
    solver.create_dataset(
        "state_commit_count",
        data=np.ones(result.times.size, dtype=np.int64),
    )
    solver.create_dataset(
        "failure_status",
        data=np.zeros(result.times.size, dtype=np.int8),
    )
    for name in responses[0].diagnostics:
        values = np.asarray([response.diagnostics[name] for response in responses])
        solver.create_dataset(name, data=values)

    phases = group.create_group("phases")
    phase_names = sorted(
        set().union(*(response.phase_diagnostics.keys() for response in responses))
    )
    for phase_name in phase_names:
        phase_group = phases.create_group(phase_name)
        diagnostics = [response.phase_diagnostics[phase_name] for response in responses]
        phase_group.create_dataset(
            "volume_fraction",
            data=np.asarray([item["volume_fraction"] for item in diagnostics]),
        )
        phase_group.create_dataset(
            "average_stress",
            data=np.stack([item["average_stress"] for item in diagnostics]),
            compression="lzf",
        )
        phase_group.create_dataset(
            "free_energy_density",
            data=np.asarray([item["free_energy_density"] for item in diagnostics]),
        )
        phase_group.create_dataset(
            "dissipation_density",
            data=np.asarray([item["dissipation_density"] for item in diagnostics]),
        )
        phase_group.attrs["state_fields_json"] = json.dumps(
            diagnostics[0]["state_fields"], sort_keys=True
        )
    validation = group.create_group("validation")
    for name, values in result.validation_diagnostics.items():
        validation.create_dataset(name, data=values)


def _quadrature_weights(model: StructuredHex8RVE) -> NDArray[np.float64]:
    """Build physical quadrature weights for dataset provenance."""
    from fem.assembly.nonlinear_solid import build_hex8_material_assembly_cache

    cache = build_hex8_material_assembly_cache(
        model.nodes,
        model.elements,
        model.quadrature_order,
    )

    return cache.jacobian_weights


def _require_effective_tangent(
    tangent: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Require the production effective tangent selected by the path schema."""
    if tangent is None:
        raise ValueError("RVE HDF5 paths require an effective tangent at every accepted step.")

    return tangent
