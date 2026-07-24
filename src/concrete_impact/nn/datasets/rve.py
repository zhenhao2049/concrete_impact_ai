"""Accepted-path HDF5 interface for continuous-time RVE-RNO training.

Contents:
    Path records, lazy HDF5 access, memory preload, batching, and pinned transfer.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from fem.rve.io import SUPPORTED_RVE_DATA_SCHEMA_VERSIONS


@dataclass(frozen=True)
class RNOPathSample:
    """Store one complete accepted path without teacher-forced hidden states."""

    path_id: str
    time: NDArray[np.float64]
    time_step: NDArray[np.float64]
    macro_strain: NDArray[np.float64]
    macro_stress: NDArray[np.float64]
    material_parameters: NDArray[np.float64]
    microstructure_features: NDArray[np.float64]
    dissipation_density: NDArray[np.float64]


@dataclass(frozen=True)
class RNOPathBatch:
    """Store padded whole paths and a validity mask for rollout training."""

    path_ids: tuple[str, ...]
    time: torch.Tensor
    time_step: torch.Tensor
    network_inputs: torch.Tensor
    macro_stress: torch.Tensor
    dissipation_density: torch.Tensor
    valid_mask: torch.Tensor

    def pin_memory(self) -> RNOPathBatch:
        """Pin every host tensor for non-blocking CUDA transfer."""
        return RNOPathBatch(
            path_ids=self.path_ids,
            time=self.time.pin_memory(),
            time_step=self.time_step.pin_memory(),
            network_inputs=self.network_inputs.pin_memory(),
            macro_stress=self.macro_stress.pin_memory(),
            dissipation_density=self.dissipation_density.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
        )


@dataclass(frozen=True)
class RNOConditioningSpec:
    """Select only varying material and microstructure conditioning features."""

    material_parameter_names: tuple[str, ...]
    microstructure_feature_names: tuple[str, ...]


class HDF5RNOPathDataset(Dataset[RNOPathSample]):
    """Lazily read whole accepted paths from supported RVE HDF5 shards."""

    def __init__(
        self,
        files: Sequence[str | Path],
        conditioning: RNOConditioningSpec,
    ) -> None:
        """Index HDF5 paths while leaving numerical arrays on disk."""
        self._index: tuple[tuple[Path, str], ...] = self._build_index(files)
        self._conditioning = conditioning

    def __len__(self) -> int:
        """Return the number of complete paths."""
        return len(self._index)

    def __getitem__(self, index: int) -> RNOPathSample:
        """Read one complete path and keep physical time step separate."""
        file_path, path_name = self._index[index]
        with h5py.File(file_path, "r") as handle:
            group = handle[f"paths/{path_name}"]
            feature_group = group["features"]
            return RNOPathSample(
                path_id=f"{file_path.name}:{path_name}",
                time=group["macro/time"][...],
                time_step=group["macro/time_step"][...],
                macro_strain=group["macro/strain"][...],
                macro_stress=group["macro/stress"][...],
                material_parameters=_select_conditioning_features(
                    feature_group,
                    "material_parameters",
                    self._conditioning.material_parameter_names,
                ),
                microstructure_features=_select_conditioning_features(
                    feature_group,
                    "microstructure_features",
                    self._conditioning.microstructure_feature_names,
                ),
                dissipation_density=group["macro/dissipation_density"][...],
            )

    @staticmethod
    def _build_index(files: Sequence[str | Path]) -> tuple[tuple[Path, str], ...]:
        """Validate shards and index only completed accepted paths."""
        index: list[tuple[Path, str]] = []
        for raw_path in files:
            path = Path(raw_path)
            with h5py.File(path, "r") as handle:
                if handle.attrs["schema_version"] not in SUPPORTED_RVE_DATA_SCHEMA_VERSIONS:
                    raise ValueError(f"Unsupported RVE data schema in {path}.")
                if handle.attrs["status"] != "complete":
                    raise ValueError(f"RVE data shard is not complete: {path}.")
                for path_name in sorted(handle["paths"]):
                    if "features" not in handle[f"paths/{path_name}"]:
                        raise ValueError(
                            f"RVE data path has no conditioning features: {path}:{path_name}."
                        )
                    index.append((path, path_name))
        return tuple(index)


class PreloadedRNOPathDataset(Dataset[RNOPathSample]):
    """Hold all accepted complete paths in host memory during training."""

    def __init__(self, source: HDF5RNOPathDataset) -> None:
        """Materialize the immutable HDF5 path dataset exactly once."""
        self._samples = tuple(source[index] for index in range(len(source)))

    def __len__(self) -> int:
        """Return the number of preloaded complete paths."""
        return len(self._samples)

    def __getitem__(self, index: int) -> RNOPathSample:
        """Return one immutable in-memory complete path."""
        return self._samples[index]


class DeviceRNOPathBatchLoader:
    """Yield deterministic device-resident whole-path batches."""

    def __init__(
        self,
        samples: Sequence[RNOPathSample],
        maximum_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        strategy: str,
        shuffle: bool,
        random_seed: int,
    ) -> None:
        """Pack accepted paths by length or as one padded device batch."""
        self.maximum_batch_size = maximum_batch_size
        self.device = device
        self.strategy = strategy
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.epoch = 0
        if strategy == "length_bucket":
            grouped: dict[int, list[RNOPathSample]] = {}
            for sample in samples:
                grouped.setdefault(sample.time.size, []).append(sample)
            self.buckets = tuple(
                _stack_equal_length_samples(grouped[length], device, dtype)
                for length in sorted(grouped)
            )
            self.single_batch = None
        elif strategy == "single_padded_batch":
            self.buckets = ()
            self.single_batch = _move_rno_path_batch(collate_rno_paths(samples), device, dtype)
        else:
            raise ValueError(f"Unsupported device RVE-RNO batching strategy: {strategy}.")

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle stream for one training epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of batches produced in one complete pass."""
        if self.single_batch is not None:
            return 1
        return sum(
            math.ceil(len(bucket.path_ids) / self.maximum_batch_size) for bucket in self.buckets
        )

    def __iter__(self) -> Iterator[RNOPathBatch]:
        """Yield every accepted path once with deterministic balanced chunks."""
        if self.single_batch is not None:
            yield self.single_batch
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed + self.epoch)
        descriptors: list[tuple[RNOPathBatch, torch.Tensor]] = []
        for bucket in self.buckets:
            count = len(bucket.path_ids)
            order = (
                torch.randperm(count, generator=generator) if self.shuffle else torch.arange(count)
            )
            batch_count = math.ceil(count / self.maximum_batch_size)
            descriptors.extend((bucket, chunk) for chunk in torch.tensor_split(order, batch_count))
        if self.shuffle:
            descriptor_order = torch.randperm(len(descriptors), generator=generator).tolist()
            descriptors = [descriptors[index] for index in descriptor_order]
        for bucket, cpu_indices in descriptors:
            device_indices = cpu_indices.to(device=self.device)
            yield RNOPathBatch(
                path_ids=tuple(bucket.path_ids[index] for index in cpu_indices.tolist()),
                time=bucket.time.index_select(0, device_indices),
                time_step=bucket.time_step.index_select(0, device_indices),
                network_inputs=bucket.network_inputs.index_select(0, device_indices),
                macro_stress=bucket.macro_stress.index_select(0, device_indices),
                dissipation_density=bucket.dissipation_density.index_select(0, device_indices),
                valid_mask=bucket.valid_mask.index_select(0, device_indices),
            )


def collate_rno_paths(samples: Sequence[RNOPathSample]) -> RNOPathBatch:
    """Pad complete paths for rollout loss without using reference internal states."""
    batch_size = len(samples)
    max_length = max(sample.time.size for sample in samples)
    material_width = samples[0].material_parameters.size
    microstructure_width = samples[0].microstructure_features.size
    input_width = 6 + material_width + microstructure_width
    time = torch.zeros((batch_size, max_length), dtype=torch.float64)
    time_step = torch.zeros_like(time)
    network_inputs = torch.zeros((batch_size, max_length, input_width), dtype=torch.float64)
    macro_stress = torch.zeros((batch_size, max_length, 6), dtype=torch.float64)
    dissipation = torch.zeros((batch_size, max_length), dtype=torch.float64)
    valid_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for sample_id, sample in enumerate(samples):
        length = sample.time.size
        static_features = np.concatenate(
            (sample.material_parameters, sample.microstructure_features)
        )
        repeated_features = np.broadcast_to(static_features, (length, static_features.size))
        inputs = np.concatenate((sample.macro_strain, repeated_features), axis=1)
        time[sample_id, :length] = torch.from_numpy(sample.time)
        time_step[sample_id, :length] = torch.from_numpy(sample.time_step)
        network_inputs[sample_id, :length] = torch.from_numpy(inputs.copy())
        macro_stress[sample_id, :length] = torch.from_numpy(sample.macro_stress)
        dissipation[sample_id, :length] = torch.from_numpy(sample.dissipation_density)
        valid_mask[sample_id, :length] = True
    return RNOPathBatch(
        path_ids=tuple(sample.path_id for sample in samples),
        time=time,
        time_step=time_step,
        network_inputs=network_inputs,
        macro_stress=macro_stress,
        dissipation_density=dissipation,
        valid_mask=valid_mask,
    )


def _stack_equal_length_samples(
    samples: Sequence[RNOPathSample],
    device: torch.device,
    dtype: torch.dtype,
) -> RNOPathBatch:
    """Stack one equal-length path group directly in its training precision."""
    batch = collate_rno_paths(samples)
    if not bool(batch.valid_mask.all()):
        raise ValueError("Equal-length RVE-RNO bucket unexpectedly contains padding.")
    return _move_rno_path_batch(batch, device, dtype)


def _move_rno_path_batch(
    batch: RNOPathBatch,
    device: torch.device,
    dtype: torch.dtype,
) -> RNOPathBatch:
    """Move one immutable whole-path batch to its configured training device."""
    return RNOPathBatch(
        path_ids=batch.path_ids,
        time=batch.time.to(device=device, dtype=dtype),
        time_step=batch.time_step.to(device=device, dtype=dtype),
        network_inputs=batch.network_inputs.to(device=device, dtype=dtype),
        macro_stress=batch.macro_stress.to(device=device, dtype=dtype),
        dissipation_density=batch.dissipation_density.to(device=device, dtype=dtype),
        valid_mask=batch.valid_mask.to(device=device),
    )


def require_disjoint_path_ids(*splits: Sequence[RNOPathSample]) -> None:
    """Reject path leakage across training, validation, and test splits."""
    observed: set[str] = set()
    for split in splits:
        split_ids = {sample.path_id for sample in split}
        if observed.intersection(split_ids):
            raise ValueError("RNO dataset splits must be disjoint by complete path.")
        observed.update(split_ids)


def batch_to_mapping(batch: RNOPathBatch) -> dict[str, Any]:
    """Expose the batch through stable names used by training loops."""
    return {
        "path_ids": batch.path_ids,
        "time": batch.time,
        "time_step": batch.time_step,
        "network_inputs": batch.network_inputs,
        "macro_stress": batch.macro_stress,
        "dissipation_density": batch.dissipation_density,
        "valid_mask": batch.valid_mask,
    }


def _select_conditioning_features(
    group: h5py.Group,
    kind: str,
    requested_names: tuple[str, ...],
) -> NDArray[np.float64]:
    """Read explicitly selected varying features in configured order."""
    stored_names = tuple(value.decode() for value in group[f"{kind}_names"][...])
    stored_values = group[kind][...]
    index = {name: position for position, name in enumerate(stored_names)}
    missing = [name for name in requested_names if name not in index]
    if missing:
        raise ValueError(f"RNO conditioning features are absent from HDF5: {missing}.")
    return np.asarray([stored_values[index[name]] for name in requested_names], dtype=np.float64)
