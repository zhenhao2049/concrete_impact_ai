"""Tests for RVE data planning and whole-path training interfaces.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from concrete_impact.experiments import (
    RVEDataTask,
    build_rve_data_task_manifest,
    load_rve_data_plan,
    validate_parallel_resources,
    write_rve_data_manifest,
)
from concrete_impact.experiments.rve_path_generation import generate_data_control_path
from concrete_impact.experiments.rve_response_generation import (
    write_impact_control_paths,
    write_response_shard,
)
from concrete_impact.nn.datasets import (
    DeviceRNOPathBatchLoader,
    HDF5RNOPathDataset,
    RNOConditioningSpec,
    collate_rno_paths,
)
from fem.rve import validate_rve_data_shard


def test_four_source_plan_builds_deterministic_manifest(tmp_path: Path) -> None:
    """Verify four source kinds, deterministic seeds, and explicit CPU limits."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_data_plan.yaml")
    manifest = build_rve_data_task_manifest(plan)

    assert {source.kind for source in plan.sources} == {
        "material_point",
        "single_phase_rve",
        "heterogeneous_rve",
        "impact_path",
    }
    assert len(manifest) == 3392
    assert manifest[0].seed == 11000
    assert manifest[-1].seed == 42515
    assert manifest[-1].macro_path_origin == "direct_fe2_explicit"
    validate_parallel_resources(plan, available_cpus=4)
    with pytest.raises(ValueError, match="exceed half"):
        validate_parallel_resources(plan, available_cpus=2)
    source = plan.sources[0]
    generated = generate_data_control_path(source, manifest[0])
    assert generated.control_kind == "macro_strain"
    assert generated.values.shape == (source.load_cases[0].time_points, 6)
    manifest_path = write_rve_data_manifest(plan, tmp_path / "manifest.json")
    assert manifest_path.is_file()


def test_hdf5_dataset_keeps_time_step_outside_network_inputs(tmp_path: Path) -> None:
    """Verify complete-path loading, padding, and continuous-time input semantics."""
    path = tmp_path / "paths.h5"
    _write_training_fixture(path)
    dataset = HDF5RNOPathDataset(
        [path],
        RNOConditioningSpec(("E", "nu"), ("volume_fraction",)),
    )
    first = dataset[0]
    second = dataset[1]
    batch = collate_rno_paths((first, second))

    assert len(dataset) == 2
    assert batch.network_inputs.shape == (2, 3, 9)
    assert batch.time_step.shape == (2, 3)
    assert np.array_equal(batch.network_inputs[0, :, :6].numpy(), first.macro_strain)
    assert batch.valid_mask.tolist() == [[True, True, True], [True, True, False]]


def test_device_length_bucket_loader_covers_paths_without_padding(tmp_path: Path) -> None:
    """Verify deterministic equal-length batches cover every accepted path once."""
    path = tmp_path / "paths.h5"
    _write_training_fixture(path)
    dataset = HDF5RNOPathDataset(
        [path],
        RNOConditioningSpec(("E", "nu"), ("volume_fraction",)),
    )
    loader = DeviceRNOPathBatchLoader(
        (dataset[0], dataset[1]),
        maximum_batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        strategy="length_bucket",
        shuffle=True,
        random_seed=72000,
    )
    loader.set_epoch(3)
    batches = tuple(loader)

    assert len(loader) == 2
    assert {batch.time.shape[1] for batch in batches} == {2, 3}
    assert all(batch.network_inputs.dtype == torch.float32 for batch in batches)
    assert all(bool(batch.valid_mask.all()) for batch in batches)
    assert {path_id for batch in batches for path_id in batch.path_ids} == {
        dataset[0].path_id,
        dataset[1].path_id,
    }


def test_material_point_response_worker_writes_valid_v3_shard(tmp_path: Path) -> None:
    """Verify a small real material-point task reaches the production HDF5 schema."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_data_plan.yaml")
    source = plan.sources[0]
    load_case = source.load_cases[0].model_copy(update={"path_count": 1, "time_points": 8})
    source = source.model_copy(update={"load_cases": (load_case,)})
    task = build_rve_data_task_manifest(
        plan.model_copy(update={"sources": (source, *plan.sources[1:])})
    )[0]
    output = write_response_shard(tmp_path, 0, source, (task,), plan.audit, None)

    assert validate_rve_data_shard(output) == {"model_count": 1, "path_count": 1}


def test_single_phase_rve_response_worker_writes_valid_v3_shard(tmp_path: Path) -> None:
    """Verify a small real RVE path is computed and stored by the response worker."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_data_plan.yaml")
    source = plan.sources[1]
    load_case = source.load_cases[0].model_copy(update={"path_count": 1, "time_points": 6})
    source = source.model_copy(update={"load_cases": (load_case,)})
    sources = (plan.sources[0], source, *plan.sources[2:])
    task = build_rve_data_task_manifest(plan.model_copy(update={"sources": sources}))[512]
    output = write_response_shard(tmp_path, 1, source, (task,), plan.audit, None)

    assert validate_rve_data_shard(output) == {"model_count": 1, "path_count": 1}


@pytest.mark.parametrize("source_index", [2, 3])
def test_two_phase_and_impact_response_workers_write_valid_v3_shards(
    tmp_path: Path,
    source_index: int,
) -> None:
    """Verify the heterogeneous direct and macro-extracted worker entrypoints."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_data_plan.yaml")
    source = plan.sources[source_index]
    load_case = source.load_cases[0].model_copy(
        update={"path_count": 1, "time_points": 4, "axial_region_count": 1}
    )
    if source.kind == "impact_path":
        load_case = load_case.model_copy(
            update={
                "expected_response": load_case.expected_response.model_copy(
                    update={
                        "regime": "elastic",
                        "require_unloading": False,
                        "require_reverse_loading": False,
                    }
                )
            }
        )
    microstructure = dict(source.microstructure_parameters)
    if "circumferential_divisions" in microstructure:
        microstructure["circumferential_divisions"] = microstructure[
            "circumferential_divisions"
        ].model_copy(update={"value": 8.0})
    else:
        from concrete_impact.experiments.rve_data_plan import FixedParameterSpec

        microstructure["circumferential_divisions"] = FixedParameterSpec(kind="fixed", value=8.0)
    source = source.model_copy(
        update={
            "load_cases": (load_case,),
            "microstructure_parameters": microstructure,
        }
    )
    task = RVEDataTask(
        task_id=f"small_{source.kind}",
        source_name=source.name,
        source_kind=source.kind,
        load_case_name=load_case.name,
        path_index=0,
        seed=load_case.seed,
        shard_index=source_index,
        lineage_id=f"small_{source.kind}_lineage",
        macro_path_origin=source.macro_path_origin,
    )
    impact_control_path = (
        write_impact_control_paths(tmp_path, source, (task,))
        if source.kind == "impact_path"
        else None
    )
    output = write_response_shard(
        tmp_path,
        source_index,
        source,
        (task,),
        plan.audit,
        impact_control_path,
    )

    assert validate_rve_data_shard(output) == {"model_count": 1, "path_count": 1}


def _write_training_fixture(path: Path) -> None:
    """Write two accepted paths using the production version-2 field layout."""
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "3.0"
        handle.attrs["status"] = "complete"
        paths = handle.create_group("paths")
        for name, length in (("path_a", 3), ("path_b", 2)):
            path_group = paths.create_group(name)
            features = path_group.create_group("features")
            features.create_dataset("material_parameters_names", data=np.asarray([b"E", b"nu"]))
            features.create_dataset("material_parameters", data=np.asarray([1.0, 2.0]))
            features.create_dataset(
                "microstructure_features_names",
                data=np.asarray([b"volume_fraction"]),
            )
            features.create_dataset("microstructure_features", data=np.asarray([0.05]))
            macro = path_group.create_group("macro")
            macro.create_dataset("time", data=np.arange(1, length + 1) * 0.1)
            macro.create_dataset("time_step", data=np.full(length, 0.1))
            macro.create_dataset("strain", data=np.arange(length * 6).reshape(length, 6))
            macro.create_dataset("stress", data=np.ones((length, 6)))
            macro.create_dataset("dissipation_density", data=np.zeros(length))
