"""Tests for fixed-cell RVE mesh and deterministic data planning.

Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from concrete_impact.benchmarks.rve_mesh_convergence import _level_passed, _select_level
from concrete_impact.core.config import load_yaml_config
from concrete_impact.experiments.rve_data_plan import (
    FixedParameterSpec,
    build_rve_data_task_manifest,
    build_rve_supplemental_task_manifest,
    load_rve_data_plan,
    validate_parallel_resources,
)
from concrete_impact.experiments.rve_path_generation import generate_data_control_path
from concrete_impact.experiments.rve_response_generation import _build_heterogeneous_rve
from concrete_impact.nn.training.acceptance import (
    PathRecord,
    SplitSettings,
    _build_split_manifest,
)
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.rve import (
    RVECompactDataShardWriter,
    iterate_prescribed_rve_path,
    solve_prescribed_rve_path,
    validate_rve_data_shard,
)


@pytest.mark.parametrize(
    ("circumferential", "radial", "axial", "element_count"),
    ((32, 4, 4, 768), (48, 6, 6, 2592), (64, 8, 8, 6144)),
)
def test_fixed_cell_mesh_counts_and_positive_center_jacobians(
    circumferential: int,
    radial: int,
    axial: int,
    element_count: int,
) -> None:
    """Verify the three declared O-grid levels and positive Hex8 mappings."""
    mesh = build_cylindrical_ogrid_hex8(
        (1.0, 1.0, 1.0),
        0.05,
        circumferential,
        radial,
        axial,
    )
    assert mesh.elements.shape == (element_count, 8)
    _, derivatives = evaluate_lagrange_shape_functions(
        "hex8", np.zeros((1, 3), dtype=np.float64)
    )
    coordinates = mesh.nodes[mesh.elements]
    jacobians = np.einsum("qia,eib->eqab", derivatives, coordinates)
    assert np.all(np.linalg.det(jacobians) > 0.0)


def test_pointwise_micro_stress_remains_diagnostic_for_fixed_cell_acceptance() -> None:
    """Verify a pointwise stress peak alone cannot reject macro RNO labels."""
    thresholds = load_yaml_config(
        "configs/benchmarks/rve/rve_rno_fixed_cell_mesh.yaml"
    )["verification"]
    record = {
        "macro_stress_error": 0.0,
        "effective_tangent_error": 0.0,
        "dissipation_error": 0.0,
        "phase_average_stress_error": 0.0,
        "maximum_q_error": 0.0,
        "maximum_micro_stress_error": 1.0,
        "active_fraction_error": 0.0,
        "maximum_hill_mandel_error": 0.0,
        "maximum_reaction_error": 0.0,
        "maximum_periodic_error": 0.0,
        "tangent_direction_error": 0.0,
        "minimum_quadrature_weight": 1.0,
    }

    assert _level_passed(record, thresholds)


def test_fixed_cell_plan_has_eighty_balanced_unique_paths() -> None:
    """Verify 40 elastic and 40 viscoplastic paths with five balanced families."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_fixed_cell_v1.yaml")
    tasks = build_rve_data_task_manifest(plan)
    assert len(tasks) == 80
    assert len({task.task_id for task in tasks}) == 80
    assert len({task.lineage_id for task in tasks}) == 80
    source = plan.sources[0]
    for load_case in source.load_cases:
        assert load_case.path_count == 40
        counts = {
            family: sum(
                load_case.path_families[index % len(load_case.path_families)] == family
                for index in range(load_case.path_count)
            )
            for family in load_case.path_families
        }
        assert set(counts.values()) == {8}


def test_c48_production_plan_has_exact_factorial_counts_and_frozen_splits() -> None:
    """Verify 2000 paths, 100 strata, 200 snapshots, and exact 1400/300/300 splits."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_c48_v2_2000.yaml")
    tasks = build_rve_data_task_manifest(plan)
    assert len(tasks) == 2000
    assert len({task.task_id for task in tasks}) == 2000
    assert len({task.design_stratum for task in tasks}) == 100
    assert sum(task.snapshot_required for task in tasks) == 200
    assert sum(task.tangent_check_required for task in tasks) == 100
    assert {
        split: sum(task.preassigned_split == split for task in tasks)
        for split in ("train", "validation", "test")
    } == {"train": 1400, "validation": 300, "test": 300}
    for stratum in {task.design_stratum for task in tasks}:
        selected = [task for task in tasks if task.design_stratum == stratum]
        assert len(selected) == 20
        assert {
            split: sum(task.preassigned_split == split for task in selected)
            for split in ("train", "validation", "test")
        } == {"train": 14, "validation": 3, "test": 3}


def test_c48_production_path_respects_amplitude_rate_and_turning_targets() -> None:
    """Verify analytic timing and nonproportional direction changes are reproducible."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_c48_v2_2000.yaml")
    tasks = build_rve_data_task_manifest(plan)
    task = next(
        item
        for item in tasks
        if item.family == "nonproportional"
        and item.turning_angle == 1.9
        and item.amplitude_band == "A3"
        and item.rate_band == "R3"
    )
    path = generate_data_control_path(plan.sources[0], task)
    weights = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    path_norms = np.sqrt(np.sum(path.values**2 * weights, axis=-1))
    assert np.isclose(np.max(path_norms), task.amplitude, rtol=0.0, atol=1.0e-14)
    rate_values = np.diff(path.values, axis=0) / np.diff(path.times)[:, None]
    rates = np.sqrt(np.sum(rate_values**2 * weights, axis=-1))
    assert float(np.max(rates)) <= float(task.target_strain_rate)
    assert np.isclose(
        float(np.max(rates)),
        float(task.target_strain_rate),
        rtol=2.0e-3,
        atol=0.0,
    )
    increments = np.diff(path.values, axis=0)
    increment_norms = np.sqrt(np.sum(increments**2 * weights, axis=-1))
    nonzero = increments[increment_norms > 1.0e-14]
    nonzero_norms = np.sqrt(np.sum(nonzero**2 * weights, axis=-1))
    unit = nonzero / nonzero_norms[:, None]
    cosine = np.sum(unit[1:] * unit[:-1] * weights, axis=1)
    turning = np.arccos(np.clip(cosine, -1.0, 1.0))
    assert np.isclose(np.max(turning), task.turning_angle, rtol=0.0, atol=1.0e-10)


def test_c48_preassigned_split_manifest_preserves_exact_per_stratum_counts() -> None:
    """Verify acceptance consumes frozen production splits without rehashing them."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_c48_v2_2000.yaml")
    tasks = build_rve_data_task_manifest(plan)
    records = tuple(
        PathRecord(
            file_path=Path(f"shard_{task.path_index:06d}.h5"),
            path_name=task.task_id,
            source_name=task.source_name,
            source_kind=task.source_kind,
            regime=str(task.expected_response_regime),
            family=str(task.family),
            lineage_id=task.lineage_id,
            macro_path_origin=task.macro_path_origin,
            preassigned_split=task.preassigned_split,
            design_stratum=task.design_stratum,
            amplitude_band=task.amplitude_band,
            rate_band=task.rate_band,
            tensor_direction=task.tensor_direction,
        )
        for task in tasks
    )
    manifest, passed, failures = _build_split_manifest(
        records,
        SplitSettings(
            seed=71000,
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
        ),
    )
    assert passed and not failures
    assert {
        split: sum(item["split"] == split for item in manifest["paths"])
        for split in ("train", "validation", "test")
    } == {"train": 1400, "validation": 300, "test": 300}


def test_optional_c48_supplement_has_exact_1000_and_700_150_150_split() -> None:
    """Verify ten new paths per original stratum without reusing base task ids."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_c48_v2_2000.yaml")
    base = build_rve_data_task_manifest(plan)
    supplement = build_rve_supplemental_task_manifest(plan)
    assert len(supplement) == 1000
    assert len({task.design_stratum for task in supplement}) == 100
    assert not ({task.task_id for task in base} & {task.task_id for task in supplement})
    assert {
        split: sum(task.preassigned_split == split for task in supplement)
        for split in ("train", "validation", "test")
    } == {"train": 700, "validation": 150, "test": 150}


def test_fixed_cell_lineage_split_is_exactly_fifty_six_twelve_twelve() -> None:
    """Verify the deterministic 70/15/15 split on two forty-path strata."""
    records = tuple(
        PathRecord(
            file_path=Path("unused.h5"),
            path_name=f"path_{regime}_{index}",
            source_name="fixed_cell",
            source_kind="heterogeneous_rve",
            regime=regime,
            family=("monotonic", "load_unload_reload", "reverse", "hold", "nonproportional")[
                index % 5
            ],
            lineage_id=f"{regime}_{index}",
            macro_path_origin="prescribed_macro_strain",
        )
        for regime in ("elastic", "viscoplastic")
        for index in range(40)
    )
    manifest, passed, failures = _build_split_manifest(
        records,
        SplitSettings(
            seed=61000,
            train_fraction=0.70,
            validation_fraction=0.15,
            test_fraction=0.15,
        ),
    )
    counts = {
        split: sum(item["split"] == split for item in manifest["paths"])
        for split in ("train", "validation", "test")
    }
    assert passed and not failures
    assert counts == {"train": 56, "validation": 12, "test": 12}


def test_parallel_budget_counts_processes_times_library_threads() -> None:
    """Reject a path-process configuration above half of available CPUs."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_fixed_cell_v1.yaml")
    validate_parallel_resources(plan, 32)
    invalid_parallel = plan.parallel.model_copy(update={"workers": 6})
    invalid = plan.model_copy(update={"parallel": invalid_parallel})
    with pytest.raises(ValueError, match="logical CPU budget"):
        validate_parallel_resources(invalid, 32)


def test_fixed_cell_selection_uses_c32_c48_c64_rule() -> None:
    """Select c32 only with a passing c48 trend and otherwise mark c64 limited."""
    levels = (
        {"reference": False},
        {"reference": False},
        {"reference": True},
    )
    base = {
        "passed": True,
        "macro_stress_error": 0.01,
        "effective_tangent_error": 0.02,
        "dissipation_error": 0.02,
    }
    comparisons = (
        base,
        {**base, "macro_stress_error": 0.005, "effective_tangent_error": 0.01},
        {**base, "macro_stress_error": 0.0, "effective_tangent_error": 0.0},
    )
    selected, quality = _select_level(
        comparisons,
        levels,
        {"selection_policy": "fixed_cell_c32_c48_c64"},
    )
    assert selected == 0
    assert "nonincreasing" in quality
    failed = tuple({**record, "passed": index == 2} for index, record in enumerate(comparisons))
    selected, quality = _select_level(
        failed,
        levels,
        {"selection_policy": "fixed_cell_c32_c48_c64"},
    )
    assert selected == 2
    assert quality == "reference_limited_preliminary_c64_data"


def test_compact_stream_matches_complete_path_macro_fields(tmp_path) -> None:
    """Verify accepted-step streaming equals the existing complete-path driver."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_fixed_cell_v1.yaml")
    source = plan.sources[0]
    micro = dict(source.microstructure_parameters)
    micro.update(
        {
            "circumferential_divisions": FixedParameterSpec(kind="fixed", value=8.0),
            "matrix_radial_divisions": FixedParameterSpec(kind="fixed", value=1.0),
            "axial_divisions": FixedParameterSpec(kind="fixed", value=1.0),
        }
    )
    source = source.model_copy(update={"microstructure_parameters": micro})
    model, update = _build_heterogeneous_rve(source)
    times = np.asarray([0.0, 1.0e-3, 2.0e-3])
    strains = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    complete = solve_prescribed_rve_path(model, times, strains, update)
    output = tmp_path / "compact.h5"
    writer = RVECompactDataShardWriter(
        output,
        model,
        {"dataset_kind": "heterogeneous_rve", "model_id": "tiny"},
        "lzf",
    )
    writer.start_path(
        "tiny",
        {
            "material_parameters": {},
            "microstructure_features": {},
        },
    )
    for accepted in iterate_prescribed_rve_path(model, times, strains, update):
        writer.append_step(accepted.time, accepted.time_step, accepted.response)
    writer.finish_path()
    writer.close_complete()
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "4.0"
        np.testing.assert_allclose(
            handle["paths/tiny/macro/stress"][...],
            np.stack([response.macro_stress for response in complete.responses]),
        )
        np.testing.assert_allclose(handle["paths/tiny/macro/time"][...], times[1:])
    assert validate_rve_data_shard(output) == {"model_count": 1, "path_count": 1}
