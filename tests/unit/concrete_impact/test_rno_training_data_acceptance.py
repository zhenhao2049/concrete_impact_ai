"""Tests for lineage splits, train-only statistics, and acceptance hashes.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from concrete_impact.nn.training.acceptance import (
    COVERAGE_AXIS_NAMES,
    CoverageBins,
    CoveragePolicy,
    PathRecord,
    SplitSettings,
    TrainingDataAcceptanceError,
    _build_split_manifest,
    _compute_coverage,
    _compute_normalization,
    _fixed_bin_coverage,
    _read_maximum_yield_activation_margin,
    _sha256_file,
    _sha256_payload,
    _turning_angles,
    require_current_training_data_acceptance,
)


def test_fixed_bin_coverage_rejects_fe2_states_in_j2_empty_bins() -> None:
    """Verify min-max overlap cannot hide unrepresented fixed state bins."""
    record, passed = _fixed_bin_coverage(
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        np.asarray([0.15, 0.55, 0.85]),
        np.asarray([0.0, 0.3, 0.7, 1.0]),
        0.05,
    )

    assert not passed
    assert record["fe2_occupied_j2_empty_bins"] == [1]
    assert record["fe2_uncovered_fraction"] == pytest.approx(1.0 / 3.0)


def test_turning_angles_project_roundoff_overshoot_to_domain_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tolerance-scale cosine overshoots produce finite endpoint angles."""
    increments = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    epsilon = np.finfo(np.float64).eps
    monkeypatch.setattr(
        "concrete_impact.nn.training.acceptance._tensor_norm",
        lambda _increments: np.asarray([1.0 - epsilon, 1.0, 1.0 - epsilon]),
    )

    with np.errstate(invalid="raise"):
        angles = _turning_angles(increments, 1.0e-14, 1.0e-10)

    np.testing.assert_array_equal(angles, np.asarray([0.0, np.pi]))


def test_turning_angles_reject_cosine_beyond_domain_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a material cosine-domain violation is not projected silently."""
    increments = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(
        "concrete_impact.nn.training.acceptance._tensor_norm",
        lambda _increments: np.asarray([0.9, 1.0]),
    )

    with pytest.raises(ValueError, match="turning cosine left"):
        _turning_angles(increments, 1.0e-14, 1.0e-10)


def test_pilot_coverage_waives_only_explicit_missing_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify pilot acceptance preserves and waives only declared empty bins."""
    settings = CoverageBins.model_validate(
        {
            **{name: [0.0, 1.0, 2.0] for name in COVERAGE_AXIS_NAMES},
            "stress_floor": 1.0e-10,
            "increment_floor": 1.0e-14,
            "invariant_domain_tolerance": 1.0e-10,
        }
    )
    axes = {name: np.asarray([0.5, 1.5]) for name in COVERAGE_AXIS_NAMES}
    axes["turning_angle"] = np.asarray([0.5])
    monkeypatch.setattr(
        "concrete_impact.nn.training.acceptance._path_state_axes",
        lambda _record, _settings: (axes, 0),
    )
    records = (
        PathRecord(
            Path("unused.h5"),
            "path",
            "source",
            "fixed_cell_rve",
            "elastic",
            "monotonic",
            "lineage",
            "prescribed_macro_strain",
        ),
    )

    _, strict_passed = _compute_coverage(records, settings)
    pilot_summary, pilot_passed = _compute_coverage(
        records,
        settings,
        CoveragePolicy(
            mode="pilot",
            allowed_missing_bin_indices={"turning_angle": (1,)},
            limitation_note="Pilot test fixture.",
        ),
    )

    assert not strict_passed
    assert pilot_passed
    assert pilot_summary["missing_bin_indices"]["turning_angle"] == [1]
    assert pilot_summary["waived_missing_bin_indices"]["turning_angle"] == [1]
    assert pilot_summary["unexpected_missing_bin_indices"]["turning_angle"] == []


def test_lineage_split_is_deterministic_and_rejects_small_strata() -> None:
    """Verify complete lineages remain together and each stratum needs three lineages."""
    records = tuple(
        PathRecord(
            Path("unused.h5"),
            f"path_{index}",
            "source",
            "material_point",
            "elastic",
            "monotonic",
            f"lineage_{index}",
            "prescribed_macro_strain",
        )
        for index in range(6)
    )
    settings = SplitSettings(
        seed=7,
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
    )

    first, passed, failures = _build_split_manifest(records, settings)
    second, _, _ = _build_split_manifest(records, settings)

    assert passed
    assert not failures
    assert first == second
    _, small_passed, small_failures = _build_split_manifest(records[:2], settings)
    assert not small_passed
    assert "2 independent lineages" in small_failures[0]

    fe2_records = tuple(
        PathRecord(
            Path("unused.h5"),
            f"fe2_path_{index}",
            "fe2_source",
            "impact_path",
            "viscoplastic",
            "impact_extracted",
            f"fe2_lineage_{index}",
            "direct_fe2_explicit",
        )
        for index in range(4)
    )
    fe2_manifest, fe2_passed, _ = _build_split_manifest(fe2_records, settings)
    assert fe2_passed
    assert {item["split"] for item in fe2_manifest["paths"]} == {"validation", "test"}


def test_design_stratum_split_overrides_old_assignment_with_exact_16_2_2() -> None:
    """Verify every complete production stratum receives the requested exact split."""
    records = tuple(
        PathRecord(
            Path("unused.h5"),
            f"path_{stratum}_{index}",
            "source",
            "fixed_cell_rve",
            "transition",
            "monotonic",
            f"lineage_{stratum}_{index}",
            "prescribed_macro_strain",
            preassigned_split=("train" if index < 14 else "validation" if index < 17 else "test"),
            design_stratum=f"monotonic|A3|R{stratum}",
            amplitude_band="A3",
            rate_band=f"R{stratum}",
            tensor_direction="deviatoric_uniaxial",
        )
        for stratum in (1, 2)
        for index in range(20)
    )
    settings = SplitSettings(
        seed=71000,
        strategy="deterministic_design_stratum",
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )

    first, passed, failures = _build_split_manifest(records, settings)
    second, _, _ = _build_split_manifest(records, settings)

    assert passed
    assert not failures
    assert first == second
    for stratum in ("monotonic|A3|R1", "monotonic|A3|R2"):
        splits = [item["split"] for item in first["paths"] if item["design_stratum"] == stratum]
        assert splits.count("train") == 16
        assert splits.count("validation") == 2
        assert splits.count("test") == 2


def test_yield_onset_reads_maximum_microscopic_margin(tmp_path: Path) -> None:
    """Verify heterogeneous RVE onset uses any active point, not all active points."""
    shard = tmp_path / "yield_margin.h5"
    with h5py.File(shard, "w") as handle:
        group = handle.create_group("paths/path")
        solver = group.create_group("solver")
        solver.create_dataset("yield_activation_margin", data=np.asarray([-0.9, -0.4]))
        solver.create_dataset("maximum_yield_activation_margin", data=np.asarray([-0.2, 0.15]))

    with h5py.File(shard, "r") as handle:
        margin = _read_maximum_yield_activation_margin(handle["paths/path"])

    np.testing.assert_array_equal(margin, np.asarray([-0.2, 0.15]))
    assert np.any(margin > 0.0)


def test_normalization_uses_training_paths_only_and_reports_zero_variance(
    tmp_path: Path,
) -> None:
    """Verify validation values do not influence training means or RMS scales."""
    shard = tmp_path / "paths.h5"
    _write_normalization_fixture(shard)
    records = tuple(
        PathRecord(
            shard,
            name,
            "source",
            "material_point",
            "elastic",
            "monotonic",
            name,
            "prescribed_macro_strain",
        )
        for name in ("train_a", "train_b", "validation")
    )
    split = {
        "paths": [
            {
                "path_id": f"{shard.name}:{name}",
                "split": "train" if name.startswith("train") else "validation",
            }
            for name in ("train_a", "train_b", "validation")
        ]
    }

    normalization, passed, failures = _compute_normalization(
        records,
        split,
        {"source": {"material_parameters", "microstructure_features"}},
    )

    assert normalization["macro_strain"]["mean"][0] == 2.0
    assert normalization["macro_stress"]["mean"][0] == 3.0
    assert not passed
    assert any("zero training variance" in failure for failure in failures)


def test_training_guard_rejects_changed_shard_hash(tmp_path: Path) -> None:
    """Verify post-audit HDF5 changes close the training gate."""
    data_plan = tmp_path / "plan.yaml"
    data_plan.write_text("plan: fixed\n", encoding="utf-8")
    shard = tmp_path / "data.h5"
    shard.write_bytes(b"initial")
    manifest = tmp_path / "response_shards.json"
    manifest.write_text(
        json.dumps({"response_shards": [str(shard)]}),
        encoding="utf-8",
    )
    split = {"paths": []}
    normalization = {"macro_strain": {"mean": [0.0], "rms_scale": [1.0]}}
    acceptance = tmp_path / "training_data_acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "passed": True,
                "data_plan_sha256": _sha256_file(data_plan),
                "response_shard_manifest_sha256": _sha256_file(manifest),
                "shard_sha256": {str(shard): _sha256_file(shard)},
                "split_manifest": split,
                "split_manifest_sha256": _sha256_payload(split),
                "normalization": normalization,
                "normalization_sha256": _sha256_payload(normalization),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    (tmp_path / "normalization.json").write_text(json.dumps(normalization), encoding="utf-8")
    shard.write_bytes(b"changed")

    with pytest.raises(TrainingDataAcceptanceError, match="shard hashes changed"):
        require_current_training_data_acceptance(acceptance, data_plan, manifest)


def _write_normalization_fixture(path: Path) -> None:
    """Write three small accepted-path groups for normalization tests."""
    with h5py.File(path, "w") as handle:
        paths = handle.create_group("paths")
        for name, strain_value, stress_value in (
            ("train_a", 1.0, 2.0),
            ("train_b", 3.0, 4.0),
            ("validation", 100.0, 200.0),
        ):
            group = paths.create_group(name)
            macro = group.create_group("macro")
            macro.create_dataset("strain", data=np.full((2, 6), strain_value))
            macro.create_dataset("stress", data=np.full((2, 6), stress_value))
            state = group.create_group("micro/state")
            state.create_dataset("equivalent_plastic_strain", data=np.zeros(2))
            features = group.create_group("features")
            features.create_dataset("material_parameters_names", data=np.asarray([b"E"]))
            features.create_dataset("material_parameters", data=np.asarray([1.0]))
            features.create_dataset("microstructure_features_names", data=np.asarray([], dtype="S"))
            features.create_dataset("microstructure_features", data=np.asarray([]))
