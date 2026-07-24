"""Tests for trained RVE-RNO evaluation, timing, and scientific figures.

Contents:
    Metric formulas, deterministic tangent states, micro-model rollout, and figure hashes.
Author:
    Zhen Hao.
Created:
    2026-07-17.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from concrete_impact.nn.config import RVERNOModelConfig
from concrete_impact.nn.datasets.rve import RNOPathBatch
from concrete_impact.nn.evaluation.rve_rno import (
    EvaluationPathSource,
    RVERNOReferenceThresholds,
    RVERNOSlurmBaselineConfig,
    _compute_model_tangents,
    _relative_frobenius_error,
    _rollout_evaluation_batch,
    compare_formal_quality_reference,
    compute_path_metrics,
    deterministic_tangent_state_selection,
    load_rve_rno_evaluation_config,
    summarize_c48_performance,
    summarize_global_split_metrics,
    summarize_timing_samples,
)
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO
from concrete_impact.reporting.rve_rno_evaluation import (
    render_rve_rno_evaluation_figures,
    select_representative_test_path_ids,
)


def test_path_metrics_use_history_energy_floors_and_physical_dissipation() -> None:
    """Verify stress, dissipation, cumulative work, and latent diagnostics."""
    reference_stress = np.ones((2, 6), dtype=np.float64)
    predicted_stress = reference_stress + 1.0
    reference_dissipation = np.asarray([1.0, 2.0])
    predicted_dissipation = np.asarray([-1.0, 3.0])
    time_step = np.asarray([0.5, 0.5])
    norms = np.asarray([1.0, 3.0])
    metrics = compute_path_metrics(
        reference_stress,
        predicted_stress,
        reference_dissipation,
        predicted_dissipation,
        time_step,
        np.ones(6),
        1.0,
        norms,
        norms,
        norms,
        0.0,
    )

    assert metrics["stress_normalized_rmse"] == pytest.approx(1.0)
    assert metrics["negative_dissipation_state_count"] == 1
    assert metrics["maximum_thermodynamic_violation"] == pytest.approx(1.0)
    assert metrics["reference_cumulative_dissipation"] == pytest.approx(1.5)
    assert metrics["predicted_cumulative_dissipation"] == pytest.approx(1.0)
    assert metrics["latent_norm_median"] == pytest.approx(2.0)


def test_tangent_state_selection_is_repeatable_and_test_only(tmp_path: Path) -> None:
    """Verify hash ordering and equal state spacing are independent of input order."""
    sources = tuple(
        EvaluationPathSource(
            path_id=f"response.h5:path_{index}",
            file_path=tmp_path / f"{index}.h5",
            group_name=f"path_{index}",
            split="test",
            metadata={},
        )
        for index in range(5)
    )
    lengths = {source.path_id: 9 + index for index, source in enumerate(sources)}
    first = deterministic_tangent_state_selection(sources, 3, 4, lengths)
    second = deterministic_tangent_state_selection(tuple(reversed(sources)), 3, 4, lengths)

    assert first == second
    assert len(first) == 3
    assert all(len(indices) == 4 for indices in first.values())


def test_micro_model_rollout_and_central_difference_tangent_are_finite() -> None:
    """Verify evaluation rollout shapes and automatic-differentiation tangent parity."""
    torch.manual_seed(17)
    model = EnergyDissipationRVERNO(_model_config())
    batch = RNOPathBatch(
        path_ids=("a", "b"),
        time=torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.0]], dtype=torch.float64),
        time_step=torch.tensor(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.0]], dtype=torch.float64
        ),
        network_inputs=torch.randn((2, 3, 6), dtype=torch.float64) * 1.0e-3,
        macro_stress=torch.zeros((2, 3, 6), dtype=torch.float64),
        dissipation_density=torch.zeros((2, 3), dtype=torch.float64),
        valid_mask=torch.tensor([[True, True, True], [True, True, False]]),
    )
    prediction = _rollout_evaluation_batch(model, batch)
    assert prediction["predicted_stress"].shape == (2, 3, 6)
    assert prediction["latent_state"].shape == (2, 3, 3)
    assert all(np.all(np.isfinite(values)) for values in prediction.values())

    strain = np.asarray([0.01, -0.002, 0.001, 0.003, 0.0, -0.001])
    latent = np.zeros(3)
    ad_tangent, finite_difference = _compute_model_tangents(
        model,
        strain,
        latent,
        1.0e-3,
        1.0e-6,
        model.prepare_energy_anchors(),
    )
    error = _relative_frobenius_error(ad_tangent, finite_difference, 1.0e-12)
    assert error < 1.0e-7


def test_timing_summary_rejects_nonfinite_and_reports_median_mad() -> None:
    """Verify repeated timing fields without running a long benchmark."""
    summary = summarize_timing_samples(10, 5, np.asarray([1.0, 2.0, 3.0]))
    assert summary["median_seconds_per_state"] == pytest.approx(0.04)
    assert summary["median_absolute_deviation_seconds_per_state"] == pytest.approx(0.02)
    assert summary["median_states_per_second"] == pytest.approx(25.0)
    with pytest.raises(FloatingPointError, match="finite and positive"):
        summarize_timing_samples(10, 5, np.asarray([1.0, np.nan]))


def test_c48_performance_uses_archived_solver_time_per_increment(
    tmp_path: Path,
) -> None:
    """Summarize complete archived c48 task records without rerunning FE solves."""
    tasks = []
    expected = []
    for index, (solver_time, increments) in enumerate(((6.4, 64), (19.2, 96))):
        attempt = tmp_path / f"task_{index:04d}" / "attempt_000"
        attempt.mkdir(parents=True)
        status_path = attempt / "status.json"
        status_path.write_text(
            json.dumps(
                {
                    "state": "succeeded",
                    "hostname": f"node_{index}",
                    "slurm_job_id": str(100 + index),
                }
            ),
            encoding="utf-8",
        )
        performance_path = attempt / "performance.json"
        performance_path.write_text(
            json.dumps(
                {
                    "solver_time": solver_time,
                    "accepted_increment_count": increments,
                }
            ),
            encoding="utf-8",
        )
        tasks.append({"status_path": str(status_path)})
        expected.append(solver_time / increments)
    subset_manifest = tmp_path / "subset_manifest.json"
    subset_manifest.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    launcher = tmp_path / "launch.sh"
    sbatch = tmp_path / "run.sbatch"
    launcher.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    baseline = RVERNOSlurmBaselineConfig(
        expected_path_count=2,
        threads_per_path=3,
        launcher=launcher,
        sbatch=sbatch,
    )

    summary = summarize_c48_performance(subset_manifest, baseline)

    assert summary["path_count"] == 2
    assert summary["median_seconds_per_increment"] == pytest.approx(np.median(expected))
    assert summary["p95_seconds_per_increment"] == pytest.approx(
        np.quantile(expected, 0.95)
    )
    assert len(summary["performance_file_sha256"]) == 2
    assert summary["threads_per_path"] == 3
    assert summary["hostnames"] == ["node_0", "node_1"]


def test_evaluation_config_freezes_production_timing_and_tangent_settings() -> None:
    """Verify the tracked 500-path evaluation request has the declared design."""
    config = load_rve_rno_evaluation_config(
        "configs/nn/rve_rno_c48_v2_500_pilot_evaluation.yaml"
    )
    assert config.performance.batch_sizes == (1, 16, 64, 256, 1024)
    assert config.performance.warmup_updates == 20
    assert config.performance.timing_groups == 10
    assert config.performance.updates_per_group == 100
    assert config.tangent.path_count == 10
    assert config.tangent.states_per_path == 4
    assert config.tangent.perturbation == pytest.approx(1.0e-5)
    assert config.certification_scope == "pilot"
    assert config.slurm_baseline.expected_path_count == 500
    assert config.slurm_baseline.threads_per_path == 3
    assert config.run_directory.name == "run_20260717_232528"


def test_global_metrics_and_formal_reference_use_energy_aggregates() -> None:
    """Verify global split metrics and Pilot-only formal threshold comparisons."""
    records = [
        _metric_record("train", "hold", 0.04, 1.0, 100.0),
        _metric_record("validation", "hold", 0.05, 1.0, 100.0),
        _metric_record("test", "hold", 0.06, 4.0, 100.0),
        _metric_record("test", "load_unload_reload", 0.08, 16.0, 100.0),
    ]
    global_metrics = summarize_global_split_metrics(records)
    expected_test = np.sqrt((4.0 + 16.0) / 200.0)
    assert global_metrics["splits"]["test"]["stress_normalized_rmse"] == pytest.approx(
        expected_test
    )

    thresholds = RVERNOReferenceThresholds(
        test_stress_rmse=0.4,
        family_stress_error=0.4,
        maximum_path_stress_error=0.25,
        normalized_dissipation_error=0.2,
        negative_dissipation_tolerance=1.0e-12,
        tangent_median_error=0.15,
        tangent_p95_error=0.30,
        macro_relative_error=0.10,
    )
    comparison = compare_formal_quality_reference(
        records,
        global_metrics,
        {
            "autodiff_vs_rve": {"median": 0.10, "p95": 0.20, "maximum": 0.25}
        },
        thresholds,
    )
    assert comparison["checks"]["test_stress_rmse"]["passed"] is True
    assert comparison["checks"]["normalized_dissipation_error"]["passed"] is False
    assert comparison["all_available_metrics_passed"] is False


def test_representative_paths_cover_groups_and_use_low_error_fillers() -> None:
    """Select the lowest-error path per group before global-best fillers."""
    records = tuple(
        {
            "path_id": f"response.h5:path_{index}",
            "split": "test",
            "family": family,
            "regime": regime,
            "stress_normalized_rmse": str(error),
        }
        for index, (family, regime, error) in enumerate(
            (
                ("hold", "elastic", 0.01),
                ("hold", "transition", 0.02),
                ("hold", "viscoplastic", 0.03),
                ("load_unload_reload", "elastic", 0.04),
                ("load_unload_reload", "transition", 0.05),
                ("hold", "elastic", 0.90),
                ("hold", "transition", 0.06),
            )
        )
    )
    selected = select_representative_test_path_ids(records, 6)
    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert "response.h5:path_5" not in selected
    assert "response.h5:path_6" in selected
    assert selected == select_representative_test_path_ids(tuple(reversed(records)), 6)


def test_figures_support_three_epochs_and_one_test_path(tmp_path: Path) -> None:
    """Verify arbitrary nonempty training and test counts with a hashed figure manifest."""
    run = tmp_path / "run"
    evaluation = tmp_path / "evaluation"
    run.mkdir()
    evaluation.mkdir()
    metrics = []
    for epoch in range(1, 4):
        for split, factor in (("train", 1.0), ("validation", 1.2)):
            metrics.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "total_loss": factor / epoch,
                    "stress_loss": factor / epoch,
                    "dissipation_loss": factor / epoch,
                    "thermodynamic_violation_count": epoch - 1,
                }
            )
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in metrics) + "\n",
        encoding="utf-8",
    )
    (run / "training_summary.json").write_text(
        json.dumps({"best_epoch": 3}), encoding="utf-8"
    )
    (run / "best_model.pt").write_bytes(b"model")
    (run / "artifact_metadata.json").write_text("{}", encoding="utf-8")
    _write_path_metric_fixture(evaluation / "path_metrics.csv")
    (evaluation / "group_metrics.json").write_text("{}", encoding="utf-8")
    (evaluation / "tangent_metrics.json").write_text("{}", encoding="utf-8")
    _write_prediction_fixture(evaluation / "predictions.h5")
    (evaluation / "inference_performance.json").write_text(
        json.dumps(
            {
                "batch_results": [
                    {"batch_size": 1, "median_states_per_second": 10.0},
                    {"batch_size": 16, "median_states_per_second": 100.0},
                ],
                "cross_platform_local_material_update_throughput_ratio": {
                    "1": 20.0,
                    "16": 200.0,
                },
            }
        ),
        encoding="utf-8",
    )
    subset_summary = tmp_path / "subset_summary.json"
    subset_summary.write_text(
        json.dumps(
            {
                "family_counts": {"hold": 2, "load_unload_reload": 1},
                "split_counts": {"train": 1, "validation": 1, "test": 1},
                "tensor_direction_counts": {"deviatoric_uniaxial": 3},
                "accepted_increment_counts": {"64": 3},
            }
        ),
        encoding="utf-8",
    )
    evaluation_config = tmp_path / "evaluation_config.yaml"
    evaluation_config.write_text("schema_version: '1.0'\n", encoding="utf-8")

    output = tmp_path / "figures"
    manifest_path = render_rve_rno_evaluation_figures(
        run_directory=run,
        evaluation_directory=evaluation,
        subset_summary_path=subset_summary,
        evaluation_config_path=evaluation_config,
        output_directory=output,
        dpi=72,
        representative_path_count=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 6
    assert len(tuple(output.glob("*.png"))) == 6
    assert len(tuple(output.glob("*.svg"))) == 6
    assert all(
        len(record["outputs"]) == 2 and all(item["sha256"] for item in record["outputs"])
        for record in manifest["figures"]
    )
    assert all(
        record["common_provenance"]["evaluation_config"]["sha256"]
        and record["common_provenance"]["model"]["sha256"]
        and len(record["common_provenance"]["evaluation_files"]) == 5
        for record in manifest["figures"]
    )
    training_figure = next(
        record
        for record in manifest["figures"]
        if record["name"] == "rno-training-validation-loss"
    )
    assert training_figure["display_filter"]["raw_metrics_modified"] is False
    assert all(
        count == 0
        for split_counts in training_figure["display_filter"][
            "omitted_point_counts"
        ].values()
        for count in split_counts.values()
    )
    training_svg = (output / "rno-training-validation-loss.svg").read_text(
        encoding="utf-8"
    )
    assert all(
        title in training_svg
        for title in ("综合损失", "应力路径损失", "耗散监督损失")
    )
    assert "负耗散状态数" not in training_svg


def _model_config() -> RVERNOModelConfig:
    """Build one small CPU direct-rate model for evaluation tests."""
    return RVERNOModelConfig(
        family="energy_rve_rno",
        time_representation="continuous",
        integrator="cs_semi_implicit_forward_euler",
        evolution="direct_rate",
        reference_time_scale=0.02,
        latent_dimension=3,
        hidden_size=8,
        hidden_layers=1,
        activation="silu",
        dtype="float64",
        device="cpu",
    )


def _metric_record(
    split: str,
    family: str,
    stress_error: float,
    stress_numerator: float,
    stress_denominator: float,
) -> dict[str, float | int | str]:
    """Build one complete metric record for aggregation tests."""
    return {
        "path_id": f"{split}:{family}",
        "split": split,
        "family": family,
        "stress_normalized_rmse": stress_error,
        "stress_squared_error_sum": stress_numerator,
        "stress_normalization_denominator": stress_denominator,
        "dissipation_normalized_rmse": 0.3,
        "dissipation_squared_error_sum": 9.0,
        "dissipation_normalization_denominator": 100.0,
        "cumulative_dissipation_relative_error": 0.2,
        "negative_dissipation_state_count": 0,
        "maximum_thermodynamic_violation": 0.0,
    }


def _write_path_metric_fixture(path: Path) -> None:
    """Write one path metric for each split."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "path_id",
                "split",
                "family",
                "regime",
                "stress_normalized_rmse",
            ),
        )
        writer.writeheader()
        for split, error in (("train", 0.1), ("validation", 0.2), ("test", 0.3)):
            writer.writerow(
                {
                    "path_id": f"response.h5:{split}",
                    "split": split,
                    "family": "hold",
                    "regime": "elastic",
                    "stress_normalized_rmse": error,
                }
            )


def _write_prediction_fixture(path: Path) -> None:
    """Write three finite prediction paths with one frozen-test path."""
    with h5py.File(path, "w") as handle:
        paths = handle.create_group("paths")
        for split in ("train", "validation", "test"):
            group = paths.create_group(split)
            group.attrs["split"] = split
            group.attrs["path_id"] = f"response.h5:{split}"
            time = np.asarray([0.1, 0.2, 0.3])
            strain = np.column_stack((time, np.zeros((3, 5))))
            stress = 2.0 * strain
            group.create_dataset("time", data=time)
            group.create_dataset("macro_strain", data=strain)
            group.create_dataset("reference_stress", data=stress)
            group.create_dataset("predicted_stress", data=stress * 0.95)
            group.create_dataset("reference_dissipation", data=time)
            group.create_dataset("predicted_dissipation", data=time * 1.05)
