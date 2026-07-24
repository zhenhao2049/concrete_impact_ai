"""Render traceable scientific figures from one RVE-RNO evaluation directory.

Contents:
    Coverage, loss, error, parity, representative-path, and throughput figures.
Author:
    Zhen Hao.
Created:
    2026-07-17.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

from concrete_impact.core.progress import write_json_atomic

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def render_rve_rno_evaluation_figures(
    run_directory: Path,
    evaluation_directory: Path,
    subset_summary_path: Path,
    evaluation_config_path: Path,
    output_directory: Path,
    dpi: int,
    representative_path_count: int,
) -> Path:
    """Render six RVE-RNO figure groups and publish their source hashes."""
    if output_directory.exists():
        raise FileExistsError(f"RVE-RNO figure output exists: {output_directory}.")
    output_directory.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    sources = {
        "subset_summary": subset_summary_path,
        "training_metrics": run_directory / "metrics.jsonl",
        "training_summary": run_directory / "training_summary.json",
        "artifact": run_directory / "best_model.pt",
        "artifact_metadata": run_directory / "artifact_metadata.json",
        "path_metrics": evaluation_directory / "path_metrics.csv",
        "group_metrics": evaluation_directory / "group_metrics.json",
        "predictions": evaluation_directory / "predictions.h5",
        "tangent_metrics": evaluation_directory / "tangent_metrics.json",
        "inference_performance": evaluation_directory / "inference_performance.json",
    }
    records = (
        _render_data_composition(sources, output_directory, dpi),
        _render_training_history(sources, output_directory, dpi),
        _render_split_errors(sources, output_directory, dpi),
        _render_stress_parity(sources, output_directory, dpi),
        _render_representative_paths(
            sources,
            output_directory,
            dpi,
            representative_path_count,
        ),
        _render_throughput(sources, output_directory, dpi),
    )
    common_provenance = {
        "evaluation_config": {
            "path": str(evaluation_config_path),
            "sha256": _sha256_file(evaluation_config_path),
        },
        "model": {
            "path": str(sources["artifact"]),
            "sha256": _sha256_file(sources["artifact"]),
        },
        "artifact_metadata": {
            "path": str(sources["artifact_metadata"]),
            "sha256": _sha256_file(sources["artifact_metadata"]),
        },
        "evaluation_files": {
            str(path): _sha256_file(path)
            for path in (
                sources["path_metrics"],
                sources["group_metrics"],
                sources["predictions"],
                sources["tangent_metrics"],
                sources["inference_performance"],
            )
        },
    }
    for record in records:
        record["common_provenance"] = common_provenance
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "figure_count": len(records),
        "common_provenance": common_provenance,
        "figures": records,
    }
    manifest_path = output_directory / "figure_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def _render_data_composition(
    sources: dict[str, Path], output: Path, dpi: int
) -> dict[str, Any]:
    """Render Pilot family, split, direction, and increment composition."""
    summary = _read_json(sources["subset_summary"])
    panels = (
        ("family_counts", "路径族"),
        ("split_counts", "数据划分"),
        ("tensor_direction_counts", "张量方向"),
        ("accepted_increment_counts", "接受增量数"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        counts = summary[key]
        labels = tuple(counts)
        values = tuple(int(counts[label]) for label in labels)
        axis.bar(np.arange(len(labels)), values, color="#4c78a8")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel("路径数")
    figure.tight_layout()
    return _save_figure(
        figure,
        output,
        "rno-data-composition",
        dpi,
        (sources["subset_summary"],),
    )


def _render_training_history(
    sources: dict[str, Path], output: Path, dpi: int
) -> dict[str, Any]:
    """Render arbitrary-length train and validation loss histories."""
    metrics = _read_json_lines(sources["training_metrics"])
    summary = _read_json(sources["training_summary"])
    by_split = {
        split: tuple(record for record in metrics if record["split"] == split)
        for split in ("train", "validation")
    }
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("RVE-RNO training figure requires train and validation records.")
    panels = (
        ("total_loss", "综合损失", True),
        ("stress_loss", "应力路径损失", True),
        ("dissipation_loss", "耗散监督损失", True),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), sharex=True)
    maximum_epoch = max(int(record["epoch"]) for record in metrics)
    warmup_epoch = min(20, maximum_epoch - 1)
    display_spike_multiplier = 8.0
    omitted_point_counts: dict[str, dict[str, int]] = {}
    for axis, (name, title, logarithmic) in zip(axes.flat, panels, strict=True):
        panel_values = []
        for split, label, color in (
            ("train", "训练集", "#1f77b4"),
            ("validation", "验证集", "#d62728"),
        ):
            epochs = np.asarray([record["epoch"] for record in by_split[split]])
            values = np.asarray([record[name] for record in by_split[split]], dtype=float)
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("RVE-RNO training figure contains non-finite metrics.")
            panel_values.append((split, label, color, epochs, values))

        positive = np.concatenate([values for _, _, _, _, values in panel_values])
        if np.any(positive <= 0.0):
            raise FloatingPointError("Logarithmic RVE-RNO losses must be positive.")
        omitted_point_counts[name] = {}
        for split, label, color, epochs, values in panel_values:
            post_warmup = epochs > warmup_epoch
            post_warmup_median = float(np.median(values[post_warmup]))
            display_upper = display_spike_multiplier * post_warmup_median
            plotted_values = values.copy()
            omitted = post_warmup & (values > display_upper)
            plotted_values[omitted] = np.nan
            omitted_point_counts[name][split] = int(np.count_nonzero(omitted))
            axis.plot(epochs, plotted_values, color=color, linewidth=1.2, label=label)
        axis.axvline(int(summary["best_epoch"]), color="black", linestyle=":")
        axis.set_title(title)
        axis.set_xlabel("训练轮次")
        if logarithmic:
            axis.set_yscale("log")
    axes[0].legend()
    figure.tight_layout()
    record = _save_figure(
        figure,
        output,
        "rno-training-validation-loss",
        dpi,
        (sources["training_metrics"], sources["training_summary"]),
    )
    record["display_filter"] = {
        "scope": "visualization_only",
        "raw_metrics_modified": False,
        "warmup_epoch": warmup_epoch,
        "post_warmup_median_multiplier": display_spike_multiplier,
        "omitted_point_counts": omitted_point_counts,
    }
    return record


def _render_split_errors(
    sources: dict[str, Path], output: Path, dpi: int
) -> dict[str, Any]:
    """Render train, validation, and frozen-test path-error distributions."""
    records = _read_csv(sources["path_metrics"])
    splits = ("train", "validation", "test")
    values = [
        np.asarray(
            [
                float(record["stress_normalized_rmse"])
                for record in records
                if record["split"] == split
            ]
        )
        for split in splits
    ]
    if any(array.size == 0 or not np.all(np.isfinite(array)) for array in values):
        raise FloatingPointError("RVE-RNO split-error figure requires finite nonempty splits.")
    figure, axis = plt.subplots(figsize=(7.5, 5.4))
    axis.boxplot(values, tick_labels=("训练", "验证", "冻结测试"), showfliers=True)
    axis.set_ylabel("归一化应力历史误差")
    axis.set_title("RVE-RNO 路径误差分布")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(
        figure,
        output,
        "rno-split-error-distribution",
        dpi,
        (sources["path_metrics"], sources["group_metrics"]),
    )


def _render_stress_parity(
    sources: dict[str, Path], output: Path, dpi: int
) -> dict[str, Any]:
    """Render six-component frozen-test stress prediction parity."""
    reference = []
    predicted = []
    with h5py.File(sources["predictions"], "r") as handle:
        for group in handle["paths"].values():
            if group.attrs["split"] == "test":
                reference.append(group["reference_stress"][...])
                predicted.append(group["predicted_stress"][...])
    if not reference:
        raise ValueError("RVE-RNO parity figure requires frozen-test predictions.")
    reference_values = np.concatenate(reference, axis=0)
    predicted_values = np.concatenate(predicted, axis=0)
    if not np.all(np.isfinite(reference_values)) or not np.all(np.isfinite(predicted_values)):
        raise FloatingPointError("RVE-RNO parity figure contains non-finite stress.")
    labels = ("xx", "yy", "zz", "yz", "xz", "xy")
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for component, axis in enumerate(axes.flat):
        x = reference_values[:, component]
        y = predicted_values[:, component]
        lower = min(float(np.min(x)), float(np.min(y)))
        upper = max(float(np.max(x)), float(np.max(y)))
        axis.scatter(x, y, s=5, alpha=0.28, color="#4c78a8")
        axis.plot((lower, upper), (lower, upper), color="black", linestyle="--")
        axis.set_title(labels[component])
        axis.set_xlabel("RVE 参照应力")
        axis.set_ylabel("RNO 预测应力")
    figure.tight_layout()
    return _save_figure(
        figure,
        output,
        "rno-test-stress-parity",
        dpi,
        (sources["predictions"], sources["artifact"], sources["artifact_metadata"]),
    )


def _render_representative_paths(
    sources: dict[str, Path],
    output: Path,
    dpi: int,
    path_count: int,
) -> dict[str, Any]:
    """Render low-error test-path stress histories and projected loops."""
    metrics = _read_csv(sources["path_metrics"])
    selected_ids = select_representative_test_path_ids(metrics, path_count)
    record_by_id = {record["path_id"]: record for record in metrics}
    with h5py.File(sources["predictions"], "r") as handle:
        group_by_id = {
            str(group.attrs["path_id"]): name
            for name, group in handle["paths"].items()
            if group.attrs["split"] == "test"
        }
        if not set(selected_ids).issubset(group_by_id):
            raise ValueError("Representative RVE-RNO paths are missing predictions.")
        selected = tuple((path_id, group_by_id[path_id]) for path_id in selected_ids)
        figure, axes = plt.subplots(2, path_count, figsize=(3.2 * path_count, 5.6))
        axes = np.asarray(axes).reshape(2, path_count)
        for column, (path_id, group_name) in enumerate(selected):
            group = handle[f"paths/{group_name}"]
            time = group["time"][...]
            strain = group["macro_strain"][...]
            reference = group["reference_stress"][...]
            predicted = group["predicted_stress"][...]
            direction = strain[int(np.argmax(np.linalg.norm(strain, axis=1)))]
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                raise ValueError(f"Representative RVE-RNO path has zero strain: {path_id}.")
            direction = direction / norm
            projected_strain = strain @ direction
            projected_reference = reference @ direction
            projected_prediction = predicted @ direction
            axes[0, column].plot(time, reference[:, 0], label="RVE", color="#1f77b4")
            axes[0, column].plot(
                time, predicted[:, 0], label="RNO", color="#d62728", linestyle="--"
            )
            axes[1, column].plot(projected_strain, projected_reference, color="#1f77b4")
            axes[1, column].plot(
                projected_strain, projected_prediction, color="#d62728", linestyle="--"
            )
            metric = record_by_id[path_id]
            axes[0, column].set_title(
                f"{metric['family']} / {metric['regime']}\n"
                f"误差={float(metric['stress_normalized_rmse']):.3f}",
                fontsize=8,
            )
            axes[0, column].set_xlabel("时间 / s")
            axes[1, column].set_xlabel("投影应变")
        axes[0, 0].set_ylabel("xx 应力")
        axes[1, 0].set_ylabel("投影应力")
        axes[0, 0].legend(fontsize=8)
        figure.tight_layout()
    record = _save_figure(
        figure,
        output,
        "rno-representative-test-paths",
        dpi,
        (sources["predictions"], sources["path_metrics"]),
    )
    record["selection"] = {
        "criterion": "minimum_stress_error_per_family_regime_then_global_minimum",
        "path_ids": list(selected_ids),
        "stress_normalized_rmse": [
            float(record_by_id[path_id]["stress_normalized_rmse"])
            for path_id in selected_ids
        ],
    }
    return record


def select_representative_test_path_ids(
    records: tuple[dict[str, str], ...],
    path_count: int,
) -> tuple[str, ...]:
    """Select low-error family-regime representatives and global-best fillers."""
    test_records = [record for record in records if record["split"] == "test"]
    if len(test_records) < path_count:
        raise ValueError("RVE-RNO representative figure requests too many test paths.")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record in test_records:
        groups.setdefault((record["family"], record["regime"]), []).append(record)

    selected = []
    for key in sorted(groups):
        group = groups[key]
        representative = min(
            group,
            key=lambda record: (
                float(record["stress_normalized_rmse"]),
                record["path_id"],
            ),
        )
        selected.append(representative["path_id"])
        if len(selected) == path_count:
            return tuple(selected)

    remaining = [record for record in test_records if record["path_id"] not in selected]
    fillers = sorted(
        remaining,
        key=lambda record: (
            float(record["stress_normalized_rmse"]),
            record["path_id"],
        ),
    )
    selected.extend(record["path_id"] for record in fillers[: path_count - len(selected)])
    return tuple(selected)


def _render_throughput(
    sources: dict[str, Path], output: Path, dpi: int
) -> dict[str, Any]:
    """Render local update throughput and cross-platform throughput ratio."""
    performance = _read_json(sources["inference_performance"])
    batch_results = performance["batch_results"]
    batch_sizes = np.asarray([record["batch_size"] for record in batch_results])
    throughput = np.asarray([record["median_states_per_second"] for record in batch_results])
    ratios = performance["cross_platform_local_material_update_throughput_ratio"]
    ratio_values = np.asarray([ratios[str(value)] for value in batch_sizes])
    if not np.all(np.isfinite(throughput)) or not np.all(np.isfinite(ratio_values)):
        raise FloatingPointError("RVE-RNO throughput figure contains non-finite values.")
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    axes[0].plot(batch_sizes, throughput, marker="o", color="#1f77b4")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("批量材料点数")
    axes[0].set_ylabel("RNO 局部更新吞吐率 / 状态每秒")
    axes[0].set_title("本地 GPU 局部更新吞吐率")
    axes[0].grid(alpha=0.25)
    axes[1].plot(batch_sizes, ratio_values, marker="s", color="#d62728")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("批量材料点数")
    axes[1].set_ylabel("跨平台局部材料更新吞吐比")
    axes[1].set_title("Slurm c48 / 本地 GPU RNO")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    return _save_figure(
        figure,
        output,
        "rno-local-update-throughput",
        dpi,
        (sources["inference_performance"],),
    )


def _save_figure(
    figure: Any,
    output: Path,
    stem: str,
    dpi: int,
    sources: tuple[Path, ...],
) -> dict[str, Any]:
    """Save one figure as PNG and SVG with source and output hashes."""
    png = output / f"{stem}.png"
    svg = output / f"{stem}.svg"
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "name": stem,
        "sources": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in sources
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in (png, svg)
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RVE-RNO figure source is not a JSON object: {path}.")
    return payload


def _read_json_lines(path: Path) -> tuple[dict[str, Any], ...]:
    """Read nonempty structured metric records."""
    records = tuple(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    )
    if not records:
        raise ValueError(f"RVE-RNO JSON-lines source is empty: {path}.")
    return records


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    """Read one nonempty CSV table."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        records = tuple(csv.DictReader(stream))
    if not records:
        raise ValueError(f"RVE-RNO CSV source is empty: {path}.")
    return records


def _sha256_file(path: Path) -> str:
    """Compute one complete file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
