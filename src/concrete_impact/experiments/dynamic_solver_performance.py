"""Equal-accuracy explicit and implicit J2 dynamics performance experiment.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import csv
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from concrete_impact.benchmarks.dynamic_viscoplastic import (
    run_dynamic_j2_viscoplastic_benchmark,
)
from fem.io.config import load_yaml_config


def run_dynamic_solver_performance(config_path: str | Path, repetitions: int = 5) -> dict:
    """Run repeated matched-accuracy timing and write performance records."""
    config = load_yaml_config(config_path)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dynamic_j2_viscoplastic_benchmark(config)  # Warm up imports and sparse kernels.
    rows = []
    for repetition in range(1, repetitions + 1):
        run_config = copy.deepcopy(config)
        run_config["analysis"]["execution_order"] = (
            ["explicit", "implicit"]
            if repetition % 2 == 1
            else ["implicit", "explicit"]
        )
        profiler = cProfile.Profile()
        start = perf_counter()
        profiler.enable()
        result = run_dynamic_j2_viscoplastic_benchmark(run_config)
        profiler.disable()
        end_to_end_time = perf_counter() - start
        profile = _summarize_profile(profiler)
        rows.append(
            {
                "repetition": repetition,
                "execution_order": "-".join(run_config["analysis"]["execution_order"]),
                "explicit_solve_time": result.metrics["explicit_solve_time"],
                "implicit_solve_time": result.metrics["implicit_solve_time"],
                "solver_speedup": result.metrics["implicit_speedup"],
                "end_to_end_time": end_to_end_time,
                "displacement_history_scaled_error": result.metrics[
                    "displacement_history_scaled_error"
                ],
                "peak_displacement_scaled_error": result.metrics[
                    "peak_displacement_scaled_error"
                ],
                "dissipated_energy_scaled_error": result.metrics[
                    "dissipated_energy_scaled_error"
                ],
                "explicit_energy_residual": result.metrics[
                    "explicit_max_scaled_energy_residual"
                ],
                "implicit_energy_residual": result.metrics[
                    "implicit_max_scaled_energy_residual"
                ],
                **profile,
            }
        )

    metrics = _summarize_rows(rows, config)
    _write_performance_csv(output_root / "performance.csv", rows)
    _write_json(output_root / "performance_metrics.json", metrics)
    _write_summary(output_root / "performance_summary.md", metrics, config)

    return metrics


def _summarize_rows(rows: list[dict], config: dict) -> dict:
    """Compute median timing, dispersion, and matched-accuracy pass status."""
    explicit_times = np.asarray([row["explicit_solve_time"] for row in rows])
    implicit_times = np.asarray([row["implicit_solve_time"] for row in rows])
    explicit_median = float(np.median(explicit_times))
    implicit_median = float(np.median(implicit_times))
    response_tolerance = float(config["verification"]["response_tolerance"])
    energy_tolerance = float(config["verification"]["energy_tolerance"])
    latest = rows[-1]
    speedup = explicit_median / implicit_median
    explicit_mad = _median_absolute_deviation(explicit_times)
    implicit_mad = _median_absolute_deviation(implicit_times)
    conservative_speedup = (explicit_median - explicit_mad) / (
        implicit_median + implicit_mad
    )
    passed = (
        latest["displacement_history_scaled_error"] <= response_tolerance
        and latest["peak_displacement_scaled_error"] <= response_tolerance
        and latest["dissipated_energy_scaled_error"] <= response_tolerance
        and latest["explicit_energy_residual"] <= energy_tolerance
        and latest["implicit_energy_residual"] <= energy_tolerance
    )
    summary = {
        "passed": passed,
        "repetitions": len(rows),
        "explicit_median_solve_time": explicit_median,
        "implicit_median_solve_time": implicit_median,
        "explicit_mad_solve_time": explicit_mad,
        "implicit_mad_solve_time": implicit_mad,
        "implicit_speedup": speedup,
        "conservative_implicit_speedup": conservative_speedup,
        "implicit_stably_faster": conservative_speedup > 1.0,
        "engineering_speed_target_met": speedup >= 1.2,
        "displacement_history_scaled_error": latest[
            "displacement_history_scaled_error"
        ],
        "peak_displacement_scaled_error": latest["peak_displacement_scaled_error"],
        "dissipated_energy_scaled_error": latest["dissipated_energy_scaled_error"],
        "explicit_max_scaled_energy_residual": latest["explicit_energy_residual"],
        "implicit_max_scaled_energy_residual": latest["implicit_energy_residual"],
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }
    for name in (
        "profile_material_time",
        "profile_assembly_time",
        "profile_linear_solve_time",
        "profile_rve_time",
        "profile_history_time",
    ):
        summary[f"median_{name}"] = float(np.median([row[name] for row in rows]))
    return summary


def _median_absolute_deviation(values: np.ndarray) -> float:
    """Compute the median absolute deviation of repeated timings."""
    median = np.median(values)

    return float(np.median(np.abs(values - median)))


def _summarize_profile(profiler: cProfile.Profile) -> dict[str, float]:
    """Aggregate exclusive Python time into non-overlapping numerical categories."""
    categories = {
        "profile_material_time": ("materials/plasticity.py", "rve/material.py"),
        "profile_assembly_time": ("assembly/nonlinear_solid.py",),
        "profile_linear_solve_time": ("solvers/linear.py",),
        "profile_rve_time": ("rve/solver.py",),
        "profile_history_time": ("dynamics/material.py",),
    }
    totals = {name: 0.0 for name in categories}
    stats = pstats.Stats(profiler)
    raw_stats = stats.stats  # type: ignore[attr-defined]
    for (filename, _line, _function), values in raw_stats.items():
        total_time = float(values[2])
        for category, fragments in categories.items():
            if any(fragment in filename for fragment in fragments):
                totals[category] += total_time
                break
    return totals


def _write_performance_csv(path: Path, rows: list[dict]) -> None:
    """Write repeated performance samples."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    """Write stable JSON output."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary(path: Path, metrics: dict, config: dict) -> None:
    """Write a compact Chinese performance conclusion."""
    status = "通过" if metrics["passed"] else "未通过"
    content = (
        "# J2 粘塑性显隐式等精度性能总结\n\n"
        f"　　验收状态：{status}。显式中位求解时间为 "
        f"{metrics['explicit_median_solve_time']:.6f} s，隐式中位求解时间为 "
        f"{metrics['implicit_median_solve_time']:.6f} s，隐式速度比为 "
        f"{metrics['implicit_speedup']:.6f}。\n\n"
        f"　　位移历史固定尺度误差为 {metrics['displacement_history_scaled_error']:.6e}，"
        f"峰值位移固定尺度误差为 {metrics['peak_displacement_scaled_error']:.6e}，"
        f"累计耗散固定尺度误差为 {metrics['dissipated_energy_scaled_error']:.6e}。"
        f"响应误差阈值为 {float(config['verification']['response_tolerance']):.6e}。\n\n"
        "　　该优势来自细网格下显式 CFL 时间步受最小单元边长限制，而当前压力脉冲"
        "的主响应时间尺度允许隐式 Newmark 使用更大的时间步；速度比较不包含网格生成、"
        "VTK、CSV 和绘图时间。\n"
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Run the configured explicit-implicit performance experiment."""
    parser = argparse.ArgumentParser(description="Run matched J2 dynamics timing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    arguments = parser.parse_args()
    run_dynamic_solver_performance(arguments.config, arguments.repetitions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
