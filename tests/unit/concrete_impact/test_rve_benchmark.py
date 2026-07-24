"""Tests for the project single-phase RVE acceptance benchmark.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from pathlib import Path

from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.core.config import load_yaml_config


def test_single_phase_rve_benchmark_passes_and_writes_dataset(tmp_path: Path) -> None:
    """Verify all homogeneous grids recover the material-point response."""
    config = load_yaml_config(
        "configs/benchmarks/rve/single_phase_j2_viscoplastic.yaml"
    )
    config["output"]["root"] = str(tmp_path / "single_phase_rve")
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result is not None
    assert result.passed
    assert result.output_paths["dataset"].is_file()
    assert result.output_paths["metrics"].is_file()
    assert result.metrics["grid_4_stress_error"] <= 1.0e-9
    assert result.metrics["effective_tangent_direction_error"] <= 1.0e-7
