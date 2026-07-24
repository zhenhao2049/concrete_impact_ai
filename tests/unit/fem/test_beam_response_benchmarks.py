"""Tests for beam response codebase benchmarks.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path

import pytest

from concrete_impact.benchmarks.registry import get_benchmark_runner
from fem.io.config import load_yaml_config


def test_elastic_tension_benchmarks_match_conjugate_response() -> None:
    """Verify static beam tension benchmarks against force-displacement conjugacy."""
    config_paths = [
        "configs/benchmarks/elastic_tension_2d_displacement.yaml",
        "configs/benchmarks/elastic_tension_2d_force.yaml",
        "configs/benchmarks/elastic_tension_3d_displacement.yaml",
        "configs/benchmarks/elastic_tension_3d_force.yaml",
    ]

    for config_path in config_paths:
        config = load_yaml_config(config_path)
        result = get_benchmark_runner(config["case"]["name"])(config)

        assert result.passed
        assert max(result.metrics.values()) < 1.0e-3
        _assert_standard_outputs_exist(result.output_paths)


def test_dynamic_tension_benchmarks_match_harmonic_response() -> None:
    """Verify dynamic beam tension benchmarks against one-dimensional rod response."""
    config_paths = [
        "configs/benchmarks/dynamic_tension_2d_explicit.yaml",
        "configs/benchmarks/dynamic_tension_2d_implicit.yaml",
        "configs/benchmarks/dynamic_tension_3d_explicit.yaml",
        "configs/benchmarks/dynamic_tension_3d_implicit.yaml",
    ]

    for config_path in config_paths:
        config = load_yaml_config(config_path)
        result = get_benchmark_runner(config["case"]["name"])(config)

        assert result.passed
        assert Path(result.output_paths["response_csv"]).is_file()
        assert Path(result.output_paths["response_png"]).is_file()
        assert Path(result.output_paths["step_final"]).is_file()


def test_wave_pulse_benchmarks_match_rod_wave_speed() -> None:
    """Verify pulse benchmarks against one-dimensional rod wave propagation."""
    config_paths = [
        "configs/benchmarks/wave_pulse_2d_explicit.yaml",
        "configs/benchmarks/wave_pulse_3d_explicit.yaml",
    ]

    for config_path in config_paths:
        config = load_yaml_config(config_path)
        result = get_benchmark_runner(config["case"]["name"])(config)

        assert result.passed
        assert "wave_speed_relative_error" in result.metrics
        _assert_standard_outputs_exist(result.output_paths)


def test_explicit_dynamic_benchmark_rejects_unstable_time_step() -> None:
    """Verify that explicit CFL violations raise directly."""
    config = load_yaml_config("configs/benchmarks/dynamic_tension_2d_explicit.yaml")
    config["analysis"]["time"]["time_step"] = 1.0e-3
    runner = get_benchmark_runner(config["case"]["name"])

    with pytest.raises(ValueError, match="CFL stability bound"):
        runner(config)


def _assert_standard_outputs_exist(output_paths: dict[str, Path]) -> None:
    """Verify standard benchmark output files."""
    assert Path(output_paths["response_csv"]).is_file()
    assert Path(output_paths["response_png"]).is_file()
    assert Path(output_paths["step_0000"]).is_file()
    assert Path(output_paths["step_mid"]).is_file()
    assert Path(output_paths["step_final"]).is_file()
    assert Path(output_paths["metrics"]).is_file()
    assert Path(output_paths["metadata"]).is_file()
    assert Path(output_paths["original_config"]).is_file()
    assert Path(output_paths["resolved_config"]).is_file()
