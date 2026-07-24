"""Tests for plasticity benchmark configurations.

Author:
    Zhen Hao.
Created:
    2026-07-08.
"""

from pathlib import Path

from concrete_impact.benchmarks.registry import get_benchmark_runner
from fem.io.config import load_yaml_config


def test_plasticity_benchmarks_match_semianalytic_curves(tmp_path: Path) -> None:
    """Verify plasticity benchmark YAML files against semi-analytic curves."""
    config_paths = [
        "configs/benchmarks/plasticity/plastic_j2_pure_shear.yaml",
        "configs/benchmarks/plasticity/plastic_j2_viscoplastic_material_point.yaml",
        "configs/benchmarks/plasticity/plastic_dp_pure_shear.yaml",
        "configs/benchmarks/plasticity/plastic_dp_cap_hydrostatic.yaml",
        "configs/benchmarks/plasticity/plastic_dp_cap_confined_triaxial.yaml",
    ]

    for config_path in config_paths:
        config = load_yaml_config(config_path)
        config["output"]["root"] = str(tmp_path / config["case"]["name"])
        result = get_benchmark_runner(config["case"]["name"])(config)

        assert result.passed
        assert Path(result.output_paths["response_csv"]).is_file()
        assert Path(result.output_paths["diagnostics_csv"]).is_file()
        assert Path(result.output_paths["response_png"]).is_file()
        assert Path(result.output_paths["metrics"]).is_file()
        assert Path(result.output_paths["metadata"]).is_file()
        assert Path(result.output_paths["original_config"]).is_file()
        assert Path(result.output_paths["resolved_config"]).is_file()


def test_static_j2_viscoplastic_cube_benchmark_matches_reference_curve(tmp_path: Path) -> None:
    """Verify the static cube J2 viscoplastic benchmark on a reduced test grid."""
    config = load_yaml_config("configs/benchmarks/plasticity/static_j2_viscoplastic_cube_3d.yaml")
    config["analysis"]["num_steps"] = 8
    config["benchmark"]["mesh_divisions"] = [1]
    config["output"]["root"] = str(tmp_path / config["case"]["name"])
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result.passed
    assert Path(result.output_paths["response_csv"]).is_file()
    assert Path(result.output_paths["diagnostics_csv"]).is_file()
    assert Path(result.output_paths["response_png"]).is_file()
    assert Path(result.output_paths["metrics"]).is_file()
    assert Path(result.output_paths["metadata"]).is_file()
    assert Path(result.output_paths["original_config"]).is_file()
    assert Path(result.output_paths["resolved_config"]).is_file()


def test_dynamic_j2_viscoplastic_pressure_pulse_is_stable(tmp_path: Path) -> None:
    """Verify matched explicit and implicit J2 dynamics on the pressure-pulse structure."""
    config = load_yaml_config(
        "configs/benchmarks/plasticity/dynamic_j2_viscoplastic_pressure_pulse_3d.yaml"
    )
    output_root = tmp_path / config["case"]["name"]
    config["output"]["root"] = str(output_root)
    config["output"]["mesh_path"] = str(output_root / "mesh.msh")
    config["output"]["vtk_path"] = str(output_root / "mesh.vtu")
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result.passed
    assert result.metrics["explicit_minimum_q_increment"] >= -1.0e-12
    assert result.metrics["implicit_minimum_q_increment"] >= -1.0e-12
    assert result.metrics["explicit_max_scaled_energy_residual"] < 0.05
    assert result.metrics["implicit_max_scaled_energy_residual"] < 0.01
    assert Path(result.output_paths["response_csv"]).is_file()
    assert Path(result.output_paths["response_png"]).is_file()
