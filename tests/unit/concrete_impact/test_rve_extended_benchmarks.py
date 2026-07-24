"""Acceptance tests for heterogeneous and macro-embedded RVE solvers.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from pathlib import Path

import meshio

from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.core.config import load_yaml_config


def test_cylindrical_inclusion_rve_benchmark(tmp_path: Path) -> None:
    """Verify geometry convergence, bounds, phases, and nonlinear state ownership."""
    config = load_yaml_config("configs/benchmarks/rve/cylindrical_inclusion.yaml")
    config["rve"]["circumferential_divisions"] = [16, 32]
    config["output"]["root"] = str(tmp_path / "heterogeneous")
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result is not None
    assert result.passed
    assert result.metrics["geometry_area_error_32"] < result.metrics["geometry_area_error_16"]
    assert result.metrics["inclusion_plastic_state_count"] == 0.0
    assert result.metrics["nonlinear_minimum_q_increment"] >= -1.0e-12
    assert result.metrics["nonlinear_minimum_dissipation"] >= -1.0e-12
    assert result.metrics["nonlinear_maximum_q"] > 1.0e-12
    assert result.metrics["nonlinear_maximum_dissipation"] > 1.0e-12
    assert result.metrics["minimum_quadrature_weight"] > 0.0
    assert result.metrics["quasistatic_applicability_passed"] in (0.0, 1.0)


def test_single_phase_rve_macro_dynamics_matches_direct_material(tmp_path: Path) -> None:
    """Verify one independent RVE per macro point in explicit and implicit dynamics."""
    config = load_yaml_config("configs/benchmarks/rve/single_phase_dynamic_embedding.yaml")
    config["output"]["root"] = str(tmp_path / "embedding")
    config["output"]["mesh_path"] = str(tmp_path / "embedding" / "mesh.msh")
    config["output"]["vtk_path"] = str(tmp_path / "embedding" / "mesh.vtu")
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result is not None
    assert result.passed
    assert max(result.metrics.values()) <= 1.0e-9


def test_multiphase_rve_macro_dynamics_has_nonuniform_microfield(tmp_path: Path) -> None:
    """Verify heterogeneous RVEs run in both macro schemes and retain microstructure."""
    config = load_yaml_config("configs/benchmarks/rve/multiphase_dynamic_embedding.yaml")
    config["rve"]["circumferential_divisions"] = 8
    config["analysis"]["explicit"]["num_steps"] = 2
    config["analysis"]["implicit"]["num_steps"] = 1
    config["output"]["root"] = str(tmp_path / "multiphase")
    config["output"]["mesh_path"] = str(tmp_path / "multiphase" / "mesh.msh")
    config["output"]["vtk_path"] = str(tmp_path / "multiphase" / "mesh.vtu")
    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result is not None
    assert result.passed
    assert result.metrics["microfield_stress_variance"] > 1.0e-8


def test_multiphase_plastic_impact_writes_paraview_series(tmp_path: Path) -> None:
    """Verify accepted plastic states are written as readable VTU/PVD animation data."""
    config = load_yaml_config("configs/benchmarks/rve/multiphase_plastic_impact.yaml")
    config["model"]["mesh"]["divisions"] = [2, 1, 1]
    config["rve"]["circumferential_divisions"] = 8
    config["rve"]["execution"] = {
        "backend": "serial",
        "workers": 1,
        "threads_per_worker": 1,
    }
    config["analysis"]["explicit"].update({"time_step": 4.0e-3, "num_steps": 8})
    config["analysis"]["implicit"].update({"time_step": 8.0e-3, "num_steps": 4})
    output_root = tmp_path / "plastic_impact"
    config["output"].update(
        {
            "root": str(output_root),
            "mesh_path": str(output_root / "mesh.msh"),
            "vtk_path": str(output_root / "mesh.vtu"),
            "save_quadrature_vtk": False,
        }
    )

    result = get_benchmark_runner(config["case"]["name"])(config)

    assert result.metrics["explicit_maximum_q"] > 1.0e-6
    assert result.metrics["implicit_maximum_q"] > 1.0e-6
    assert result.output_paths["explicit_collection"].is_file()
    assert result.output_paths["implicit_collection"].is_file()
    explicit_pvd = result.output_paths["explicit_collection"].read_text(encoding="utf-8")
    assert explicit_pvd.count("<DataSet") == 9
    final_mesh = meshio.read(result.output_paths["explicit_last_frame"])
    assert "displacement" in final_mesh.point_data
    assert "von_mises_stress" in final_mesh.cell_data
    assert "phase_matrix__maximum_equivalent_plastic_strain_max" in final_mesh.cell_data
