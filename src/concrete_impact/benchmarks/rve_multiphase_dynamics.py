"""Multiphase quasi-static RVE embedding benchmark for macro dynamics.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from concrete_impact.benchmarks.dynamic_viscoplastic import (
    _build_left_dirichlet,
    _build_material_update_settings,
    _build_model_def,
    _build_newton_settings,
    _build_output_def,
    _build_pressure_load_function,
    _build_time_settings,
)
from concrete_impact.benchmarks.rve_heterogeneous import (
    _elastic_material,
    _newton_settings,
    _viscoplastic_material,
)
from concrete_impact.core.config import load_yaml_config
from fem.cases import RunResult
from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.preprocess import build_preprocess_data
from fem.rve import (
    RVEExecutionSettings,
    RVEHomogenizedMaterial,
    RVERequest,
    build_cylindrical_two_phase_rve,
    initialize_rve_state,
    solve_rve_microequilibrium,
)


def run_multiphase_rve_dynamic_embedding(config: dict[str, Any]) -> RunResult:
    """Run explicit and implicit macro dynamics with one heterogeneous RVE per point."""
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    direct_bundle = build_preprocess_data(model_def, output_def)
    heterogeneous_config = load_yaml_config(config["rve"]["reference_config"])
    rve_model = _build_heterogeneous_model(config, heterogeneous_config)
    execution = config["rve"]["execution"]
    rve_material = RVEHomogenizedMaterial(
        rve_model,
        "multiphase_rve_material",
        RVEExecutionSettings(
            backend=execution["backend"],
            workers=int(execution["workers"]),
            threads_per_worker=int(execution["threads_per_worker"]),
        ),
    )
    bundle = replace(direct_bundle, material=rve_material)
    dirichlet = _build_left_dirichlet(bundle)
    load_function = _build_pressure_load_function(bundle, config)
    update = _build_material_update_settings(config)
    initial = np.zeros(bundle.mesh_info.dof_map.size)
    try:
        explicit = solve_material_explicit(
            bundle,
            _build_time_settings(config["analysis"]["explicit"]),
            initial,
            initial,
            dirichlet,
            load_function,
            "three_dimensional",
            update,
        )
        implicit = solve_material_implicit_newmark(
            bundle,
            _build_time_settings(config["analysis"]["implicit"]),
            initial,
            initial,
            dirichlet,
            load_function,
            "three_dimensional",
            update,
            _build_newton_settings(config),
        )
    finally:
        rve_material.close()
    verification_strain = np.asarray(
        config["verification"]["microfield_macro_strain"], dtype=np.float64
    )
    micro_response = solve_rve_microequilibrium(
        rve_model,
        RVERequest(verification_strain, 1.0e-3, update),
        initialize_rve_state(rve_model),
    )
    microfield_variance = float(
        np.mean(np.sum((micro_response.micro_stresses - micro_response.macro_stress) ** 2, axis=1))
        / float(config["verification"]["stress_scale"]) ** 2
    )
    phase_key = "phase_matrix__equivalent_plastic_strain"
    metrics = {
        "explicit_energy_residual": _maximum_scaled(
            explicit.energy_residual, float(config["verification"]["energy_scale"])
        ),
        "implicit_energy_residual": _maximum_scaled(
            implicit.energy_residual, float(config["verification"]["energy_scale"])
        ),
        "explicit_matrix_q_maximum": float(np.max(explicit.material_diagnostics[phase_key])),
        "implicit_matrix_q_maximum": float(np.max(implicit.material_diagnostics[phase_key])),
        "microfield_stress_variance": microfield_variance,
        "explicit_solve_time": explicit.solve_time,
        "implicit_solve_time": implicit.solve_time,
    }
    passed = (
        metrics["explicit_energy_residual"] <= float(config["verification"]["energy_tolerance"])
        and metrics["implicit_energy_residual"] <= float(config["verification"]["energy_tolerance"])
        and metrics["microfield_stress_variance"]
        >= float(config["verification"]["minimum_microfield_variance"])
    )
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps({"passed": passed, "metrics": metrics}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={"metrics": metrics_path},
        passed=passed,
    )


def _build_heterogeneous_model(
    config: dict[str, Any],
    reference: dict[str, Any],
):
    """Build the configured reduced-cost heterogeneous benchmark cell."""
    lengths = reference["rve"]["lengths"]
    mesh = build_cylindrical_ogrid_hex8(
        (float(lengths[0]), float(lengths[1]), float(lengths[2])),
        float(reference["rve"]["inclusion_volume_fraction"]),
        int(config["rve"]["circumferential_divisions"]),
        int(config["rve"]["matrix_radial_divisions"]),
        int(config["rve"]["axial_divisions"]),
    )
    return build_cylindrical_two_phase_rve(
        mesh,
        _viscoplastic_material(reference["materials"]["matrix_viscoplastic"]),
        _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
        _newton_settings(reference),
        float(reference["verification"]["hill_mandel_power_scale"]),
    )


def _maximum_scaled(values: np.ndarray, scale: float) -> float:
    """Compute the maximum absolute fixed-scale history value."""
    return float(np.max(np.abs(values)) / scale)
