"""Single-phase RVE embedding benchmark for macro impact dynamics.

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
from fem.cases import RunResult
from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.mesh.structured import build_structured_hex8_box
from fem.preprocess import build_preprocess_data
from fem.rve import (
    RVEHomogenizedMaterial,
    StructuredHex8RVE,
    build_structured_hex8_periodic_constraint,
)


def run_single_phase_rve_dynamic_embedding(config: dict[str, Any]) -> RunResult:
    """Compare direct J2 and one-RVE-per-Gauss-point macro dynamics."""
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    direct_bundle = build_preprocess_data(model_def, output_def)
    rve_model = _build_single_phase_rve(config, direct_bundle.material)
    rve_material = RVEHomogenizedMaterial(rve_model, "single_phase_rve_material")
    rve_bundle = replace(direct_bundle, material=rve_material)
    dirichlet = _build_left_dirichlet(direct_bundle)
    load_function = _build_pressure_load_function(direct_bundle, config)
    update_settings = _build_material_update_settings(config)
    explicit_time = _build_time_settings(config["analysis"]["explicit"])
    implicit_time = _build_time_settings(config["analysis"]["implicit"])
    newton = _build_newton_settings(config)
    dof_count = direct_bundle.mesh_info.dof_map.size
    initial_displacement = np.zeros(dof_count)
    initial_velocity = np.zeros(dof_count)

    direct_explicit = solve_material_explicit(
        direct_bundle,
        explicit_time,
        initial_displacement,
        initial_velocity,
        dirichlet,
        load_function,
        "three_dimensional",
        update_settings,
    )
    rve_explicit = solve_material_explicit(
        rve_bundle,
        explicit_time,
        initial_displacement,
        initial_velocity,
        dirichlet,
        load_function,
        "three_dimensional",
        update_settings,
    )
    direct_implicit = solve_material_implicit_newmark(
        direct_bundle,
        implicit_time,
        initial_displacement,
        initial_velocity,
        dirichlet,
        load_function,
        "three_dimensional",
        update_settings,
        newton,
    )
    rve_implicit = solve_material_implicit_newmark(
        rve_bundle,
        implicit_time,
        initial_displacement,
        initial_velocity,
        dirichlet,
        load_function,
        "three_dimensional",
        update_settings,
        newton,
    )
    scales = config["verification"]
    metrics = {}
    for scheme, direct, rve in (
        ("explicit", direct_explicit, rve_explicit),
        ("implicit", direct_implicit, rve_implicit),
    ):
        metrics[f"{scheme}_displacement_error"] = _scaled_error(
            direct.displacement,
            rve.displacement,
            float(scales["displacement_scale"]),
        )
        metrics[f"{scheme}_velocity_error"] = _scaled_error(
            direct.velocity,
            rve.velocity,
            float(scales["velocity_scale"]),
        )
        metrics[f"{scheme}_stress_error"] = _scaled_error(
            direct.stresses,
            rve.stresses,
            float(scales["stress_scale"]),
        )
        metrics[f"{scheme}_energy_error"] = _scaled_error(
            direct.energy_residual,
            rve.energy_residual,
            float(scales["energy_scale"]),
        )
        metrics[f"{scheme}_q_error"] = _scaled_error(
            direct.equivalent_plastic_strain,
            rve.equivalent_plastic_strain,
            float(scales["strain_scale"]),
        )
    tolerance = float(scales["rve_embedding_tolerance"])
    passed = all(value <= tolerance for value in metrics.values())
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


def _build_single_phase_rve(config: dict[str, Any], material: Any) -> StructuredHex8RVE:
    """Build the configured homogeneous microcell used by every macro point."""
    divisions = (
        int(config["rve"]["divisions"][0]),
        int(config["rve"]["divisions"][1]),
        int(config["rve"]["divisions"][2]),
    )
    lengths = (
        float(config["rve"]["lengths"][0]),
        float(config["rve"]["lengths"][1]),
        float(config["rve"]["lengths"][2]),
    )
    nodes, elements = build_structured_hex8_box(lengths, divisions)
    return StructuredHex8RVE(
        nodes=nodes,
        elements=elements,
        divisions=divisions,
        material=material,
        constraint=build_structured_hex8_periodic_constraint(nodes, divisions),
        newton_settings=_build_newton_settings(config),
        hill_mandel_power_scale=float(config["rve"]["hill_mandel_power_scale"]),
    )


def _scaled_error(reference: np.ndarray, candidate: np.ndarray, scale: float) -> float:
    """Compute a fixed-scale maximum absolute difference."""
    return float(np.max(np.abs(candidate - reference)) / scale)
