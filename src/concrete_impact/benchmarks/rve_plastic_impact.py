"""Plastic loading-unloading impact benchmark with heterogeneous FE2 material.

Contents:
    Impact benchmark orchestration and multiphase RVE solution schemes.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

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
    _update_settings,
    _viscoplastic_material,
)
from concrete_impact.benchmarks.rve_multiphase_dynamics import _maximum_scaled
from concrete_impact.core.config import load_yaml_config
from fem.cases import RunResult
from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.post.material_dynamic import (
    compute_strain_rate_spectrum,
    extract_quadrature_strain_history,
    write_material_dynamic_vtk_series,
)
from fem.preprocess import build_preprocess_data
from fem.rve import (
    RVEExecutionSettings,
    RVEHomogenizedMaterial,
    build_cylindrical_two_phase_rve,
    compute_linear_rve_dynamic_tangent,
    compute_rve_dynamic_screening,
)
from fem.solvers.progress import ProgressCallback


@dataclass(frozen=True)
class PlasticImpactSimulation:
    """Store accepted FE2 impact histories for independent convergence checks."""

    bundle: Any
    rve_model: Any
    material_update_settings: Any
    explicit: Any
    implicit: Any
    spectrum: Any
    screening: list[dict[str, object]]


@dataclass(frozen=True)
class PlasticImpactSchemeSimulation:
    """Store one accepted FE2 impact history for a selected time integrator."""

    bundle: Any
    rve_model: Any
    material_update_settings: Any
    solution: Any


def run_multiphase_rve_plastic_impact(config: dict[str, Any]) -> RunResult:
    """Run a fixed plastic impact and verify loading-unloading histories."""
    simulation = solve_multiphase_rve_plastic_impact(config)
    comparison = _build_common_grid_comparison(
        simulation.bundle,
        simulation.explicit,
        simulation.implicit,
    )
    metrics = _compute_metrics(
        config,
        simulation.bundle,
        simulation.explicit,
        simulation.implicit,
        comparison,
        simulation.spectrum,
        simulation.screening,
    )
    passed = _evaluate_pass(config["verification"], metrics)
    output_paths = _write_outputs(
        config,
        simulation.bundle,
        simulation.explicit,
        simulation.implicit,
        simulation.spectrum,
        simulation.screening,
        metrics,
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths=output_paths,
        passed=passed,
    )


def solve_multiphase_rve_plastic_impact(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> PlasticImpactSimulation:
    """Solve explicit and implicit FE2 impact histories without writing outputs."""
    execution_order = tuple(config["analysis"]["execution_order"])
    if len(execution_order) != 2 or set(execution_order) != {"explicit", "implicit"}:
        raise ValueError(
            "Plastic FE2 impact execution_order must contain explicit and implicit once."
        )
    simulations = {
        scheme: solve_multiphase_rve_plastic_impact_scheme(
            config,
            cast(Literal["explicit", "implicit"], scheme),
            progress_callback,
        )
        for scheme in execution_order
    }
    explicit_run = simulations["explicit"]
    implicit_run = simulations["implicit"]
    bundle = explicit_run.bundle
    explicit = explicit_run.solution
    implicit = implicit_run.solution
    nonlinear_model = explicit_run.rve_model
    update = explicit_run.material_update_settings

    comparison = _build_common_grid_comparison(bundle, explicit, implicit)
    spectrum = compute_strain_rate_spectrum(
        explicit.times,
        extract_quadrature_strain_history(bundle, explicit.displacement),
    )
    reference = load_yaml_config(config["rve"]["reference_config"])
    mesh = _build_rve_mesh(config, reference)
    elastic_model = build_cylindrical_two_phase_rve(
        mesh,
        _elastic_material("matrix", reference["materials"]["matrix_elastic"]),
        _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
        _newton_settings(reference),
        float(reference["verification"]["hill_mandel_power_scale"]),
    )
    screening = _compute_scale_screening(config, reference, elastic_model, spectrum)
    del comparison
    return PlasticImpactSimulation(
        bundle=bundle,
        rve_model=nonlinear_model,
        material_update_settings=update,
        explicit=explicit,
        implicit=implicit,
        spectrum=spectrum,
        screening=screening,
    )


def solve_multiphase_rve_plastic_impact_scheme(
    config: dict[str, Any],
    scheme: Literal["explicit", "implicit"],
    progress_callback: ProgressCallback | None = None,
) -> PlasticImpactSchemeSimulation:
    """Solve one explicitly selected FE2 impact time integration scheme."""
    if scheme not in {"explicit", "implicit"}:
        raise ValueError(f"Unsupported FE2 time integration scheme: {scheme}.")
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    direct_bundle = build_preprocess_data(model_def, output_def)
    reference = load_yaml_config(config["rve"]["reference_config"])
    mesh = _build_rve_mesh(config, reference)
    nonlinear_model = build_cylindrical_two_phase_rve(
        mesh,
        _viscoplastic_material(reference["materials"]["matrix_viscoplastic"]),
        _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
        _newton_settings(reference),
        float(reference["verification"]["hill_mandel_power_scale"]),
    )
    execution = config["rve"]["execution"]
    rve_material = RVEHomogenizedMaterial(
        nonlinear_model,
        "multiphase_rve_plastic_impact",
        RVEExecutionSettings(
            cast(Literal["serial", "process_pool"], str(execution["backend"])),
            int(execution["workers"]),
            int(execution["threads_per_worker"]),
        ),
    )
    bundle = replace(direct_bundle, material=rve_material)
    initial = np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64)
    update = _build_material_update_settings(config)
    load_function = _build_pressure_load_function(bundle, config)
    dirichlet = _build_left_dirichlet(bundle)
    try:
        if scheme == "explicit":
            solution = solve_material_explicit(
                bundle,
                _build_time_settings(config["analysis"]["explicit"]),
                initial,
                initial,
                dirichlet,
                load_function,
                "three_dimensional",
                update,
                progress_callback,
            )
        elif scheme == "implicit":
            solution = solve_material_implicit_newmark(
                bundle,
                _build_time_settings(config["analysis"]["implicit"]),
                initial,
                initial,
                dirichlet,
                load_function,
                "three_dimensional",
                update,
                _build_newton_settings(config),
                progress_callback,
            )
    finally:
        rve_material.close()
    return PlasticImpactSchemeSimulation(
        bundle=bundle,
        rve_model=nonlinear_model,
        material_update_settings=update,
        solution=solution,
    )


def _build_rve_mesh(config: dict[str, Any], reference: dict[str, Any]):
    """Build the explicitly configured reduced-cost O-grid cell."""
    lengths = reference["rve"]["lengths"]
    return build_cylindrical_ogrid_hex8(
        (float(lengths[0]), float(lengths[1]), float(lengths[2])),
        float(reference["rve"]["inclusion_volume_fraction"]),
        int(config["rve"]["circumferential_divisions"]),
        int(config["rve"]["matrix_radial_divisions"]),
        int(config["rve"]["axial_divisions"]),
    )


def _build_common_grid_comparison(bundle, explicit, implicit) -> dict[str, np.ndarray]:
    """Extract exact common-grid responses without interpolation."""
    right_nodes = bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = bundle.mesh_info.dof_map[right_nodes, 0]
    indices = np.asarray(
        [int(np.argmin(np.abs(explicit.times - time))) for time in implicit.times],
        dtype=np.int64,
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, float(implicit.times[-1])) * 32.0
    if not np.allclose(explicit.times[indices], implicit.times, rtol=0.0, atol=tolerance):
        raise ValueError("Plastic FE2 impact histories have no exact common time grid.")
    q_key = "phase_matrix__equivalent_plastic_strain"
    return {
        "explicit_indices": indices,
        "explicit_right_displacement": np.mean(explicit.displacement[:, right_dofs], axis=1)[
            indices
        ],
        "implicit_right_displacement": np.mean(implicit.displacement[:, right_dofs], axis=1),
        "explicit_q_maximum": np.max(explicit.material_diagnostics[q_key], axis=1)[indices],
        "implicit_q_maximum": np.max(implicit.material_diagnostics[q_key], axis=1),
        "explicit_dissipation": explicit.dissipated_energy[indices],
        "implicit_dissipation": implicit.dissipated_energy,
    }


def _compute_scale_screening(config, reference, elastic_model, spectrum) -> list[dict[str, object]]:
    """Evaluate nondimensional dynamic indicators for prescribed cell-size ratios."""
    update = _update_settings(reference)
    unit_screening = compute_rve_dynamic_screening(elastic_model, update, 1.0)
    static_tangent = compute_linear_rve_dynamic_tangent(elastic_model, update, 0.0)
    macro_length = float(config["model"]["geometry"]["size"][0])
    omega_values = {
        "median": float(np.median(spectrum.omega_99)),
        "p95": float(np.percentile(spectrum.omega_99, 95.0)),
        "maximum": float(np.max(spectrum.omega_99)),
    }
    records = []
    for length_ratio in config["spectrum"]["cell_to_macro_length_ratios"]:
        physical_cell_length = float(length_ratio) * macro_length
        for statistic, omega in omega_values.items():
            scaled_frequency = omega * physical_cell_length
            dynamic_tangent = compute_linear_rve_dynamic_tangent(
                elastic_model, update, scaled_frequency
            )
            scale_ratio = omega * unit_screening.transit_time * physical_cell_length
            frequency_ratio = omega * physical_cell_length / unit_screening.first_angular_frequency
            tangent_difference = float(
                np.linalg.norm(dynamic_tangent - static_tangent) / np.linalg.norm(static_tangent)
            )
            ratio_tolerance = float(reference["verification"]["tolerances"]["dynamic_ratio"])
            tangent_tolerance = float(reference["verification"]["tolerances"]["dynamic_tangent"])
            records.append(
                {
                    "cell_to_macro_length_ratio": float(length_ratio),
                    "omega_statistic": statistic,
                    "omega_99": omega,
                    "scale_ratio": scale_ratio,
                    "frequency_ratio": frequency_ratio,
                    "dynamic_tangent_difference": tangent_difference,
                    "conservative_screening_passed": bool(
                        scale_ratio <= ratio_tolerance
                        and frequency_ratio <= ratio_tolerance
                        and tangent_difference <= tangent_tolerance
                    ),
                }
            )
    return records


def _compute_metrics(config, bundle, explicit, implicit, comparison, spectrum, screening):
    """Compute fixed-scale accuracy, plasticity, unloading, and spectrum metrics."""
    verification = config["verification"]
    displacement_scale = float(verification["displacement_scale"])
    dissipation_scale = float(verification["dissipation_scale"])
    q_scale = float(verification["q_scale"])
    explicit_q = comparison["explicit_q_maximum"]
    implicit_q = comparison["implicit_q_maximum"]
    explicit_eq = _mean_equivalent_stress(explicit.stresses)
    implicit_eq = _mean_equivalent_stress(implicit.stresses)
    return {
        "displacement_history_scaled_error": float(
            np.linalg.norm(
                comparison["explicit_right_displacement"]
                - comparison["implicit_right_displacement"]
            )
            / (np.sqrt(implicit.times.size) * displacement_scale)
        ),
        "q_history_scaled_error": float(
            np.linalg.norm(explicit_q - implicit_q) / (np.sqrt(implicit.times.size) * q_scale)
        ),
        "maximum_q_scaled_error": float(
            abs(float(np.max(explicit_q)) - float(np.max(implicit_q))) / q_scale
        ),
        "dissipated_energy_scaled_error": float(
            abs(comparison["explicit_dissipation"][-1] - comparison["implicit_dissipation"][-1])
            / dissipation_scale
        ),
        "explicit_maximum_q": float(np.max(explicit_q)),
        "implicit_maximum_q": float(np.max(implicit_q)),
        "explicit_maximum_active_fraction": float(
            np.max(explicit.viscoplastic_active_volume_fraction)
        ),
        "implicit_maximum_active_fraction": float(
            np.max(implicit.viscoplastic_active_volume_fraction)
        ),
        "explicit_minimum_negative_work": float(np.min(explicit.negative_incremental_work)),
        "implicit_minimum_negative_work": float(np.min(implicit.negative_incremental_work)),
        "explicit_equivalent_stress_drop": _maximum_postpeak_drop(explicit_eq),
        "implicit_equivalent_stress_drop": _maximum_postpeak_drop(implicit_eq),
        "explicit_minimum_q_increment": _minimum_increment(
            explicit.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
        ),
        "implicit_minimum_q_increment": _minimum_increment(
            implicit.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
        ),
        "explicit_minimum_dissipation_density": float(
            np.min(explicit.material_diagnostics["dissipation_density"])
        ),
        "implicit_minimum_dissipation_density": float(
            np.min(implicit.material_diagnostics["dissipation_density"])
        ),
        "explicit_residual_displacement": float(abs(comparison["explicit_right_displacement"][-1])),
        "implicit_residual_displacement": float(abs(comparison["implicit_right_displacement"][-1])),
        "explicit_wave_arrival_time": _wave_arrival_time(explicit.times, explicit_eq),
        "implicit_wave_arrival_time": _wave_arrival_time(implicit.times, implicit_eq),
        "explicit_energy_residual": _maximum_scaled(
            explicit.energy_residual, float(verification["energy_scale"])
        ),
        "implicit_energy_residual": _maximum_scaled(
            implicit.energy_residual, float(verification["energy_scale"])
        ),
        "omega_99_median": float(np.median(spectrum.omega_99)),
        "omega_99_p95": float(np.percentile(spectrum.omega_99, 95.0)),
        "omega_99_maximum": float(np.max(spectrum.omega_99)),
        "screening_maximum_scale_ratio": max(item["scale_ratio"] for item in screening),
        "screening_maximum_frequency_ratio": max(item["frequency_ratio"] for item in screening),
        "screening_maximum_tangent_difference": max(
            item["dynamic_tangent_difference"] for item in screening
        ),
        "explicit_solve_time": explicit.solve_time,
        "implicit_solve_time": implicit.solve_time,
    }


def _evaluate_pass(verification, metrics: dict[str, float]) -> bool:
    """Evaluate fixed physical and numerical benchmark criteria."""
    tolerance = float(verification["response_tolerance"])
    plastic_threshold = float(verification["plastic_q_threshold"])
    work_threshold = float(verification["negative_work_threshold"])
    stress_drop_threshold = float(verification["stress_drop_threshold"])
    residual_threshold = float(verification["residual_displacement_threshold"])
    arrival_scale = float(verification["arrival_time_scale"])
    return (
        metrics["displacement_history_scaled_error"] <= tolerance
        and metrics["maximum_q_scaled_error"] <= tolerance
        and metrics["dissipated_energy_scaled_error"] <= tolerance
        and metrics["explicit_maximum_q"] >= plastic_threshold
        and metrics["implicit_maximum_q"] >= plastic_threshold
        and metrics["explicit_maximum_active_fraction"] > 0.0
        and metrics["implicit_maximum_active_fraction"] > 0.0
        and metrics["explicit_minimum_negative_work"] <= -work_threshold
        and metrics["implicit_minimum_negative_work"] <= -work_threshold
        and metrics["explicit_equivalent_stress_drop"] >= stress_drop_threshold
        and metrics["implicit_equivalent_stress_drop"] >= stress_drop_threshold
        and metrics["explicit_minimum_q_increment"] >= -1.0e-12
        and metrics["implicit_minimum_q_increment"] >= -1.0e-12
        and metrics["explicit_minimum_dissipation_density"] >= -1.0e-12
        and metrics["implicit_minimum_dissipation_density"] >= -1.0e-12
        and metrics["explicit_residual_displacement"] >= residual_threshold
        and metrics["implicit_residual_displacement"] >= residual_threshold
        and abs(metrics["explicit_wave_arrival_time"] - metrics["implicit_wave_arrival_time"])
        / arrival_scale
        <= tolerance
        and metrics["explicit_energy_residual"] <= float(verification["energy_tolerance"])
        and metrics["implicit_energy_residual"] <= float(verification["energy_tolerance"])
    )


def _write_outputs(config, bundle, explicit, implicit, spectrum, screening, metrics):
    """Write VTK animation, spectra, screening records, and metrics."""
    root = Path(config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if bool(config["output"]["save_vtk"]):
        stride = int(config["output"]["vtk_step_stride"])
        explicit_paths = write_material_dynamic_vtk_series(
            bundle,
            explicit,
            root / "vtk" / "explicit",
            "explicit_evolution",
            stride,
            bool(config["output"]["save_quadrature_vtk"]),
        )
        implicit_paths = write_material_dynamic_vtk_series(
            bundle,
            implicit,
            root / "vtk" / "implicit",
            "implicit_evolution",
            stride,
            bool(config["output"]["save_quadrature_vtk"]),
        )
        paths.update({f"explicit_{key}": value for key, value in explicit_paths.items()})
        paths.update({f"implicit_{key}": value for key, value in implicit_paths.items()})
    spectrum_path = root / "curves" / "strain_rate_spectrum.csv"
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    mean_energy = np.mean(spectrum.point_energy, axis=0)
    cumulative = np.cumsum(mean_energy) / np.sum(mean_energy)
    with spectrum_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("angular_frequency", "mean_energy", "cumulative_energy"))
        writer.writerows(zip(spectrum.angular_frequencies, mean_energy, cumulative, strict=True))
    screening_path = root / "dynamic_screening.json"
    screening_payload = {
        "engineering_conclusion": "undetermined_without_physical_cell_scale",
        "records": screening,
    }
    screening_path.write_text(
        json.dumps(screening_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    paths.update(
        {
            "spectrum_csv": spectrum_path,
            "dynamic_screening": screening_path,
            "metrics": metrics_path,
        }
    )
    return paths


def _mean_equivalent_stress(stresses: np.ndarray) -> np.ndarray:
    """Compute the point-mean J2 equivalent-stress history."""
    mean = np.mean(stresses[..., :3], axis=2)
    deviator = stresses[..., :3] - mean[..., None]
    squared = np.sum(deviator**2, axis=2) + 2.0 * np.sum(stresses[..., 3:] ** 2, axis=2)
    return np.mean(np.sqrt(1.5 * squared), axis=1)


def _maximum_postpeak_drop(values: np.ndarray) -> float:
    """Measure the largest stress reduction after the history peak."""
    peak_id = int(np.argmax(values))
    return float(values[peak_id] - np.min(values[peak_id:]))


def _minimum_increment(values: np.ndarray) -> float:
    """Return the smallest accepted increment of an internal variable."""
    return float(np.min(np.diff(values, axis=0)))


def _wave_arrival_time(times: np.ndarray, equivalent_stress: np.ndarray) -> float:
    """Return the first time reaching ten percent of the history peak stress."""
    threshold = 0.1 * float(np.max(equivalent_stress))
    ids = np.flatnonzero(equivalent_stress >= threshold)
    if ids.size == 0:
        raise ValueError("Impact history has no measurable stress-wave arrival.")
    return float(times[ids[0]])
