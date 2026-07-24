"""J2 viscoplastic explicit-implicit structural dynamics benchmark.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from concrete_impact.core.failure_records import record_material_point_failures
from fem.assembly.load import assemble_uniform_boundary_pressure
from fem.cases import CaseDef, OutputDef, RunResult, write_run_records
from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.materials.data import MaterialUpdateSettings
from fem.post.curves import write_response_csv, write_response_plot
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.solvers.data import (
    AdaptiveTimeStepSettings,
    ArmijoSettings,
    DirichletDofSet,
    FEMRunDef,
    LinearSolverSettings,
    NewtonSettings,
    NonlinearNewtonSettings,
    SurrogateSettings,
    TimeIntegrationSettings,
)


@record_material_point_failures
def run_dynamic_j2_viscoplastic_benchmark(config: dict[str, Any]) -> RunResult:
    """Run matched explicit and implicit J2 pressure-pulse simulations."""
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    bundle = build_preprocess_data(model_def, output_def)
    dirichlet = _build_left_dirichlet(bundle)
    update_settings = _build_material_update_settings(config)
    load_function = _build_pressure_load_function(bundle, config)
    dof_count = bundle.mesh_info.dof_map.size
    initial_displacement = np.zeros(dof_count, dtype=np.float64)
    initial_velocity = np.zeros(dof_count, dtype=np.float64)
    explicit_time = _build_time_settings(config["analysis"]["explicit"])
    implicit_time = _build_time_settings(config["analysis"]["implicit"])

    execution_order = tuple(config["analysis"].get("execution_order", ("explicit", "implicit")))
    if set(execution_order) != {"explicit", "implicit"} or len(execution_order) != 2:
        raise ValueError("Dynamic comparison execution_order must contain explicit and implicit.")
    solutions = {}
    for scheme in execution_order:
        if scheme == "explicit":
            solutions[scheme] = solve_material_explicit(
                bundle,
                explicit_time,
                initial_displacement,
                initial_velocity,
                dirichlet,
                load_function,
                "three_dimensional",
                update_settings,
            )
        else:
            solutions[scheme] = solve_material_implicit_newmark(
                bundle,
                implicit_time,
                initial_displacement,
                initial_velocity,
                dirichlet,
                load_function,
                "three_dimensional",
                update_settings,
                _build_newton_settings(config),
            )
    explicit = solutions["explicit"]
    implicit = solutions["implicit"]
    data = _build_comparison_data(bundle, explicit, implicit)
    metrics = _compute_metrics(data, explicit, implicit, config["verification"])
    output_paths = _write_outputs(output_def.root, data)
    tolerance = float(config["verification"]["response_tolerance"])
    energy_tolerance = float(config["verification"]["energy_tolerance"])
    passed = (
        metrics["displacement_history_scaled_error"] <= tolerance
        and metrics["peak_displacement_scaled_error"] <= tolerance
        and metrics["dissipated_energy_scaled_error"] <= tolerance
        and metrics["explicit_max_scaled_energy_residual"] <= energy_tolerance
        and metrics["implicit_max_scaled_energy_residual"] <= energy_tolerance
        and metrics["explicit_minimum_q_increment"] >= -1.0e-12
        and metrics["implicit_minimum_q_increment"] >= -1.0e-12
    )
    case_def = _build_case_def(config, model_def, output_def, explicit_time, dirichlet, dof_count)
    result = RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths=output_paths,
        passed=passed,
    )
    record_paths = write_run_records(case_def, result, config, config)

    return RunResult(
        name=result.name,
        metrics=result.metrics,
        output_paths={**result.output_paths, **record_paths},
        passed=result.passed,
    )


def _build_model_def(config: dict[str, Any]) -> ModelDef:
    """Build the pressure-pulse model definition."""
    return ModelDef(
        name=str(config["name"]),
        dimension=int(config["dimension"]),
        geometry=config["geometry"],
        mesh=config["mesh"],
        material=config["material"],
        quadrature=config["quadrature"],
        boundary_conditions=config["boundary_conditions"],
    )


def _build_output_def(config: dict[str, Any]) -> OutputDef:
    """Build benchmark output settings."""
    return OutputDef(
        root=Path(config["root"]),
        mesh_path=Path(config["mesh_path"]),
        vtk_path=Path(config["vtk_path"]),
        save_vtk=bool(config["save_vtk"]),
        save_history=bool(config["save_history"]),
        fields=tuple(config["fields"]),
    )


def _build_left_dirichlet(bundle: PreprocessBundle) -> DirichletDofSet:
    """Fix every displacement component on the left boundary."""
    nodes = bundle.mesh_info.boundary_groups["left"].nodes
    dofs = bundle.mesh_info.dof_map[nodes, :].reshape(-1)

    return DirichletDofSet(dofs=dofs, values=np.zeros(dofs.size, dtype=np.float64))


def _build_pressure_load_function(bundle: PreprocessBundle, config: dict[str, Any]):
    """Build a half-sine normal-pressure load function."""
    load = config["loading"]
    amplitude = float(load["amplitude"])
    duration = float(load["duration"])
    normal = np.asarray(load["outward_normal"], dtype=np.float64)
    unit_force = assemble_uniform_boundary_pressure(
        bundle.mesh_info,
        str(load["group"]),
        1.0,
        normal,
        1.0,
    )

    def load_function(time: float) -> NDArray[np.float64]:
        """Evaluate the pressure-pulse equivalent nodal force."""
        if 0.0 <= time <= duration:
            pressure = amplitude * np.sin(np.pi * time / duration)
        else:
            pressure = 0.0

        return pressure * unit_force

    return load_function


def _build_material_update_settings(config: dict[str, Any]) -> MaterialUpdateSettings:
    """Build local constitutive integration settings."""
    settings = config["analysis"]["material_update"]

    return MaterialUpdateSettings(
        max_iterations=int(settings["max_iterations"]),
        yield_relative_tolerance=float(settings["yield_relative_tolerance"]),
        residual_absolute_tolerance=float(settings["residual_absolute_tolerance"]),
        residual_relative_tolerance=float(settings["residual_relative_tolerance"]),
    )


def _build_time_settings(config: dict[str, Any]) -> TimeIntegrationSettings:
    """Build one dynamic time-integration definition."""
    adaptive = config.get("adaptive_time_step", {})
    return TimeIntegrationSettings(
        scheme=str(config["scheme"]),
        time_step=float(config["time_step"]),
        num_steps=int(config["num_steps"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        cfl_safety_factor=float(config["cfl_safety_factor"]),
        adaptive=AdaptiveTimeStepSettings(
            enabled=bool(adaptive.get("enabled", False)),
            minimum_factor=float(adaptive.get("minimum_factor", 1.0 / 16.0)),
            maximum_factor=float(adaptive.get("maximum_factor", 2.0)),
            cutback_factor=float(adaptive.get("cutback_factor", 0.5)),
            growth_factor=float(adaptive.get("growth_factor", 2.0)),
            easy_iteration_limit=int(adaptive.get("easy_iteration_limit", 4)),
            easy_step_count=int(adaptive.get("easy_step_count", 3)),
        ),
    )


def _build_newton_settings(config: dict[str, Any]) -> NonlinearNewtonSettings:
    """Build scaled global Newton settings."""
    settings = config["analysis"]["implicit"]["newton"]
    armijo = settings.get("armijo", {})

    return NonlinearNewtonSettings(
        max_iterations=int(settings["max_iterations"]),
        residual_absolute_tolerance=float(settings["residual_absolute_tolerance"]),
        residual_relative_tolerance=float(settings["residual_relative_tolerance"]),
        increment_absolute_tolerance=float(settings["increment_absolute_tolerance"]),
        increment_relative_tolerance=float(settings["increment_relative_tolerance"]),
        armijo=ArmijoSettings(
            enabled=bool(armijo.get("enabled", False)),
            sufficient_decrease=float(armijo.get("sufficient_decrease", 1.0e-4)),
            reduction_factor=float(armijo.get("reduction_factor", 0.5)),
            max_backtracks=int(armijo.get("max_backtracks", 20)),
        ),
    )


def _build_comparison_data(bundle, explicit, implicit) -> dict[str, NDArray[np.float64]]:
    """Extract explicit and implicit responses at identical stored times."""
    right_nodes = bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = bundle.mesh_info.dof_map[right_nodes, 0]
    explicit_displacement = np.mean(explicit.displacement[:, right_dofs], axis=1)
    implicit_displacement = np.mean(implicit.displacement[:, right_dofs], axis=1)
    explicit_indices = np.asarray(
        [int(np.argmin(np.abs(explicit.times - time))) for time in implicit.times],
        dtype=np.int64,
    )
    matched_times = explicit.times[explicit_indices]
    time_tolerance = np.finfo(np.float64).eps * max(1.0, float(implicit.times[-1]))
    if not np.allclose(matched_times, implicit.times, rtol=0.0, atol=time_tolerance):
        raise ValueError("Explicit and implicit histories have no exact common time grid.")

    return {
        "time": implicit.times,
        "explicit_right_displacement": explicit_displacement[explicit_indices],
        "implicit_right_displacement": implicit_displacement,
        "explicit_dissipated_energy": explicit.dissipated_energy[explicit_indices],
        "implicit_dissipated_energy": implicit.dissipated_energy,
    }


def _compute_metrics(data, explicit, implicit, verification) -> dict[str, float]:
    """Compute fixed-scale response, stability, and timing metrics."""
    explicit_displacement = data["explicit_right_displacement"]
    implicit_displacement = data["implicit_right_displacement"]
    displacement_scale = float(verification["displacement_scale"])
    dissipation_scale = float(verification["dissipation_scale"])
    energy_scale = float(verification["energy_scale"])
    if displacement_scale <= 0.0 or dissipation_scale <= 0.0 or energy_scale <= 0.0:
        raise ValueError("Verification scales must be strictly positive.")

    return {
        "displacement_history_scaled_error": float(
            np.linalg.norm(implicit_displacement - explicit_displacement)
            / (np.sqrt(explicit_displacement.size) * displacement_scale)
        ),
        "peak_displacement_scaled_error": float(
            abs(np.max(np.abs(implicit_displacement)) - np.max(np.abs(explicit_displacement)))
            / displacement_scale
        ),
        "dissipated_energy_scaled_error": float(
            abs(implicit.dissipated_energy[-1] - explicit.dissipated_energy[-1])
            / dissipation_scale
        ),
        "explicit_max_scaled_energy_residual": _scaled_energy_residual(explicit, energy_scale),
        "implicit_max_scaled_energy_residual": _scaled_energy_residual(implicit, energy_scale),
        "explicit_minimum_q_increment": _minimum_q_increment(explicit),
        "implicit_minimum_q_increment": _minimum_q_increment(implicit),
        "explicit_solve_time": explicit.solve_time,
        "implicit_solve_time": implicit.solve_time,
        "implicit_speedup": explicit.solve_time / implicit.solve_time,
        "implicit_total_newton_iterations": float(np.sum(implicit.newton_iterations)),
    }


def _scaled_energy_residual(solution, energy_scale: float) -> float:
    """Compute the maximum energy residual using a fixed physical scale."""
    return float(np.max(np.abs(solution.energy_residual)) / energy_scale)


def _minimum_q_increment(solution) -> float:
    """Compute the minimum equivalent-plastic-strain increment."""
    if solution.equivalent_plastic_strain.shape[0] < 2:
        return 0.0

    return float(np.min(np.diff(solution.equivalent_plastic_strain, axis=0)))


def _write_outputs(output_root: Path, data: dict[str, NDArray[np.float64]]) -> dict[str, Path]:
    """Write response curves for the matched dynamic comparison."""
    csv_path = write_response_csv(data, output_root / "curves" / "response.csv")
    png_path = write_response_plot(
        data,
        output_root / "curves" / "response.png",
        "time",
        (("explicit_right_displacement", "implicit_right_displacement"),),
    )

    return {"response_csv": csv_path, "response_png": png_path}


def _build_case_def(config, model_def, output_def, time, dirichlet, dof_count) -> CaseDef:
    """Build a record-compatible case definition for the comparison benchmark."""
    return CaseDef(
        name=str(config["case"]["name"]),
        model=model_def,
        run=FEMRunDef(
            name=model_def.name,
            analysis_type="dynamic_material_comparison",
            plane_state="three_dimensional",
            dirichlet=dirichlet,
            external_force=np.zeros(dof_count),
            initial_displacement=np.zeros(dof_count),
            initial_velocity=np.zeros(dof_count),
            exact_displacement=lambda nodes, current_time: np.zeros_like(nodes),
            linear_solver=LinearSolverSettings(backend="scipy", method="direct"),
            newton=NewtonSettings(
                max_iterations=1,
                residual_tolerance=0.0,
                increment_tolerance=0.0,
            ),
            time=time,
            surrogate=SurrogateSettings(enabled=False),
        ),
        output=output_def,
        metadata={
            "description": str(config["case"]["description"]),
            "benchmark_family": "dynamic_j2_viscoplastic",
        },
    )
