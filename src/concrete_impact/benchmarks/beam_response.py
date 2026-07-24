"""Beam response codebase benchmark entrypoints.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from fem.assembly.boundary import apply_dirichlet_to_linear_system, impose_values_on_vector
from fem.assembly.elasticity import (
    assemble_consistent_mass_matrix,
    assemble_lumped_mass_vector,
    assemble_stiffness_matrix,
)
from fem.assembly.load import assemble_uniform_boundary_traction
from fem.cases import CaseDef, OutputDef, RunResult, write_run_records
from fem.dynamics.stability import check_explicit_stability, compute_explicit_stability_report
from fem.materials.linear_elastic import LinearElasticMaterial
from fem.post.curves import write_response_csv, write_response_plot
from fem.post.fields import average_quadrature_fields, compute_linear_elastic_quadrature_fields
from fem.post.response import (
    average_boundary_component,
    average_gauge_component,
    sum_boundary_component,
)
from fem.post.vtk import write_solution_vtk
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.solvers.data import (
    DirichletDofSet,
    DynamicSolution,
    FEMRunDef,
    LinearSolverSettings,
    NewtonSettings,
    StaticSolution,
    SurrogateSettings,
    TimeIntegrationSettings,
)
from fem.solvers.linear import solve_sparse_direct

VectorFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


def run_elastic_tension_benchmark(config: dict[str, Any]) -> RunResult:
    """Run a static beam tension benchmark with force-displacement conjugacy."""
    case_def, bundle = _prepare_case(config)
    result = _ELASTIC_CONTROL_RUNNERS[str(config["loading"]["control"])](case_def, bundle, config)
    record_paths = write_run_records(case_def, result, config, config)

    return _merge_record_paths(result, record_paths)


def run_dynamic_tension_benchmark(config: dict[str, Any]) -> RunResult:
    """Run a dynamic harmonic beam tension benchmark with conjugate response curves."""
    case_def, bundle = _prepare_case(config)
    _DYNAMIC_STABILITY_CHECKERS[case_def.run.time.scheme](bundle, case_def.run.time)
    solution = _DYNAMIC_SOLVERS[case_def.run.time.scheme](case_def, bundle, config)
    result = _postprocess_dynamic_tension(case_def, bundle, config, solution)
    record_paths = write_run_records(case_def, result, config, config)

    return _merge_record_paths(result, record_paths)


def run_wave_pulse_benchmark(config: dict[str, Any]) -> RunResult:
    """Run an explicit beam pulse benchmark for longitudinal wave speed verification."""
    case_def, bundle = _prepare_case(config)
    check_explicit_stability(bundle, case_def.run.time)
    solution = _solve_wave_explicit(case_def, bundle, config)
    result = _postprocess_wave_pulse(case_def, bundle, config, solution)
    record_paths = write_run_records(case_def, result, config, config)

    return _merge_record_paths(result, record_paths)


def _prepare_case(config: dict[str, Any]) -> tuple[CaseDef, PreprocessBundle]:
    """Prepare a benchmark case and preprocess bundle."""
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    bundle = build_preprocess_data(model_def, output_def)
    run_def = _build_run_def(config, bundle)

    case_def = CaseDef(
        name=str(config["case"]["name"]),
        model=model_def,
        run=run_def,
        output=output_def,
        metadata={
            "description": str(config["case"]["description"]),
            "benchmark_family": "beam_response",
            "verification_tolerance": float(config["verification"]["tolerance"]),
        },
    )

    return case_def, bundle


def _build_model_def(config: dict[str, Any]) -> ModelDef:
    """Build a preprocess model definition."""
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
    """Build a benchmark output definition."""
    return OutputDef(
        root=Path(config["root"]),
        mesh_path=Path(config["mesh_path"]),
        vtk_path=Path(config["vtk_path"]),
        save_vtk=bool(config["save_vtk"]),
        save_history=bool(config["save_history"]),
        fields=tuple(config["fields"]),
    )


def _build_run_def(config: dict[str, Any], bundle: PreprocessBundle) -> FEMRunDef:
    """Build solver settings and initial fields."""
    dof_count = bundle.mesh_info.dof_map.size
    time = _TIME_BUILDERS[str(config["analysis"]["type"])](config)

    return FEMRunDef(
        name=str(config["model"]["name"]),
        analysis_type=str(config["analysis"]["type"]),
        plane_state=str(config["analysis"]["plane_state"]),
        dirichlet=_build_anchor_dirichlet(bundle),
        external_force=np.zeros(dof_count, dtype=np.float64),
        initial_displacement=np.zeros(dof_count, dtype=np.float64),
        initial_velocity=np.zeros(dof_count, dtype=np.float64),
        exact_displacement=_zero_displacement,
        linear_solver=LinearSolverSettings(
            backend=str(config["analysis"]["linear_solver"]["backend"]),
            method=str(config["analysis"]["linear_solver"]["method"]),
        ),
        newton=NewtonSettings(
            max_iterations=int(config["analysis"]["newton"]["max_iterations"]),
            residual_tolerance=float(config["analysis"]["newton"]["residual_tolerance"]),
            increment_tolerance=float(config["analysis"]["newton"]["increment_tolerance"]),
        ),
        time=time,
        surrogate=SurrogateSettings(enabled=False),
    )


def _run_elastic_displacement_control(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
) -> RunResult:
    """Run a static displacement-control beam benchmark."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    load_factors = np.linspace(0.0, 1.0, int(config["loading"]["steps"]), dtype=np.float64)
    target_displacement = float(config["loading"]["right_displacement"])
    external_force = np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64)

    numerical_force: list[float] = []
    exact_force: list[float] = []
    displacements: list[NDArray[np.float64]] = []

    for load_factor in load_factors:
        right_displacement = target_displacement * load_factor
        dirichlet = _build_beam_dirichlet(bundle, right_displacement)
        solution = _solve_static_system(stiffness, external_force, dirichlet)
        physical_force = _physical_force_scale(bundle, config) * sum_boundary_component(
            bundle.mesh_info,
            solution.reaction,
            "right",
            0,
        )

        numerical_force.append(physical_force)
        exact_force.append(_static_exact_force(bundle, config, right_displacement))
        displacements.append(solution.displacement)

    data = {
        "right_displacement": target_displacement * load_factors,
        "numerical_right_force": np.asarray(numerical_force, dtype=np.float64),
        "exact_right_force": np.asarray(exact_force, dtype=np.float64),
    }
    output_paths = _write_elastic_outputs(case_def, bundle, data, displacements)
    error = _relative_l2_error(data["numerical_right_force"], data["exact_right_force"])

    return _build_result(case_def, {"force_relative_error": error}, output_paths)


def _run_elastic_force_control(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
) -> RunResult:
    """Run a static force-control beam benchmark."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    load_factors = np.linspace(0.0, 1.0, int(config["loading"]["steps"]), dtype=np.float64)
    target_force = float(config["loading"]["right_force"])

    numerical_displacement: list[float] = []
    exact_displacement: list[float] = []
    displacements: list[NDArray[np.float64]] = []

    for load_factor in load_factors:
        physical_force = target_force * load_factor
        external_force = _build_right_force_vector(bundle, config, physical_force)
        solution = _solve_static_system(
            stiffness,
            external_force,
            _build_left_anchor_dirichlet(bundle),
        )
        right_displacement = average_boundary_component(
            bundle.mesh_info,
            solution.displacement,
            "right",
            0,
        )

        numerical_displacement.append(right_displacement)
        exact_displacement.append(_static_exact_displacement(bundle, config, physical_force))
        displacements.append(solution.displacement)

    data = {
        "right_force": target_force * load_factors,
        "numerical_right_displacement": np.asarray(numerical_displacement, dtype=np.float64),
        "exact_right_displacement": np.asarray(exact_displacement, dtype=np.float64),
    }
    output_paths = _write_elastic_outputs(case_def, bundle, data, displacements)
    error = _relative_l2_error(
        data["numerical_right_displacement"],
        data["exact_right_displacement"],
    )

    return _build_result(case_def, {"displacement_relative_error": error}, output_paths)


def _solve_dynamic_explicit(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
) -> DynamicSolution:
    """Solve a harmonic displacement-control beam problem explicitly."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    mass_lumped = assemble_lumped_mass_vector(bundle)
    time_settings = case_def.run.time
    time_step = time_settings.time_step
    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step
    dof_count = bundle.mesh_info.dof_map.size

    displacement = np.zeros((times.shape[0], dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    displacement[0, :] = _dynamic_exact_displacement(bundle, config, 0.0)
    velocity[0, :] = _dynamic_exact_velocity(bundle, config, 0.0)
    dirichlet = _build_dynamic_dirichlet(bundle, config, 0.0)
    displacement[0, :] = impose_values_on_vector(
        displacement[0, :],
        dirichlet.dofs,
        dirichlet.values,
    )
    acceleration[0, :] = _compute_lumped_acceleration(
        stiffness,
        mass_lumped,
        displacement[0, :],
        np.zeros(dof_count, dtype=np.float64),
        dirichlet.dofs,
    )
    previous_displacement = (
        displacement[0, :]
        - time_step * velocity[0, :]
        + 0.5 * time_step**2 * acceleration[0, :]
    )
    previous_dirichlet = _build_dynamic_dirichlet(bundle, config, -time_step)
    previous_displacement = impose_values_on_vector(
        previous_displacement,
        previous_dirichlet.dofs,
        previous_dirichlet.values,
    )

    for step_id in range(time_settings.num_steps):
        current_displacement = displacement[step_id, :]
        current_dirichlet = _build_dynamic_dirichlet(bundle, config, times[step_id])
        current_acceleration = _compute_lumped_acceleration(
            stiffness,
            mass_lumped,
            current_displacement,
            np.zeros(dof_count, dtype=np.float64),
            current_dirichlet.dofs,
        )
        next_displacement = (
            2.0 * current_displacement
            - previous_displacement
            + time_step**2 * current_acceleration
        )
        next_dirichlet = _build_dynamic_dirichlet(bundle, config, times[step_id + 1])
        next_displacement = impose_values_on_vector(
            next_displacement,
            next_dirichlet.dofs,
            next_dirichlet.values,
        )

        displacement[step_id + 1, :] = next_displacement
        velocity[step_id + 1, :] = (next_displacement - previous_displacement) / (
            2.0 * time_step
        )
        acceleration[step_id + 1, :] = current_acceleration
        previous_displacement = current_displacement

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def _solve_dynamic_implicit(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
) -> DynamicSolution:
    """Solve a harmonic displacement-control beam problem with implicit Newmark."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    mass = assemble_consistent_mass_matrix(bundle)
    time_settings = case_def.run.time
    time_step = time_settings.time_step
    beta = time_settings.beta
    gamma = time_settings.gamma
    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step
    dof_count = bundle.mesh_info.dof_map.size

    displacement = np.zeros((times.shape[0], dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    displacement[0, :] = _dynamic_exact_displacement(bundle, config, 0.0)
    velocity[0, :] = _dynamic_exact_velocity(bundle, config, 0.0)
    dirichlet = _build_dynamic_dirichlet(bundle, config, 0.0)
    displacement[0, :] = impose_values_on_vector(
        displacement[0, :],
        dirichlet.dofs,
        dirichlet.values,
    )
    acceleration[0, :] = _compute_consistent_acceleration(
        mass,
        stiffness,
        displacement[0, :],
        np.zeros(dof_count, dtype=np.float64),
        dirichlet.dofs,
    )

    a0 = 1.0 / (beta * time_step**2)
    a2 = 1.0 / (beta * time_step)
    a3 = 1.0 / (2.0 * beta) - 1.0
    effective_stiffness = stiffness + a0 * mass

    for step_id in range(time_settings.num_steps):
        rhs = mass @ (
            a0 * displacement[step_id, :]
            + a2 * velocity[step_id, :]
            + a3 * acceleration[step_id, :]
        )
        next_dirichlet = _build_dynamic_dirichlet(bundle, config, times[step_id + 1])
        constrained_matrix, constrained_rhs = apply_dirichlet_to_linear_system(
            effective_stiffness,
            np.asarray(rhs, dtype=np.float64),
            next_dirichlet.dofs,
            next_dirichlet.values,
        )
        displacement[step_id + 1, :] = solve_sparse_direct(constrained_matrix, constrained_rhs)
        acceleration[step_id + 1, :] = (
            a0 * (displacement[step_id + 1, :] - displacement[step_id, :])
            - a2 * velocity[step_id, :]
            - a3 * acceleration[step_id, :]
        )
        velocity[step_id + 1, :] = velocity[step_id, :] + time_step * (
            (1.0 - gamma) * acceleration[step_id, :] + gamma * acceleration[step_id + 1, :]
        )

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def _solve_wave_explicit(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
) -> DynamicSolution:
    """Solve a free longitudinal pulse benchmark with explicit central difference."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    mass_lumped = assemble_lumped_mass_vector(bundle)
    time_settings = case_def.run.time
    time_step = time_settings.time_step
    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step
    dof_count = bundle.mesh_info.dof_map.size
    external_force = np.zeros(dof_count, dtype=np.float64)
    dirichlet = _build_anchor_dirichlet(bundle)

    displacement = np.zeros((times.shape[0], dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    displacement[0, :] = _wave_exact_displacement(bundle, config, 0.0)
    velocity[0, :] = _wave_exact_velocity(bundle, config, 0.0)
    displacement[0, :] = impose_values_on_vector(
        displacement[0, :],
        dirichlet.dofs,
        dirichlet.values,
    )
    velocity[0, :] = impose_values_on_vector(velocity[0, :], dirichlet.dofs, dirichlet.values)
    acceleration[0, :] = _compute_lumped_acceleration(
        stiffness,
        mass_lumped,
        displacement[0, :],
        external_force,
        dirichlet.dofs,
    )
    previous_displacement = (
        displacement[0, :]
        - time_step * velocity[0, :]
        + 0.5 * time_step**2 * acceleration[0, :]
    )
    previous_displacement = impose_values_on_vector(
        previous_displacement,
        dirichlet.dofs,
        dirichlet.values,
    )

    for step_id in range(time_settings.num_steps):
        current_displacement = displacement[step_id, :]
        current_acceleration = _compute_lumped_acceleration(
            stiffness,
            mass_lumped,
            current_displacement,
            external_force,
            dirichlet.dofs,
        )
        next_displacement = (
            2.0 * current_displacement
            - previous_displacement
            + time_step**2 * current_acceleration
        )
        next_displacement = impose_values_on_vector(
            next_displacement,
            dirichlet.dofs,
            dirichlet.values,
        )

        displacement[step_id + 1, :] = next_displacement
        velocity[step_id, :] = (next_displacement - previous_displacement) / (
            2.0 * time_step
        )
        acceleration[step_id, :] = current_acceleration
        previous_displacement = current_displacement

    acceleration[-1, :] = _compute_lumped_acceleration(
        stiffness,
        mass_lumped,
        displacement[-1, :],
        external_force,
        dirichlet.dofs,
    )
    velocity[-1, :] = velocity[-2, :] + 0.5 * time_step * (
        acceleration[-2, :] + acceleration[-1, :]
    )

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def _postprocess_dynamic_tension(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
    solution: DynamicSolution,
) -> RunResult:
    """Postprocess a dynamic tension solution."""
    stiffness = assemble_stiffness_matrix(bundle, case_def.run.plane_state)
    numerical_displacement = np.asarray(
        [
            average_boundary_component(bundle.mesh_info, displacement, "right", 0)
            for displacement in solution.displacement
        ],
        dtype=np.float64,
    )
    numerical_force = _physical_force_scale(bundle, config) * np.asarray(
        [
            sum_boundary_component(
                bundle.mesh_info,
                stiffness @ displacement,
                "right",
                0,
            )
            for displacement in solution.displacement
        ],
        dtype=np.float64,
    )
    exact_displacement = _dynamic_exact_right_displacement(config, solution.times)
    exact_force = _dynamic_exact_right_force(bundle, config, solution.times)
    data = {
        "time": solution.times,
        "numerical_right_displacement": numerical_displacement,
        "exact_right_displacement": exact_displacement,
        "numerical_right_force": numerical_force,
        "exact_right_force": exact_force,
    }
    output_paths = _write_dynamic_outputs(case_def, bundle, data, solution.displacement)
    displacement_error = _relative_l2_error(numerical_displacement, exact_displacement)
    force_error = _relative_l2_error(numerical_force, exact_force)

    return _build_result(
        case_def,
        {
            "displacement_history_relative_error": displacement_error,
            "force_history_relative_error": force_error,
        },
        output_paths,
    )


def _postprocess_wave_pulse(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    config: dict[str, Any],
    solution: DynamicSolution,
) -> RunResult:
    """Postprocess a wave pulse solution."""
    gauge_x = np.asarray(config["wave"]["gauges"], dtype=np.float64)
    numerical_velocity = np.column_stack(
        [
            np.asarray(
                [
                    average_gauge_component(bundle.mesh_info, velocity, float(gauge), 0)
                    for velocity in solution.velocity
                ],
                dtype=np.float64,
            )
            for gauge in gauge_x
        ],
    )
    exact_velocity = np.column_stack(
        [
            _wave_exact_gauge_velocity(config, float(gauge), solution.times)
            for gauge in gauge_x
        ],
    )
    numerical_displacement = np.column_stack(
        [
            np.asarray(
                [
                    average_gauge_component(bundle.mesh_info, displacement, float(gauge), 0)
                    for displacement in solution.displacement
                ],
                dtype=np.float64,
            )
            for gauge in gauge_x
        ],
    )
    data = {
        "time": solution.times,
        "numerical_velocity_gauge_0": numerical_velocity[:, 0],
        "exact_velocity_gauge_0": exact_velocity[:, 0],
        "numerical_velocity_gauge_1": numerical_velocity[:, 1],
        "exact_velocity_gauge_1": exact_velocity[:, 1],
    }
    output_paths = _write_wave_outputs(case_def, bundle, data, solution.displacement)
    history_error = _relative_l2_error(numerical_velocity.reshape(-1), exact_velocity.reshape(-1))
    arrival_error = _compute_arrival_time_error(
        solution.times,
        numerical_velocity,
        exact_velocity,
    )
    speed_error = _compute_wave_speed_error(solution.times, numerical_displacement, gauge_x, bundle)

    return _build_result(
        case_def,
        {
            "velocity_history_relative_error": history_error,
            "arrival_time_relative_error": arrival_error,
            "wave_speed_relative_error": speed_error,
        },
        output_paths,
    )


def _write_elastic_outputs(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    data: dict[str, NDArray[np.float64]],
    displacements: list[NDArray[np.float64]],
) -> dict[str, Path]:
    """Write static response curves and VTK snapshots."""
    csv_path = write_response_csv(data, case_def.output.root / "curves" / "response.csv")
    x_name = tuple(data.keys())[0]
    y_pairs = ((tuple(data.keys())[1], tuple(data.keys())[2]),)
    png_path = write_response_plot(
        data,
        case_def.output.root / "curves" / "response.png",
        x_name,
        y_pairs,
    )
    vtk_paths = _write_vtk_snapshots(case_def, bundle, np.asarray(displacements))

    return {"response_csv": csv_path, "response_png": png_path, **vtk_paths}


def _write_dynamic_outputs(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    data: dict[str, NDArray[np.float64]],
    displacements: NDArray[np.float64],
) -> dict[str, Path]:
    """Write dynamic response curves and VTK snapshots."""
    csv_path = write_response_csv(data, case_def.output.root / "curves" / "response.csv")
    png_path = write_response_plot(
        data,
        case_def.output.root / "curves" / "response.png",
        "time",
        (
            ("numerical_right_displacement", "exact_right_displacement"),
            ("numerical_right_force", "exact_right_force"),
        ),
    )
    vtk_paths = _write_vtk_snapshots(case_def, bundle, displacements)

    return {"response_csv": csv_path, "response_png": png_path, **vtk_paths}


def _write_wave_outputs(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    data: dict[str, NDArray[np.float64]],
    displacements: NDArray[np.float64],
) -> dict[str, Path]:
    """Write wave response curves and VTK snapshots."""
    csv_path = write_response_csv(data, case_def.output.root / "curves" / "response.csv")
    png_path = write_response_plot(
        data,
        case_def.output.root / "curves" / "response.png",
        "time",
        (
            ("numerical_velocity_gauge_0", "exact_velocity_gauge_0"),
            ("numerical_velocity_gauge_1", "exact_velocity_gauge_1"),
        ),
    )
    vtk_paths = _write_vtk_snapshots(case_def, bundle, displacements)

    return {"response_csv": csv_path, "response_png": png_path, **vtk_paths}


def _write_vtk_snapshots(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    displacements: NDArray[np.float64],
) -> dict[str, Path]:
    """Write first, middle, and final VTK solution snapshots."""
    step_ids = (0, displacements.shape[0] // 2, displacements.shape[0] - 1)
    names = ("step_0000", "step_mid", "step_final")
    paths: dict[str, Path] = {}

    for name, step_id in zip(names, step_ids, strict=True):
        displacement = displacements[step_id, :]
        quadrature_fields = compute_linear_elastic_quadrature_fields(
            bundle,
            displacement,
            case_def.run.plane_state,
        )
        cell_fields = average_quadrature_fields(quadrature_fields)
        path = write_solution_vtk(
            bundle.mesh_info,
            case_def.output.root / "vtk" / f"{case_def.name}_{name}.vtu",
            displacement,
            cell_fields,
        )
        paths[name] = path

    return paths


def _solve_static_system(
    stiffness: csr_matrix,
    external_force: NDArray[np.float64],
    dirichlet: DirichletDofSet,
) -> StaticSolution:
    """Solve one constrained linear static system."""
    constrained_matrix, constrained_rhs = apply_dirichlet_to_linear_system(
        stiffness,
        external_force,
        dirichlet.dofs,
        dirichlet.values,
    )
    displacement = solve_sparse_direct(constrained_matrix, constrained_rhs)
    reaction = stiffness @ displacement - external_force

    return StaticSolution(displacement=np.asarray(displacement), reaction=np.asarray(reaction))


def _build_beam_dirichlet(
    bundle: PreprocessBundle,
    right_displacement: float,
) -> DirichletDofSet:
    """Build beam axial constraints and a right-end displacement condition."""
    mesh_info = bundle.mesh_info
    right_nodes = mesh_info.boundary_groups["right"].nodes
    right_dofs = mesh_info.dof_map[right_nodes, 0]
    left_anchor = _build_left_anchor_dirichlet(bundle)

    dofs = np.concatenate([left_anchor.dofs, right_dofs])
    values = np.concatenate(
        [
            left_anchor.values,
            np.full(right_dofs.shape[0], right_displacement, dtype=np.float64),
        ],
    )

    return DirichletDofSet(dofs=dofs, values=values)


def _build_left_anchor_dirichlet(bundle: PreprocessBundle) -> DirichletDofSet:
    """Build left-end axial constraints with minimal rigid-mode anchors."""
    mesh_info = bundle.mesh_info
    left_nodes = mesh_info.boundary_groups["left"].nodes
    left_dofs = mesh_info.dof_map[left_nodes, 0]
    anchor = _build_anchor_dirichlet(bundle)
    dofs = np.concatenate([left_dofs, anchor.dofs])
    values = np.concatenate(
        [
            np.zeros(left_dofs.shape[0], dtype=np.float64),
            anchor.values,
        ],
    )

    return DirichletDofSet(dofs=dofs, values=values)


def _build_dynamic_dirichlet(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    time: float,
) -> DirichletDofSet:
    """Build time-dependent harmonic right-end displacement constraints."""
    amplitude = float(config["loading"]["amplitude"])
    angular_frequency = float(config["loading"]["angular_frequency"])

    return _build_beam_dirichlet(bundle, amplitude * np.sin(angular_frequency * time))


def _build_anchor_dirichlet(bundle: PreprocessBundle) -> DirichletDofSet:
    """Build minimal rigid-mode anchor constraints."""
    return ANCHOR_BUILDERS[bundle.mesh_info.dimension](bundle)


def _build_anchor_2d(bundle: PreprocessBundle) -> DirichletDofSet:
    """Build a 2D transverse anchor."""
    node = _node_closest_to(bundle.mesh_info.nodes, np.asarray([0.0, 0.0], dtype=np.float64))
    dof = bundle.mesh_info.dof_map[node, 1]

    return DirichletDofSet(
        dofs=np.asarray([dof], dtype=np.int64),
        values=np.zeros(1, dtype=np.float64),
    )


def _build_anchor_3d(bundle: PreprocessBundle) -> DirichletDofSet:
    """Build 3D transverse translation and twist anchors."""
    nodes = bundle.mesh_info.nodes
    size = np.asarray(bundle.model_def.geometry["size"], dtype=np.float64)
    node_a = _node_closest_to(nodes, np.asarray([0.0, 0.0, 0.0], dtype=np.float64))
    node_b = _node_closest_to(nodes, np.asarray([0.0, size[1], 0.0], dtype=np.float64))
    dofs = np.asarray(
        [
            bundle.mesh_info.dof_map[node_a, 1],
            bundle.mesh_info.dof_map[node_a, 2],
            bundle.mesh_info.dof_map[node_b, 2],
        ],
        dtype=np.int64,
    )

    return DirichletDofSet(dofs=dofs, values=np.zeros(dofs.shape[0], dtype=np.float64))


def _build_right_force_vector(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    physical_force: float,
) -> NDArray[np.float64]:
    """Build a right-end axial force vector."""
    area = _cross_section_area(bundle, config)
    traction = np.zeros(bundle.mesh_info.dimension, dtype=np.float64)
    traction[0] = physical_force / area

    return assemble_uniform_boundary_traction(
        bundle.mesh_info,
        "right",
        traction,
        LOAD_MEASURE_SCALE[bundle.mesh_info.dimension](config),
    )


def _static_exact_force(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    right_displacement: float,
) -> float:
    """Compute the exact axial force in a static beam."""
    material = _require_linear_elastic_material(bundle)

    return (
        material.young_modulus
        * _cross_section_area(bundle, config)
        * right_displacement
        / _length(bundle)
    )


def _static_exact_displacement(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    physical_force: float,
) -> float:
    """Compute the exact right-end displacement in a static beam."""
    material = _require_linear_elastic_material(bundle)

    return (
        physical_force
        * _length(bundle)
        / (material.young_modulus * _cross_section_area(bundle, config))
    )


def _dynamic_exact_displacement(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    time: float,
) -> NDArray[np.float64]:
    """Compute the exact harmonic rod displacement field."""
    nodes = bundle.mesh_info.nodes
    values = np.zeros((nodes.shape[0], bundle.mesh_info.dimension), dtype=np.float64)
    values[:, 0] = _dynamic_mode_shape(bundle, config) * np.sin(
        float(config["loading"]["angular_frequency"]) * time
    )

    return values.reshape(-1)


def _dynamic_exact_velocity(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    time: float,
) -> NDArray[np.float64]:
    """Compute the exact harmonic rod velocity field."""
    omega = float(config["loading"]["angular_frequency"])
    nodes = bundle.mesh_info.nodes
    values = np.zeros((nodes.shape[0], bundle.mesh_info.dimension), dtype=np.float64)
    values[:, 0] = _dynamic_mode_shape(bundle, config) * omega * np.cos(omega * time)

    return values.reshape(-1)


def _dynamic_mode_shape(bundle: PreprocessBundle, config: dict[str, Any]) -> NDArray[np.float64]:
    """Compute the one-dimensional harmonic rod mode shape."""
    omega = float(config["loading"]["angular_frequency"])
    wave_speed = _rod_wave_speed(bundle)
    wave_number = omega / wave_speed
    amplitude = float(config["loading"]["amplitude"])
    length = _length(bundle)

    return amplitude * np.sin(wave_number * bundle.mesh_info.nodes[:, 0]) / np.sin(
        wave_number * length
    )


def _dynamic_exact_right_displacement(
    config: dict[str, Any],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the prescribed exact right-end displacement history."""
    return float(config["loading"]["amplitude"]) * np.sin(
        float(config["loading"]["angular_frequency"]) * times
    )


def _dynamic_exact_right_force(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the exact harmonic axial force history."""
    omega = float(config["loading"]["angular_frequency"])
    wave_number = omega / _rod_wave_speed(bundle)
    length = _length(bundle)
    dynamic_stiffness = (
        _require_linear_elastic_material(bundle).young_modulus
        * _cross_section_area(bundle, config)
        * wave_number
        / np.tan(wave_number * length)
    )

    return dynamic_stiffness * _dynamic_exact_right_displacement(config, times)


def _wave_exact_displacement(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    time: float,
) -> NDArray[np.float64]:
    """Compute a right-traveling Gaussian displacement pulse."""
    nodes = bundle.mesh_info.nodes
    values = np.zeros((nodes.shape[0], bundle.mesh_info.dimension), dtype=np.float64)
    values[:, 0] = _wave_pulse_displacement(config, nodes[:, 0], time, _rod_wave_speed(bundle))

    return values.reshape(-1)


def _wave_exact_velocity(
    bundle: PreprocessBundle,
    config: dict[str, Any],
    time: float,
) -> NDArray[np.float64]:
    """Compute a right-traveling Gaussian velocity pulse."""
    nodes = bundle.mesh_info.nodes
    values = np.zeros((nodes.shape[0], bundle.mesh_info.dimension), dtype=np.float64)
    values[:, 0] = _wave_pulse_velocity(config, nodes[:, 0], time, _rod_wave_speed(bundle))

    return values.reshape(-1)


def _wave_exact_gauge_velocity(
    config: dict[str, Any],
    gauge_x: float,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exact velocity at one gauge point."""
    wave_speed = np.sqrt(
        float(config["model"]["material"]["young_modulus"])
        / float(config["model"]["material"]["density"])
    )

    return _wave_pulse_velocity(config, gauge_x, times, wave_speed)


def _wave_pulse_displacement(
    config: dict[str, Any],
    coordinate_x: NDArray[np.float64] | float,
    time: NDArray[np.float64] | float,
    wave_speed: float,
) -> NDArray[np.float64]:
    """Evaluate the Gaussian pulse displacement."""
    amplitude = float(config["wave"]["amplitude"])
    center = float(config["wave"]["center"])
    width = float(config["wave"]["width"])
    argument = (coordinate_x - wave_speed * time - center) / width

    return amplitude * np.exp(-(argument**2))


def _wave_pulse_velocity(
    config: dict[str, Any],
    coordinate_x: NDArray[np.float64] | float,
    time: NDArray[np.float64] | float,
    wave_speed: float,
) -> NDArray[np.float64]:
    """Evaluate the Gaussian pulse velocity."""
    amplitude = float(config["wave"]["amplitude"])
    center = float(config["wave"]["center"])
    width = float(config["wave"]["width"])
    argument = (coordinate_x - wave_speed * time - center) / width

    return 2.0 * amplitude * wave_speed * argument * np.exp(-(argument**2)) / width


def _compute_lumped_acceleration(
    stiffness: csr_matrix,
    mass_lumped: NDArray[np.float64],
    displacement: NDArray[np.float64],
    external_force: NDArray[np.float64],
    constrained_dofs: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute acceleration from row-sum lumped mass."""
    acceleration = (external_force - stiffness @ displacement) / mass_lumped
    acceleration[constrained_dofs] = 0.0

    return np.asarray(acceleration, dtype=np.float64)


def _compute_consistent_acceleration(
    mass: csr_matrix,
    stiffness: csr_matrix,
    displacement: NDArray[np.float64],
    external_force: NDArray[np.float64],
    constrained_dofs: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute acceleration from a consistent mass matrix."""
    rhs = external_force - stiffness @ displacement
    values = np.zeros(constrained_dofs.shape[0], dtype=np.float64)
    constrained_mass, constrained_rhs = apply_dirichlet_to_linear_system(
        mass,
        np.asarray(rhs, dtype=np.float64),
        constrained_dofs,
        values,
    )

    return solve_sparse_direct(constrained_mass, constrained_rhs)


def _compute_arrival_time_error(
    times: NDArray[np.float64],
    numerical_velocity: NDArray[np.float64],
    exact_velocity: NDArray[np.float64],
) -> float:
    """Compute relative wave-front arrival error at five percent of the exact peak."""
    threshold = 0.05 * np.max(np.abs(exact_velocity), axis=0)
    numerical_arrival = np.asarray(
        [
            times[np.flatnonzero(np.abs(values) >= limit)[0]]
            for values, limit in zip(numerical_velocity.T, threshold, strict=True)
        ],
        dtype=np.float64,
    )
    exact_arrival = np.asarray(
        [
            times[np.flatnonzero(np.abs(values) >= limit)[0]]
            for values, limit in zip(exact_velocity.T, threshold, strict=True)
        ],
        dtype=np.float64,
    )

    return _relative_l2_error(numerical_arrival, exact_arrival)


def _compute_wave_speed_error(
    times: NDArray[np.float64],
    numerical_displacement: NDArray[np.float64],
    gauge_x: NDArray[np.float64],
    bundle: PreprocessBundle,
) -> float:
    """Compute relative wave speed error from pulse-center passage at two gauges."""
    arrival = times[np.argmax(np.abs(numerical_displacement), axis=0)]
    numerical_speed = (gauge_x[1] - gauge_x[0]) / (arrival[1] - arrival[0])

    return abs(float(numerical_speed) - _rod_wave_speed(bundle)) / _rod_wave_speed(bundle)


def _cross_section_area(bundle: PreprocessBundle, config: dict[str, Any]) -> float:
    """Compute the physical beam cross-section area."""
    return CROSS_SECTION_BUILDERS[bundle.mesh_info.dimension](bundle, config)


def _cross_section_2d(bundle: PreprocessBundle, config: dict[str, Any]) -> float:
    """Compute the physical 2D beam area with YAML thickness."""
    return float(config["section"]["thickness"]) * float(bundle.model_def.geometry["size"][1])


def _cross_section_3d(bundle: PreprocessBundle, config: dict[str, Any]) -> float:
    """Compute the physical 3D beam area."""
    size = bundle.model_def.geometry["size"]

    return float(size[1]) * float(size[2])


def _physical_force_scale(bundle: PreprocessBundle, config: dict[str, Any]) -> float:
    """Return the scale from numerical reaction to physical force."""
    return PHYSICAL_FORCE_SCALE[bundle.mesh_info.dimension](config)


def _length(bundle: PreprocessBundle) -> float:
    """Return the beam length."""
    return float(bundle.model_def.geometry["size"][0])


def _rod_wave_speed(bundle: PreprocessBundle) -> float:
    """Compute the one-dimensional rod wave speed."""
    material = _require_linear_elastic_material(bundle)

    return float(np.sqrt(material.young_modulus / material.density))


def _require_linear_elastic_material(bundle: PreprocessBundle) -> LinearElasticMaterial:
    """Require the material type used by analytic beam-response formulas."""
    if not isinstance(bundle.material, LinearElasticMaterial):
        raise TypeError("Beam-response analytic formulas require LinearElasticMaterial.")

    return bundle.material


def _node_closest_to(nodes: NDArray[np.float64], point: NDArray[np.float64]) -> int:
    """Return the node closest to a point."""
    return int(np.argmin(np.linalg.norm(nodes - point, axis=1)))


def _zero_displacement(nodes: NDArray[np.float64], time: float) -> NDArray[np.float64]:
    """Return a zero displacement field."""
    return np.zeros_like(nodes)


def _static_time(config: dict[str, Any]) -> TimeIntegrationSettings:
    """Build inactive time settings."""
    return TimeIntegrationSettings(
        scheme="static",
        time_step=0.0,
        num_steps=0,
        beta=0.25,
        gamma=0.5,
        cfl_safety_factor=0.8,
    )


def _dynamic_time(config: dict[str, Any]) -> TimeIntegrationSettings:
    """Build active time-integration settings."""
    time = config["analysis"]["time"]

    return TimeIntegrationSettings(
        scheme=str(time["scheme"]),
        time_step=float(time["time_step"]),
        num_steps=int(time["num_steps"]),
        beta=float(time["beta"]),
        gamma=float(time["gamma"]),
        cfl_safety_factor=float(time["cfl_safety_factor"]),
    )


def _implicit_stability_report(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
) -> None:
    """Compute CFL diagnostics for implicit runs."""
    compute_explicit_stability_report(bundle, time_settings)


def _build_result(
    case_def: CaseDef,
    metrics: dict[str, float],
    output_paths: dict[str, Path],
) -> RunResult:
    """Build a benchmark result with tolerance-based pass status."""
    tolerance = float(case_def.metadata["verification_tolerance"])
    passed = bool(max(metrics.values()) <= tolerance)

    return RunResult(name=case_def.name, metrics=metrics, output_paths=output_paths, passed=passed)


def _merge_record_paths(
    result: RunResult,
    record_paths: dict[str, Path],
) -> RunResult:
    """Return a result with standard record paths included."""
    return RunResult(
        name=result.name,
        metrics=result.metrics,
        output_paths={**result.output_paths, **record_paths},
        passed=result.passed,
    )


def _relative_l2_error(
    numerical: NDArray[np.float64],
    exact: NDArray[np.float64],
) -> float:
    """Compute a relative Euclidean norm error."""
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


ANCHOR_BUILDERS = {
    2: _build_anchor_2d,
    3: _build_anchor_3d,
}

CROSS_SECTION_BUILDERS = {
    2: _cross_section_2d,
    3: _cross_section_3d,
}

LOAD_MEASURE_SCALE = {
    2: lambda config: 1.0,
    3: lambda config: 1.0,
}

PHYSICAL_FORCE_SCALE = {
    2: lambda config: float(config["section"]["thickness"]),
    3: lambda config: 1.0,
}

_TIME_BUILDERS = {
    "elastic": _static_time,
    "dynamic": _dynamic_time,
    "wave": _dynamic_time,
}

_ELASTIC_CONTROL_RUNNERS = {
    "displacement": _run_elastic_displacement_control,
    "force": _run_elastic_force_control,
}

_DYNAMIC_STABILITY_CHECKERS = {
    "explicit_central_difference": check_explicit_stability,
    "implicit_newmark": _implicit_stability_report,
}

_DYNAMIC_SOLVERS = {
    "explicit_central_difference": _solve_dynamic_explicit,
    "implicit_newmark": _solve_dynamic_implicit,
}
