"""Linear elastic benchmark entrypoints.

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

from fem.cases import CaseDef, OutputDef, RunResult, write_run_records
from fem.dynamics.newmark import solve_explicit_central_difference, solve_implicit_newmark
from fem.dynamics.stability import check_explicit_stability, compute_explicit_stability_report
from fem.materials.linear_elastic import LinearElasticMaterial
from fem.post.fields import average_quadrature_fields, compute_linear_elastic_quadrature_fields
from fem.post.vtk import write_quadrature_points_vtk, write_solution_vtk
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.solvers.data import (
    DirichletDofSet,
    FEMRunDef,
    LinearSolverSettings,
    NewtonSettings,
    SurrogateSettings,
    TimeIntegrationSettings,
)
from fem.solvers.static import solve_linear_static

ExactDisplacement = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


def run_linear_elastic(config: dict[str, Any]) -> RunResult:
    """Run one YAML-configured linear elastic benchmark."""
    case_def, bundle = prepare_linear_elastic_case(config)
    result = run_linear_elastic_case(case_def, bundle)
    record_paths = write_run_records(case_def, result, config, config)

    return RunResult(
        name=result.name,
        metrics=result.metrics,
        output_paths={**result.output_paths, **record_paths},
        passed=result.passed,
    )


def prepare_linear_elastic_case(
    config: dict[str, Any],
) -> tuple[CaseDef, PreprocessBundle]:
    """Prepare one linear elastic benchmark case and its preprocess bundle."""
    model_def = _build_model_def(config["model"])
    output_def = _build_output_def(config["output"])
    bundle = build_preprocess_data(model_def, output_def)
    exact_displacement = _EXACT_DISPLACEMENT_BUILDERS[
        str(config["analysis"]["exact_solution"])
    ](bundle)
    run_def = _RUN_DEF_BUILDERS[str(config["analysis"]["type"])](
        config,
        bundle,
        exact_displacement,
    )

    case_def = CaseDef(
        name=str(config["case"]["name"]),
        model=model_def,
        run=run_def,
        output=output_def,
        metadata={
            "description": str(config["case"]["description"]),
            "benchmark_family": "linear_elastic",
            "verification_metric": str(config["verification"]["metric"]),
            "verification_tolerance": float(config["verification"]["tolerance"]),
        },
    )

    return case_def, bundle


def build_linear_elastic_case(config: dict[str, Any]) -> CaseDef:
    """Build one linear elastic benchmark case definition."""
    case_def, _ = prepare_linear_elastic_case(config)

    return case_def


def run_linear_elastic_case(
    case_def: CaseDef,
    bundle: PreprocessBundle,
) -> RunResult:
    """Run a prepared linear elastic benchmark case."""
    result = _CASE_RUNNERS[case_def.run.analysis_type](case_def, bundle)
    tolerance = float(case_def.metadata["verification_tolerance"])
    error = result.metrics["relative_displacement_error"]

    return RunResult(
        name=result.name,
        metrics=result.metrics,
        output_paths=result.output_paths,
        passed=bool(error <= tolerance),
    )


def _build_model_def(config: dict[str, Any]) -> ModelDef:
    """Build the preprocess model definition."""
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
    """Build the case output definition."""
    return OutputDef(
        root=Path(config["root"]),
        mesh_path=Path(config["mesh_path"]),
        vtk_path=Path(config["vtk_path"]),
        save_vtk=bool(config["save_vtk"]),
        save_history=bool(config["save_history"]),
        fields=tuple(config["fields"]),
    )


def _build_static_run_def(
    config: dict[str, Any],
    bundle: PreprocessBundle,
    exact_displacement: ExactDisplacement,
) -> FEMRunDef:
    """Build solver settings for a static benchmark."""
    dof_count = bundle.mesh_info.dof_map.size
    dirichlet = _build_all_boundary_dirichlet(bundle, exact_displacement, 0.0)

    return _build_run_def(
        config=config,
        bundle=bundle,
        analysis_type="static",
        dirichlet=dirichlet,
        external_force=np.zeros(dof_count, dtype=np.float64),
        initial_displacement=np.zeros(dof_count, dtype=np.float64),
        initial_velocity=np.zeros(dof_count, dtype=np.float64),
        exact_displacement=exact_displacement,
        time=_make_static_time_settings(),
    )


def _build_dynamic_run_def(
    config: dict[str, Any],
    bundle: PreprocessBundle,
    exact_displacement: ExactDisplacement,
) -> FEMRunDef:
    """Build solver settings for a dynamic benchmark."""
    initial_displacement = _flatten_node_field(
        exact_displacement(bundle.mesh_info.nodes, 0.0),
    )
    initial_velocity = np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64)
    dirichlet = _build_longitudinal_mode_dirichlet(bundle)
    time = _make_dynamic_time_settings(config["analysis"]["time"])
    _DYNAMIC_STABILITY_CHECKERS[time.scheme](bundle, time)

    return _build_run_def(
        config=config,
        bundle=bundle,
        analysis_type="dynamic",
        dirichlet=dirichlet,
        external_force=np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64),
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        exact_displacement=exact_displacement,
        time=time,
    )


def _build_run_def(
    config: dict[str, Any],
    bundle: PreprocessBundle,
    analysis_type: str,
    dirichlet: DirichletDofSet,
    external_force: NDArray[np.float64],
    initial_displacement: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
    exact_displacement: ExactDisplacement,
    time: TimeIntegrationSettings,
) -> FEMRunDef:
    """Build a complete solver definition."""
    analysis = config["analysis"]

    return FEMRunDef(
        name=str(config["model"]["name"]),
        analysis_type=analysis_type,
        plane_state=str(analysis["plane_state"]),
        dirichlet=dirichlet,
        external_force=external_force,
        initial_displacement=initial_displacement,
        initial_velocity=initial_velocity,
        exact_displacement=exact_displacement,
        linear_solver=LinearSolverSettings(
            backend=str(analysis["linear_solver"]["backend"]),
            method=str(analysis["linear_solver"]["method"]),
        ),
        newton=NewtonSettings(
            max_iterations=int(analysis["newton"]["max_iterations"]),
            residual_tolerance=float(analysis["newton"]["residual_tolerance"]),
            increment_tolerance=float(analysis["newton"]["increment_tolerance"]),
        ),
        time=time,
        surrogate=_SURROGATE_BUILDERS[str(analysis["surrogate"]["mode"])](analysis),
    )


def _run_static_case(
    case_def: CaseDef,
    bundle: PreprocessBundle,
) -> RunResult:
    """Run a static linear elastic benchmark."""
    solution = solve_linear_static(bundle, case_def.run)
    exact = _flatten_node_field(
        case_def.run.exact_displacement(bundle.mesh_info.nodes, 0.0),
    )
    relative_error = _relative_l2_error(solution.displacement, exact)
    output_paths = _write_linear_elastic_outputs(
        case_def,
        bundle,
        solution.displacement,
        "static",
    )

    return RunResult(
        name=case_def.name,
        metrics={"relative_displacement_error": relative_error},
        output_paths=output_paths,
        passed=True,
    )


def _run_dynamic_case(
    case_def: CaseDef,
    bundle: PreprocessBundle,
) -> RunResult:
    """Run a dynamic linear elastic benchmark."""
    solution = _DYNAMIC_SOLVERS[case_def.run.time.scheme](bundle, case_def.run)
    final_time = float(solution.times[-1])
    exact = _flatten_node_field(
        case_def.run.exact_displacement(bundle.mesh_info.nodes, final_time),
    )
    relative_error = _relative_l2_error(solution.displacement[-1, :], exact)
    output_paths = _write_linear_elastic_outputs(
        case_def,
        bundle,
        solution.displacement[-1, :],
        "dynamic",
    )

    return RunResult(
        name=case_def.name,
        metrics={"relative_displacement_error": relative_error},
        output_paths=output_paths,
        passed=True,
    )


def _build_all_boundary_dirichlet(
    bundle: PreprocessBundle,
    exact_displacement: ExactDisplacement,
    time: float,
) -> DirichletDofSet:
    """Build prescribed values on all boundary nodes."""
    boundary_nodes = _collect_boundary_nodes(bundle)
    dofs = bundle.mesh_info.dof_map[boundary_nodes, :].reshape(-1)
    exact_values = _flatten_node_field(exact_displacement(bundle.mesh_info.nodes, time))

    return DirichletDofSet(dofs=dofs, values=exact_values[dofs])


def _build_longitudinal_mode_dirichlet(bundle: PreprocessBundle) -> DirichletDofSet:
    """Build zero constraints for the longitudinal wave benchmark."""
    mesh_info = bundle.mesh_info
    left_right_nodes = np.union1d(
        mesh_info.boundary_groups["left"].nodes,
        mesh_info.boundary_groups["right"].nodes,
    )
    axial_dofs = mesh_info.dof_map[left_right_nodes, 0]
    lateral_dofs = mesh_info.dof_map[:, 1:].reshape(-1)
    dofs = np.unique(np.concatenate([axial_dofs, lateral_dofs]))

    return DirichletDofSet(dofs=dofs, values=np.zeros(dofs.shape[0], dtype=np.float64))


def _collect_boundary_nodes(bundle: PreprocessBundle) -> NDArray[np.int64]:
    """Collect all unique boundary nodes."""
    boundary_nodes = [
        boundary_group.nodes
        for boundary_group in bundle.mesh_info.boundary_groups.values()
    ]

    return np.unique(np.concatenate(boundary_nodes))


def _make_static_affine_displacement_2d(
    bundle: PreprocessBundle,
) -> ExactDisplacement:
    """Create the exact affine displacement field for the 2D patch test."""
    gradient = np.asarray([[1.0e-4, 2.0e-5], [-1.0e-5, 8.0e-5]], dtype=np.float64)

    def exact(nodes: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        """Evaluate the exact 2D static displacement."""
        return nodes @ gradient.T

    return exact


def _make_static_affine_displacement_3d(
    bundle: PreprocessBundle,
) -> ExactDisplacement:
    """Create the exact affine displacement field for the 3D patch test."""
    gradient = np.asarray(
        [
            [1.0e-4, 2.0e-5, -1.0e-5],
            [-1.0e-5, 8.0e-5, 1.5e-5],
            [0.5e-5, -2.0e-5, 6.0e-5],
        ],
        dtype=np.float64,
    )

    def exact(nodes: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        """Evaluate the exact 3D static displacement."""
        return nodes @ gradient.T

    return exact


def _make_longitudinal_mode_displacement(
    bundle: PreprocessBundle,
) -> ExactDisplacement:
    """Create the exact longitudinal standing-wave displacement field."""
    length = float(bundle.model_def.geometry["size"][0])
    amplitude = 1.0e-6
    wave_speed = _compute_constrained_longitudinal_wave_speed(bundle)
    angular_frequency = np.pi * wave_speed / length

    def exact(nodes: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        """Evaluate the exact longitudinal wave displacement."""
        values = np.zeros((nodes.shape[0], bundle.mesh_info.dimension), dtype=np.float64)
        values[:, 0] = amplitude * np.sin(np.pi * nodes[:, 0] / length) * np.cos(
            angular_frequency * time
        )

        return values

    return exact


def _compute_constrained_longitudinal_wave_speed(bundle: PreprocessBundle) -> float:
    """Compute the constrained longitudinal wave speed."""
    material = _require_linear_elastic_material(bundle)
    young_modulus = material.young_modulus
    poisson_ratio = material.poisson_ratio
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    return float(np.sqrt((lame_lambda + 2.0 * shear_modulus) / material.density))


def _require_linear_elastic_material(bundle: PreprocessBundle) -> LinearElasticMaterial:
    """Require the material type used by analytic elastic benchmark formulas."""
    if not isinstance(bundle.material, LinearElasticMaterial):
        raise TypeError("Linear-elastic benchmark formulas require LinearElasticMaterial.")

    return bundle.material


def _make_static_time_settings() -> TimeIntegrationSettings:
    """Create inactive time settings for static benchmarks."""
    return TimeIntegrationSettings(
        scheme="static",
        time_step=0.0,
        num_steps=0,
        beta=0.25,
        gamma=0.5,
        cfl_safety_factor=0.8,
    )


def _make_dynamic_time_settings(config: dict[str, Any]) -> TimeIntegrationSettings:
    """Create time settings for dynamic benchmarks."""
    return TimeIntegrationSettings(
        scheme=str(config["scheme"]),
        time_step=float(config["time_step"]),
        num_steps=int(config["num_steps"]),
        beta=float(config["beta"]),
        gamma=float(config["gamma"]),
        cfl_safety_factor=float(config["cfl_safety_factor"]),
    )


def _make_disabled_surrogate_settings(config: dict[str, Any]) -> SurrogateSettings:
    """Create inactive surrogate settings."""
    if str(config["surrogate"]["mode"]) != "disabled":
        raise ValueError("Linear-elastic benchmark currently requires surrogate.mode=disabled.")

    return SurrogateSettings(enabled=False)


def _write_linear_elastic_outputs(
    case_def: CaseDef,
    bundle: PreprocessBundle,
    displacement: NDArray[np.float64],
    tag: str,
) -> dict[str, Path]:
    """Write solution and quadrature fields for one benchmark."""
    quadrature_fields = compute_linear_elastic_quadrature_fields(
        bundle,
        displacement,
        case_def.run.plane_state,
    )
    cell_fields = average_quadrature_fields(quadrature_fields)
    solution_path = write_solution_vtk(
        bundle.mesh_info,
        case_def.output.root / f"{case_def.run.name}_{tag}_solution.vtu",
        displacement,
        cell_fields,
    )
    quadrature_path = write_quadrature_points_vtk(
        case_def.output.root / f"{case_def.run.name}_{tag}_quadrature.vtu",
        quadrature_fields,
    )

    return {"solution": solution_path, "quadrature": quadrature_path}


def _flatten_node_field(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten a nodal vector field by node-major DOF ordering."""
    return field.reshape(-1)


def _relative_l2_error(
    numerical: NDArray[np.float64],
    exact: NDArray[np.float64],
) -> float:
    """Compute the relative Euclidean error."""
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def _implicit_stability_report(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
) -> None:
    """Evaluate the explicit CFL bound for an implicit comparison run."""
    compute_explicit_stability_report(bundle, time_settings)


_RUN_DEF_BUILDERS = {
    "static": _build_static_run_def,
    "dynamic": _build_dynamic_run_def,
}

_CASE_RUNNERS = {
    "static": _run_static_case,
    "dynamic": _run_dynamic_case,
}

_EXACT_DISPLACEMENT_BUILDERS = {
    "static_affine_2d": _make_static_affine_displacement_2d,
    "static_affine_3d": _make_static_affine_displacement_3d,
    "longitudinal_mode": _make_longitudinal_mode_displacement,
}

_DYNAMIC_STABILITY_CHECKERS = {
    "explicit_central_difference": check_explicit_stability,
    "implicit_newmark": _implicit_stability_report,
}

_DYNAMIC_SOLVERS = {
    "explicit_central_difference": solve_explicit_central_difference,
    "implicit_newmark": solve_implicit_newmark,
}

_SURROGATE_BUILDERS = {
    "disabled": _make_disabled_surrogate_settings,
}
