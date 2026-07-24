"""Static nonlinear viscoplastic solid benchmarks.

Contents:
    Benchmark paths, local solves, reference response, metrics, plots, and result records.
Author:
    Zhen Hao.
Created:
    2026-07-09.
"""

from __future__ import annotations

import copy
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from concrete_impact.core.failure_records import record_material_point_failures
from fem.assembly.nonlinear_solid import (
    assemble_bundle_material_response,
    build_material_assembly_cache,
    initialize_bundle_material_state,
)
from fem.assembly.solid import build_solid_dof_map
from fem.cases import OutputDef, RunResult
from fem.io.config import write_yaml_config
from fem.materials import (
    J2ViscoplasticMaterial,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
    build_material,
)
from fem.post.curves import write_response_csv
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.solvers.data import NewtonSettings
from fem.solvers.newton import solve_newton

VOIGT_DIM = 6


@record_material_point_failures
def run_static_viscoplastic_benchmark(config: dict[str, Any]) -> RunResult:
    """Run a static nonlinear viscoplastic solid benchmark."""
    case_name = str(config["case"]["name"])
    output_root = Path(str(config["output"]["root"]))
    material = build_material(config["model"]["material"])
    if not isinstance(material, J2ViscoplasticMaterial):
        raise TypeError("Static viscoplastic benchmark requires J2ViscoplasticMaterial.")

    curve_data = _build_static_cube_curves(config, material)
    output_paths = _build_output_paths(output_root)
    response_csv = write_response_csv(curve_data, output_paths["response_csv"])
    diagnostics_csv = write_response_csv(curve_data, output_paths["diagnostics_csv"])
    response_png = _write_static_cube_plot(curve_data, output_paths["response_png"])
    output_paths["response_csv"] = response_csv
    output_paths["diagnostics_csv"] = diagnostics_csv
    output_paths["response_png"] = response_png
    metrics = _compute_static_metrics(
        curve_data,
        float(config["verification"]["absolute_tolerance"]),
    )
    passed = _passes_metrics(
        metrics,
        float(config["verification"]["relative_tolerance"]),
        float(config["verification"]["absolute_tolerance"]),
    )
    result = RunResult(name=case_name, metrics=metrics, output_paths=output_paths, passed=passed)
    _write_records(config, result, output_root)

    return result


def _build_static_cube_curves(
    config: dict[str, Any],
    material: J2ViscoplasticMaterial,
) -> dict[str, NDArray[np.float64]]:
    """Build all mesh-division static cube curves."""
    records: dict[str, list[float]] = {
        "mesh_divisions": [],
        "step": [],
        "time": [],
        "axial_strain": [],
        "reaction_numerical": [],
        "reaction_exact": [],
        "lateral_strain_numerical": [],
        "lateral_strain_exact": [],
        "equivalent_plastic_strain": [],
        "newton_iterations": [],
        "newton_residual_norm": [],
    }
    for division in config["benchmark"]["mesh_divisions"]:
        _append_static_cube_path(records, config, material, int(division))

    return {name: np.asarray(values, dtype=np.float64) for name, values in records.items()}


def _append_static_cube_path(
    records: dict[str, list[float]],
    config: dict[str, Any],
    material: J2ViscoplasticMaterial,
    mesh_division: int,
) -> None:
    """Append one static cube mesh-resolution path."""
    bundle = _build_mesh_bundle(config, mesh_division)
    nodes = bundle.mesh_info.nodes
    length_values = tuple(float(value) for value in config["model"]["geometry"]["size"])
    if len(length_values) != 3:
        raise ValueError("Static viscoplastic cube requires exactly three geometry lengths.")
    lengths = (length_values[0], length_values[1], length_values[2])
    time_step = float(config["analysis"]["time_step"])
    strain_rate = float(config["benchmark"]["axial_strain_rate"])
    num_steps = int(config["analysis"]["num_steps"])
    newton_settings = NewtonSettings(
        max_iterations=int(config["solver"]["newton"]["max_iterations"]),
        residual_tolerance=float(config["solver"]["newton"]["residual_tolerance"]),
        increment_tolerance=float(config["solver"]["newton"]["increment_tolerance"]),
    )
    material_update_settings = _build_material_update_settings(config)
    state = initialize_bundle_material_state(bundle)
    assembly_cache = build_material_assembly_cache(bundle)
    displacement = np.zeros(nodes.shape[0] * 3, dtype=np.float64)
    exact_state = material.initialize_state(1)
    exact_lateral_strain = 0.0

    for step_id in range(num_steps + 1):
        axial_strain = strain_rate * time_step * step_id
        exact_lateral_strain, exact_response = _solve_uniaxial_stress_reference(
            material,
            exact_state,
            axial_strain,
            exact_lateral_strain,
            config,
        )
        exact_state = _require_response_state(exact_response)
        step_result = _solve_static_cube_step(
            bundle,
            state,
            displacement,
            axial_strain,
            lengths,
            time_step,
            newton_settings,
            material_update_settings,
            assembly_cache,
        )
        state = step_result["assembly"].state
        displacement = step_result["displacement"]
        reaction_dofs = _right_face_x_dofs(nodes, lengths[0])
        reaction = float(np.sum(step_result["assembly"].internal_force[reaction_dofs]))
        lateral_strain = _compute_average_lateral_strain(displacement, nodes, lengths)
        equivalent_plastic_strain = float(
            np.mean(step_result["assembly"].diagnostics["equivalent_plastic_strain"])
        )

        records["mesh_divisions"].append(float(mesh_division))
        records["step"].append(float(step_id))
        records["time"].append(time_step * step_id)
        records["axial_strain"].append(axial_strain)
        records["reaction_numerical"].append(reaction)
        exact_reaction = float(exact_response.stresses[0, 0] * lengths[1] * lengths[2])
        records["reaction_exact"].append(exact_reaction)
        records["lateral_strain_numerical"].append(lateral_strain)
        records["lateral_strain_exact"].append(exact_lateral_strain)
        records["equivalent_plastic_strain"].append(equivalent_plastic_strain)
        records["newton_iterations"].append(float(step_result["newton"].iterations))
        records["newton_residual_norm"].append(float(step_result["newton"].residual_norm))


def _solve_static_cube_step(
    bundle: PreprocessBundle,
    state: MaterialState,
    previous_displacement: NDArray[np.float64],
    axial_strain: float,
    lengths: tuple[float, float, float],
    time_step: float,
    newton_settings: NewtonSettings,
    material_update_settings: MaterialUpdateSettings,
    assembly_cache,
) -> dict[str, Any]:
    """Solve one displacement-controlled static cube step."""
    nodes = bundle.mesh_info.nodes
    prescribed_dofs, prescribed_values = _build_uniaxial_dirichlet(nodes, lengths, axial_strain)
    dof_count = nodes.shape[0] * 3
    all_dofs = np.arange(dof_count, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs, assume_unique=False)
    template = previous_displacement.copy()
    template[prescribed_dofs] = prescribed_values

    def build_displacement(free_values: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = template.copy()
        displacement[free_dofs] = free_values

        return displacement

    def residual_function(free_values: NDArray[np.float64]) -> NDArray[np.float64]:
        displacement = build_displacement(free_values)
        assembly = assemble_bundle_material_response(
            bundle,
            state,
            displacement,
            previous_displacement,
            time_step,
            "three_dimensional",
            material_update_settings,
            assembly_cache,
            need_tangent=True,
        )

        return assembly.internal_force[free_dofs]

    def tangent_function(free_values: NDArray[np.float64]):
        displacement = build_displacement(free_values)
        assembly = assemble_bundle_material_response(
            bundle,
            state,
            displacement,
            previous_displacement,
            time_step,
            "three_dimensional",
            material_update_settings,
            assembly_cache,
            need_tangent=True,
        )

        return assembly.tangent[free_dofs, :][:, free_dofs].tocsr()

    newton = solve_newton(template[free_dofs], residual_function, tangent_function, newton_settings)
    displacement = build_displacement(newton.solution)
    assembly = assemble_bundle_material_response(
        bundle,
        state,
        displacement,
        previous_displacement,
        time_step,
        "three_dimensional",
        material_update_settings,
        assembly_cache,
        need_tangent=True,
    )

    return {
        "displacement": displacement,
        "newton": newton,
        "assembly": assembly,
    }


def _build_mesh_bundle(config: dict[str, Any], mesh_division: int) -> PreprocessBundle:
    """Build one fully preprocessed cube mesh for a refinement level."""
    model = copy.deepcopy(config["model"])
    model["mesh"]["divisions"] = [mesh_division, mesh_division, mesh_division]
    model_def = ModelDef(
        name=f"{model['name']}_mesh_{mesh_division}",
        dimension=int(model["dimension"]),
        geometry=model["geometry"],
        mesh=model["mesh"],
        material=model["material"],
        quadrature=model["quadrature"],
        boundary_conditions=model["boundary_conditions"],
    )
    output_root = Path(str(config["output"]["root"])) / f"mesh_{mesh_division}"
    output_def = OutputDef(
        root=output_root,
        mesh_path=output_root / "mesh.msh",
        vtk_path=output_root / "mesh.vtu",
        save_vtk=False,
        save_history=False,
        fields=("mesh",),
    )

    return build_preprocess_data(model_def, output_def)


def _solve_uniaxial_stress_reference(
    material: J2ViscoplasticMaterial,
    state: MaterialState,
    axial_strain: float,
    initial_lateral_strain: float,
    config: dict[str, Any],
) -> tuple[float, MaterialPointResponse]:
    """Solve the material-point lateral strain for zero lateral stress."""
    lateral_strain = initial_lateral_strain
    tolerance = float(config["benchmark"]["lateral_stress_tolerance"])
    max_iterations = int(config["benchmark"]["lateral_newton_max_iterations"])

    for _ in range(max_iterations):
        response = _update_uniaxial_reference_point(
            material,
            state,
            axial_strain,
            lateral_strain,
            config,
        )
        residual = response.stresses[0, 1]
        if abs(residual) <= tolerance:
            return lateral_strain, response

        step = 1.0e-8 * max(1.0, abs(lateral_strain))
        response_plus = _update_uniaxial_reference_point(
            material,
            state,
            axial_strain,
            lateral_strain + step,
            config,
        )
        response_minus = _update_uniaxial_reference_point(
            material,
            state,
            axial_strain,
            lateral_strain - step,
            config,
        )
        derivative = (response_plus.stresses[0, 1] - response_minus.stresses[0, 1]) / (
            2.0 * step
        )
        if abs(derivative) <= 1.0e-12:
            raise RuntimeError("Static cube reference lateral Newton derivative is singular.")
        lateral_strain -= residual / derivative

    raise RuntimeError("Static cube reference lateral Newton iteration did not converge.")


def _update_uniaxial_reference_point(
    material: J2ViscoplasticMaterial,
    state: MaterialState,
    axial_strain: float,
    lateral_strain: float,
    config: dict[str, Any],
) -> MaterialPointResponse:
    """Update the material point on an axisymmetric uniaxial-stress strain path."""
    strain = np.zeros((1, VOIGT_DIM), dtype=np.float64)
    strain[0, 0] = axial_strain
    strain[0, 1] = lateral_strain
    strain[0, 2] = lateral_strain
    request = MaterialPointRequest(
        strains=strain,
        strain_rates=np.zeros_like(strain),
        time_step=float(config["analysis"]["time_step"]),
        kinematics="three_dimensional",
        update_settings=_build_material_update_settings(config),
    )

    return material.update(
        request,
        state,
        MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
    )


def _build_material_update_settings(config: dict[str, Any]) -> MaterialUpdateSettings:
    """Build local constitutive integration settings."""
    settings = config["analysis"]["material_update"]

    return MaterialUpdateSettings(
        max_iterations=int(settings["max_iterations"]),
        yield_relative_tolerance=float(settings["yield_relative_tolerance"]),
        residual_absolute_tolerance=float(settings["residual_absolute_tolerance"]),
        residual_relative_tolerance=float(settings["residual_relative_tolerance"]),
    )


def _build_uniaxial_dirichlet(
    nodes: NDArray[np.float64],
    lengths: tuple[float, float, float],
    axial_strain: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Build displacement-control DOFs for a free-lateral cube."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    left_nodes = np.flatnonzero(np.isclose(nodes[:, 0], 0.0))
    right_nodes = np.flatnonzero(np.isclose(nodes[:, 0], lengths[0]))
    origin_node = _find_node(nodes, (0.0, 0.0, 0.0))
    y_anchor_node = _find_node(nodes, (0.0, lengths[1], 0.0))
    dof_values: dict[int, float] = {}

    for node_id in left_nodes:
        dof_values[int(dof_map[node_id, 0])] = 0.0
    for node_id in right_nodes:
        dof_values[int(dof_map[node_id, 0])] = axial_strain * lengths[0]
    dof_values[int(dof_map[origin_node, 1])] = 0.0
    dof_values[int(dof_map[origin_node, 2])] = 0.0
    dof_values[int(dof_map[y_anchor_node, 2])] = 0.0

    prescribed_dofs = np.asarray(sorted(dof_values), dtype=np.int64)
    prescribed_values = np.asarray([dof_values[int(dof)] for dof in prescribed_dofs])

    return prescribed_dofs, prescribed_values


def _find_node(nodes: NDArray[np.float64], point: tuple[float, float, float]) -> int:
    """Find one structured-grid node by physical coordinates."""
    matches = np.flatnonzero(np.all(np.isclose(nodes, np.asarray(point)), axis=1))
    if matches.shape[0] != 1:
        raise ValueError("Structured cube benchmark expected exactly one matching node.")

    return int(matches[0])


def _right_face_x_dofs(nodes: NDArray[np.float64], length_x: float) -> NDArray[np.int64]:
    """Return x-DOFs on the right cube face."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    right_nodes = np.flatnonzero(np.isclose(nodes[:, 0], length_x))

    return dof_map[right_nodes, 0]


def _compute_average_lateral_strain(
    displacement: NDArray[np.float64],
    nodes: NDArray[np.float64],
    lengths: tuple[float, float, float],
) -> float:
    """Compute average lateral strain from free-surface displacements."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    y_face_nodes = np.flatnonzero(np.isclose(nodes[:, 1], lengths[1]))
    z_face_nodes = np.flatnonzero(np.isclose(nodes[:, 2], lengths[2]))
    y_strain = float(np.mean(displacement[dof_map[y_face_nodes, 1]]) / lengths[1])
    z_strain = float(np.mean(displacement[dof_map[z_face_nodes, 2]]) / lengths[2])

    return 0.5 * (y_strain + z_strain)


def _require_response_state(response: MaterialPointResponse) -> MaterialState:
    """Return the updated material state from a response."""
    if response.state is None:
        raise ValueError("Static benchmark requires material responses with updated state.")

    return response.state


def _compute_static_metrics(
    data: dict[str, NDArray[np.float64]],
    absolute_tolerance: float,
) -> dict[str, float]:
    """Compute static benchmark curve errors."""
    metrics: dict[str, float] = {}
    for numerical_name, exact_name in (
        ("reaction_numerical", "reaction_exact"),
        ("lateral_strain_numerical", "lateral_strain_exact"),
    ):
        difference = data[numerical_name] - data[exact_name]
        scale = max(float(np.linalg.norm(data[exact_name], ord=2)), absolute_tolerance)
        metrics[f"{numerical_name}_relative_l2_error"] = float(
            np.linalg.norm(difference, ord=2) / scale,
        )
        metrics[f"{numerical_name}_max_abs_error"] = float(np.max(np.abs(difference)))
        for mesh_division in np.unique(data["mesh_divisions"]):
            mask = data["mesh_divisions"] == mesh_division
            mesh_difference = difference[mask]
            mesh_exact = data[exact_name][mask]
            mesh_scale = max(float(np.linalg.norm(mesh_exact, ord=2)), absolute_tolerance)
            label = f"{int(mesh_division)}x{int(mesh_division)}x{int(mesh_division)}"
            metrics[f"{numerical_name}_{label}_relative_l2_error"] = float(
                np.linalg.norm(mesh_difference, ord=2) / mesh_scale,
            )
            metrics[f"{numerical_name}_{label}_max_abs_error"] = float(
                np.max(np.abs(mesh_difference)),
            )

    return metrics


def _passes_metrics(
    metrics: dict[str, float],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    """Evaluate pass/fail from curve metrics."""
    return all(
        (
            value <= relative_tolerance
            if metric_name.endswith("relative_l2_error")
            else value <= absolute_tolerance
        )
        for metric_name, value in metrics.items()
    )


def _write_static_cube_plot(
    data: dict[str, NDArray[np.float64]],
    output_path: str | Path,
) -> Path:
    """Write static cube response plots."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.2))

    for mesh_division in np.unique(data["mesh_divisions"]):
        mask = data["mesh_divisions"] == mesh_division
        label = f"{int(mesh_division)}x{int(mesh_division)}x{int(mesh_division)}"
        axes[0].plot(data["axial_strain"][mask], data["reaction_numerical"][mask], label=label)
        axes[1].plot(
            data["axial_strain"][mask],
            data["lateral_strain_numerical"][mask],
            label=label,
        )
        axes[2].plot(data["axial_strain"][mask], data["newton_iterations"][mask], label=label)

    first_mesh = data["mesh_divisions"] == np.unique(data["mesh_divisions"])[0]
    axes[0].plot(
        data["axial_strain"][first_mesh],
        data["reaction_exact"][first_mesh],
        "--",
        label="exact",
    )
    axes[1].plot(
        data["axial_strain"][first_mesh],
        data["lateral_strain_exact"][first_mesh],
        "--",
        label="exact",
    )
    axes[0].set_ylabel("reaction")
    axes[1].set_ylabel("lateral_strain")
    axes[2].set_ylabel("newton_iterations")
    axes[2].set_xlabel("axial_strain")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

    return path


def _build_output_paths(output_root: Path) -> dict[str, Path]:
    """Build standard output paths for static viscoplastic benchmarks."""
    return {
        "response_csv": output_root / "response.csv",
        "diagnostics_csv": output_root / "diagnostics.csv",
        "response_png": output_root / "response.png",
        "original_config": output_root / "original_config.yaml",
        "resolved_config": output_root / "resolved_config.yaml",
        "metadata": output_root / "run_metadata.json",
        "metrics": output_root / "metrics.json",
    }


def _write_records(config: dict[str, Any], result: RunResult, output_root: Path) -> None:
    """Write static benchmark reproducibility records."""
    output_root.mkdir(parents=True, exist_ok=True)
    write_yaml_config(config, result.output_paths["original_config"])
    write_yaml_config(copy.deepcopy(config), result.output_paths["resolved_config"])
    _write_json(result.output_paths["metadata"], _build_metadata_record(config))
    _write_json(result.output_paths["metrics"], _build_metrics_record(result))


def _build_metadata_record(config: dict[str, Any]) -> dict[str, Any]:
    """Build static benchmark metadata."""
    return {
        "case_name": str(config["case"]["name"]),
        "benchmark_type": str(config["benchmark"]["type"]),
        "analysis_type": "static_nonlinear_viscoplastic_solid",
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "metadata": config["case"],
    }


def _build_metrics_record(result: RunResult) -> dict[str, Any]:
    """Build static benchmark metrics record."""
    return {
        "case_name": result.name,
        "passed": result.passed,
        "metrics": result.metrics,
        "output_paths": {name: str(path) for name, path in result.output_paths.items()},
    }


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
