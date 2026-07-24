"""Reference-path generation for fixed linear-rod velocity-Verlet INC.

Contents:
    Preprocessing, nested time refinement, defect labels, acceptance, and HDF5 output.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from concrete_impact.nn.config import (
    VelocityVerletINCControlConfig,
    VelocityVerletINCDataGenerationConfig,
)
from concrete_impact.nn.datasets.velocity_verlet_inc import INC_DATA_SCHEMA_VERSION
from concrete_impact.nn.deployment_inc import build_velocity_verlet_inc_system_data
from concrete_impact.nn.inc_dynamics import compute_velocity_verlet_defect_labels
from fem.assembly.load import assemble_uniform_boundary_traction
from fem.cases import OutputDef
from fem.dynamics.material import solve_material_explicit
from fem.dynamics.stability import check_explicit_stability
from fem.materials.data import MaterialUpdateSettings
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.solvers.data import DirichletDofSet, MaterialDynamicSolution, TimeIntegrationSettings

CONTROL_PARAMETER_ORDER = ("pressure_amplitude", "pulse_duration")


def generate_velocity_verlet_inc_dataset(
    config: VelocityVerletINCDataGenerationConfig,
) -> Path:
    """Generate and validate one complete fixed-system INC HDF5 dataset."""
    output_path = config.output_path
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_inc_linear_rod_bundle(config, output_path.parent)
    dirichlet = build_inc_linear_rod_dirichlet(bundle)
    coarse_time = _time_settings(config, 1)
    check_explicit_stability(bundle, coarse_time)
    system = build_velocity_verlet_inc_system_data(
        bundle,
        coarse_time,
        dirichlet,
        config.plane_state,
        CONTROL_PARAMETER_ORDER,
        config.model["material"],
    )
    gauge_free_id = _arrival_free_dof(bundle, system.free_dofs, config.arrival_coordinate)

    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = INC_DATA_SCHEMA_VERSION
        handle.attrs["status"] = "writing"
        _write_system_group(handle, system)
        paths_group = handle.create_group("paths")
        for control in config.controls:
            load_function = build_inc_pressure_load_function(bundle, config, control)
            fine = solve_material_explicit(
                bundle,
                _time_settings(config, config.fine_ratio),
                np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64),
                np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64),
                dirichlet,
                load_function,
                config.plane_state,
                _linear_update_settings(),
            )
            verification = solve_material_explicit(
                bundle,
                _time_settings(config, config.verification_ratio),
                np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64),
                np.zeros(bundle.mesh_info.dof_map.size, dtype=np.float64),
                dirichlet,
                load_function,
                config.plane_state,
                _linear_update_settings(),
            )
            reference_metrics = _validate_reference_convergence(
                fine,
                verification,
                system.mass_lumped_free,
                system.free_dofs,
                gauge_free_id,
                config,
                control.pulse_duration,
            )
            path_data = _build_path_data(fine, system, config)
            _validate_path_labels(path_data, system, config.time_step)
            _write_path_group(
                paths_group.create_group(control.path_id),
                control.split,
                np.asarray(
                    [control.pressure_amplitude, control.pulse_duration],
                    dtype=np.float64,
                ),
                path_data,
                reference_metrics,
            )
        handle.attrs.modify("status", "complete")

    return output_path


def build_inc_linear_rod_bundle(
    config: VelocityVerletINCDataGenerationConfig,
    output_root: Path,
) -> PreprocessBundle:
    """Build the fixed linear-rod preprocessing bundle."""
    model = config.model
    model_def = ModelDef(
        name=str(model["name"]),
        dimension=int(model["dimension"]),
        geometry=model["geometry"],
        mesh=model["mesh"],
        material=model["material"],
        quadrature=model["quadrature"],
        boundary_conditions=model["boundary_conditions"],
    )
    output = OutputDef(
        root=output_root / "preprocess",
        mesh_path=output_root / "preprocess" / "mesh.msh",
        vtk_path=output_root / "preprocess" / "mesh.vtu",
        save_vtk=False,
        save_history=False,
        fields=(),
    )
    return build_preprocess_data(model_def, output)


def build_inc_linear_rod_dirichlet(bundle: PreprocessBundle) -> DirichletDofSet:
    """Constrain every transverse DOF and the left longitudinal boundary."""
    nodes = bundle.mesh_info.nodes
    dof_map = bundle.mesh_info.dof_map
    transverse = dof_map[:, 1]
    left_nodes = np.flatnonzero(np.isclose(nodes[:, 0], np.min(nodes[:, 0])))
    longitudinal_left = dof_map[left_nodes, 0]
    dofs = np.unique(np.concatenate((transverse, longitudinal_left))).astype(np.int64)
    return DirichletDofSet(dofs=dofs, values=np.zeros(dofs.size, dtype=np.float64))


def build_inc_pressure_load_function(
    bundle: PreprocessBundle,
    config: VelocityVerletINCDataGenerationConfig,
    control: VelocityVerletINCControlConfig,
) -> Callable[[float], NDArray[np.float64]]:
    """Build one deterministic right-boundary pressure-pulse load function."""
    def load_function(time: float) -> NDArray[np.float64]:
        """Assemble the current smooth longitudinal pressure pulse."""
        if time <= control.pulse_duration:
            pressure = control.pressure_amplitude * np.sin(
                np.pi * time / control.pulse_duration
            ) ** 2
        else:
            pressure = 0.0
        traction = np.asarray([-pressure, 0.0], dtype=np.float64)
        return assemble_uniform_boundary_traction(
            bundle.mesh_info,
            config.pressure_boundary,
            traction,
            config.thickness,
        )

    return load_function


def _time_settings(
    config: VelocityVerletINCDataGenerationConfig,
    ratio: int,
) -> TimeIntegrationSettings:
    """Build one nested explicit time grid."""
    return TimeIntegrationSettings(
        scheme="velocity_verlet",
        time_step=config.time_step / ratio,
        num_steps=config.num_steps * ratio,
        beta=0.0,
        gamma=0.5,
        cfl_safety_factor=config.cfl_safety_factor,
    )


def _linear_update_settings() -> MaterialUpdateSettings:
    """Build inactive local iteration settings required by the common interface."""
    return MaterialUpdateSettings(
        max_iterations=1,
        yield_relative_tolerance=1.0e-12,
        residual_absolute_tolerance=1.0e-12,
        residual_relative_tolerance=1.0e-12,
    )


def _arrival_free_dof(
    bundle: PreprocessBundle,
    free_dofs: NDArray[np.int64],
    coordinate: float,
) -> int:
    """Return the free-vector index of the nearest longitudinal gauge node."""
    node_id = int(np.argmin(np.abs(bundle.mesh_info.nodes[:, 0] - coordinate)))
    dof = int(bundle.mesh_info.dof_map[node_id, 0])
    matches = np.flatnonzero(free_dofs == dof)
    if matches.size != 1:
        raise ValueError("INC arrival gauge must select exactly one free longitudinal DOF.")
    return int(matches[0])


def _validate_reference_convergence(
    fine: MaterialDynamicSolution,
    verification: MaterialDynamicSolution,
    mass_free: NDArray[np.float64],
    free_dofs: NDArray[np.int64],
    gauge_free_id: int,
    config: VelocityVerletINCDataGenerationConfig,
    pulse_duration: float,
) -> dict[str, float]:
    """Require nested-grid state, arrival, and energy convergence."""
    nested_ratio = config.verification_ratio // config.fine_ratio
    verification_u = verification.displacement[::nested_ratio, free_dofs]
    verification_v = verification.velocity[::nested_ratio, free_dofs]
    fine_u = fine.displacement[:, free_dofs]
    fine_v = fine.velocity[:, free_dofs]
    displacement_error = _relative_mass_error(fine_u, verification_u, mass_free)
    velocity_error = _relative_mass_error(fine_v, verification_v, mass_free)
    verification_energy = (
        verification.kinetic_energy + verification.free_energy
    )[::nested_ratio]
    fine_energy = fine.kinetic_energy + fine.free_energy
    energy_scale = float(np.max(np.abs(verification_energy)))
    if energy_scale == 0.0:
        raise ValueError("INC reference trajectory has zero mechanical energy scale.")
    energy_error = float(np.max(np.abs(fine_energy - verification_energy)) / energy_scale)
    threshold = config.arrival_threshold * float(
        np.max(np.abs(verification_u[:, gauge_free_id]))
    )
    if threshold == 0.0:
        raise ValueError("INC arrival gauge has zero reference response.")
    fine_arrival = _threshold_arrival(fine.times, fine_u[:, gauge_free_id], threshold)
    verification_arrival = _threshold_arrival(
        verification.times,
        verification.displacement[:, free_dofs[gauge_free_id]],
        threshold,
    )
    arrival_error = abs(fine_arrival - verification_arrival) / pulse_duration
    metrics = {
        "displacement_reference_error": displacement_error,
        "velocity_reference_error": velocity_error,
        "arrival_reference_error": arrival_error,
        "energy_reference_error": energy_error,
    }
    tolerances = {
        "displacement_reference_error": config.displacement_reference_tolerance,
        "velocity_reference_error": config.velocity_reference_tolerance,
        "arrival_reference_error": config.arrival_reference_tolerance,
        "energy_reference_error": config.energy_reference_tolerance,
    }
    failed = {
        name: {"value": value, "tolerance": tolerances[name]}
        for name, value in metrics.items()
        if value > tolerances[name]
    }
    if failed:
        raise ValueError(f"INC fine reference convergence failed: {failed}.")
    return metrics


def _build_path_data(
    fine: MaterialDynamicSolution,
    system,
    config: VelocityVerletINCDataGenerationConfig,
) -> dict[str, NDArray[np.float64]]:
    """Downsample a fine path and construct unique two-stage residual labels."""
    sample = np.arange(config.num_steps + 1, dtype=np.int64) * config.fine_ratio
    free = system.free_dofs
    displacement = fine.displacement[sample][:, free]
    velocity = fine.velocity[sample][:, free]
    internal_force = fine.internal_force[sample][:, free]
    external_force = fine.external_force[sample][:, free]
    baseline_acceleration = (external_force - internal_force) / system.mass_lumped_free
    residual_start, residual_end = compute_velocity_verlet_defect_labels(
        displacement,
        velocity,
        baseline_acceleration,
        system.mass_lumped_free,
        config.time_step,
    )

    return {
        "time": fine.times[sample],
        "displacement": displacement,
        "velocity": velocity,
        "baseline_acceleration": baseline_acceleration,
        "internal_force": internal_force,
        "external_force": external_force,
        "residual_force_start": residual_start,
        "residual_force_end": residual_end,
        "mechanical_energy": (fine.kinetic_energy + fine.free_energy)[sample],
    }


def _validate_path_labels(
    data: dict[str, NDArray[np.float64]],
    system,
    time_step: float,
) -> None:
    """Require algebraic label reconstruction and linear stiffness consistency."""
    mass = system.mass_lumped_free
    source_start = -data["residual_force_start"] / mass
    source_end = -data["residual_force_end"] / mass
    reconstructed_u = (
        data["displacement"][:-1]
        + time_step * data["velocity"][:-1]
        + 0.5 * time_step**2 * (data["baseline_acceleration"][:-1] + source_start)
    )
    reconstructed_v = data["velocity"][:-1] + 0.5 * time_step * (
        data["baseline_acceleration"][:-1]
        + source_start
        + data["baseline_acceleration"][1:]
        + source_end
    )
    displacement_error = _relative_euclidean_error(reconstructed_u, data["displacement"][1:])
    velocity_error = _relative_euclidean_error(reconstructed_v, data["velocity"][1:])
    internal_from_stiffness = data["displacement"] @ system.stiffness_free.T
    stiffness_error = _relative_euclidean_error(internal_from_stiffness, data["internal_force"])
    if max(displacement_error, velocity_error, stiffness_error) > 1.0e-11:
        raise ValueError(
            "INC path algebraic validation failed: "
            f"u={displacement_error:.6e}, v={velocity_error:.6e}, "
            f"stiffness={stiffness_error:.6e}."
        )


def _write_system_group(handle: h5py.File, system) -> None:
    """Write fixed operators and compatibility metadata."""
    group = handle.create_group("system")
    group.create_dataset("free_dofs", data=system.free_dofs)
    group.create_dataset("mass_lumped_free", data=system.mass_lumped_free)
    group.create_dataset("stiffness_free", data=system.stiffness_free)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "control_parameter_order",
        data=np.asarray(system.control_parameter_order, dtype=string_dtype),
    )
    group.attrs["time_step"] = system.time_step
    group.attrs["cfl_ratio"] = system.cfl_ratio
    group.attrs["mesh_sha256"] = system.mesh_sha256
    group.attrs["lumped_mass_sha256"] = system.lumped_mass_sha256
    group.attrs["material_config_sha256"] = system.material_config_sha256
    group.attrs["boundary_config_sha256"] = system.boundary_config_sha256
    group.attrs["free_dof_order_sha256"] = system.free_dof_order_sha256


def _write_path_group(
    group: h5py.Group,
    split: str,
    controls: NDArray[np.float64],
    data: dict[str, NDArray[np.float64]],
    reference_metrics: dict[str, float],
) -> None:
    """Write one complete accepted path using semantic subgroups."""
    group.attrs["split"] = split
    for name, value in reference_metrics.items():
        group.attrs[name] = value
    group.create_dataset("time", data=data["time"])
    group.create_dataset("control_parameters", data=controls)
    state = group.create_group("state")
    state.create_dataset("displacement", data=data["displacement"])
    state.create_dataset("velocity", data=data["velocity"])
    baseline = group.create_group("baseline")
    baseline.create_dataset("acceleration", data=data["baseline_acceleration"])
    baseline.create_dataset("internal_force", data=data["internal_force"])
    baseline.create_dataset("external_force", data=data["external_force"])
    target = group.create_group("target")
    target.create_dataset("residual_force_start", data=data["residual_force_start"])
    target.create_dataset("residual_force_end", data=data["residual_force_end"])
    reference = group.create_group("reference")
    reference.create_dataset("mechanical_energy", data=data["mechanical_energy"])


def _relative_mass_error(
    values: NDArray[np.float64],
    reference: NDArray[np.float64],
    mass: NDArray[np.float64],
) -> float:
    """Compute a trajectory relative lumped-mass norm."""
    numerator = np.sum((values - reference) ** 2 * mass[None, :])
    denominator = np.sum(reference**2 * mass[None, :])
    return float(np.sqrt(numerator / denominator))


def _relative_euclidean_error(
    values: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> float:
    """Compute a relative Euclidean error with a physical nonzero reference."""
    norm = np.linalg.norm(reference)
    if norm == 0.0:
        raise ValueError("INC algebraic validation requires a nonzero reference field.")
    return float(np.linalg.norm(values - reference) / norm)


def _threshold_arrival(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    threshold: float,
) -> float:
    """Return the first configured threshold crossing time."""
    matches = np.flatnonzero(np.abs(values) >= threshold)
    if matches.size == 0:
        raise ValueError("INC reference path never reaches the arrival threshold.")
    return float(times[matches[0]])
