"""Memory-bounded generation of fixed-system linear-elastic INC paths.

Author:
    Zhen Hao.
Created:
    2026-09-15.
"""

from __future__ import annotations

import json
import resource
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from concrete_impact.experiments.inc_elastic_plan import (
    ElasticINCPlan,
    ElasticINCTask,
    build_control_parameters,
    control_parameter_order,
)
from concrete_impact.nn.inc_dynamics import compute_velocity_verlet_defect_labels
from fem.assembly.elasticity import assemble_lumped_mass_vector, assemble_stiffness_matrix
from fem.assembly.load import assemble_boundary_traction
from fem.cases import OutputDef
from fem.dynamics.stability import estimate_minimum_edge_length
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import ModelDef


@dataclass(frozen=True)
class ElasticINCSystem:
    """Store the fixed free system used by every path."""

    free_dofs: NDArray[np.int64]
    mass: NDArray[np.float64]
    stiffness: csr_matrix
    load_basis: NDArray[np.float64]
    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    dof_map: NDArray[np.int64]
    time_step: float
    cfl_ratio: float


class ElasticINCAcceptanceError(RuntimeError):
    """Report a failed path with structured numerical diagnostics."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        """Store the failed acceptance metrics."""
        super().__init__(f"Elastic INC path acceptance failed: {diagnostics}.")
        self.diagnostics = diagnostics


def build_elastic_inc_preflight(
    plan: ElasticINCPlan,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build, validate, and write one fixed elastic INC system."""
    root = Path(output_root)
    system_path = root / "system.h5"
    report_path = root / "preflight.json"
    if system_path.exists():
        validate_elastic_inc_system_plan(system_path, plan)
        system = load_elastic_inc_system(system_path)
        _validate_existing_system(system, plan)
        report = _read_json(report_path)
        if report["passed"] is not True:
            raise ValueError("Existing INC preflight report did not pass.")
        return report

    bundle = _build_bundle(plan, root / "preprocess")
    fixed_dofs = _build_fixed_dofs(plan, bundle.mesh_info.nodes, bundle.mesh_info.dof_map)
    free_dofs = np.setdiff1d(
        np.arange(bundle.mesh_info.dof_map.size, dtype=np.int64),
        fixed_dofs,
    )
    mass = assemble_lumped_mass_vector(bundle)[free_dofs]
    stiffness = assemble_stiffness_matrix(bundle, plan.plane_state)[free_dofs, :][
        :, free_dofs
    ].tocsr()
    load_basis_full = _build_load_basis(plan, bundle.mesh_info)
    load_basis = load_basis_full[:, free_dofs]
    minimum_length = estimate_minimum_edge_length(bundle.mesh_info)
    wave_speed = float(bundle.material.maximum_wave_speed)
    cfl_ratio = plan.time_step * wave_speed / minimum_length
    if cfl_ratio > plan.coarse_cfl_ratio * (1.0 + 1.0e-12):
        raise ValueError(
            "Elastic INC time step exceeds the requested CFL ratio: "
            f"actual={cfl_ratio:.12e}, requested={plan.coarse_cfl_ratio:.12e}."
        )
    if plan.design != "p0_7" and np.linalg.matrix_rank(load_basis) != 4:
        raise ValueError("P1 spatial load basis does not have rank four.")

    eigenvalues = eigh(
        stiffness.toarray(),
        np.diag(mass),
        subset_by_index=(0, min(5, free_dofs.size - 1)),
        eigvals_only=True,
    )
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        raise ValueError("Elastic INC constrained system has no positive eigenvalue.")
    frequencies = np.sqrt(positive) / (2.0 * np.pi)
    system = ElasticINCSystem(
        free_dofs=free_dofs,
        mass=mass,
        stiffness=stiffness,
        load_basis=load_basis,
        nodes=bundle.mesh_info.nodes,
        elements=bundle.mesh_info.elements,
        dof_map=bundle.mesh_info.dof_map,
        time_step=plan.time_step,
        cfl_ratio=cfl_ratio,
    )
    _write_system(system_path, system, plan)
    report = {
        "schema_version": "2.0",
        "passed": True,
        "design": plan.design,
        "node_count": int(system.nodes.shape[0]),
        "element_count": int(system.elements.shape[0]),
        "free_dof_count": int(system.free_dofs.size),
        "minimum_element_length": minimum_length,
        "maximum_wave_speed": wave_speed,
        "coarse_cfl_ratio": cfl_ratio,
        "time_step": plan.time_step,
        "num_steps": plan.num_steps,
        "final_time": plan.time_step * plan.num_steps,
        "first_frequencies_hz": frequencies.tolist(),
        "first_period_seconds": float(1.0 / frequencies[0]),
        "spatial_load_rank": int(np.linalg.matrix_rank(load_basis)),
    }
    _write_json_atomic(report_path, report)
    return report


def load_elastic_inc_system(path: str | Path) -> ElasticINCSystem:
    """Load one completed fixed elastic INC system."""
    with h5py.File(Path(path), "r") as handle:
        if handle.attrs["schema_version"] != "2.0":
            raise ValueError("Unsupported elastic INC system schema.")
        if handle.attrs["status"] != "complete":
            raise ValueError("Elastic INC system is incomplete.")
        stiffness_group = handle["stiffness_free"]
        stiffness = csr_matrix(
            (
                stiffness_group["data"][...],
                stiffness_group["indices"][...],
                stiffness_group["indptr"][...],
            ),
            shape=tuple(int(value) for value in stiffness_group.attrs["shape"]),
        )
        return ElasticINCSystem(
            free_dofs=handle["free_dofs"][...],
            mass=handle["mass_lumped_free"][...],
            stiffness=stiffness,
            load_basis=handle["load_basis_free"][...],
            nodes=handle["mesh/nodes"][...],
            elements=handle["mesh/elements"][...],
            dof_map=handle["mesh/dof_map"][...],
            time_step=float(handle.attrs["time_step"]),
            cfl_ratio=float(handle.attrs["cfl_ratio"]),
        )


def validate_elastic_inc_system_plan(path: str | Path, plan: ElasticINCPlan) -> None:
    """Compare one stored system plan directly with the requested fields."""
    with h5py.File(Path(path), "r") as handle:
        stored = json.loads(str(handle.attrs["plan_json"]))
    requested = plan.model_dump(mode="json")
    if stored != requested:
        raise ValueError("Stored elastic INC system plan differs from the requested plan.")


def generate_elastic_inc_path(
    plan: ElasticINCPlan,
    task: ElasticINCTask,
    output_root: str | Path,
) -> dict[str, Any]:
    """Generate one nested-reference path with bounded memory."""
    root = Path(output_root)
    task_root = root / "tasks" / task.task_id
    response_path = task_root / "response.h5"
    partial_path = task_root / "response.h5.partial"
    metrics_path = task_root / "metrics.json"
    progress_path = task_root / "progress.json"
    if response_path.exists():
        raise FileExistsError(f"Elastic INC response already exists: {response_path}.")
    if partial_path.exists():
        raise FileExistsError(f"Elastic INC partial response requires diagnosis: {partial_path}.")
    task_root.mkdir(parents=True, exist_ok=True)
    system = load_elastic_inc_system(root / "system.h5")
    validate_elastic_inc_system_plan(root / "system.h5", plan)
    _validate_existing_system(system, plan)
    unit_force = np.asarray(task.spatial_coefficients) @ system.load_basis
    if np.linalg.norm(unit_force) == 0.0:
        raise ValueError(f"Elastic INC task has zero spatial load: {task.task_id}.")

    started = perf_counter()
    progress_stride = max(1, plan.num_steps * plan.fine_ratio // plan.progress_updates)
    _write_progress(progress_path, task, "running", 0, plan.num_steps * plan.fine_ratio, 0.0)
    metrics = _integrate_nested_path(
        plan,
        task,
        system,
        unit_force,
        partial_path,
        progress_path,
        progress_stride,
        started,
    )
    failed = _failed_metrics(plan, metrics)
    metrics["passed"] = not failed
    metrics["failed_metrics"] = failed
    metrics["wall_seconds"] = perf_counter() - started
    metrics["peak_memory_mb"] = _peak_memory_mb()
    _write_json_atomic(metrics_path, metrics)
    if failed:
        raise ElasticINCAcceptanceError(metrics)

    with h5py.File(partial_path, "r+") as handle:
        handle.attrs.modify("status", "complete")
    partial_path.replace(response_path)
    _write_progress(
        progress_path,
        task,
        "complete",
        plan.num_steps * plan.fine_ratio,
        plan.num_steps * plan.fine_ratio,
        metrics["wall_seconds"],
    )
    return metrics


def pulse_shape(kind: str, time: float, duration: float) -> float:
    """Evaluate one fixed compact-support pulse family."""
    normalized = time / duration
    if normalized < 0.0 or normalized > 1.0:
        return 0.0
    builders = {
        "sine_squared": lambda value: np.sin(np.pi * value) ** 2,
        "half_sine": lambda value: np.sin(np.pi * value),
        "raised_cosine": lambda value: 0.5 * (1.0 - np.cos(2.0 * np.pi * value)),
        "double_pulse": _double_pulse_shape,
    }
    return float(builders[kind](normalized))


def _integrate_nested_path(
    plan: ElasticINCPlan,
    task: ElasticINCTask,
    system: ElasticINCSystem,
    unit_force: NDArray[np.float64],
    partial_path: Path,
    progress_path: Path,
    progress_stride: int,
    started: float,
) -> dict[str, Any]:
    """Integrate fine and reference grids while writing only coarse nodes."""
    free_count = system.free_dofs.size
    fine_step = plan.time_step / plan.fine_ratio
    reference_step = plan.time_step / plan.reference_ratio
    reference_substeps = plan.reference_ratio // plan.fine_ratio
    total_fine_steps = plan.num_steps * plan.fine_ratio
    fine_u = np.zeros(free_count, dtype=np.float64)
    fine_v = np.zeros(free_count, dtype=np.float64)
    reference_u = np.zeros(free_count, dtype=np.float64)
    reference_v = np.zeros(free_count, dtype=np.float64)
    fine_a = _acceleration(system, unit_force, task, 0.0, fine_u)
    reference_a = fine_a.copy()
    previous_coarse_u = fine_u.copy()
    previous_coarse_v = fine_v.copy()
    previous_coarse_a = fine_a.copy()
    error_sums = {
        "displacement_numerator": 0.0,
        "displacement_denominator": 0.0,
        "velocity_numerator": 0.0,
        "velocity_denominator": 0.0,
        "label_displacement_numerator": 0.0,
        "label_displacement_denominator": 0.0,
        "label_velocity_numerator": 0.0,
        "label_velocity_denominator": 0.0,
        "pulse_displacement_numerator": 0.0,
        "pulse_displacement_denominator": 0.0,
        "pulse_velocity_numerator": 0.0,
        "pulse_velocity_denominator": 0.0,
        "post_displacement_numerator": 0.0,
        "post_displacement_denominator": 0.0,
        "post_velocity_numerator": 0.0,
        "post_velocity_denominator": 0.0,
    }
    segment_sample_counts = {"pulse": 0, "post": 0}
    energy_difference_max = 0.0
    reference_energy_max = 0.0
    response_max = 0.0

    with h5py.File(partial_path, "w") as handle:
        datasets = _initialize_response_file(handle, plan, task, system)
        datasets["displacement"][0] = fine_u
        datasets["velocity"][0] = fine_v
        datasets["mechanical_energy"][0] = 0.0
        buffers = _PathBuffers(plan.chunk_steps, free_count)

        reference_time = 0.0
        for step_id in range(1, total_fine_steps + 1):
            fine_time = step_id * fine_step
            fine_u, fine_v, fine_a = _advance_velocity_verlet(
                system,
                unit_force,
                task,
                fine_time - fine_step,
                fine_step,
                fine_u,
                fine_v,
                fine_a,
            )
            for _ in range(reference_substeps):
                reference_u, reference_v, reference_a = _advance_velocity_verlet(
                    system,
                    unit_force,
                    task,
                    reference_time,
                    reference_step,
                    reference_u,
                    reference_v,
                    reference_a,
                )
                reference_time += reference_step

            displacement_difference = fine_u - reference_u
            velocity_difference = fine_v - reference_v
            error_sums["displacement_numerator"] += float(
                np.sum(system.mass * displacement_difference**2)
            )
            error_sums["displacement_denominator"] += float(np.sum(system.mass * reference_u**2))
            error_sums["velocity_numerator"] += float(np.sum(system.mass * velocity_difference**2))
            error_sums["velocity_denominator"] += float(np.sum(system.mass * reference_v**2))
            segment = "pulse" if fine_time <= task.pulse_duration else "post"
            segment_sample_counts[segment] += 1
            error_sums[f"{segment}_displacement_numerator"] += float(
                np.sum(system.mass * displacement_difference**2)
            )
            error_sums[f"{segment}_displacement_denominator"] += float(
                np.sum(system.mass * reference_u**2)
            )
            error_sums[f"{segment}_velocity_numerator"] += float(
                np.sum(system.mass * velocity_difference**2)
            )
            error_sums[f"{segment}_velocity_denominator"] += float(
                np.sum(system.mass * reference_v**2)
            )
            fine_energy = _mechanical_energy(system, fine_u, fine_v)
            reference_energy = _mechanical_energy(system, reference_u, reference_v)
            energy_difference_max = max(
                energy_difference_max,
                abs(fine_energy - reference_energy),
            )
            reference_energy_max = max(reference_energy_max, abs(reference_energy))
            response_max = max(
                response_max,
                float(np.sqrt(np.sum(system.mass * reference_u**2))),
            )

            if step_id % plan.fine_ratio == 0:
                coarse_id = step_id // plan.fine_ratio
                residual_start, residual_end = compute_velocity_verlet_defect_labels(
                    np.stack((previous_coarse_u, fine_u)),
                    np.stack((previous_coarse_v, fine_v)),
                    np.stack((previous_coarse_a, fine_a)),
                    system.mass,
                    plan.time_step,
                )
                reconstructed_u, reconstructed_v = _reconstruct_coarse_step(
                    previous_coarse_u,
                    previous_coarse_v,
                    previous_coarse_a,
                    fine_a,
                    system.mass,
                    residual_start[0],
                    residual_end[0],
                    plan.time_step,
                )
                error_sums["label_displacement_numerator"] += float(
                    np.sum((reconstructed_u - fine_u) ** 2)
                )
                error_sums["label_displacement_denominator"] += float(np.sum(fine_u**2))
                error_sums["label_velocity_numerator"] += float(
                    np.sum((reconstructed_v - fine_v) ** 2)
                )
                error_sums["label_velocity_denominator"] += float(np.sum(fine_v**2))
                buffers.append(
                    coarse_id,
                    fine_u,
                    fine_v,
                    fine_energy,
                    residual_start[0],
                    residual_end[0],
                )
                if buffers.full:
                    buffers.flush(datasets)
                previous_coarse_u = fine_u.copy()
                previous_coarse_v = fine_v.copy()
                previous_coarse_a = fine_a.copy()

            if step_id % progress_stride == 0 or step_id == total_fine_steps:
                _write_progress(
                    progress_path,
                    task,
                    "nested_reference",
                    step_id,
                    total_fine_steps,
                    perf_counter() - started,
                )

        buffers.flush(datasets)

    metrics = {
        "schema_version": "2.0",
        "task_id": task.task_id,
        "displacement_reference_error": _relative_error(
            error_sums["displacement_numerator"],
            error_sums["displacement_denominator"],
            "reference displacement",
        ),
        "velocity_reference_error": _relative_error(
            error_sums["velocity_numerator"],
            error_sums["velocity_denominator"],
            "reference velocity",
        ),
        "energy_reference_error": energy_difference_max / reference_energy_max,
        "label_displacement_error": _relative_error(
            error_sums["label_displacement_numerator"],
            error_sums["label_displacement_denominator"],
            "label displacement",
        ),
        "label_velocity_error": _relative_error(
            error_sums["label_velocity_numerator"],
            error_sums["label_velocity_denominator"],
            "label velocity",
        ),
        "maximum_mass_displacement_norm": response_max,
        "displacement_error_l2_mass_time": float(
            np.sqrt(fine_step * error_sums["displacement_numerator"])
        ),
        "reference_displacement_l2_mass_time": float(
            np.sqrt(fine_step * error_sums["displacement_denominator"])
        ),
        "velocity_error_l2_mass_time": float(np.sqrt(fine_step * error_sums["velocity_numerator"])),
        "reference_velocity_l2_mass_time": float(
            np.sqrt(fine_step * error_sums["velocity_denominator"])
        ),
    }
    for segment in ("pulse", "post"):
        if segment_sample_counts[segment] == 0:
            metrics[f"displacement_reference_error_{segment}"] = None
            metrics[f"velocity_reference_error_{segment}"] = None
            continue
        metrics[f"displacement_reference_error_{segment}"] = _relative_error(
            error_sums[f"{segment}_displacement_numerator"],
            error_sums[f"{segment}_displacement_denominator"],
            f"{segment} reference displacement",
        )
        metrics[f"velocity_reference_error_{segment}"] = _relative_error(
            error_sums[f"{segment}_velocity_numerator"],
            error_sums[f"{segment}_velocity_denominator"],
            f"{segment} reference velocity",
        )
    return metrics


class _PathBuffers:
    """Buffer coarse-node HDF5 writes without retaining a complete path."""

    def __init__(self, capacity: int, free_count: int) -> None:
        """Allocate one fixed-size path buffer."""
        self.capacity = capacity
        self.count = 0
        self.coarse_ids = np.zeros(capacity, dtype=np.int64)
        self.displacement = np.zeros((capacity, free_count), dtype=np.float64)
        self.velocity = np.zeros((capacity, free_count), dtype=np.float64)
        self.energy = np.zeros(capacity, dtype=np.float64)
        self.residual_start = np.zeros((capacity, free_count), dtype=np.float64)
        self.residual_end = np.zeros((capacity, free_count), dtype=np.float64)

    @property
    def full(self) -> bool:
        """Return whether the buffer reached its fixed capacity."""
        return self.count == self.capacity

    def append(
        self,
        coarse_id: int,
        displacement: NDArray[np.float64],
        velocity: NDArray[np.float64],
        energy: float,
        residual_start: NDArray[np.float64],
        residual_end: NDArray[np.float64],
    ) -> None:
        """Append one coarse node and its preceding interval labels."""
        index = self.count
        self.coarse_ids[index] = coarse_id
        self.displacement[index] = displacement
        self.velocity[index] = velocity
        self.energy[index] = energy
        self.residual_start[index] = residual_start
        self.residual_end[index] = residual_end
        self.count += 1

    def flush(self, datasets: dict[str, h5py.Dataset]) -> None:
        """Write all buffered consecutive coarse nodes."""
        if self.count == 0:
            return
        node_start = int(self.coarse_ids[0])
        node_end = node_start + self.count
        if not np.array_equal(self.coarse_ids[: self.count], np.arange(node_start, node_end)):
            raise ValueError("Elastic INC write buffer contains nonconsecutive coarse nodes.")
        datasets["displacement"][node_start:node_end] = self.displacement[: self.count]
        datasets["velocity"][node_start:node_end] = self.velocity[: self.count]
        datasets["mechanical_energy"][node_start:node_end] = self.energy[: self.count]
        datasets["residual_start"][node_start - 1 : node_end - 1] = self.residual_start[
            : self.count
        ]
        datasets["residual_end"][node_start - 1 : node_end - 1] = self.residual_end[: self.count]
        self.count = 0


def _build_bundle(plan: ElasticINCPlan, output_root: Path):
    """Build the reviewed rectangular linear-elastic preprocessing bundle."""
    model = plan.model
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
        root=output_root,
        mesh_path=output_root / "mesh.msh",
        vtk_path=output_root / "mesh.vtu",
        save_vtk=False,
        save_history=False,
        fields=(),
    )
    return build_preprocess_data(model_def, output)


def _build_fixed_dofs(
    plan: ElasticINCPlan,
    nodes: NDArray[np.float64],
    dof_map: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Build the P0 one-dimensional or P1 clamped-left constraints."""
    left_nodes = np.flatnonzero(np.isclose(nodes[:, 0], np.min(nodes[:, 0])))
    if plan.design == "p0_7":
        dofs = np.concatenate((dof_map[:, 1], dof_map[left_nodes, 0]))
    else:
        dofs = dof_map[left_nodes, :].reshape(-1)
    return np.unique(dofs).astype(np.int64)


def _build_load_basis(plan: ElasticINCPlan, mesh_info) -> NDArray[np.float64]:
    """Assemble the four reviewed right-boundary unit traction patterns."""
    coordinates = mesh_info.nodes
    height_min = float(np.min(coordinates[:, 1]))
    height_max = float(np.max(coordinates[:, 1]))
    height = height_max - height_min
    height_mid = 0.5 * (height_min + height_max)

    functions = (
        lambda point: np.asarray((-1.0, 0.0), dtype=np.float64),
        lambda point: np.asarray((0.0, 1.0), dtype=np.float64),
        lambda point: np.asarray(
            (-2.0 * (point[1] - height_mid) / height, 0.0),
            dtype=np.float64,
        ),
        lambda point: np.asarray(
            (-float(point[1] > height_mid), 0.0),
            dtype=np.float64,
        ),
    )
    return np.stack(
        [
            assemble_boundary_traction(
                mesh_info,
                plan.load_boundary,
                function,
                plan.thickness,
            )
            for function in functions
        ]
    )


def _write_system(path: Path, system: ElasticINCSystem, plan: ElasticINCPlan) -> None:
    """Write one fixed system without file-digest metadata."""
    partial_path = path.with_suffix(".h5.partial")
    if partial_path.exists():
        raise FileExistsError(partial_path)
    with h5py.File(partial_path, "w") as handle:
        handle.attrs["schema_version"] = "2.0"
        handle.attrs["status"] = "writing"
        handle.attrs["design"] = plan.design
        handle.attrs["time_step"] = system.time_step
        handle.attrs["cfl_ratio"] = system.cfl_ratio
        handle.attrs["plan_json"] = json.dumps(
            plan.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.create_dataset("free_dofs", data=system.free_dofs)
        handle.create_dataset("mass_lumped_free", data=system.mass)
        handle.create_dataset("load_basis_free", data=system.load_basis)
        strings = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "control_parameter_order",
            data=np.asarray(control_parameter_order(plan), dtype=strings),
        )
        mesh = handle.create_group("mesh")
        mesh.create_dataset("nodes", data=system.nodes)
        mesh.create_dataset("elements", data=system.elements)
        mesh.create_dataset("dof_map", data=system.dof_map)
        stiffness = handle.create_group("stiffness_free")
        stiffness.create_dataset("data", data=system.stiffness.data)
        stiffness.create_dataset("indices", data=system.stiffness.indices)
        stiffness.create_dataset("indptr", data=system.stiffness.indptr)
        stiffness.attrs["shape"] = system.stiffness.shape
        handle.attrs.modify("status", "complete")
    partial_path.replace(path)


def _initialize_response_file(
    handle: h5py.File,
    plan: ElasticINCPlan,
    task: ElasticINCTask,
    system: ElasticINCSystem,
) -> dict[str, h5py.Dataset]:
    """Create one preallocated sharded response file."""
    handle.attrs["schema_version"] = "2.0"
    handle.attrs["status"] = "writing"
    handle.attrs["task_id"] = task.task_id
    handle.attrs["split"] = task.split
    handle.attrs["pulse_kind"] = task.pulse_kind
    handle.attrs["pressure_amplitude"] = task.pressure_amplitude
    handle.attrs["pulse_duration"] = task.pulse_duration
    handle.create_dataset(
        "time",
        data=np.arange(plan.num_steps + 1, dtype=np.float64) * plan.time_step,
    )
    handle.create_dataset("control_parameters", data=build_control_parameters(plan, task))
    handle.create_dataset("spatial_coefficients", data=task.spatial_coefficients)
    chunk_nodes = min(plan.chunk_steps, plan.num_steps + 1)
    state = handle.create_group("state")
    target = handle.create_group("target")
    reference = handle.create_group("reference")
    options = {
        "compression": plan.compression,
        "compression_opts": plan.compression_level,
        "shuffle": True,
    }
    return {
        "displacement": state.create_dataset(
            "displacement",
            shape=(plan.num_steps + 1, system.free_dofs.size),
            dtype=np.float64,
            chunks=(chunk_nodes, system.free_dofs.size),
            **options,
        ),
        "velocity": state.create_dataset(
            "velocity",
            shape=(plan.num_steps + 1, system.free_dofs.size),
            dtype=np.float64,
            chunks=(chunk_nodes, system.free_dofs.size),
            **options,
        ),
        "residual_start": target.create_dataset(
            "residual_force_start",
            shape=(plan.num_steps, system.free_dofs.size),
            dtype=np.float64,
            chunks=(min(plan.chunk_steps, plan.num_steps), system.free_dofs.size),
            **options,
        ),
        "residual_end": target.create_dataset(
            "residual_force_end",
            shape=(plan.num_steps, system.free_dofs.size),
            dtype=np.float64,
            chunks=(min(plan.chunk_steps, plan.num_steps), system.free_dofs.size),
            **options,
        ),
        "mechanical_energy": reference.create_dataset(
            "mechanical_energy",
            shape=(plan.num_steps + 1,),
            dtype=np.float64,
            chunks=(chunk_nodes,),
            **options,
        ),
    }


def _advance_velocity_verlet(
    system: ElasticINCSystem,
    unit_force: NDArray[np.float64],
    task: ElasticINCTask,
    time: float,
    time_step: float,
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    acceleration: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Advance one exact matrix-form velocity-Verlet step."""
    next_displacement = displacement + time_step * velocity + 0.5 * time_step**2 * acceleration
    next_acceleration = _acceleration(
        system,
        unit_force,
        task,
        time + time_step,
        next_displacement,
    )
    next_velocity = velocity + 0.5 * time_step * (acceleration + next_acceleration)
    return next_displacement, next_velocity, next_acceleration


def _acceleration(
    system: ElasticINCSystem,
    unit_force: NDArray[np.float64],
    task: ElasticINCTask,
    time: float,
    displacement: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate one free-DOF semidiscrete acceleration."""
    external = (
        task.pressure_amplitude
        * pulse_shape(
            task.pulse_kind,
            time,
            task.pulse_duration,
        )
        * unit_force
    )
    return np.asarray((external - system.stiffness @ displacement) / system.mass)


def _mechanical_energy(
    system: ElasticINCSystem,
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> float:
    """Compute lumped kinetic plus elastic strain energy."""
    kinetic = 0.5 * np.sum(system.mass * velocity**2)
    elastic = 0.5 * displacement @ (system.stiffness @ displacement)
    return float(kinetic + elastic)


def _reconstruct_coarse_step(
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    acceleration_start: NDArray[np.float64],
    acceleration_end: NDArray[np.float64],
    mass: NDArray[np.float64],
    residual_start: NDArray[np.float64],
    residual_end: NDArray[np.float64],
    time_step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct one coarse step from the two residual labels."""
    source_start = -residual_start / mass
    source_end = -residual_end / mass
    next_displacement = (
        displacement
        + time_step * velocity
        + 0.5 * time_step**2 * (acceleration_start + source_start)
    )
    next_velocity = velocity + 0.5 * time_step * (
        acceleration_start + source_start + acceleration_end + source_end
    )
    return next_displacement, next_velocity


def _failed_metrics(plan: ElasticINCPlan, metrics: dict[str, Any]) -> dict[str, Any]:
    """Return every failed configured path metric."""
    limits = {
        "displacement_reference_error": plan.thresholds.displacement,
        "velocity_reference_error": plan.thresholds.velocity,
        "energy_reference_error": plan.thresholds.energy,
        "label_displacement_error": plan.thresholds.label_reconstruction,
        "label_velocity_error": plan.thresholds.label_reconstruction,
    }
    failed = {
        name: {"value": metrics[name], "limit": limit}
        for name, limit in limits.items()
        if metrics[name] > limit
    }
    if metrics["maximum_mass_displacement_norm"] < plan.thresholds.minimum_response:
        failed["maximum_mass_displacement_norm"] = {
            "value": metrics["maximum_mass_displacement_norm"],
            "minimum": plan.thresholds.minimum_response,
        }
    return failed


def _relative_error(numerator: float, denominator: float, name: str) -> float:
    """Compute one relative trajectory error with a nonzero scale."""
    if denominator <= 0.0:
        raise ValueError(f"Elastic INC {name} scale is zero.")
    return float(np.sqrt(numerator / denominator))


def _double_pulse_shape(normalized: float) -> float:
    """Evaluate the reviewed separated double pulse."""
    if 0.0 <= normalized <= 0.4:
        return float(np.sin(np.pi * normalized / 0.4))
    if 0.6 <= normalized <= 1.0:
        return float(np.sin(np.pi * (normalized - 0.6) / 0.4))
    return 0.0


def _validate_existing_system(system: ElasticINCSystem, plan: ElasticINCPlan) -> None:
    """Require direct fixed-system dimensions and time settings."""
    expected_free = 80 if plan.design == "p0_7" else 400
    if system.free_dofs.size != expected_free:
        raise ValueError(
            "Elastic INC free-DOF count differs from the plan: "
            f"expected={expected_free}, received={system.free_dofs.size}."
        )
    if system.time_step != plan.time_step:
        raise ValueError("Elastic INC system time step differs from the plan.")
    if system.cfl_ratio > plan.coarse_cfl_ratio * (1.0 + 1.0e-12):
        raise ValueError("Elastic INC system violates the configured CFL ratio.")


def _write_progress(
    path: Path,
    task: ElasticINCTask,
    stage: str,
    completed: int,
    total: int,
    elapsed_seconds: float,
) -> None:
    """Write one compact task progress record."""
    _write_json_atomic(
        path,
        {
            "task_id": task.task_id,
            "stage": stage,
            "completed_increments": completed,
            "total_increments": total,
            "progress_percent": 100.0 * completed / total,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def _peak_memory_mb() -> float:
    """Return the process peak resident memory on Linux."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one structured JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON object through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
