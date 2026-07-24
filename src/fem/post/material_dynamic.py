"""Accepted-state field reconstruction and spectra for material dynamics.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fem.assembly.nonlinear_solid import build_material_assembly_cache
from fem.post.animation import write_pvd_collection
from fem.post.vtk import write_named_point_cloud_vtk, write_named_solution_vtk
from fem.preprocess.data import PreprocessBundle
from fem.solvers.data import MaterialDynamicSolution

ENGINEERING_TENSOR_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


@dataclass(frozen=True)
class StrainRateSpectrum:
    """Store one-sided strain-rate spectral energy and percentile frequencies."""

    angular_frequencies: NDArray[np.float64]
    point_energy: NDArray[np.float64]
    cumulative_point_energy: NDArray[np.float64]
    omega_99: NDArray[np.float64]


def extract_quadrature_strain_history(
    bundle: PreprocessBundle,
    displacement_history: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reconstruct accepted integration-point engineering strains."""
    cache = build_material_assembly_cache(bundle)
    element_count, quadrature_count = cache.b_matrices.shape[:2]
    result = np.zeros(
        (displacement_history.shape[0], element_count * quadrature_count, 6),
        dtype=np.float64,
    )
    for element_id, connectivity in enumerate(bundle.mesh_info.elements):
        element_dofs = bundle.mesh_info.dof_map[connectivity].reshape(-1)
        element_displacements = displacement_history[:, element_dofs]
        for quadrature_id in range(quadrature_count):
            point_id = element_id * quadrature_count + quadrature_id
            result[:, point_id] = (
                element_displacements @ cache.b_matrices[element_id, quadrature_id].T
            )
    return result


def compute_strain_rate_spectrum(
    times: NDArray[np.float64],
    strain_history: NDArray[np.float64],
    energy_fraction: float = 0.99,
) -> StrainRateSpectrum:
    """Compute the exact discrete spectrum of accepted fixed-step strain rates."""
    time_steps = np.diff(times)
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(float(times[-1]))) * 32.0
    if not np.allclose(time_steps, time_steps[0], rtol=0.0, atol=tolerance):
        raise ValueError("Strain-rate spectrum requires a strictly uniform accepted time grid.")
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("Spectral cumulative-energy fraction must lie in (0, 1].")

    strain_rate = np.diff(strain_history, axis=0) / time_steps[0]
    coefficients = np.fft.rfft(strain_rate, axis=0)
    angular_frequencies = np.asarray(
        2.0 * np.pi * np.fft.rfftfreq(strain_rate.shape[0], d=float(time_steps[0])),
        dtype=np.float64,
    )
    point_energy = np.sum(
        np.abs(coefficients) ** 2 * ENGINEERING_TENSOR_WEIGHTS[None, None, :],
        axis=2,
    ).T
    total = np.sum(point_energy, axis=1)
    if np.any(total <= 0.0):
        point_ids = np.flatnonzero(total <= 0.0).tolist()
        raise ValueError(f"Strain-rate spectrum has zero-energy integration points: {point_ids}.")
    cumulative = np.cumsum(point_energy, axis=1) / total[:, None]
    frequency_ids = np.argmax(cumulative >= energy_fraction, axis=1)
    omega_99 = angular_frequencies[frequency_ids]
    return StrainRateSpectrum(
        angular_frequencies=angular_frequencies,
        point_energy=point_energy,
        cumulative_point_energy=cumulative,
        omega_99=omega_99,
    )


def write_material_dynamic_vtk_series(
    bundle: PreprocessBundle,
    solution: MaterialDynamicSolution,
    output_directory: str | Path,
    collection_name: str,
    step_stride: int,
    write_quadrature_points: bool = False,
) -> dict[str, Path]:
    """Write accepted nonlinear dynamic states as a ParaView time series."""
    if step_stride <= 0:
        raise ValueError("VTK step stride must be strictly positive.")
    output_root = Path(output_directory)
    _clear_managed_vtk_frames(output_root)
    strain_history = extract_quadrature_strain_history(bundle, solution.displacement)
    step_ids = list(range(0, solution.times.size, step_stride))
    if step_ids[-1] != solution.times.size - 1:
        step_ids.append(solution.times.size - 1)
    vtk_paths = []
    quadrature_paths = []
    times = []
    for step_id in step_ids:
        point_fields = _build_nodal_fields(bundle, solution, step_id)
        cell_fields = _build_cell_fields(bundle, solution, strain_history, step_id)
        vtk_path = write_named_solution_vtk(
            bundle.mesh_info,
            output_root / f"step_{step_id:05d}.vtu",
            point_fields,
            cell_fields,
        )
        vtk_paths.append(vtk_path)
        if write_quadrature_points:
            quadrature_path = write_named_point_cloud_vtk(
                output_root / "quadrature" / f"step_{step_id:05d}.vtu",
                _quadrature_coordinates(bundle),
                _build_quadrature_fields(solution, strain_history, step_id),
            )
            quadrature_paths.append(quadrature_path)
        times.append(float(solution.times[step_id]))
    pvd_path = write_pvd_collection(
        output_root / f"{collection_name}.pvd",
        tuple(vtk_paths),
        tuple(times),
    )
    paths = {"collection": pvd_path, "first_frame": vtk_paths[0], "last_frame": vtk_paths[-1]}
    if write_quadrature_points:
        paths["quadrature_collection"] = write_pvd_collection(
            output_root / "quadrature" / f"{collection_name}_quadrature.pvd",
            tuple(quadrature_paths),
            tuple(times),
        )
    return paths


def _clear_managed_vtk_frames(output_root: Path) -> None:
    """Remove only frame files managed by this deterministic series writer."""
    for directory in (output_root, output_root / "quadrature"):
        if directory.is_dir():
            for path in directory.glob("step_*.vtu"):
                path.unlink()


def _build_nodal_fields(
    bundle: PreprocessBundle,
    solution: MaterialDynamicSolution,
    step_id: int,
) -> dict[str, NDArray[np.float64]]:
    """Build vector-valued accepted nodal fields."""
    node_count = bundle.mesh_info.nodes.shape[0]
    return {
        "displacement": solution.displacement[step_id].reshape(node_count, 3),
        "velocity": solution.velocity[step_id].reshape(node_count, 3),
        "acceleration": solution.acceleration[step_id].reshape(node_count, 3),
    }


def _build_cell_fields(
    bundle: PreprocessBundle,
    solution: MaterialDynamicSolution,
    strain_history: NDArray[np.float64],
    step_id: int,
) -> dict[str, NDArray[np.float64]]:
    """Volume-average accepted quadrature fields onto macro elements."""
    cache = build_material_assembly_cache(bundle)
    element_count, quadrature_count = cache.jacobian_weights.shape
    strains = strain_history[step_id].reshape(element_count, quadrature_count, 6)
    stresses = solution.stresses[step_id].reshape(element_count, quadrature_count, 6)
    fields: dict[str, NDArray[np.float64]] = {
        "strain": _weighted_average(strains, cache.jacobian_weights),
        "stress": _weighted_average(stresses, cache.jacobian_weights),
        "von_mises_stress": _weighted_average(
            _von_mises(stresses), cache.jacobian_weights
        ),
    }
    diagnostics = solution.material_diagnostics
    for name in (
        "equivalent_plastic_strain",
        "phase_matrix__equivalent_plastic_strain",
        "phase_matrix__maximum_equivalent_plastic_strain",
        "viscoplastic_active",
        "incremental_work_density",
        "free_energy_density",
        "dissipation_density",
        "phase_matrix__volume_fraction",
        "phase_inclusion__volume_fraction",
    ):
        if name not in diagnostics:
            continue
        values = diagnostics[name][step_id].reshape(element_count, quadrature_count)
        fields[name] = _weighted_average(values, cache.jacobian_weights)
        if "equivalent_plastic_strain" in name:
            fields[f"{name}_max"] = np.max(values, axis=1)
    if "incremental_work_density" in diagnostics:
        work = diagnostics["incremental_work_density"][step_id].reshape(
            element_count, quadrature_count
        )
        fields["positive_work_density"] = _weighted_average(
            np.maximum(work, 0.0), cache.jacobian_weights
        )
        fields["negative_work_density"] = _weighted_average(
            np.minimum(work, 0.0), cache.jacobian_weights
        )
        fields["unloading_fraction"] = _weighted_average(
            (work < 0.0).astype(np.float64), cache.jacobian_weights
        )
    for phase_name in ("matrix", "inclusion"):
        component_names = [
            f"phase_{phase_name}__stress_{component}"
            for component in ("xx", "yy", "zz", "yz", "xz", "xy")
        ]
        if all(name in diagnostics for name in component_names):
            phase_stress = np.stack(
                [diagnostics[name][step_id] for name in component_names], axis=1
            ).reshape(element_count, quadrature_count, 6)
            fields[f"phase_{phase_name}__stress"] = _weighted_average(
                phase_stress, cache.jacobian_weights
            )
    return fields


def _quadrature_coordinates(bundle: PreprocessBundle) -> NDArray[np.float64]:
    """Evaluate fixed physical quadrature coordinates in assembly ordering."""
    coordinates = []
    for connectivity in bundle.mesh_info.elements:
        element_coordinates = bundle.mesh_info.nodes[connectivity]
        coordinates.extend(bundle.shape_function_cache.shape_values @ element_coordinates)
    return np.asarray(coordinates, dtype=np.float64)


def _build_quadrature_fields(
    solution: MaterialDynamicSolution,
    strain_history: NDArray[np.float64],
    step_id: int,
) -> dict[str, NDArray[np.float64]]:
    """Build full accepted integration-point fields for detailed inspection."""
    fields = {
        "strain": strain_history[step_id],
        "stress": solution.stresses[step_id],
        "von_mises_stress": _von_mises(solution.stresses[step_id]),
    }
    for name, history in solution.material_diagnostics.items():
        if history.ndim == 2 and history.shape[1] == solution.stresses.shape[1]:
            fields[name] = history[step_id]
    return fields


def _weighted_average(values: NDArray[np.float64], weights: NDArray[np.float64]):
    """Average scalar or vector quadrature data with physical cell weights."""
    denominator = np.sum(weights, axis=1)
    if values.ndim == 2:
        return np.sum(values * weights, axis=1) / denominator
    return np.sum(values * weights[:, :, None], axis=1) / denominator[:, None]


def _von_mises(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute three-dimensional von Mises stress from tensorial shear components."""
    mean = np.mean(stress[..., :3], axis=-1)
    deviator = stress[..., :3] - mean[..., None]
    norm_squared = np.sum(deviator**2, axis=-1) + 2.0 * np.sum(stress[..., 3:] ** 2, axis=-1)
    return np.sqrt(1.5 * norm_squared)
