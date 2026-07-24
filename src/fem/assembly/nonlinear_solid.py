"""Nonlinear three-dimensional solid internal-force assembly.

Contents:
    Material-state initialization, Hex8 caches, strain fields, forces, and tangents.
Author:
    Zhen Hao.
Created:
    2026-07-09.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fem.assembly.elasticity import build_strain_displacement_matrix
from fem.assembly.solid import SOLID_DOFS_PER_NODE, build_solid_dof_map
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.materials.data import (
    MaterialModel,
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
)
from fem.materials.errors import MaterialPointConvergenceError
from fem.preprocess.data import PreprocessBundle
from fem.quadrature.rules import make_quadrature_rule

VOIGT_DIM = 6


@dataclass(frozen=True)
class NonlinearSolidAssemblyResult:
    """Store one nonlinear solid assembly result."""

    internal_force: NDArray[np.float64]
    tangent: csr_matrix
    state: MaterialState
    strains: NDArray[np.float64]
    strain_rates: NDArray[np.float64]
    stresses: NDArray[np.float64]
    diagnostics: dict[str, NDArray[np.float64]]
    free_energy: NDArray[np.float64]
    dissipation: NDArray[np.float64]


@dataclass(frozen=True)
class MaterialAssemblyCache:
    """Store displacement-strain operators and quadrature weights."""

    b_matrices: NDArray[np.float64]
    jacobian_weights: NDArray[np.float64]


def initialize_hex8_material_state(
    material: MaterialModel,
    element_count: int,
    quadrature_order: int = 2,
) -> MaterialState:
    """Initialize integration-point state for Hex8 solid elements."""
    _, weights = make_quadrature_rule("hex8", quadrature_order)

    return material.initialize_state(element_count * weights.shape[0])


def build_hex8_material_assembly_cache(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    quadrature_order: int = 2,
) -> MaterialAssemblyCache:
    """Build reusable Hex8 strain operators and quadrature weights."""
    geometry = _build_hex8_quadrature_geometry(nodes, elements, quadrature_order)

    return MaterialAssemblyCache(
        b_matrices=geometry["b_matrices"],
        jacobian_weights=geometry["jacobian_weights"],
    )


def assemble_hex8_material_response(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    material: MaterialModel,
    committed_state: MaterialState,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
    need_tangent: bool = True,
    quadrature_order: int = 2,
) -> NonlinearSolidAssemblyResult:
    """Assemble internal force and tangent from material-point responses."""
    cache = build_hex8_material_assembly_cache(nodes, elements, quadrature_order)

    return assemble_hex8_material_response_cached(
        nodes,
        elements,
        material,
        committed_state,
        displacement,
        previous_displacement,
        time_step,
        update_settings,
        cache,
        need_tangent,
    )


def assemble_hex8_material_response_cached(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    material: MaterialModel,
    committed_state: MaterialState,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    time_step: float,
    update_settings: MaterialUpdateSettings,
    cache: MaterialAssemblyCache,
    need_tangent: bool = True,
) -> NonlinearSolidAssemblyResult:
    """Assemble one Hex8 response using immutable precomputed geometry data."""
    strains, strain_rates = _compute_hex8_strain_fields(
        nodes.shape[0],
        elements,
        displacement,
        previous_displacement,
        time_step,
        cache.b_matrices,
    )
    request = MaterialPointRequest(
        strains=strains,
        strain_rates=strain_rates,
        time_step=time_step,
        kinematics="three_dimensional",
        update_settings=update_settings,
    )
    try:
        response = material.update(
            request,
            committed_state,
            MaterialResponseRequirements(
                tangent=need_tangent,
                free_energy=True,
                dissipation=True,
            ),
        )
    except MaterialPointConvergenceError as error:
        quadrature_count = cache.jacobian_weights.shape[1]
        raise _with_assembly_point_context(error, quadrature_count) from error
    _require_assembly_outputs(
        response.tangents,
        response.free_energy,
        response.dissipation,
        need_tangent,
    )

    internal_force, tangent = _assemble_force_and_tangent(
        nodes.shape[0],
        elements,
        response.stresses,
        response.tangents,
        cache.b_matrices,
        cache.jacobian_weights,
    )

    return NonlinearSolidAssemblyResult(
        internal_force=internal_force,
        tangent=tangent,
        state=response.state,
        strains=strains,
        strain_rates=strain_rates,
        stresses=response.stresses,
        diagnostics=response.diagnostics,
        free_energy=response.free_energy,
        dissipation=response.dissipation,
    )


def initialize_bundle_material_state(bundle: PreprocessBundle) -> MaterialState:
    """Initialize material state for every quadrature point in a preprocess bundle."""
    point_count = (
        bundle.mesh_info.elements.shape[0]
        * bundle.shape_function_cache.quadrature_weights.shape[0]
    )

    return bundle.material.initialize_state(point_count)


def build_material_assembly_cache(bundle: PreprocessBundle) -> MaterialAssemblyCache:
    """Precompute strain-displacement matrices for repeated material assembly."""
    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache
    element_count = mesh_info.elements.shape[0]
    quadrature_count = cache.quadrature_weights.shape[0]
    strain_dim = 6 if mesh_info.dimension == 3 else 3
    element_dof_count = mesh_info.elements.shape[1] * mesh_info.dimension
    b_matrices = np.zeros(
        (element_count, quadrature_count, strain_dim, element_dof_count),
        dtype=np.float64,
    )
    jacobian_weights = np.zeros((element_count, quadrature_count), dtype=np.float64)
    for element_id in range(element_count):
        for quadrature_id, weight in enumerate(cache.quadrature_weights):
            b_matrices[element_id, quadrature_id] = build_strain_displacement_matrix(
                cache.shape_gradients_physical[element_id, quadrature_id],
                mesh_info.dimension,
            )
            jacobian_weights[element_id, quadrature_id] = (
                cache.jacobian_determinants[element_id, quadrature_id] * weight
            )

    return MaterialAssemblyCache(
        b_matrices=b_matrices,
        jacobian_weights=jacobian_weights,
    )


def assemble_bundle_material_response(
    bundle: PreprocessBundle,
    committed_state: MaterialState,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    time_step: float,
    kinematics: str,
    update_settings: MaterialUpdateSettings,
    assembly_cache: MaterialAssemblyCache,
    need_tangent: bool = True,
) -> NonlinearSolidAssemblyResult:
    """Assemble a cached-geometry material response for one displacement trial."""
    mesh_info = bundle.mesh_info
    element_count = mesh_info.elements.shape[0]
    quadrature_count = assembly_cache.jacobian_weights.shape[1]
    strain_dim = 6 if mesh_info.dimension == 3 else 3
    point_count = element_count * quadrature_count
    strains = np.zeros((point_count, strain_dim), dtype=np.float64)
    strain_rates = np.zeros_like(strains)

    for element_id, connectivity in enumerate(mesh_info.elements):
        element_dofs = mesh_info.dof_map[connectivity, :].reshape(-1)
        element_displacement = displacement[element_dofs]
        previous_element_displacement = previous_displacement[element_dofs]
        for quadrature_id in range(quadrature_count):
            b_matrix = assembly_cache.b_matrices[element_id, quadrature_id]
            point_id = element_id * quadrature_count + quadrature_id
            strains[point_id] = b_matrix @ element_displacement
            strain_rates[point_id] = (
                b_matrix @ (element_displacement - previous_element_displacement)
            ) / time_step

    request = MaterialPointRequest(
        strains=strains,
        strain_rates=strain_rates,
        time_step=time_step,
        kinematics=kinematics,
        update_settings=update_settings,
    )
    try:
        response = bundle.material.update(
            request,
            committed_state,
            MaterialResponseRequirements(
                tangent=need_tangent,
                free_energy=True,
                dissipation=True,
            ),
        )
    except MaterialPointConvergenceError as error:
        raise _with_assembly_point_context(error, quadrature_count) from error
    _require_assembly_outputs(
        response.tangents,
        response.free_energy,
        response.dissipation,
        need_tangent,
    )

    internal_force, tangent = _assemble_cached_force_and_tangent(
        mesh_info.dof_map,
        mesh_info.elements,
        response.stresses,
        response.tangents,
        assembly_cache.b_matrices,
        assembly_cache.jacobian_weights,
        need_tangent,
    )

    return NonlinearSolidAssemblyResult(
        internal_force=internal_force,
        tangent=tangent,
        state=response.state,
        strains=strains,
        strain_rates=strain_rates,
        stresses=response.stresses,
        diagnostics=response.diagnostics,
        free_energy=response.free_energy,
        dissipation=response.dissipation,
    )


def _require_assembly_outputs(
    tangents: NDArray[np.float64] | None,
    free_energy: NDArray[np.float64] | None,
    dissipation: NDArray[np.float64] | None,
    need_tangent: bool,
) -> None:
    """Require every constitutive output consumed by solid assembly."""
    if need_tangent and tangents is None:
        raise ValueError("Material assembly requested a tangent but received None.")
    if free_energy is None:
        raise ValueError("Material assembly requested free energy but received None.")
    if dissipation is None:
        raise ValueError("Material assembly requested dissipation but received None.")


def integrate_quadrature_density(
    bundle: PreprocessBundle,
    density: NDArray[np.float64],
) -> float:
    """Integrate one scalar quadrature-point density over the domain."""
    cache = bundle.shape_function_cache
    quadrature_count = cache.quadrature_weights.shape[0]
    total = 0.0
    for element_id in range(bundle.mesh_info.elements.shape[0]):
        for quadrature_id, weight in enumerate(cache.quadrature_weights):
            point_id = element_id * quadrature_count + quadrature_id
            total += (
                density[point_id]
                * cache.jacobian_determinants[element_id, quadrature_id]
                * weight
            )

    return float(total)


def _with_assembly_point_context(
    error: MaterialPointConvergenceError,
    quadrature_count: int,
) -> MaterialPointConvergenceError:
    """Add element and quadrature identifiers to one material-point failure."""
    point_id = int(error.diagnostics["point_id"])

    return error.with_context(
        element_id=point_id // quadrature_count,
        quadrature_id=point_id % quadrature_count,
    )


def _assemble_cached_force_and_tangent(
    dof_map: NDArray[np.int64],
    elements: NDArray[np.int64],
    stresses: NDArray[np.float64],
    tangents: NDArray[np.float64],
    b_matrices: NDArray[np.float64],
    jacobian_weights: NDArray[np.float64],
    need_tangent: bool,
) -> tuple[NDArray[np.float64], csr_matrix]:
    """Assemble force and tangent from cached kinematics."""
    dof_count = dof_map.size
    internal_force = np.zeros(dof_count, dtype=np.float64)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    quadrature_count = jacobian_weights.shape[1]

    for element_id, connectivity in enumerate(elements):
        element_dofs = dof_map[connectivity, :].reshape(-1)
        element_force = np.zeros(element_dofs.shape[0], dtype=np.float64)
        element_tangent = np.zeros((element_dofs.shape[0], element_dofs.shape[0]))
        for quadrature_id in range(quadrature_count):
            point_id = element_id * quadrature_count + quadrature_id
            b_matrix = b_matrices[element_id, quadrature_id]
            weight = jacobian_weights[element_id, quadrature_id]
            element_force += b_matrix.T @ stresses[point_id] * weight
            if need_tangent:
                element_tangent += b_matrix.T @ tangents[point_id] @ b_matrix * weight
        internal_force[element_dofs] += element_force
        if need_tangent:
            row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")
            rows.extend(row_ids.reshape(-1).tolist())
            columns.extend(column_ids.reshape(-1).tolist())
            values.extend(element_tangent.reshape(-1).tolist())

    tangent = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count)).tocsr()

    return internal_force, tangent


def _build_hex8_quadrature_geometry(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    quadrature_order: int,
) -> dict[str, NDArray[np.float64]]:
    """Build B matrices and Jacobian weights for Hex8 quadrature points."""
    points, weights = make_quadrature_rule("hex8", quadrature_order)
    _, gradients_reference = evaluate_lagrange_shape_functions("hex8", points)
    quadrature_count = weights.shape[0]
    b_matrices = np.zeros(
        (elements.shape[0], quadrature_count, VOIGT_DIM, 8 * SOLID_DOFS_PER_NODE),
        dtype=np.float64,
    )
    jacobian_weights = np.zeros((elements.shape[0], quadrature_count), dtype=np.float64)

    for element_id, connectivity in enumerate(elements):
        element_nodes = nodes[connectivity, :]
        for quadrature_id, weight in enumerate(weights):
            jacobian = element_nodes.T @ gradients_reference[quadrature_id, :, :]
            determinant = float(np.linalg.det(jacobian))
            if determinant <= 0.0:
                raise ValueError("Hex8 nonlinear assembly requires positive Jacobian.")
            gradients_physical = gradients_reference[quadrature_id, :, :] @ np.linalg.inv(
                jacobian,
            )
            b_matrices[element_id, quadrature_id, :, :] = build_strain_displacement_matrix(
                gradients_physical,
                3,
            )
            jacobian_weights[element_id, quadrature_id] = determinant * weight

    return {
        "b_matrices": b_matrices,
        "jacobian_weights": jacobian_weights,
    }


def _compute_hex8_strain_fields(
    node_count: int,
    elements: NDArray[np.int64],
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    time_step: float,
    b_matrices: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute total strains and strain rates at Hex8 quadrature points."""
    dof_map = build_solid_dof_map(node_count)
    quadrature_count = b_matrices.shape[1]
    strains = np.zeros((elements.shape[0] * quadrature_count, VOIGT_DIM), dtype=np.float64)
    strain_rates = np.zeros_like(strains)

    for element_id, connectivity in enumerate(elements):
        element_dofs = dof_map[connectivity, :].reshape(-1)
        element_displacement = displacement[element_dofs]
        previous_element_displacement = previous_displacement[element_dofs]
        for quadrature_id in range(quadrature_count):
            point_id = element_id * quadrature_count + quadrature_id
            b_matrix = b_matrices[element_id, quadrature_id, :, :]
            strains[point_id, :] = b_matrix @ element_displacement
            strain_rates[point_id, :] = (
                b_matrix @ (element_displacement - previous_element_displacement)
            ) / time_step

    return strains, strain_rates


def _assemble_force_and_tangent(
    node_count: int,
    elements: NDArray[np.int64],
    stresses: NDArray[np.float64],
    tangents: NDArray[np.float64],
    b_matrices: NDArray[np.float64],
    jacobian_weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], csr_matrix]:
    """Assemble global internal force and tangent matrix."""
    dof_map = build_solid_dof_map(node_count)
    dof_count = node_count * SOLID_DOFS_PER_NODE
    internal_force = np.zeros(dof_count, dtype=np.float64)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    quadrature_count = b_matrices.shape[1]

    for element_id, connectivity in enumerate(elements):
        element_dofs = dof_map[connectivity, :].reshape(-1)
        element_force = np.zeros(element_dofs.shape[0], dtype=np.float64)
        element_tangent = np.zeros((element_dofs.shape[0], element_dofs.shape[0]), dtype=np.float64)
        for quadrature_id in range(quadrature_count):
            point_id = element_id * quadrature_count + quadrature_id
            b_matrix = b_matrices[element_id, quadrature_id, :, :]
            weight = jacobian_weights[element_id, quadrature_id]
            element_force += b_matrix.T @ stresses[point_id, :] * weight
            element_tangent += b_matrix.T @ tangents[point_id, :, :] @ b_matrix * weight

        internal_force[element_dofs] += element_force
        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")
        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_tangent.reshape(-1).tolist())

    tangent = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count)).tocsr()

    return internal_force, tangent
