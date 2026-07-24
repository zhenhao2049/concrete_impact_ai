"""Linear elastic finite element assembly.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fem.materials.linear_elastic import build_elasticity_matrix

if TYPE_CHECKING:
    from fem.preprocess.data import PreprocessBundle


def assemble_stiffness_matrix(
    bundle: PreprocessBundle,
    plane_state: str,
) -> csr_matrix:
    """Assemble the global linear elastic stiffness matrix."""
    mesh_info = bundle.mesh_info
    elasticity_matrix = build_elasticity_matrix(bundle.material, mesh_info.dimension, plane_state)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for element_id, connectivity in enumerate(mesh_info.elements):
        element_dofs = mesh_info.dof_map[connectivity, :].reshape(-1)
        element_matrix = _assemble_element_stiffness(bundle, element_id, elasticity_matrix)

        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")
        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_matrix.reshape(-1).tolist())

    dof_count = mesh_info.dof_map.size
    matrix = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count))

    return matrix.tocsr()


def assemble_consistent_mass_matrix(bundle: PreprocessBundle) -> csr_matrix:
    """Assemble the global consistent mass matrix."""
    mesh_info = bundle.mesh_info

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for element_id, connectivity in enumerate(mesh_info.elements):
        element_dofs = mesh_info.dof_map[connectivity, :].reshape(-1)
        element_matrix = _assemble_element_mass(bundle, element_id)

        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")
        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_matrix.reshape(-1).tolist())

    dof_count = mesh_info.dof_map.size
    matrix = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count))

    return matrix.tocsr()


def assemble_lumped_mass_vector(bundle: PreprocessBundle) -> NDArray[np.float64]:
    """Assemble the row-sum lumped mass vector."""
    mass_matrix = assemble_consistent_mass_matrix(bundle)

    return np.asarray(mass_matrix.sum(axis=1)).reshape(-1)


def build_strain_displacement_matrix(
    gradients: NDArray[np.float64],
    dimension: int,
) -> NDArray[np.float64]:
    """Build the strain-displacement matrix at one quadrature point."""
    return STRAIN_DISPLACEMENT_BUILDERS[dimension](gradients)


def _assemble_element_stiffness(
    bundle: PreprocessBundle,
    element_id: int,
    elasticity_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble one element stiffness matrix."""
    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache
    node_count = mesh_info.elements.shape[1]
    element_dof_count = node_count * mesh_info.dimension
    element_matrix = np.zeros((element_dof_count, element_dof_count), dtype=np.float64)

    for quadrature_id, weight in enumerate(cache.quadrature_weights):
        gradients = cache.shape_gradients_physical[element_id, quadrature_id, :, :]
        b_matrix = build_strain_displacement_matrix(gradients, mesh_info.dimension)
        jacobian_weight = cache.jacobian_determinants[element_id, quadrature_id] * weight
        element_matrix += b_matrix.T @ elasticity_matrix @ b_matrix * jacobian_weight

    return element_matrix


def _assemble_element_mass(
    bundle: PreprocessBundle,
    element_id: int,
) -> NDArray[np.float64]:
    """Assemble one element mass matrix."""
    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache
    node_count = mesh_info.elements.shape[1]
    element_dof_count = node_count * mesh_info.dimension
    element_matrix = np.zeros((element_dof_count, element_dof_count), dtype=np.float64)

    for quadrature_id, weight in enumerate(cache.quadrature_weights):
        shape_values = cache.shape_values[quadrature_id, :]
        shape_matrix = _build_shape_matrix(shape_values, mesh_info.dimension)
        jacobian_weight = cache.jacobian_determinants[element_id, quadrature_id] * weight
        element_matrix += bundle.material.density * shape_matrix.T @ shape_matrix * jacobian_weight

    return element_matrix


def _build_shape_matrix(
    shape_values: NDArray[np.float64],
    dimension: int,
) -> NDArray[np.float64]:
    """Build the vector-valued displacement interpolation matrix."""
    node_count = shape_values.shape[0]
    matrix = np.zeros((dimension, node_count * dimension), dtype=np.float64)

    for node_id, shape_value in enumerate(shape_values):
        for component_id in range(dimension):
            matrix[component_id, node_id * dimension + component_id] = shape_value

    return matrix


def _build_plane_strain_b_matrix(gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the plane-strain B matrix."""
    node_count = gradients.shape[0]
    matrix = np.zeros((3, 2 * node_count), dtype=np.float64)

    for node_id in range(node_count):
        dof_x = 2 * node_id
        dof_y = dof_x + 1
        matrix[0, dof_x] = gradients[node_id, 0]
        matrix[1, dof_y] = gradients[node_id, 1]
        matrix[2, dof_x] = gradients[node_id, 1]
        matrix[2, dof_y] = gradients[node_id, 0]

    return matrix


def _build_three_dimensional_b_matrix(gradients: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the three-dimensional B matrix."""
    node_count = gradients.shape[0]
    matrix = np.zeros((6, 3 * node_count), dtype=np.float64)

    for node_id in range(node_count):
        dof_x = 3 * node_id
        dof_y = dof_x + 1
        dof_z = dof_x + 2
        gradient_x = gradients[node_id, 0]
        gradient_y = gradients[node_id, 1]
        gradient_z = gradients[node_id, 2]

        matrix[0, dof_x] = gradient_x
        matrix[1, dof_y] = gradient_y
        matrix[2, dof_z] = gradient_z
        matrix[3, dof_y] = gradient_z
        matrix[3, dof_z] = gradient_y
        matrix[4, dof_x] = gradient_z
        matrix[4, dof_z] = gradient_x
        matrix[5, dof_x] = gradient_y
        matrix[5, dof_y] = gradient_x

    return matrix


STRAIN_DISPLACEMENT_BUILDERS = {
    2: _build_plane_strain_b_matrix,
    3: _build_three_dimensional_b_matrix,
}
