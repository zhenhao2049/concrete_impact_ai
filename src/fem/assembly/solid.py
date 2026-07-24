"""Three-dimensional solid finite element assembly.

Author:
    Zhen Hao.
Created:
    2026-07-03.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fem.assembly.elasticity import build_strain_displacement_matrix
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.materials.linear_elastic import LinearElasticMaterial, build_elasticity_matrix
from fem.quadrature.rules import make_quadrature_rule

SOLID_DOFS_PER_NODE = 3


def build_solid_dof_map(node_count: int) -> NDArray[np.int64]:
    """Build the three-DOF solid dof map."""
    return np.arange(node_count * SOLID_DOFS_PER_NODE, dtype=np.int64).reshape(
        node_count,
        SOLID_DOFS_PER_NODE,
    )


def assemble_hex8_stiffness_matrix(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    material: LinearElasticMaterial,
) -> csr_matrix:
    """Assemble the global hex8 solid stiffness matrix."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    elasticity = build_elasticity_matrix(material, 3, "three_dimensional")
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for connectivity in elements:
        element_nodes = nodes[connectivity, :]
        element_dofs = dof_map[connectivity, :].reshape(-1)
        element_matrix = assemble_hex8_element_stiffness(element_nodes, elasticity)
        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")

        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_matrix.reshape(-1).tolist())

    dof_count = nodes.shape[0] * SOLID_DOFS_PER_NODE
    matrix = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count))

    return matrix.tocsr()


def assemble_hex8_mass_matrix(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    material: LinearElasticMaterial,
) -> csr_matrix:
    """Assemble the global consistent hex8 solid mass matrix."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for connectivity in elements:
        element_nodes = nodes[connectivity, :]
        element_dofs = dof_map[connectivity, :].reshape(-1)
        element_matrix = assemble_hex8_element_mass(element_nodes, material.density)
        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")

        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_matrix.reshape(-1).tolist())

    dof_count = nodes.shape[0] * SOLID_DOFS_PER_NODE
    matrix = coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count))

    return matrix.tocsr()


def assemble_hex8_top_pressure_load(
    nodes: NDArray[np.float64],
    top_faces: NDArray[np.int64],
    pressure: float,
) -> NDArray[np.float64]:
    """Assemble uniform pressure on top quadrilateral faces."""
    dof_map = build_solid_dof_map(nodes.shape[0])
    force = np.zeros(nodes.shape[0] * SOLID_DOFS_PER_NODE, dtype=np.float64)
    points, weights = make_quadrature_rule("quad4", 2)

    for face in top_faces:
        face_nodes = nodes[face, :]
        face_force = np.zeros(4 * SOLID_DOFS_PER_NODE, dtype=np.float64)

        for point, weight in zip(points, weights, strict=True):
            shape_values, gradients_reference = evaluate_lagrange_shape_functions(
                "quad4",
                point.reshape(1, 2),
            )
            tangent_xi = face_nodes.T @ gradients_reference[0, :, 0]
            tangent_eta = face_nodes.T @ gradients_reference[0, :, 1]
            area_weight = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta))) * weight

            for node_id, shape_value in enumerate(shape_values[0, :]):
                face_force[node_id * SOLID_DOFS_PER_NODE + 2] += (
                    shape_value * pressure * area_weight
                )

        force[dof_map[face, :].reshape(-1)] += face_force

    return force


def assemble_hex8_element_stiffness(
    node_coordinates: NDArray[np.float64],
    elasticity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble one hex8 stiffness matrix."""
    points, weights = make_quadrature_rule("hex8", 2)
    shape_values, gradients_reference = evaluate_lagrange_shape_functions("hex8", points)
    matrix = np.zeros((8 * SOLID_DOFS_PER_NODE, 8 * SOLID_DOFS_PER_NODE), dtype=np.float64)

    for quadrature_id, weight in enumerate(weights):
        jacobian = node_coordinates.T @ gradients_reference[quadrature_id, :, :]
        gradients_physical = gradients_reference[quadrature_id, :, :] @ np.linalg.inv(jacobian)
        b_matrix = build_strain_displacement_matrix(gradients_physical, 3)
        jacobian_weight = float(np.linalg.det(jacobian)) * weight
        matrix += b_matrix.T @ elasticity @ b_matrix * jacobian_weight

    return matrix


def assemble_hex8_element_mass(
    node_coordinates: NDArray[np.float64],
    density: float,
) -> NDArray[np.float64]:
    """Assemble one hex8 consistent mass matrix."""
    points, weights = make_quadrature_rule("hex8", 2)
    shape_values, gradients_reference = evaluate_lagrange_shape_functions("hex8", points)
    matrix = np.zeros((8 * SOLID_DOFS_PER_NODE, 8 * SOLID_DOFS_PER_NODE), dtype=np.float64)

    for quadrature_id, weight in enumerate(weights):
        jacobian = node_coordinates.T @ gradients_reference[quadrature_id, :, :]
        jacobian_weight = float(np.linalg.det(jacobian)) * weight

        for node_i, shape_i in enumerate(shape_values[quadrature_id, :]):
            for node_j, shape_j in enumerate(shape_values[quadrature_id, :]):
                value = density * shape_i * shape_j * jacobian_weight
                for component_id in range(SOLID_DOFS_PER_NODE):
                    row = node_i * SOLID_DOFS_PER_NODE + component_id
                    column = node_j * SOLID_DOFS_PER_NODE + component_id
                    matrix[row, column] += value

    return matrix
