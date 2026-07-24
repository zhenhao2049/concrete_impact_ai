"""Shape function and Jacobian precomputation.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np

from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.mesh.data import MeshInfo
from fem.preprocess.data import ModelDef
from fem.quadrature.data import ShapeFunctionCache
from fem.quadrature.rules import make_quadrature_rule


def precompute_shape_functions(mesh_info: MeshInfo, model_def: ModelDef) -> ShapeFunctionCache:
    """Precompute Lagrange shape functions and element Jacobians."""
    order = int(model_def.quadrature["order"])
    quadrature_points, quadrature_weights = make_quadrature_rule(mesh_info.element_type, order)
    shape_values, gradients_reference = evaluate_lagrange_shape_functions(
        mesh_info.element_type,
        quadrature_points,
    )

    jacobians, determinants, gradients_physical = _compute_element_jacobians(
        mesh_info,
        gradients_reference,
    )

    return ShapeFunctionCache(
        element_type=mesh_info.element_type,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        shape_values=shape_values,
        shape_gradients_reference=gradients_reference,
        jacobians=jacobians,
        jacobian_determinants=determinants,
        shape_gradients_physical=gradients_physical,
    )


def _compute_element_jacobians(
    mesh_info: MeshInfo,
    gradients_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Jacobian matrices and physical gradients for all elements."""
    element_count = mesh_info.elements.shape[0]
    quadrature_count = gradients_reference.shape[0]
    node_count = gradients_reference.shape[1]
    dimension = mesh_info.dimension

    jacobians = np.zeros((element_count, quadrature_count, dimension, dimension), dtype=np.float64)
    determinants = np.zeros((element_count, quadrature_count), dtype=np.float64)
    gradients_physical = np.zeros(
        (element_count, quadrature_count, node_count, dimension),
        dtype=np.float64,
    )

    for element_id, connectivity in enumerate(mesh_info.elements):
        element_coordinates = mesh_info.nodes[connectivity, :]

        for quadrature_id in range(quadrature_count):
            gradient_reference = gradients_reference[quadrature_id, :, :]
            jacobian = element_coordinates.T @ gradient_reference
            inverse_jacobian = np.linalg.inv(jacobian)

            jacobians[element_id, quadrature_id, :, :] = jacobian
            determinants[element_id, quadrature_id] = np.linalg.det(jacobian)
            gradients_physical[element_id, quadrature_id, :, :] = (
                gradient_reference @ inverse_jacobian
            )

    return jacobians, determinants, gradients_physical
