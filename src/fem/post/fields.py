"""Finite element field post-processing.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fem.assembly.elasticity import build_strain_displacement_matrix
from fem.materials.linear_elastic import build_elasticity_matrix

if TYPE_CHECKING:
    from fem.preprocess.data import PreprocessBundle


@dataclass(frozen=True)
class QuadratureFieldData:
    """Store fields evaluated at quadrature points."""

    coordinates: NDArray[np.float64]
    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    energy_density: NDArray[np.float64]


@dataclass(frozen=True)
class CellFieldData:
    """Store element-averaged fields."""

    strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    energy_density: NDArray[np.float64]


def compute_linear_elastic_quadrature_fields(
    bundle: PreprocessBundle,
    displacement: NDArray[np.float64],
    plane_state: str,
) -> QuadratureFieldData:
    """Compute strain, stress and energy density at quadrature points."""
    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache
    elasticity_matrix = build_elasticity_matrix(bundle.material, mesh_info.dimension, plane_state)
    element_count = mesh_info.elements.shape[0]
    quadrature_count = cache.quadrature_weights.shape[0]
    strain_count = elasticity_matrix.shape[0]

    coordinates = np.zeros((element_count, quadrature_count, mesh_info.dimension), dtype=np.float64)
    strain = np.zeros((element_count, quadrature_count, strain_count), dtype=np.float64)
    stress = np.zeros_like(strain)
    energy_density = np.zeros((element_count, quadrature_count), dtype=np.float64)

    for element_id, connectivity in enumerate(mesh_info.elements):
        element_coordinates = mesh_info.nodes[connectivity, :]
        element_dofs = mesh_info.dof_map[connectivity, :].reshape(-1)
        element_displacement = displacement[element_dofs]

        for quadrature_id in range(quadrature_count):
            shape_values = cache.shape_values[quadrature_id, :]
            gradients = cache.shape_gradients_physical[element_id, quadrature_id, :, :]
            b_matrix = build_strain_displacement_matrix(gradients, mesh_info.dimension)
            coordinates[element_id, quadrature_id, :] = shape_values @ element_coordinates
            strain[element_id, quadrature_id, :] = b_matrix @ element_displacement
            stress[element_id, quadrature_id, :] = (
                elasticity_matrix @ strain[element_id, quadrature_id, :]
            )
            energy_density[element_id, quadrature_id] = 0.5 * (
                strain[element_id, quadrature_id, :] @ stress[element_id, quadrature_id, :]
            )

    return QuadratureFieldData(
        coordinates=coordinates,
        strain=strain,
        stress=stress,
        energy_density=energy_density,
    )


def average_quadrature_fields(
    quadrature_fields: QuadratureFieldData,
) -> CellFieldData:
    """Average quadrature fields onto elements."""
    return CellFieldData(
        strain=np.mean(quadrature_fields.strain, axis=1),
        stress=np.mean(quadrature_fields.stress, axis=1),
        energy_density=np.mean(quadrature_fields.energy_density, axis=1),
    )
