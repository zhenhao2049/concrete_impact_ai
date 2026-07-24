"""Finite element external load assembly helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray

from fem.mesh.data import MeshInfo


def assemble_uniform_boundary_traction(
    mesh_info: MeshInfo,
    group_name: str,
    traction: NDArray[np.float64],
    thickness: float,
) -> NDArray[np.float64]:
    """Assemble equivalent nodal forces for one uniform boundary traction."""
    boundary_group = mesh_info.boundary_groups[group_name]
    force = np.zeros(mesh_info.dof_map.size, dtype=np.float64)

    for cell in boundary_group.cells:
        measure = BOUNDARY_MEASURE_BUILDERS[boundary_group.cell_type](
            mesh_info.nodes[cell, :],
        )
        nodal_force = traction * measure * thickness / float(cell.shape[0])

        for node_id in cell:
            dofs = mesh_info.dof_map[node_id, :]
            force[dofs] += nodal_force

    return force


def assemble_uniform_boundary_pressure(
    mesh_info: MeshInfo,
    group_name: str,
    pressure: float,
    outward_normal: NDArray[np.float64],
    thickness: float,
) -> NDArray[np.float64]:
    """Assemble a scalar pressure acting opposite to a supplied outward normal."""
    normal_norm = float(np.linalg.norm(outward_normal))
    if not np.isclose(normal_norm, 1.0):
        raise ValueError("Boundary pressure requires a unit outward normal.")
    traction = -pressure * outward_normal

    return assemble_uniform_boundary_traction(
        mesh_info,
        group_name,
        traction,
        thickness,
    )


def _line_measure(coordinates: NDArray[np.float64]) -> float:
    """Compute the physical length of one line boundary cell."""
    return float(np.linalg.norm(coordinates[1, :] - coordinates[0, :]))


def _quad_measure(coordinates: NDArray[np.float64]) -> float:
    """Compute the physical area of one rectangular quadrilateral boundary cell."""
    extents = np.ptp(coordinates, axis=0)
    positive_extents = extents[extents > 0.0]

    return float(np.prod(positive_extents))


BOUNDARY_MEASURE_BUILDERS = {
    "line": _line_measure,
    "quad": _quad_measure,
}
