"""Finite element external load assembly helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from collections.abc import Callable

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


def assemble_boundary_traction(
    mesh_info: MeshInfo,
    group_name: str,
    traction_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    thickness: float,
) -> NDArray[np.float64]:
    """Assemble a coordinate-dependent traction on a line boundary."""
    boundary_group = mesh_info.boundary_groups[group_name]
    assembler = BOUNDARY_TRACTION_ASSEMBLERS[boundary_group.cell_type]
    force = np.zeros(mesh_info.dof_map.size, dtype=np.float64)

    for cell in boundary_group.cells:
        nodal_force = assembler(
            mesh_info.nodes[cell, :],
            traction_function,
            thickness,
        )
        for local_node_id, node_id in enumerate(cell):
            dofs = mesh_info.dof_map[node_id, :]
            force[dofs] += nodal_force[local_node_id]

    return force


def _line_measure(coordinates: NDArray[np.float64]) -> float:
    """Compute the physical length of one line boundary cell."""
    return float(np.linalg.norm(coordinates[1, :] - coordinates[0, :]))


def _quad_measure(coordinates: NDArray[np.float64]) -> float:
    """Compute the physical area of one rectangular quadrilateral boundary cell."""
    extents = np.ptp(coordinates, axis=0)
    positive_extents = extents[extents > 0.0]

    return float(np.prod(positive_extents))


def _assemble_line_traction(
    coordinates: NDArray[np.float64],
    traction_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    thickness: float,
) -> NDArray[np.float64]:
    """Integrate one two-node line traction with two Gauss points."""
    line_length = _line_measure(coordinates)
    jacobian = 0.5 * line_length
    nodal_force = np.zeros_like(coordinates, dtype=np.float64)
    gauss_points = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    for coordinate in gauss_points:
        shape_values = np.asarray(
            (0.5 * (1.0 - coordinate), 0.5 * (1.0 + coordinate)),
            dtype=np.float64,
        )
        physical_coordinate = shape_values @ coordinates
        traction = np.asarray(traction_function(physical_coordinate), dtype=np.float64)
        mesh_dimension = coordinates.shape[1]
        if traction.shape != (mesh_dimension,):
            raise ValueError(
                "Boundary traction dimension differs from the mesh dimension: "
                f"expected={mesh_dimension}, received={traction.shape}."
            )
        nodal_force += shape_values[:, None] * traction[None, :] * jacobian * thickness

    return nodal_force


BOUNDARY_MEASURE_BUILDERS = {
    "line": _line_measure,
    "quad": _quad_measure,
}

BOUNDARY_TRACTION_ASSEMBLERS = {
    "line": _assemble_line_traction,
}
