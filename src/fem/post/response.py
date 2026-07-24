"""Finite element response quantity extraction helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray

from fem.mesh.data import MeshInfo


def average_boundary_component(
    mesh_info: MeshInfo,
    vector: NDArray[np.float64],
    group_name: str,
    component_id: int,
) -> float:
    """Compute one average component over a named boundary group."""
    nodes = mesh_info.boundary_groups[group_name].nodes
    dofs = mesh_info.dof_map[nodes, component_id]

    return float(np.mean(vector[dofs]))


def sum_boundary_component(
    mesh_info: MeshInfo,
    vector: NDArray[np.float64],
    group_name: str,
    component_id: int,
) -> float:
    """Sum one vector component over a named boundary group."""
    nodes = mesh_info.boundary_groups[group_name].nodes
    dofs = mesh_info.dof_map[nodes, component_id]

    return float(np.sum(vector[dofs]))


def average_gauge_component(
    mesh_info: MeshInfo,
    vector: NDArray[np.float64],
    coordinate_x: float,
    component_id: int,
) -> float:
    """Average one component on the mesh section closest to a gauge coordinate."""
    distances = np.abs(mesh_info.nodes[:, 0] - coordinate_x)
    section_nodes = np.where(np.isclose(distances, np.min(distances)))[0]
    dofs = mesh_info.dof_map[section_nodes, component_id]

    return float(np.mean(vector[dofs]))
