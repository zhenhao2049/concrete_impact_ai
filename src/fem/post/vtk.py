"""VTK mesh and field output helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path

import meshio
import numpy as np

from fem.mesh.data import MeshInfo
from fem.mesh.ordering import INTERNAL_TO_VTK
from fem.post.fields import CellFieldData, QuadratureFieldData


def write_mesh_vtk(mesh_info: MeshInfo, output_path: str | Path) -> Path:
    """Write the normalized finite element mesh to a VTK file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = POINT_PADDING[mesh_info.dimension](mesh_info.nodes)
    cells = mesh_info.elements[:, INTERNAL_TO_VTK[mesh_info.element_type]]
    mesh = meshio.Mesh(
        points=points,
        cells=[(mesh_info.element_cell_type, cells)],
        cell_data={"physical_tag": [mesh_info.element_physical_tags]},
    )
    mesh.write(path)

    return path


def write_solution_vtk(
    mesh_info: MeshInfo,
    output_path: str | Path,
    displacement: np.ndarray,
    cell_fields: CellFieldData,
) -> Path:
    """Write nodal displacement and cell fields to a VTK file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = POINT_PADDING[mesh_info.dimension](mesh_info.nodes)
    cells = mesh_info.elements[:, INTERNAL_TO_VTK[mesh_info.element_type]]
    displacement_vectors = POINT_PADDING[mesh_info.dimension](
        displacement.reshape(mesh_info.nodes.shape[0], mesh_info.dimension)
    )
    mesh = meshio.Mesh(
        points=points,
        cells=[(mesh_info.element_cell_type, cells)],
        point_data={"displacement": displacement_vectors},
        cell_data={
            "strain": [cell_fields.strain],
            "stress": [cell_fields.stress],
            "energy_density": [cell_fields.energy_density],
        },
    )
    mesh.write(path)

    return path


def write_named_solution_vtk(
    mesh_info: MeshInfo,
    output_path: str | Path,
    point_fields: dict[str, np.ndarray],
    cell_fields: dict[str, np.ndarray],
) -> Path:
    """Write explicitly named nodal and element fields to one VTU file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = POINT_PADDING[mesh_info.dimension](mesh_info.nodes)
    cells = mesh_info.elements[:, INTERNAL_TO_VTK[mesh_info.element_type]]
    mesh = meshio.Mesh(
        points=points,
        cells=[(mesh_info.element_cell_type, cells)],
        point_data=point_fields,
        cell_data={name: [values] for name, values in cell_fields.items()},
    )
    mesh.write(path)

    return path


def write_named_point_cloud_vtk(
    output_path: str | Path,
    coordinates: np.ndarray,
    point_fields: dict[str, np.ndarray],
) -> Path:
    """Write named integration-point fields as a VTK vertex cloud."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = POINT_PADDING[coordinates.shape[1]](coordinates)
    vertices = np.arange(points.shape[0], dtype=np.int64).reshape(-1, 1)
    meshio.Mesh(
        points=points,
        cells=[("vertex", vertices)],
        point_data=point_fields,
    ).write(path)
    return path


def write_quadrature_points_vtk(
    output_path: str | Path,
    quadrature_fields: QuadratureFieldData,
) -> Path:
    """Write quadrature-point fields as a VTK point cloud."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coordinates = quadrature_fields.coordinates.reshape(-1, quadrature_fields.coordinates.shape[2])
    points = POINT_PADDING[coordinates.shape[1]](coordinates)
    point_count = points.shape[0]
    mesh = meshio.Mesh(
        points=points,
        cells=[("vertex", np.arange(point_count, dtype=np.int64).reshape(-1, 1))],
        point_data={
            "strain": quadrature_fields.strain.reshape(point_count, -1),
            "stress": quadrature_fields.stress.reshape(point_count, -1),
            "energy_density": quadrature_fields.energy_density.reshape(point_count),
        },
    )
    mesh.write(path)

    return path


def _pad_2d_points(nodes: np.ndarray) -> np.ndarray:
    """Pad two-dimensional points with zero z-coordinates."""
    return np.column_stack([nodes, np.zeros(nodes.shape[0], dtype=np.float64)])


POINT_PADDING = {
    2: _pad_2d_points,
    3: lambda nodes: nodes,
}
