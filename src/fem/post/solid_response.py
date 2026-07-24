"""Three-dimensional solid response VTK helpers.

Author:
    Zhen Hao.
Created:
    2026-07-03.
"""

from pathlib import Path

import meshio
import numpy as np
from numpy.typing import NDArray


def write_solid_vtk(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    displacement: NDArray[np.float64],
    output_path: str | Path,
    velocity: NDArray[np.float64] | None = None,
    cell_stress: NDArray[np.float64] | None = None,
    field_data: dict[str, float] | None = None,
) -> Path:
    """Write hex8 solid nodal fields to VTK."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    displacement_vectors = displacement.reshape(nodes.shape[0], 3)
    point_data = {"displacement": displacement_vectors}

    if velocity is not None:
        velocity_vectors = velocity.reshape(nodes.shape[0], 3)
        point_data["velocity"] = velocity_vectors
        point_data["kinetic_energy_density"] = 0.5 * np.sum(velocity_vectors**2, axis=1)

    mesh = meshio.Mesh(
        points=nodes,
        cells=[("hexahedron", elements[:, [0, 1, 3, 2, 4, 5, 7, 6]])],
        point_data=point_data,
        cell_data=_build_cell_data(cell_stress),
        field_data=_build_field_data(field_data),
    )
    mesh.write(path)

    return path


def _build_field_data(field_data: dict[str, float] | None) -> dict[str, NDArray[np.float64]]:
    """Build meshio-compatible scalar field data."""
    data = {}

    for key, value in (field_data or {}).items():
        data[key] = np.asarray([float(value)], dtype=np.float64)

    return data


def _build_cell_data(
    cell_stress: NDArray[np.float64] | None,
) -> dict[str, list[NDArray[np.float64]]]:
    """Build optional solid cell data."""
    data = {}

    if cell_stress is not None:
        data["stress"] = [cell_stress]

    return data
