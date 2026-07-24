"""Mesh data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhysicalGroup:
    """Gmsh physical group metadata."""

    name: str
    tag: int
    dimension: int


@dataclass(frozen=True)
class BoundaryGroup:
    """Boundary entity group extracted from a mesh."""

    name: str
    tag: int
    dimension: int
    cell_type: str
    cells: NDArray[np.int64]
    nodes: NDArray[np.int64]


@dataclass(frozen=True)
class RawMesh:
    """Raw mesh read from a mesh file."""

    points: NDArray[np.float64]
    cells: dict[str, NDArray[np.int64]]
    cell_physical_tags: dict[str, NDArray[np.int64]]
    physical_groups: dict[str, PhysicalGroup]
    source_path: Path


@dataclass(frozen=True)
class MeshInfo:
    """Finite element mesh data after project-level normalization."""

    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    element_type: str
    element_cell_type: str
    dimension: int
    dof_map: NDArray[np.int64]
    element_physical_tags: NDArray[np.int64]
    element_group_names: tuple[str, ...]
    boundary_groups: dict[str, BoundaryGroup]
    physical_groups: dict[str, PhysicalGroup]
    element_ordering: str
