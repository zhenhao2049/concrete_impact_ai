"""Preprocess data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fem.boundary.data import BoundaryConditionSet
from fem.geometry.data import GeometryDef
from fem.materials.data import MaterialModel
from fem.mesh.data import MeshInfo
from fem.quadrature.data import ShapeFunctionCache


@dataclass(frozen=True)
class ModelDef:
    """Raw model definition loaded from one YAML file."""

    name: str
    dimension: int
    geometry: dict[str, Any]
    mesh: dict[str, Any]
    material: dict[str, Any]
    quadrature: dict[str, Any]
    boundary_conditions: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class PreprocessBundle:
    """Complete finite element preprocess output."""

    model_def: ModelDef
    geometry: GeometryDef
    material: MaterialModel
    mesh_info: MeshInfo
    shape_function_cache: ShapeFunctionCache
    boundary_conditions: BoundaryConditionSet
    mesh_path: Path
    vtk_path: Path
