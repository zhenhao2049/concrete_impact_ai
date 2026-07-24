"""Finite element preprocess pipeline.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fem.boundary.extractors import extract_boundary_conditions
from fem.geometry.builders import build_geometry
from fem.materials.registry import build_material
from fem.mesh.gmsh_io import (
    build_mesh_info,
    convert_element_ordering,
    generate_gmsh_mesh,
    read_gmsh_mesh,
)
from fem.post.vtk import write_mesh_vtk
from fem.preprocess.data import ModelDef, PreprocessBundle
from fem.quadrature.cache import precompute_shape_functions

if TYPE_CHECKING:
    from fem.cases.data import OutputDef

MeshFileGenerator = Callable[[ModelDef], Path]


def build_preprocess_data(model_def: ModelDef, output_def: OutputDef) -> PreprocessBundle:
    """Build the complete finite element preprocess bundle."""
    geometry = build_geometry(model_def)
    material = build_material(model_def.material)

    mesh_path = MESH_FILE_GENERATORS[str(model_def.mesh["generator"])](model_def, output_def)
    raw_mesh = read_gmsh_mesh(mesh_path)
    ordered_mesh = convert_element_ordering(raw_mesh, model_def)
    mesh_info = build_mesh_info(ordered_mesh, model_def)

    shape_function_cache = precompute_shape_functions(mesh_info, model_def)
    boundary_conditions = extract_boundary_conditions(mesh_info, model_def)
    vtk_path = write_mesh_vtk(mesh_info, output_def.vtk_path)

    return PreprocessBundle(
        model_def=model_def,
        geometry=geometry,
        material=material,
        mesh_info=mesh_info,
        shape_function_cache=shape_function_cache,
        boundary_conditions=boundary_conditions,
        mesh_path=mesh_path,
        vtk_path=vtk_path,
    )


def _generate_gmsh_mesh_file(model_def: ModelDef, output_def: OutputDef) -> Path:
    """Generate a Gmsh mesh file with external output settings."""
    return generate_gmsh_mesh(model_def, output_def.mesh_path)


def _generate_iga_mesh_file(model_def: ModelDef, output_def: OutputDef) -> Path:
    """Reserve the IGA mesh generation entry in the preprocess pipeline."""
    raise NotImplementedError("IGA preprocessing requires a selected IGA backend.")


MESH_FILE_GENERATORS: dict[str, MeshFileGenerator] = {
    "gmsh": _generate_gmsh_mesh_file,
    "iga": _generate_iga_mesh_file,
}
