"""Gmsh mesh generation and mesh normalization.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import gmsh
import meshio
import numpy as np

from fem.mesh.data import BoundaryGroup, MeshInfo, PhysicalGroup, RawMesh
from fem.mesh.ordering import (
    BOUNDARY_CELL_TYPES,
    GMSH_TO_INTERNAL,
    INTERNAL_ORDERING_NAMES,
    MESHIO_CELL_TYPES,
    reorder_cells,
)
from fem.preprocess.data import ModelDef

GeometryBuilder = Callable[[ModelDef], None]


def generate_gmsh_mesh(model_def: ModelDef, mesh_path: str | Path) -> Path:
    """Generate a Gmsh mesh file from the model definition."""
    path = Path(mesh_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(model_def.name)

        GMSH_GEOMETRY_BUILDERS[str(model_def.geometry["type"])](model_def)

        gmsh.model.mesh.generate(model_def.dimension)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    return path


def read_gmsh_mesh(mesh_path: str | Path) -> RawMesh:
    """Read a Gmsh mesh file through meshio."""
    path = Path(mesh_path)
    mesh = meshio.read(path)
    cell_physical_tags = mesh.cell_data_dict["gmsh:physical"]

    physical_groups = {
        name: PhysicalGroup(name=name, tag=int(data[0]), dimension=int(data[1]))
        for name, data in mesh.field_data.items()
    }

    cells = {
        cell_type: np.asarray(values, dtype=np.int64)
        for cell_type, values in mesh.cells_dict.items()
    }
    tags = {
        cell_type: np.asarray(values, dtype=np.int64)
        for cell_type, values in cell_physical_tags.items()
    }

    return RawMesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=cells,
        cell_physical_tags=tags,
        physical_groups=physical_groups,
        source_path=path,
    )


def convert_element_ordering(raw_mesh: RawMesh, model_def: ModelDef) -> RawMesh:
    """Convert Gmsh element ordering to the project internal ordering."""
    element_type = str(model_def.mesh["element_type"])
    cell_type = MESHIO_CELL_TYPES[element_type]
    cells = dict(raw_mesh.cells)
    cells[cell_type] = reorder_cells(raw_mesh.cells[cell_type], GMSH_TO_INTERNAL[element_type])

    return replace(raw_mesh, cells=cells)


def build_mesh_info(raw_mesh: RawMesh, model_def: ModelDef) -> MeshInfo:
    """Build normalized mesh information for FEM computations."""
    element_type = str(model_def.mesh["element_type"])
    element_cell_type = MESHIO_CELL_TYPES[element_type]
    boundary_cell_type = BOUNDARY_CELL_TYPES[model_def.dimension]

    nodes = raw_mesh.points[:, : model_def.dimension]
    elements = raw_mesh.cells[element_cell_type]
    element_physical_tags = raw_mesh.cell_physical_tags[element_cell_type]

    dof_map = np.arange(nodes.shape[0] * model_def.dimension, dtype=np.int64)
    dof_map = dof_map.reshape(nodes.shape[0], model_def.dimension)

    domain_tag_to_name = {
        group.tag: group.name
        for group in raw_mesh.physical_groups.values()
        if group.dimension == model_def.dimension
    }
    element_group_names = tuple(domain_tag_to_name[int(tag)] for tag in element_physical_tags)

    boundary_groups = {
        group.name: _build_boundary_group(raw_mesh, group, boundary_cell_type)
        for group in raw_mesh.physical_groups.values()
        if group.dimension == model_def.dimension - 1
    }

    return MeshInfo(
        nodes=nodes,
        elements=elements,
        element_type=element_type,
        element_cell_type=element_cell_type,
        dimension=model_def.dimension,
        dof_map=dof_map,
        element_physical_tags=element_physical_tags,
        element_group_names=element_group_names,
        boundary_groups=boundary_groups,
        physical_groups=raw_mesh.physical_groups,
        element_ordering=INTERNAL_ORDERING_NAMES[element_type],
    )


def _build_boundary_group(
    raw_mesh: RawMesh,
    physical_group: PhysicalGroup,
    boundary_cell_type: str,
) -> BoundaryGroup:
    """Build one boundary group from physical tags."""
    cell_tags = raw_mesh.cell_physical_tags[boundary_cell_type]
    cells = raw_mesh.cells[boundary_cell_type][cell_tags == physical_group.tag]
    nodes = np.unique(cells.reshape(-1))

    return BoundaryGroup(
        name=physical_group.name,
        tag=physical_group.tag,
        dimension=physical_group.dimension,
        cell_type=boundary_cell_type,
        cells=cells,
        nodes=nodes,
    )


def _build_rectangle_gmsh(model_def: ModelDef) -> None:
    """Build a transfinite quadrilateral rectangle in the active Gmsh model."""
    origin = model_def.geometry["origin"]
    size = model_def.geometry["size"]
    divisions = model_def.mesh["divisions"]
    x0, y0 = float(origin[0]), float(origin[1])
    lx, ly = float(size[0]), float(size[1])
    nx, ny = int(divisions[0]), int(divisions[1])

    p1 = gmsh.model.geo.addPoint(x0, y0, 0.0)
    p2 = gmsh.model.geo.addPoint(x0 + lx, y0, 0.0)
    p3 = gmsh.model.geo.addPoint(x0 + lx, y0 + ly, 0.0)
    p4 = gmsh.model.geo.addPoint(x0, y0 + ly, 0.0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteSurface(surface, cornerTags=[p1, p2, p3, p4])
    gmsh.model.geo.mesh.setRecombine(2, surface)
    gmsh.model.geo.synchronize()

    _add_physical_group(2, [surface], 1, "domain")
    _add_physical_group(1, [l4], 11, "left")
    _add_physical_group(1, [l2], 12, "right")
    _add_physical_group(1, [l1], 13, "bottom")
    _add_physical_group(1, [l3], 14, "top")


def _build_box_gmsh(model_def: ModelDef) -> None:
    """Build a transfinite hexahedral box in the active Gmsh model."""
    origin = model_def.geometry["origin"]
    size = model_def.geometry["size"]
    divisions = model_def.mesh["divisions"]
    x0, y0, z0 = float(origin[0]), float(origin[1]), float(origin[2])
    lx, ly, lz = float(size[0]), float(size[1]), float(size[2])
    nx, ny, nz = int(divisions[0]), int(divisions[1]), int(divisions[2])

    p000 = gmsh.model.geo.addPoint(x0, y0, z0)
    p100 = gmsh.model.geo.addPoint(x0 + lx, y0, z0)
    p110 = gmsh.model.geo.addPoint(x0 + lx, y0 + ly, z0)
    p010 = gmsh.model.geo.addPoint(x0, y0 + ly, z0)
    p001 = gmsh.model.geo.addPoint(x0, y0, z0 + lz)
    p101 = gmsh.model.geo.addPoint(x0 + lx, y0, z0 + lz)
    p111 = gmsh.model.geo.addPoint(x0 + lx, y0 + ly, z0 + lz)
    p011 = gmsh.model.geo.addPoint(x0, y0 + ly, z0 + lz)

    l1 = gmsh.model.geo.addLine(p000, p100)
    l2 = gmsh.model.geo.addLine(p100, p110)
    l3 = gmsh.model.geo.addLine(p110, p010)
    l4 = gmsh.model.geo.addLine(p010, p000)
    l5 = gmsh.model.geo.addLine(p001, p101)
    l6 = gmsh.model.geo.addLine(p101, p111)
    l7 = gmsh.model.geo.addLine(p111, p011)
    l8 = gmsh.model.geo.addLine(p011, p001)
    l9 = gmsh.model.geo.addLine(p000, p001)
    l10 = gmsh.model.geo.addLine(p100, p101)
    l11 = gmsh.model.geo.addLine(p110, p111)
    l12 = gmsh.model.geo.addLine(p010, p011)

    bottom = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])])
    top = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])])
    front = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])])
    right = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])])
    back = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])])
    left = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])])

    surface_loop = gmsh.model.geo.addSurfaceLoop([bottom, right, back, left, front, top])
    volume = gmsh.model.geo.addVolume([surface_loop])

    for line in [l1, l3, l5, l7]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, nx + 1)
    for line in [l2, l4, l6, l8]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, ny + 1)
    for line in [l9, l10, l11, l12]:
        gmsh.model.geo.mesh.setTransfiniteCurve(line, nz + 1)
    for surface in [bottom, top, front, right, back, left]:
        gmsh.model.geo.mesh.setTransfiniteSurface(surface)
        gmsh.model.geo.mesh.setRecombine(2, surface)

    gmsh.model.geo.mesh.setTransfiniteVolume(
        volume,
        cornerTags=[p000, p100, p110, p010, p001, p101, p111, p011],
    )
    gmsh.model.geo.synchronize()

    _add_physical_group(3, [volume], 1, "domain")
    _add_physical_group(2, [left], 21, "left")
    _add_physical_group(2, [right], 22, "right")
    _add_physical_group(2, [front], 23, "front")
    _add_physical_group(2, [back], 24, "back")
    _add_physical_group(2, [bottom], 25, "bottom")
    _add_physical_group(2, [top], 26, "top")


def _add_physical_group(dimension: int, tags: list[int], physical_tag: int, name: str) -> None:
    """Add one named physical group to the active Gmsh model."""
    gmsh.model.addPhysicalGroup(dimension, tags, physical_tag)
    gmsh.model.setPhysicalName(dimension, physical_tag, name)


GMSH_GEOMETRY_BUILDERS: dict[str, GeometryBuilder] = {
    "rectangle": _build_rectangle_gmsh,
    "box": _build_box_gmsh,
}
