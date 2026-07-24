"""Boundary condition extraction from mesh groups.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np

from fem.boundary.data import BoundaryCondition, BoundaryConditionSet
from fem.mesh.data import MeshInfo
from fem.preprocess.data import ModelDef

COMPONENT_INDEX = {
    "ux": 0,
    "uy": 1,
    "uz": 2,
    "vx": 0,
    "vy": 1,
    "vz": 2,
    "tx": 0,
    "ty": 1,
    "tz": 2,
}


def extract_boundary_conditions(mesh_info: MeshInfo, model_def: ModelDef) -> BoundaryConditionSet:
    """Extract boundary conditions from named mesh boundary groups."""
    return BoundaryConditionSet(
        dirichlet=_build_condition_tuple("dirichlet", mesh_info, model_def),
        velocity=_build_condition_tuple("velocity", mesh_info, model_def),
        traction=_build_condition_tuple("traction", mesh_info, model_def),
        pressure=_build_condition_tuple("pressure", mesh_info, model_def),
    )


def _build_condition_tuple(
    kind: str,
    mesh_info: MeshInfo,
    model_def: ModelDef,
) -> tuple[BoundaryCondition, ...]:
    """Build all boundary conditions of one kind."""
    return tuple(
        _build_condition(kind, mesh_info, condition_spec)
        for condition_spec in model_def.boundary_conditions[kind]
    )


def _build_condition(
    kind: str,
    mesh_info: MeshInfo,
    condition_spec: dict[str, object],
) -> BoundaryCondition:
    """Build one boundary condition entry."""
    group_name = str(condition_spec["group"])
    boundary_group = mesh_info.boundary_groups[group_name]
    components = _resolve_components(condition_spec["components"])
    dofs = mesh_info.dof_map[np.ix_(boundary_group.nodes, np.asarray(components))]

    return BoundaryCondition(
        name=str(condition_spec["name"]),
        kind=kind,
        group=group_name,
        components=components,
        values=np.asarray(condition_spec["values"], dtype=np.float64),
        nodes=boundary_group.nodes,
        dofs=dofs.reshape(-1),
    )


def _resolve_components(component_names: object) -> tuple[int, ...]:
    """Resolve component names to integer component ids."""
    return tuple(COMPONENT_INDEX[str(name)] for name in component_names)
