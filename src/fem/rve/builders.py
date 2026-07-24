"""Builders for single- and multi-phase quasi-static Hex8 RVEs.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import numpy as np

from fem.assembly.nonlinear_solid import build_hex8_material_assembly_cache
from fem.materials.data import MaterialModel
from fem.materials.phased import PhasedMaterialModel, RVEPhase
from fem.mesh.structured import CylindricalOGridHex8Mesh
from fem.rve.data import StructuredHex8RVE
from fem.rve.periodic import build_periodic_constraint_from_classes
from fem.solvers.data import NonlinearNewtonSettings


def build_cylindrical_two_phase_rve(
    mesh: CylindricalOGridHex8Mesh,
    matrix_material: MaterialModel,
    inclusion_material: MaterialModel,
    newton_settings: NonlinearNewtonSettings,
    hill_mandel_power_scale: float,
    quadrature_order: int = 2,
) -> StructuredHex8RVE:
    """Build a centered z-cylinder two-phase RVE from a conforming O-grid."""
    cache = build_hex8_material_assembly_cache(
        mesh.nodes,
        mesh.elements,
        quadrature_order,
    )
    element_volumes = np.sum(cache.jacobian_weights, axis=1)
    matrix_volume = float(np.sum(element_volumes[mesh.element_phase_ids == 0]))
    inclusion_volume = float(np.sum(element_volumes[mesh.element_phase_ids == 1]))
    total_volume = matrix_volume + inclusion_volume
    effective_density = (
        matrix_material.density * matrix_volume
        + inclusion_material.density * inclusion_volume
    ) / total_volume
    quadrature_count = cache.jacobian_weights.shape[1]
    point_phase_ids = np.repeat(mesh.element_phase_ids, quadrature_count)
    phased_material = PhasedMaterialModel(
        name=f"{matrix_material.name}__{inclusion_material.name}",
        phases=(
            RVEPhase("matrix", matrix_material),
            RVEPhase("inclusion", inclusion_material),
        ),
        point_phase_ids=point_phase_ids,
        effective_density=effective_density,
    )
    constraint = build_periodic_constraint_from_classes(
        mesh.nodes,
        mesh.periodic_class_ids,
        (0, 0, 0),
        mesh.opposite_face_nodes,
    )

    return StructuredHex8RVE(
        nodes=mesh.nodes,
        elements=mesh.elements,
        divisions=(0, 0, 0),
        material=phased_material,
        constraint=constraint,
        newton_settings=newton_settings,
        element_phase_ids=mesh.element_phase_ids,
        element_densities=np.where(
            mesh.element_phase_ids == 0,
            matrix_material.density,
            inclusion_material.density,
        ),
        phase_names=("matrix", "inclusion"),
        quadrature_order=quadrature_order,
        hill_mandel_power_scale=hill_mandel_power_scale,
    )
