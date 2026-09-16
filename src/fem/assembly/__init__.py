"""Global and element-level assembly modules.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.assembly.boundary import apply_dirichlet_to_linear_system, impose_values_on_vector
from fem.assembly.elasticity import (
    assemble_consistent_mass_matrix,
    assemble_lumped_mass_vector,
    assemble_stiffness_matrix,
    build_strain_displacement_matrix,
)
from fem.assembly.load import (
    assemble_boundary_traction,
    assemble_uniform_boundary_pressure,
    assemble_uniform_boundary_traction,
)
from fem.assembly.nonlinear_solid import (
    assemble_bundle_material_response,
    assemble_hex8_material_response,
    build_hex8_material_assembly_cache,
    build_material_assembly_cache,
    initialize_bundle_material_state,
    integrate_quadrature_density,
)

__all__ = [
    "apply_dirichlet_to_linear_system",
    "assemble_boundary_traction",
    "assemble_consistent_mass_matrix",
    "assemble_lumped_mass_vector",
    "assemble_stiffness_matrix",
    "assemble_uniform_boundary_traction",
    "assemble_uniform_boundary_pressure",
    "assemble_bundle_material_response",
    "assemble_hex8_material_response",
    "build_hex8_material_assembly_cache",
    "build_material_assembly_cache",
    "build_strain_displacement_matrix",
    "impose_values_on_vector",
    "initialize_bundle_material_state",
    "integrate_quadrature_density",
]
