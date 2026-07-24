"""Linear dynamic screening operators for periodic Hex8 RVEs.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, solve
from scipy.sparse import coo_matrix, csr_matrix

from fem.assembly.nonlinear_solid import assemble_hex8_material_response_cached
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.materials.data import MaterialUpdateSettings
from fem.materials.linear_elastic import LinearElasticMaterial
from fem.materials.phased import PhasedMaterialModel
from fem.quadrature.rules import make_quadrature_rule
from fem.rve.data import RVEOperatorCache, StructuredHex8RVE
from fem.rve.solver import build_rve_operator_cache, initialize_rve_state


@dataclass(frozen=True)
class RVEDynamicScreening:
    """Store scale-separation and first constrained-cell frequency metrics."""

    cell_length: float
    minimum_shear_wave_speed: float
    transit_time: float
    first_angular_frequency: float
    maximum_loading_angular_frequency: float
    scale_ratio: float
    frequency_ratio: float


def assemble_rve_consistent_mass(
    model: StructuredHex8RVE,
    operator_cache: RVEOperatorCache,
) -> csr_matrix:
    """Assemble the exact consistent translational mass matrix for one Hex8 RVE."""
    points, _ = make_quadrature_rule("hex8", model.quadrature_order)
    shape_values, _ = evaluate_lagrange_shape_functions("hex8", points)
    element_densities = (
        np.full(model.elements.shape[0], model.material.density)
        if model.element_densities is None
        else model.element_densities
    )
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for element_id, connectivity in enumerate(model.elements):
        scalar_mass = np.zeros((8, 8), dtype=np.float64)
        for quadrature_id in range(shape_values.shape[0]):
            shape = shape_values[quadrature_id]
            scalar_mass += (
                element_densities[element_id]
                * np.outer(shape, shape)
                * operator_cache.assembly.jacobian_weights[element_id, quadrature_id]
            )
        element_mass = np.kron(scalar_mass, np.eye(3))
        element_dofs = np.asarray(
            [3 * int(node) + component for node in connectivity for component in range(3)]
        )
        row_ids, column_ids = np.meshgrid(element_dofs, element_dofs, indexing="ij")
        rows.extend(row_ids.reshape(-1).tolist())
        columns.extend(column_ids.reshape(-1).tolist())
        values.extend(element_mass.reshape(-1).tolist())
    dof_count = 3 * model.nodes.shape[0]
    return coo_matrix((values, (rows, columns)), shape=(dof_count, dof_count)).tocsr()


def compute_rve_dynamic_screening(
    model: StructuredHex8RVE,
    update_settings: MaterialUpdateSettings,
    maximum_loading_angular_frequency: float,
    operator_cache: RVEOperatorCache | None = None,
) -> RVEDynamicScreening:
    """Compute transit-time and constrained-cell eigenfrequency indicators."""
    cache = operator_cache or build_rve_operator_cache(model)
    stiffness = _assemble_zero_state_stiffness(model, update_settings, cache)
    mass = assemble_rve_consistent_mass(model, cache)
    transformation = model.constraint.transformation
    reduced_stiffness = (transformation.T @ stiffness @ transformation).toarray()
    reduced_mass = (transformation.T @ mass @ transformation).toarray()
    eigenvalues = eigh(reduced_stiffness, reduced_mass, eigvals_only=True)
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        raise ValueError("Constrained RVE eigenproblem has no positive eigenvalue.")
    first_frequency = float(np.sqrt(positive[0]))
    lengths = np.ptp(model.nodes, axis=0)
    cell_length = float(np.max(lengths))
    minimum_speed = _minimum_phase_shear_wave_speed(model)
    transit_time = cell_length / minimum_speed
    return RVEDynamicScreening(
        cell_length=cell_length,
        minimum_shear_wave_speed=minimum_speed,
        transit_time=transit_time,
        first_angular_frequency=first_frequency,
        maximum_loading_angular_frequency=maximum_loading_angular_frequency,
        scale_ratio=maximum_loading_angular_frequency * transit_time,
        frequency_ratio=maximum_loading_angular_frequency / first_frequency,
    )


def compute_linear_rve_dynamic_tangent(
    model: StructuredHex8RVE,
    update_settings: MaterialUpdateSettings,
    angular_frequency: float,
    operator_cache: RVEOperatorCache | None = None,
) -> NDArray[np.float64]:
    """Condense the harmonic linear RVE operator at one angular frequency."""
    cache = operator_cache or build_rve_operator_cache(model)
    stiffness = _assemble_zero_state_stiffness(model, update_settings, cache)
    mass = assemble_rve_consistent_mass(model, cache)
    dynamic_operator = stiffness - angular_frequency**2 * mass
    transformation = model.constraint.transformation
    affine = model.constraint.affine_matrix
    k_ww = (transformation.T @ dynamic_operator @ transformation).toarray()
    k_we = np.asarray(transformation.T @ dynamic_operator @ affine)
    sensitivity = solve(k_ww, k_we, assume_a="sym")
    k_ee = np.asarray(affine.T @ dynamic_operator @ affine)
    k_ew = np.asarray(affine.T @ dynamic_operator @ transformation)
    return (k_ee - k_ew @ sensitivity) / cache.volume


def _assemble_zero_state_stiffness(
    model: StructuredHex8RVE,
    update_settings: MaterialUpdateSettings,
    operator_cache: RVEOperatorCache,
) -> csr_matrix:
    """Assemble the elastic tangent at the undeformed, uncommitted RVE state."""
    _require_linear_phases(model)
    displacement = np.zeros(3 * model.nodes.shape[0], dtype=np.float64)
    state = initialize_rve_state(model)
    assembly = assemble_hex8_material_response_cached(
        model.nodes,
        model.elements,
        model.material,
        state.material_state,
        displacement,
        displacement,
        1.0,
        update_settings,
        operator_cache.assembly,
        True,
    )
    return assembly.tangent


def _require_linear_phases(model: StructuredHex8RVE) -> tuple[LinearElasticMaterial, ...]:
    """Return all phases and reject nonlinear materials in the screening solver."""
    if isinstance(model.material, LinearElasticMaterial):
        return (model.material,)
    if isinstance(model.material, PhasedMaterialModel):
        materials: list[LinearElasticMaterial] = []
        for phase in model.material.phases:
            if not isinstance(phase.material, LinearElasticMaterial):
                raise ValueError("Linear dynamic RVE screening requires linear-elastic phases.")
            materials.append(phase.material)
        return tuple(materials)
    raise ValueError("Linear dynamic RVE screening received an unsupported material model.")


def _minimum_phase_shear_wave_speed(model: StructuredHex8RVE) -> float:
    """Compute the minimum phase shear-wave speed used by the transit indicator."""
    speeds = []
    for material in _require_linear_phases(model):
        shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
        speeds.append(float(np.sqrt(shear_modulus / material.density)))
    return min(speeds)
