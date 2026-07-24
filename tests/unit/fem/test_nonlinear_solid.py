"""Tests for nonlinear solid assembly and effective Newmark terms.

Author:
    Zhen Hao.
Created:
    2026-07-09.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity

from fem.assembly.nonlinear_solid import (
    assemble_hex8_material_response,
    initialize_hex8_material_state,
)
from fem.assembly.solid import build_solid_dof_map
from fem.dynamics.material import (
    _compute_consistent_kinetic_energy,
    _compute_lumped_kinetic_energy,
)
from fem.dynamics.nonlinear import build_newmark_effective_tangent
from fem.materials.data import MaterialUpdateSettings
from fem.materials.errors import MaterialPointConvergenceError
from fem.materials.plasticity import J2ViscoplasticMaterial
from fem.mesh.structured import build_structured_hex8_box


def test_hex8_nonlinear_assembly_uniform_elastic_reaction() -> None:
    """Verify Hex8 material assembly under a uniform elastic strain field."""
    material = J2ViscoplasticMaterial(
        name="assembly_test",
        density=1.0,
        young_modulus=1000.0,
        poisson_ratio=0.25,
        yield_stress=1.0e6,
        hardening_modulus=80.0,
        time_scale=30.0,
        reference_stress=10.0,
        rate_exponent=1.0,
    )
    nodes, elements = build_structured_hex8_box((1.0, 1.0, 1.0), (1, 1, 1))
    state = initialize_hex8_material_state(material, elements.shape[0])
    snapshot = {name: values.copy() for name, values in state.variables.items()}
    dof_map = build_solid_dof_map(nodes.shape[0])
    displacement = np.zeros(nodes.shape[0] * 3, dtype=np.float64)
    axial_strain = 1.0e-3
    displacement[dof_map[:, 0]] = axial_strain * nodes[:, 0]

    result = assemble_hex8_material_response(
        nodes,
        elements,
        material,
        state,
        displacement,
        np.zeros_like(displacement),
        1.0,
        need_tangent=True,
        update_settings=MaterialUpdateSettings(
            max_iterations=50,
            yield_relative_tolerance=1.0e-10,
            residual_absolute_tolerance=1.0e-12,
            residual_relative_tolerance=1.0e-10,
        ),
    )
    right_nodes = np.flatnonzero(np.isclose(nodes[:, 0], 1.0))
    reaction = float(np.sum(result.internal_force[dof_map[right_nodes, 0]]))
    shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    lame_lambda = material.young_modulus * material.poisson_ratio / (
        (1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)
    )
    exact_stress = (lame_lambda + 2.0 * shear_modulus) * axial_strain

    assert reaction == pytest.approx(exact_stress, rel=1.0e-12, abs=1.0e-12)
    for name, values in state.variables.items():
        assert np.array_equal(values, snapshot[name])


def test_newmark_effective_tangent_adds_inertia_and_damping_terms() -> None:
    """Verify nonlinear implicit Newmark tangent composition."""
    material_tangent = 2.0 * identity(3, format="csr")
    mass = 3.0 * identity(3, format="csr")
    damping = 5.0 * identity(3, format="csr")
    effective = build_newmark_effective_tangent(
        material_tangent,
        mass,
        damping,
        time_step=0.2,
        beta=0.25,
        gamma=0.5,
    )
    expected_diagonal = 2.0 + 3.0 / (0.25 * 0.2**2) + 0.5 * 5.0 / (0.25 * 0.2)

    assert np.allclose(effective.diagonal(), expected_diagonal)


def test_material_dynamics_kinetic_energy_matches_selected_mass_operator() -> None:
    """Verify explicit and implicit energy records use their actual mass operators."""
    velocity = np.asarray([1.0, -2.0], dtype=np.float64)
    mass_lumped = np.asarray([2.0, 3.0], dtype=np.float64)
    mass_consistent = csr_matrix(np.asarray([[2.0, 0.5], [0.5, 3.0]]))

    lumped_energy = _compute_lumped_kinetic_energy(mass_lumped, velocity)
    consistent_energy = _compute_consistent_kinetic_energy(mass_consistent, velocity)

    assert lumped_energy == pytest.approx(7.0)
    assert consistent_energy == pytest.approx(6.0)


def test_hex8_material_failure_reports_element_and_quadrature_context() -> None:
    """Verify assembly enriches a failed local Newton without committing its state."""
    material = J2ViscoplasticMaterial(
        name="assembly_failure_test",
        density=1.0,
        young_modulus=1000.0,
        poisson_ratio=0.25,
        yield_stress=1.0,
        hardening_modulus=10.0,
        time_scale=0.02,
        reference_stress=1.0,
        rate_exponent=2.0,
    )
    nodes, elements = build_structured_hex8_box((1.0, 1.0, 1.0), (1, 1, 1))
    state = initialize_hex8_material_state(material, elements.shape[0])
    snapshot = {name: values.copy() for name, values in state.variables.items()}
    dof_map = build_solid_dof_map(nodes.shape[0])
    displacement = np.zeros(nodes.shape[0] * 3, dtype=np.float64)
    displacement[dof_map[:, 0]] = 0.02 * nodes[:, 0]

    with pytest.raises(MaterialPointConvergenceError) as error_info:
        assemble_hex8_material_response(
            nodes,
            elements,
            material,
            state,
            displacement,
            np.zeros_like(displacement),
            1.0e-3,
            update_settings=MaterialUpdateSettings(
                max_iterations=1,
                yield_relative_tolerance=1.0e-10,
                residual_absolute_tolerance=1.0e-30,
                residual_relative_tolerance=1.0e-30,
            ),
        )

    error = error_info.value
    assert error.diagnostics["point_id"] == 0
    assert error.diagnostics["element_id"] == 0
    assert error.diagnostics["quadrature_id"] == 0
    for name, values in state.variables.items():
        assert np.array_equal(values, snapshot[name])
