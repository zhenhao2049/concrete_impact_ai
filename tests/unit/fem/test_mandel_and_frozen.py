"""Tests for Mandel coordinates and frozen-history material evaluation.

Author:
    Zhen Hao.
Created:
    2026-07-22.
"""

import numpy as np

from fem.materials import (
    J2ViscoplasticMaterial,
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
    build_frozen_history_material,
)
from fem.tensors import (
    d4_mandel_operators,
    engineering_stiffness_to_mandel,
    engineering_strain_to_mandel,
    engineering_stress_to_mandel,
    mandel_stiffness_to_engineering,
    mandel_strain_to_engineering,
    mandel_stress_to_engineering,
    project_mandel_stiffness_to_d4,
)


def test_mandel_round_trip_preserves_power_and_stiffness_action() -> None:
    """Verify power-conjugate vector and stiffness transformations."""
    strain = np.asarray([0.01, -0.004, 0.002, 0.006, -0.008, 0.003])
    stress = np.asarray([12.0, -3.0, 5.0, 4.0, -2.0, 7.0])
    stiffness = np.arange(36, dtype=np.float64).reshape(6, 6) + 10.0 * np.eye(6)

    strain_mandel = engineering_strain_to_mandel(strain)
    stress_mandel = engineering_stress_to_mandel(stress)
    stiffness_mandel = engineering_stiffness_to_mandel(stiffness)

    assert np.allclose(mandel_strain_to_engineering(strain_mandel), strain)
    assert np.allclose(mandel_stress_to_engineering(stress_mandel), stress)
    assert np.allclose(mandel_stiffness_to_engineering(stiffness_mandel), stiffness)
    np.testing.assert_allclose(
        stress_mandel @ strain_mandel,
        stress @ strain,
    )
    assert np.allclose(
        stress_mandel,
        engineering_stress_to_mandel(stress),
    )
    assert np.allclose(
        stiffness_mandel @ strain_mandel,
        engineering_stress_to_mandel(stiffness @ strain),
    )


def test_d4_projection_is_invariant_and_idempotent() -> None:
    """Verify square-group averaging defines an invariant projector."""
    rng = np.random.default_rng(72000)
    matrix = rng.normal(size=(6, 6))
    matrix = matrix.T @ matrix + np.eye(6)
    projected = project_mandel_stiffness_to_d4(matrix)

    assert np.allclose(project_mandel_stiffness_to_d4(projected), projected)
    for operator in d4_mandel_operators():
        assert np.allclose(operator @ projected @ operator.T, projected, atol=1.0e-12)


def test_frozen_j2_keeps_history_and_returns_elastic_tangent() -> None:
    """Verify J2 frozen evaluation changes neither plastic strain nor hardening state."""
    material = J2ViscoplasticMaterial(
        name="matrix",
        density=1.0,
        young_modulus=1000.0,
        poisson_ratio=0.2,
        yield_stress=10.0,
        hardening_modulus=20.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    state = material.initialize_state(2)
    state.variables["plastic_strain"][:] = np.asarray(
        [[0.01, -0.005, -0.005, 0.002, 0.0, 0.0], [0.0] * 6]
    )
    state.variables["equivalent_plastic_strain"][:] = [0.012, 0.0]
    state.variables["viscoplastic_multiplier"][:] = [0.012, 0.0]
    frozen = build_frozen_history_material(material)
    request = MaterialPointRequest(
        strains=np.asarray(
            [[0.02, -0.006, -0.004, 0.004, 0.0, 0.0], [0.001] * 6]
        ),
        strain_rates=np.asarray([[0.1] * 6, [0.2] * 6]),
        time_step=0.01,
        kinematics="three_dimensional",
        update_settings=MaterialUpdateSettings(30, 1.0e-10, 1.0e-12, 1.0e-10),
    )
    response = frozen.update(
        request,
        state,
        MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
    )

    for name, values in state.variables.items():
        assert np.array_equal(response.state.variables[name], values)
        assert not np.shares_memory(response.state.variables[name], values)
    assert response.tangents is not None
    assert np.allclose(response.tangents[0], response.tangents[1])
    assert response.dissipation is not None
    assert np.array_equal(response.dissipation, np.zeros(2))
    assert np.array_equal(response.diagnostics["viscoplastic_active"], np.zeros(2))
