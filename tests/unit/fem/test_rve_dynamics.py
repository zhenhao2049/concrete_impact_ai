"""Tests for linear dynamic RVE applicability screening.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

import numpy as np

from fem.materials import (
    J2ViscoplasticMaterial,
    LinearElasticMaterial,
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
)
from fem.mesh.structured import build_cylindrical_ogrid_hex8
from fem.rve import (
    RVEExecutionSettings,
    RVEHomogenizedMaterial,
    build_cylindrical_two_phase_rve,
    compute_linear_rve_dynamic_tangent,
    compute_rve_dynamic_screening,
)
from fem.solvers import ArmijoSettings, NonlinearNewtonSettings


def test_linear_dynamic_screening_has_positive_mode_and_static_limit() -> None:
    """Verify the constrained mode and zero-frequency Schur complement."""
    mesh = build_cylindrical_ogrid_hex8((1.0, 1.0, 1.0), 0.05, 16, 2, 1)
    matrix = LinearElasticMaterial("matrix", 2400.0, 1000.0, 0.2)
    inclusion = LinearElasticMaterial("inclusion", 7850.0, 6666.666666666667, 0.3)
    model = build_cylindrical_two_phase_rve(
        mesh,
        matrix,
        inclusion,
        NonlinearNewtonSettings(
            20,
            1.0e-10,
            1.0e-10,
            1.0e-12,
            1.0e-10,
            ArmijoSettings(enabled=True),
        ),
        10.0,
    )
    update = MaterialUpdateSettings(30, 1.0e-10, 1.0e-12, 1.0e-10)
    screening = compute_rve_dynamic_screening(model, update, 0.1)
    static_tangent = compute_linear_rve_dynamic_tangent(model, update, 0.0)
    low_frequency_tangent = compute_linear_rve_dynamic_tangent(model, update, 1.0e-6)

    assert screening.first_angular_frequency > 0.0
    assert screening.minimum_shear_wave_speed > 0.0
    assert screening.scale_ratio > 0.0
    assert np.linalg.norm(low_frequency_tangent - static_tangent) / np.linalg.norm(
        static_tangent
    ) < 1.0e-12


def test_multiphase_rve_adapter_preserves_independent_macro_point_states() -> None:
    """Verify two macro points own separate phased RVE states and diagnostics."""
    mesh = build_cylindrical_ogrid_hex8((1.0, 1.0, 1.0), 0.05, 16, 2, 1)
    matrix = J2ViscoplasticMaterial(
        "matrix", 2400.0, 1000.0, 0.2, 10.0, 20.0, 0.02, 10.0, 2.0
    )
    inclusion = LinearElasticMaterial("inclusion", 7850.0, 6666.666666666667, 0.3)
    model = build_cylindrical_two_phase_rve(
        mesh,
        matrix,
        inclusion,
        NonlinearNewtonSettings(
            20, 1.0e-10, 1.0e-10, 1.0e-12, 1.0e-10, ArmijoSettings(enabled=True)
        ),
        10.0,
    )
    adapter = RVEHomogenizedMaterial(model, "two_phase")
    state = adapter.initialize_state(2)
    response = adapter.update(
        MaterialPointRequest(
            strains=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.03],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.002],
                ]
            ),
            strain_rates=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 30.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                ]
            ),
            time_step=1.0e-3,
            kinematics="three_dimensional",
            update_settings=MaterialUpdateSettings(50, 1.0e-10, 1.0e-12, 1.0e-10),
        ),
        state,
        MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
    )

    assert adapter.maximum_wave_speed == inclusion.maximum_wave_speed
    phase_q = response.diagnostics["phase_matrix__equivalent_plastic_strain"]
    assert phase_q[0] > phase_q[1]
    assert not np.shares_memory(
        response.state.variables["rve_macro_strain"][0],
        response.state.variables["rve_macro_strain"][1],
    )
    process_adapter = RVEHomogenizedMaterial(
        model,
        "two_phase_process",
        RVEExecutionSettings(backend="process_pool", workers=1),
    )
    try:
        process_response = process_adapter.update(
            MaterialPointRequest(
                strains=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.002]]),
                strain_rates=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]),
                time_step=1.0e-3,
                kinematics="three_dimensional",
                update_settings=MaterialUpdateSettings(50, 1.0e-10, 1.0e-12, 1.0e-10),
            ),
            process_adapter.initialize_state(1),
            MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
        )
    finally:
        process_adapter.close()
    assert np.allclose(process_response.stresses[0], response.stresses[1], atol=1.0e-12)
