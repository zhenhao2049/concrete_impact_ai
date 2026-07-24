"""Tests for rate-independent plastic material updates.

Author:
    Zhen Hao.
Created:
    2026-07-08.
"""

import numpy as np
import pytest

from fem.materials.data import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
)
from fem.materials.errors import MaterialPointConvergenceError
from fem.materials.plasticity import (
    DruckerPragerCapMaterial,
    DruckerPragerMaterial,
    J2PlasticMaterial,
    J2ViscoplasticMaterial,
    _cap_pressure_derivative,
    _cap_q_derivative,
    _compute_dp_cap_branch_values,
    _compute_dp_cap_geometry,
    _numerical_tangent,
    _update_j2_viscoplastic_point,
)

VOIGT_DIM = 6


def test_j2_pure_shear_matches_perfect_plastic_curve() -> None:
    """Verify J2 radial return against a pure-shear closed-form curve."""
    material = J2PlasticMaterial(
        name="j2_test",
        density=7850.0,
        young_modulus=2.0e11,
        poisson_ratio=0.30,
        yield_stress=4.0e8,
        hardening_modulus=0.0,
        kinematic_fraction=0.0,
    )
    shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    yield_shear = material.yield_stress / np.sqrt(3.0)
    state = material.initialize_state(1)

    for shear_strain in np.linspace(0.0, 6.0e-3, 31):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[5] = shear_strain
        response = material.update(_request(strain), state, _full_requirements())
        state = _state(response.state)

        exact_shear = min(shear_modulus * shear_strain, yield_shear)
        assert response.stresses[0, 5] == pytest.approx(exact_shear, rel=1.0e-12, abs=1.0e-5)
        assert response.diagnostics["yield_value"][0] <= 1.0e-5

    assert response.tangents[0, 5, 5] == pytest.approx(0.0, abs=1.0e5)


def test_j2_viscoplastic_multiplier_depends_on_physical_time_step() -> None:
    """Verify J2 viscoplastic flow depends on the physical time increment."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_dt_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=30.0,
        reference_stress=10.0,
        rate_exponent=1.0,
    )
    strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    strain[5] = 0.02
    state = material.initialize_state(1)
    response_small_dt = material.update(_request(strain, time_step=0.5), state)
    response_large_dt = material.update(_request(strain, time_step=5.0), state)

    multiplier_small = response_small_dt.diagnostics["viscoplastic_multiplier"][0]
    multiplier_large = response_large_dt.diagnostics["viscoplastic_multiplier"][0]

    assert multiplier_large > multiplier_small
    assert response_large_dt.stresses[0, 5] < response_small_dt.stresses[0, 5]


def test_j2_viscoplastic_yield_activation_uses_relative_stress_scale() -> None:
    """Verify the elastic branch uses epsilon times a stress-valued scale."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_activation_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    settings = MaterialUpdateSettings(
        max_iterations=50,
        yield_relative_tolerance=1.0e-3,
        residual_absolute_tolerance=1.0e-12,
        residual_relative_tolerance=1.0e-10,
    )

    responses = []
    for trial_overstress in (5.0e-3, 2.0e-2):
        strain = np.zeros((1, VOIGT_DIM), dtype=np.float64)
        strain[0, 5] = (material.yield_stress + trial_overstress) / (
            np.sqrt(3.0) * shear_modulus
        )
        request = MaterialPointRequest(
            strains=strain,
            strain_rates=None,
            time_step=1.0e-3,
            kinematics="three_dimensional",
            update_settings=settings,
        )
        responses.append(material.update(request, material.initialize_state(1)))

    elastic_response, plastic_response = responses
    assert elastic_response.diagnostics["trial_yield_value"][0] > 0.0
    assert elastic_response.diagnostics["trial_yield_value"][0] < elastic_response.diagnostics[
        "yield_activation_threshold"
    ][0]
    assert elastic_response.diagnostics["plastic_multiplier"][0] == 0.0
    assert elastic_response.diagnostics["yield_activation_margin"][0] < 0.0
    assert elastic_response.diagnostics["viscoplastic_active"][0] == 0.0
    assert plastic_response.diagnostics["trial_yield_value"][0] > plastic_response.diagnostics[
        "yield_activation_threshold"
    ][0]
    assert plastic_response.diagnostics["plastic_multiplier"][0] > 0.0
    assert plastic_response.diagnostics["yield_activation_margin"][0] > 0.0
    assert plastic_response.diagnostics["viscoplastic_active"][0] == 1.0


def test_j2_viscoplastic_loading_diagnostics_separate_work_stress_and_flow() -> None:
    """Verify signed work, stress change, and viscoplastic activity remain distinct."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_path_diagnostics",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    state = material.initialize_state(1)
    loaded_strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    loaded_strain[5] = 0.02
    loaded = material.update(
        _request_with_increment(loaded_strain, np.zeros(VOIGT_DIM), 1.0e-3),
        state,
        _full_requirements(),
    )

    assert loaded.diagnostics["incremental_work_density"][0] > 0.0
    assert loaded.diagnostics["equivalent_stress_increment"][0] > 0.0
    assert loaded.diagnostics["viscoplastic_active"][0] == 1.0

    committed = _state(loaded.state)
    relaxed = material.update(
        _request_with_increment(loaded_strain, loaded_strain, 1.0e-3),
        committed,
        _full_requirements(),
    )

    assert relaxed.diagnostics["incremental_work_density"][0] == pytest.approx(0.0)
    assert relaxed.diagnostics["equivalent_stress_increment"][0] < 0.0
    assert relaxed.diagnostics["viscoplastic_active"][0] == 1.0


def test_j2_viscoplastic_consistent_tangent_matches_finite_difference() -> None:
    """Verify the analytic J2 viscoplastic tangent against a local perturbation test."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_tangent_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=30.0,
        reference_stress=10.0,
        rate_exponent=1.0,
    )
    strain = np.asarray([0.004, -0.0015, -0.001, 0.002, -0.003, 0.02], dtype=np.float64)
    state = material.initialize_state(1)
    response = material.update(
        _request(strain, time_step=2.0), state, _full_requirements()
    )
    numerical_tangent = _numerical_tangent(
        material,
        strain,
        state,
        _update_j2_viscoplastic_point,
        2.0,
        _material_update_settings(),
    )

    assert response.diagnostics["active_surface"][0] == pytest.approx(4.0)
    assert np.allclose(response.tangents[0], numerical_tangent, rtol=1.0e-5, atol=1.0e-6)


def test_j2_viscoplastic_quadratic_rate_root_tangent_and_dissipation() -> None:
    """Verify the pure-Newton m=2 update, tangent, free energy, and dissipation identity."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_quadratic_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    strain = np.asarray([0.004, -0.0015, -0.001, 0.002, -0.003, 0.02], dtype=np.float64)
    state = material.initialize_state(1)
    request = _request(strain, time_step=1.0e-3)
    response = material.update(request, state, _full_requirements())
    numerical_tangent = _numerical_tangent(
        material,
        strain,
        state,
        _update_j2_viscoplastic_point,
        request.time_step,
        request.update_settings,
    )
    next_state = _state(response.state)
    plastic_strain = next_state.variables["plastic_strain"][0]
    q_value = float(next_state.variables["equivalent_plastic_strain"][0])
    elastic_strain = strain - plastic_strain
    expected_free_energy = 0.5 * response.stresses[0] @ elastic_strain + 0.5 * (
        material.hardening_modulus * q_value**2
    )

    assert response.diagnostics["local_residual"][0] == pytest.approx(0.0, abs=1.0e-12)
    assert response.diagnostics["plastic_multiplier"][0] > 0.0
    assert response.diagnostics["plastic_multiplier"][0] < response.diagnostics[
        "root_upper_bound"
    ][0]
    assert response.dissipation[0] >= 0.0
    assert response.dissipation[0] == pytest.approx(
        response.diagnostics["plastic_power"][0]
        - response.diagnostics["hardening_storage_rate"][0],
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert response.free_energy[0] == pytest.approx(expected_free_energy, rel=1.0e-12)
    assert np.allclose(response.tangents[0], numerical_tangent, rtol=1.0e-5, atol=1.0e-6)


def test_j2_viscoplastic_local_newton_failure_preserves_committed_state() -> None:
    """Verify an unconverged local Newton reports diagnostics without committing state."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_failure_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    strain = np.asarray([0.004, -0.0015, -0.001, 0.002, -0.003, 0.02], dtype=np.float64)
    state = material.initialize_state(1)
    snapshot = {name: values.copy() for name, values in state.variables.items()}
    request = MaterialPointRequest(
        strains=strain.reshape(1, VOIGT_DIM),
        strain_rates=np.zeros((1, VOIGT_DIM), dtype=np.float64),
        time_step=1.0e-3,
        kinematics="three_dimensional",
        update_settings=MaterialUpdateSettings(
            max_iterations=1,
            yield_relative_tolerance=1.0e-10,
            residual_absolute_tolerance=1.0e-30,
            residual_relative_tolerance=1.0e-30,
        ),
    )

    with pytest.raises(MaterialPointConvergenceError) as error_info:
        material.update(request, state, _full_requirements())

    error = error_info.value
    assert error.reason == "maximum_iterations_exceeded"
    assert error.diagnostics["point_id"] == 0
    assert error.diagnostics["iteration"] == 1
    assert error.diagnostics["newton_candidate"] > 0.0
    assert error.diagnostics["physical_upper_bound"] > error.diagnostics["newton_candidate"]
    for name, values in state.variables.items():
        assert np.array_equal(values, snapshot[name])


def test_j2_viscoplastic_non_finite_iteration_fails_directly() -> None:
    """Verify a non-finite Perzyna rate is reported instead of entering another method."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_non_finite_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=1.0e-300,
        rate_exponent=2.0,
    )
    strain = np.asarray([0.004, -0.0015, -0.001, 0.002, -0.003, 0.02], dtype=np.float64)

    with pytest.raises(MaterialPointConvergenceError) as error_info:
        material.update(_request(strain, time_step=1.0e-3), material.initialize_state(1))

    error = error_info.value
    assert error.reason == "non_finite_iteration_value"
    assert error.diagnostics["residual"] in {"inf", "-inf"}


def test_j2_non_finite_invariant_fails_without_truncation() -> None:
    """Verify an invalid J2 invariant raises structured diagnostics directly."""
    material = J2ViscoplasticMaterial(
        name="j2_vp_invariant_test",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )
    strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    strain[5] = np.nan

    with pytest.raises(MaterialPointConvergenceError) as error_info:
        material.update(_request(strain), material.initialize_state(1))

    assert error_info.value.reason == "non_finite_j2_invariant"
    assert error_info.value.diagnostics["algorithm"] == "j2_equivalent_stress"
    assert error_info.value.diagnostics["point_id"] == 0


def test_dp_pure_shear_matches_semianalytic_nonassociated_curve() -> None:
    """Verify DP return mapping against a pure-shear semi-analytic path."""
    material = DruckerPragerMaterial(
        name="dp_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.0,
        cohesion=3.0e6,
        flow_parameter=0.20,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
    )
    shear_modulus = material.young_modulus / (2.0 * (1.0 + material.poisson_ratio))
    bulk_modulus = material.young_modulus / (3.0 * (1.0 - 2.0 * material.poisson_ratio))
    state = material.initialize_state(1)

    for shear_strain in np.linspace(0.0, 5.0e-4, 31):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[5] = shear_strain
        response = material.update(_request(strain), state)
        state = _state(response.state)
        exact = _dp_pure_shear_exact(material, shear_modulus, bulk_modulus, shear_strain)

        assert response.stresses[0, 5] == pytest.approx(
            exact["shear_stress"],
            rel=1.0e-12,
            abs=1.0e-8,
        )
        assert response.diagnostics["pressure"][0] == pytest.approx(
            exact["pressure"],
            rel=1.0e-12,
            abs=1.0e-8,
        )
        assert response.diagnostics["plastic_volumetric_strain"][0] == pytest.approx(
            exact["plastic_volumetric_strain"],
            rel=1.0e-12,
            abs=1.0e-15,
        )


def test_dp_and_dp_cap_state_variables_are_distinct() -> None:
    """Verify that only DP-Cap carries cap-apex state variables."""
    dp_material = DruckerPragerMaterial(
        name="dp_state_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.8,
        cohesion=3.0e6,
        flow_parameter=0.05,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
    )
    cap_material = DruckerPragerCapMaterial(
        name="cap_state_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.8,
        cohesion=3.0e6,
        flow_parameter=0.05,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
        initial_cap_apex_pressure=3.5e7,
        cap_transition_ratio=0.45,
        cap_hardening=0.0,
        max_local_iterations=30,
        local_tolerance=1.0e-10,
    )

    assert "cap_apex_pressure" not in dp_material.initialize_state(1).variables
    assert "cap_apex_pressure" in cap_material.initialize_state(1).variables


def test_dp_cap_hydrostatic_compression_matches_semianalytic_curve() -> None:
    """Verify cap return mapping against a hydrostatic semi-analytic path."""
    material = DruckerPragerCapMaterial(
        name="cap_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.5,
        cohesion=3.0e6,
        flow_parameter=0.0,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
        initial_cap_apex_pressure=2.5e7,
        cap_transition_ratio=0.45,
        cap_hardening=5.0e9,
        max_local_iterations=30,
        local_tolerance=1.0e-10,
    )
    bulk_modulus = material.young_modulus / (3.0 * (1.0 - 2.0 * material.poisson_ratio))
    state = material.initialize_state(1)
    exact_state = {
        "cap_apex_pressure": material.initial_cap_apex_pressure,
        "plastic_volumetric_strain": 0.0,
    }

    for compression_strain in np.linspace(0.0, 3.0e-3, 31):
        strain = np.zeros(VOIGT_DIM, dtype=np.float64)
        strain[:3] = -compression_strain / 3.0
        response = material.update(_request(strain), state)
        state = _state(response.state)
        exact = _cap_hydrostatic_exact(material, bulk_modulus, compression_strain, exact_state)

        assert response.diagnostics["pressure"][0] == pytest.approx(
            exact["pressure"],
            rel=1.0e-10,
            abs=1.0e-4,
        )
        assert response.diagnostics["cap_apex_pressure"][0] == pytest.approx(
            exact["cap_apex_pressure"],
            rel=1.0e-10,
            abs=1.0e-4,
        )
        assert response.diagnostics["plastic_volumetric_strain"][0] == pytest.approx(
            exact["plastic_volumetric_strain"],
            rel=1.0e-10,
            abs=1.0e-14,
        )


def test_material_update_does_not_mutate_committed_state() -> None:
    """Verify predictor-corrector updates leave the input committed state untouched."""
    material = J2PlasticMaterial(
        name="j2_state_test",
        density=7850.0,
        young_modulus=2.0e11,
        poisson_ratio=0.30,
        yield_stress=4.0e8,
        hardening_modulus=0.0,
        kinematic_fraction=0.0,
    )
    committed_state = material.initialize_state(1)
    snapshot = {name: values.copy() for name, values in committed_state.variables.items()}
    strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    strain[5] = 6.0e-3

    first_response = material.update(_request(strain), committed_state)
    second_response = material.update(_request(strain), committed_state)

    for name, values in committed_state.variables.items():
        assert np.array_equal(values, snapshot[name])
    assert np.allclose(first_response.stresses, second_response.stresses)
    assert _state(first_response.state).variables["equivalent_plastic_strain"][0] > 0.0


def test_dp_cap_composite_surface_returns_old_corner_path() -> None:
    """Verify composite DP-Cap returns finite stresses for old corner paths."""
    material = DruckerPragerCapMaterial(
        name="cap_corner_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.0,
        cohesion=1.0e6,
        flow_parameter=0.0,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
        initial_cap_apex_pressure=5.0e6,
        cap_transition_ratio=0.4,
        cap_hardening=0.0,
        max_local_iterations=30,
        local_tolerance=1.0e-10,
    )
    strain = np.zeros(VOIGT_DIM, dtype=np.float64)
    strain[:3] = -1.0e-3 / 3.0
    strain[5] = 1.0e-3

    response = material.update(_request(strain), material.initialize_state(1))

    assert np.all(np.isfinite(response.stresses))
    assert response.diagnostics["composite_yield_value"][0] <= 1.0e-6
    assert response.diagnostics["active_surface"][0] in (2.0, 3.0)


def test_dp_cap_composite_surface_is_c1_at_transition() -> None:
    """Verify value and slope continuity at the DP-Cap transition."""
    material = DruckerPragerCapMaterial(
        name="cap_geometry_test",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.20,
        friction_parameter=0.8,
        cohesion=3.0e6,
        flow_parameter=0.05,
        cohesion_hardening=0.0,
        kappa_rate=1.0,
        initial_cap_apex_pressure=3.5e7,
        cap_transition_ratio=0.45,
        cap_hardening=0.0,
        max_local_iterations=30,
        local_tolerance=1.0e-10,
    )
    geometry = _compute_dp_cap_geometry(material, material.initial_cap_apex_pressure)
    composite_value, dp_value, cap_value = _compute_dp_cap_branch_values(
        material,
        geometry.transition_pressure,
        geometry.transition_q,
        0.0,
        material.initial_cap_apex_pressure,
    )
    cap_slope = -_cap_pressure_derivative(
        material,
        geometry.transition_pressure,
        material.initial_cap_apex_pressure,
    ) / _cap_q_derivative(
        material,
        geometry.transition_q,
        material.initial_cap_apex_pressure,
    )

    assert composite_value == pytest.approx(0.0, abs=1.0e-10)
    assert dp_value == pytest.approx(0.0, abs=1.0e-10)
    assert cap_value == pytest.approx(0.0, abs=1.0e-10)
    assert cap_slope == pytest.approx(material.friction_parameter, rel=1.0e-12)


def _request(strain: np.ndarray, time_step: float = 1.0) -> MaterialPointRequest:
    """Build a single-point material request."""
    return MaterialPointRequest(
        strains=strain.reshape(1, VOIGT_DIM),
        strain_rates=np.zeros((1, VOIGT_DIM), dtype=np.float64),
        time_step=time_step,
        kinematics="three_dimensional",
        update_settings=_material_update_settings(),
    )


def _request_with_increment(
    strain: np.ndarray,
    previous_strain: np.ndarray,
    time_step: float,
) -> MaterialPointRequest:
    """Build a request with an exact step-average strain rate."""
    return MaterialPointRequest(
        strains=strain.reshape(1, VOIGT_DIM),
        strain_rates=((strain - previous_strain) / time_step).reshape(1, VOIGT_DIM),
        time_step=time_step,
        kinematics="three_dimensional",
        update_settings=_material_update_settings(),
    )


def _full_requirements() -> MaterialResponseRequirements:
    """Request tangent and thermodynamic outputs for constitutive tests."""
    return MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True)


def _material_update_settings() -> MaterialUpdateSettings:
    """Build local constitutive settings for material tests."""
    return MaterialUpdateSettings(
        max_iterations=50,
        yield_relative_tolerance=1.0e-10,
        residual_absolute_tolerance=1.0e-12,
        residual_relative_tolerance=1.0e-10,
    )


def _state(state: MaterialState | None) -> MaterialState:
    """Require an updated material state."""
    if state is None:
        raise ValueError("Plastic material response did not include an updated state.")

    return state


def _dp_pure_shear_exact(
    material: DruckerPragerMaterial,
    shear_modulus: float,
    bulk_modulus: float,
    shear_strain: float,
) -> dict[str, float]:
    """Evaluate the DP pure-shear closed response."""
    q_trial = np.sqrt(3.0) * shear_modulus * shear_strain
    yield_trial = q_trial - material.cohesion
    if yield_trial <= 0.0:
        return {
            "shear_stress": shear_modulus * shear_strain,
            "pressure": 0.0,
            "plastic_volumetric_strain": 0.0,
        }

    denominator = (
        3.0 * shear_modulus
        + material.friction_parameter * bulk_modulus * material.flow_parameter
        + material.cohesion_hardening * material.kappa_rate
    )
    delta_gamma = yield_trial / denominator
    q_value = q_trial - 3.0 * shear_modulus * delta_gamma

    return {
        "shear_stress": q_value / np.sqrt(3.0),
        "pressure": bulk_modulus * material.flow_parameter * delta_gamma,
        "plastic_volumetric_strain": material.flow_parameter * delta_gamma,
    }


def _cap_hydrostatic_exact(
    material: DruckerPragerCapMaterial,
    bulk_modulus: float,
    compression_strain: float,
    exact_state: dict[str, float],
) -> dict[str, float]:
    """Evaluate the cap hydrostatic closed response."""
    apex_pressure_n = exact_state["cap_apex_pressure"]
    plastic_volumetric_strain_n = exact_state["plastic_volumetric_strain"]
    pressure_trial = bulk_modulus * (compression_strain - plastic_volumetric_strain_n)

    if pressure_trial <= apex_pressure_n:
        pressure = pressure_trial
        apex_pressure = apex_pressure_n
        plastic_volumetric_strain = plastic_volumetric_strain_n
    else:
        plastic_increment = (pressure_trial - apex_pressure_n) / (
            bulk_modulus + material.cap_hardening
        )
        pressure = pressure_trial - bulk_modulus * plastic_increment
        apex_pressure = apex_pressure_n + material.cap_hardening * plastic_increment
        plastic_volumetric_strain = plastic_volumetric_strain_n + plastic_increment

    exact_state["cap_apex_pressure"] = apex_pressure
    exact_state["plastic_volumetric_strain"] = plastic_volumetric_strain

    return {
        "pressure": pressure,
        "cap_apex_pressure": apex_pressure,
        "plastic_volumetric_strain": plastic_volumetric_strain,
    }
