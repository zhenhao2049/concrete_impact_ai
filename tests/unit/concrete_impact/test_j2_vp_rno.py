"""Tests for the first structured continuous-time J2-VP-RNO.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

import numpy as np
import pytest
import torch

from concrete_impact.nn.config import J2VPRNOModelConfig
from concrete_impact.nn.models.j2_vp_rno import StructuredJ2VPRNO
from concrete_impact.nn.registry import load_j2_vp_rno_state_dict
from fem.assembly.nonlinear_solid import assemble_hex8_material_response
from fem.materials.data import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
)
from fem.materials.errors import MaterialPointConvergenceError
from fem.materials.plasticity import J2ViscoplasticMaterial
from fem.surrogates import save_surrogate_metadata


def test_zero_correction_rno_matches_j2_perzyna_on_reversal_path() -> None:
    """Verify stress, state, energy, dissipation, and tangent on a reversal path."""
    baseline, rno = _build_models()
    baseline_state = baseline.initialize_state(1)
    rno_state = rno.initialize_state(1)
    q_history = []
    for shear_strain in (0.0, 0.008, 0.020, 0.010, 0.0, -0.012, -0.020):
        strain = np.zeros((1, 6), dtype=np.float64)
        strain[0, 5] = shear_strain
        request = _request(strain)
        requirements = MaterialResponseRequirements(
            tangent=True,
            free_energy=True,
            dissipation=True,
        )
        baseline_response = baseline.update(request, baseline_state, requirements)
        rno_response = rno.update(request, rno_state, requirements)

        assert np.allclose(rno_response.stresses, baseline_response.stresses, atol=1.0e-12)
        assert np.allclose(rno_response.tangents, baseline_response.tangents, atol=1.0e-9)
        assert np.allclose(rno_response.free_energy, baseline_response.free_energy, atol=1.0e-12)
        assert np.allclose(rno_response.dissipation, baseline_response.dissipation, atol=1.0e-10)
        assert np.allclose(
            rno_response.state.variables["plastic_strain"],
            baseline_response.state.variables["plastic_strain"],
            atol=1.0e-14,
        )
        assert np.allclose(
            rno_response.state.variables["equivalent_plastic_strain"],
            baseline_response.state.variables["equivalent_plastic_strain"],
            atol=1.0e-14,
        )
        baseline_state = baseline_response.state
        rno_state = rno_response.state
        q_history.append(float(rno_state.variables["equivalent_plastic_strain"][0]))

    assert np.all(np.diff(q_history) >= 0.0)


def test_rno_continuous_metadata_excludes_time_step_and_strain_rate() -> None:
    """Verify the continuous network reads state invariants rather than numerical rates."""
    _, rno = _build_models()
    input_names = tuple(field.name for field in rno.metadata.input_fields)

    assert "time_step" not in input_names
    assert "strain_rate" not in input_names
    assert rno.metadata.capabilities.time_representation == "continuous"
    assert rno.metadata.capabilities.supports_autodiff_tangent


def test_rno_uses_the_same_relative_yield_activation_as_j2_perzyna() -> None:
    """Verify the zero-correction RNO and baseline share the activation band."""
    baseline, rno = _build_models()
    shear_modulus = baseline.young_modulus / (2.0 * (1.0 + baseline.poisson_ratio))
    settings = MaterialUpdateSettings(
        max_iterations=50,
        yield_relative_tolerance=1.0e-3,
        residual_absolute_tolerance=1.0e-12,
        residual_relative_tolerance=1.0e-10,
    )
    for trial_overstress in (5.0e-3, 2.0e-2):
        strain = np.zeros((1, 6), dtype=np.float64)
        strain[0, 5] = (baseline.yield_stress + trial_overstress) / (
            np.sqrt(3.0) * shear_modulus
        )
        request = MaterialPointRequest(
            strains=strain,
            strain_rates=None,
            time_step=1.0e-3,
            kinematics="three_dimensional",
            update_settings=settings,
        )
        baseline_response = baseline.update(request, baseline.initialize_state(1))
        rno_response = rno.update(request, rno.initialize_state(1))

        assert np.array_equal(rno_response.stresses, baseline_response.stresses)
        assert np.array_equal(
            rno_response.diagnostics["plastic_multiplier"],
            baseline_response.diagnostics["plastic_multiplier"],
        )
        assert np.array_equal(
            rno_response.diagnostics["yield_activation_threshold"],
            baseline_response.diagnostics["yield_activation_threshold"],
        )


def test_rno_local_newton_failure_preserves_committed_state() -> None:
    """Verify an unconverged RNO step raises diagnostics without state mutation."""
    _, rno = _build_models()
    state = rno.initialize_state(1)
    snapshot = {name: values.copy() for name, values in state.variables.items()}
    strain = np.asarray([[0.004, -0.0015, -0.001, 0.002, -0.003, 0.02]])
    request = MaterialPointRequest(
        strains=strain,
        strain_rates=None,
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
        rno.update(request, state)

    assert error_info.value.reason == "maximum_iterations_exceeded"
    assert error_info.value.diagnostics["algorithm"] == "structured_j2_vp_rno_backward_euler"
    assert error_info.value.diagnostics["point_id"] == 0
    for name, values in state.variables.items():
        assert np.array_equal(values, snapshot[name])


def test_rno_embeds_in_hex8_material_assembly_with_consistent_tangent() -> None:
    """Verify the RNO satisfies the generic nonlinear solid material boundary."""
    _, rno = _build_models()
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.arange(8, dtype=np.int64).reshape(1, 8)
    displacement = np.zeros((8, 3), dtype=np.float64)
    displacement[:, 0] = 0.02 * nodes[:, 1]
    state = rno.initialize_state(8)

    assembly = assemble_hex8_material_response(
        nodes,
        elements,
        rno,
        state,
        displacement.reshape(-1),
        np.zeros(24, dtype=np.float64),
        1.0e-3,
        _request(np.zeros((1, 6))).update_settings,
        need_tangent=True,
    )

    assert np.all(np.isfinite(assembly.internal_force))
    assert np.all(np.isfinite(assembly.tangent.data))
    assert np.all(assembly.state.variables["equivalent_plastic_strain"] > 0.0)


def test_rno_state_dict_loader_preserves_tangent_capability(tmp_path) -> None:
    """Verify trained state dictionaries reload through the structured RNO contract."""
    _, source = _build_models()
    _, target = _build_models()
    model_path = tmp_path / "j2_vp_rno.pt"
    metadata_path = tmp_path / "j2_vp_rno_metadata.json"
    torch.save(source.state_dict(), model_path)
    save_surrogate_metadata(source.metadata, metadata_path)

    loaded = load_j2_vp_rno_state_dict(target, model_path, metadata_path)
    request = _request(np.asarray([[0.004, -0.0015, -0.001, 0.002, -0.003, 0.02]]))
    requirements = MaterialResponseRequirements(tangent=True)
    source_response = source.update(request, source.initialize_state(1), requirements)
    loaded_response = loaded.update(request, loaded.initialize_state(1), requirements)

    assert np.array_equal(loaded_response.stresses, source_response.stresses)
    assert np.array_equal(loaded_response.tangents, source_response.tangents)


def _build_models() -> tuple[J2ViscoplasticMaterial, StructuredJ2VPRNO]:
    """Build matching baseline and exact-zero-correction RNO models."""
    parameters = {
        "name": "j2_vp_rno_test",
        "density": 1.0,
        "young_modulus": 3339.9160679991464,
        "poisson_ratio": 0.36994096308414537,
        "yield_stress": 10.0,
        "hardening_modulus": 80.0,
        "time_scale": 0.02,
        "reference_stress": 10.0,
        "rate_exponent": 2.0,
    }
    config = J2VPRNOModelConfig(
        family="j2_vp_rno",
        time_representation="continuous",
        hidden_size=8,
        hidden_layers=1,
        activation="tanh",
        correction_scale=1.0,
        dtype="float64",
        device="cpu",
    )

    return (
        J2ViscoplasticMaterial(**parameters),
        StructuredJ2VPRNO(
            **parameters,
            equivalent_plastic_strain_scale=0.1,
            model_config=config,
        ),
    )


def _request(strain: np.ndarray) -> MaterialPointRequest:
    """Build one strict material-point update request."""
    return MaterialPointRequest(
        strains=strain,
        strain_rates=None,
        time_step=1.0e-3,
        kinematics="three_dimensional",
        update_settings=MaterialUpdateSettings(
            max_iterations=50,
            yield_relative_tolerance=1.0e-10,
            residual_absolute_tolerance=1.0e-12,
            residual_relative_tolerance=1.0e-10,
        ),
    )
