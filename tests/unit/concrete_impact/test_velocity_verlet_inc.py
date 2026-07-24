"""Tests for two-stage velocity-Verlet INC algebra and fixed-mesh MLP.

Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

import numpy as np
import pytest
import torch

from concrete_impact.nn.config import (
    VelocityVerletINCModelConfig,
    load_velocity_verlet_inc_data_config,
    load_velocity_verlet_inc_training_config,
)
from concrete_impact.nn.datasets.velocity_verlet_inc import (
    VelocityVerletINCPath,
    compute_inc_normalization,
)
from concrete_impact.nn.inc_dynamics import (
    advance_velocity_verlet_with_residuals,
    compute_velocity_verlet_defect_labels,
)
from concrete_impact.nn.models.velocity_verlet_inc import VelocityVerletINCMLP
from fem.dynamics.material import _validate_corrector_response
from fem.surrogates.data import VelocityVerletCorrectionResponse


def test_two_stage_labels_reconstruct_exact_oscillator_states() -> None:
    """Verify unique stage labels reproduce one analytical oscillator step."""
    mass = np.asarray([1.0], dtype=np.float64)
    stiffness = 100.0
    omega = np.sqrt(stiffness / mass[0])
    time_step = 0.18
    times = np.arange(21, dtype=np.float64) * time_step
    displacement = np.cos(omega * times)[:, None]
    velocity = (-omega * np.sin(omega * times))[:, None]
    acceleration = -stiffness * displacement / mass
    residual_start, residual_end = compute_velocity_verlet_defect_labels(
        displacement,
        velocity,
        acceleration,
        mass,
        time_step,
    )

    for step_id in range(times.size - 1):
        reconstructed_u, reconstructed_v = advance_velocity_verlet_with_residuals(
            displacement[step_id],
            velocity[step_id],
            acceleration[step_id],
            acceleration[step_id + 1],
            mass,
            residual_start[step_id],
            residual_end[step_id],
            time_step,
        )

        assert np.allclose(reconstructed_u, displacement[step_id + 1], rtol=1.0e-12)
        assert np.allclose(reconstructed_v, velocity[step_id + 1], rtol=1.0e-12)


def test_velocity_verlet_inc_mlp_has_two_float64_stage_heads() -> None:
    """Verify the fixed-mesh MLP returns two declared free-DOF fields."""
    config = VelocityVerletINCModelConfig(
        family="velocity_verlet_inc_mlp",
        time_representation="discrete",
        hidden_size=16,
        residual_blocks=2,
        activation="silu",
        dtype="float64",
        device="cpu",
    )
    model = VelocityVerletINCMLP(config, input_width=20, free_dof_count=3)

    start, end = model(torch.zeros((4, 20), dtype=torch.float64))

    assert start.shape == (4, 3)
    assert end.shape == (4, 3)
    assert start.dtype == torch.float64
    assert end.dtype == torch.float64


def test_output_normalization_matches_mass_dual_stage_scale() -> None:
    """Verify normalized residual coordinates reproduce the mass-dual loss."""
    mass = np.asarray([2.0, 8.0], dtype=np.float64)
    paths = tuple(
        _normalization_path(path_id, amplitude)
        for path_id, amplitude in enumerate((1.0, 2.0))
    )
    normalization = compute_inc_normalization(paths, mass)
    residual = paths[0].residual_force_start[0]
    scale = normalization.statistics["residual_force_start"]["scale"]
    normalized_norm = np.sum((residual / scale) ** 2)
    all_labels = np.concatenate([path.residual_force_start for path in paths], axis=0)
    dual_scale_squared = np.mean(np.sum(all_labels**2 / mass[None, :], axis=1))
    physical_ratio = np.sum(residual**2 / mass) / dual_scale_squared

    assert normalized_norm == pytest.approx(physical_ratio)


def test_corrector_response_rejects_nonfinite_stage_force() -> None:
    """Verify non-finite deployed residual forces terminate immediately."""
    response = VelocityVerletCorrectionResponse(
        residual_force_start_free=np.asarray([np.nan]),
        residual_force_end_free=np.asarray([0.0]),
    )

    with pytest.raises(FloatingPointError, match="non-finite residual"):
        _validate_corrector_response(response, 1)


def test_inc_reference_configs_lock_fixed_mesh_and_two_stage_training() -> None:
    """Verify the committed data and training requests match the first-model boundary."""
    data = load_velocity_verlet_inc_data_config(
        "configs/nn/velocity_verlet_inc_linear_rod_data.yaml"
    )
    training = load_velocity_verlet_inc_training_config(
        "configs/nn/velocity_verlet_inc_linear_rod_training.yaml"
    )

    assert data.model["mesh"]["divisions"] == [40, 1]
    assert data.fine_ratio == 8
    assert data.verification_ratio == 16
    assert len(data.controls) == 30
    assert training.model.family == "velocity_verlet_inc_mlp"
    assert training.rollout_steps == 16
    assert training.rollout_label_weight == pytest.approx(0.1)


def _normalization_path(path_id: int, amplitude: float) -> VelocityVerletINCPath:
    """Build one nondegenerate synthetic path for normalization tests."""
    time = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    base = amplitude * np.asarray([[1.0, -0.5], [1.5, 0.25], [2.0, 0.75]])
    return VelocityVerletINCPath(
        path_id=f"path_{path_id}",
        split="train",
        time=time,
        control_parameters=np.asarray([amplitude, 0.1 + 0.01 * path_id]),
        displacement=base,
        velocity=2.0 * base,
        baseline_acceleration=3.0 * base,
        internal_force=4.0 * base,
        external_force=5.0 * base,
        residual_force_start=6.0 * base[:-1],
        residual_force_end=7.0 * base[:-1],
        mechanical_energy=np.sum(base**2, axis=1),
    )
