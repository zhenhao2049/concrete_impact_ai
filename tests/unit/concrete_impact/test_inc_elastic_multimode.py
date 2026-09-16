"""Tests for the fixed-system elastic multimode INC delivery model.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from pathlib import Path

import numpy as np
import torch

from concrete_impact.nn.config_inc_elastic import load_inc_elastic_multimode_config
from concrete_impact.nn.datasets.inc_elastic_multimode import (
    INCElasticNormalization,
    build_inc_elastic_features,
    elastic_inc_feature_layout,
    elastic_inc_input_width,
)
from concrete_impact.nn.evaluation.inc_elastic_multimode import (
    _passes_kinematic_delivery,
    kinematic_validation_score,
    validation_score,
)
from concrete_impact.nn.models.inc_elastic_multimode import INCElasticMultimodeMLP

CONFIG_PATH = Path("configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml")


def test_delivery_multimode_paths_are_disjoint_and_fixed() -> None:
    """Require the exact half-sine 12/2/2 path ownership."""
    config = load_inc_elastic_multimode_config(CONFIG_PATH)
    all_paths = config.data.train_paths + config.data.validation_paths + config.data.test_paths

    assert len(config.data.train_paths) == 12
    assert len(config.data.validation_paths) == 2
    assert len(config.data.test_paths) == 2
    assert len(set(all_paths)) == 16
    assert config.data.source_split_policy == "training_pool_resplit"
    assert all("_w1_" in path_id for path_id in all_paths)
    assert {path_id.split("_")[1] for path_id in config.data.train_paths} == {
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    }
    assert {path_id.split("_")[1] for path_id in config.data.validation_paths} == {"s7"}
    assert {path_id.split("_")[1] for path_id in config.data.test_paths} == {"s8"}
    assert {path_id.split("_")[3] for path_id in all_paths} == {"d3", "d4"}


def test_feature_map_has_reviewed_808_field_order() -> None:
    """Require eight loading features followed by 400 sine and cosine pairs."""
    layout = elastic_inc_feature_layout(400)

    assert elastic_inc_input_width(400) == 808
    assert layout["modal_sine"] == {"start": 8, "count": 400}
    assert layout["modal_cosine"] == {"start": 408, "count": 400}


def test_feature_map_is_pressure_independent_and_finite() -> None:
    """Verify linear pressure scaling is external to the neural input."""
    normalization = INCElasticNormalization(
        duration_mean=2.0,
        duration_scale=0.5,
        residual_start_scale=np.ones(3),
        residual_end_scale=np.ones(3),
        displacement_scale=1.0,
        velocity_scale=1.0,
        energy_scale=1.0,
    )
    features = build_inc_elastic_features(
        np.asarray((0.0, 0.5, 1.0)),
        time_step=0.5,
        pulse_duration=2.0,
        spatial_coefficients=np.asarray((1.0, 0.0, 0.0, 0.0)),
        angular_frequencies=np.asarray((1.0, 2.0, 3.0)),
        normalization=normalization,
        final_time=1.5,
    )

    assert features.shape == (3, 14)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    assert np.array_equal(features[:, 2:6], np.asarray(((1.0, 0.0, 0.0, 0.0),) * 3))


def test_duration_modal_phase_extends_the_feature_map_deterministically() -> None:
    """Expose pulse-duration phase for every fixed-system mode."""
    normalization = INCElasticNormalization(
        duration_mean=2.0,
        duration_scale=0.5,
        residual_start_scale=np.ones(3),
        residual_end_scale=np.ones(3),
        displacement_scale=1.0,
        velocity_scale=1.0,
        energy_scale=1.0,
    )
    frequencies = np.asarray((1.0, 2.0, 3.0))
    features = build_inc_elastic_features(
        np.asarray((0.0, 0.5)),
        time_step=0.5,
        pulse_duration=2.0,
        spatial_coefficients=np.asarray((1.0, 0.0, 0.0, 0.0)),
        angular_frequencies=frequencies,
        normalization=normalization,
        final_time=1.5,
        include_duration_modal_phase=True,
    )
    layout = elastic_inc_feature_layout(3, include_duration_modal_phase=True)

    assert elastic_inc_input_width(3, include_duration_modal_phase=True) == 20
    assert features.shape == (2, 20)
    assert layout["duration_modal_sine"] == {"start": 14, "count": 3}
    assert layout["duration_modal_cosine"] == {"start": 17, "count": 3}
    assert np.allclose(features[:, 14:17], np.sin(2.0 * frequencies)[None, :])
    assert np.allclose(features[:, 17:20], np.cos(2.0 * frequencies)[None, :])


def test_direct_model_has_two_full_width_float32_heads() -> None:
    """Require two 400-DOF outputs without a hidden output-rank bottleneck."""
    config = load_inc_elastic_multimode_config(CONFIG_PATH)
    cpu_model_config = config.model.model_copy(update={"device": "cpu"})
    model = INCElasticMultimodeMLP(cpu_model_config, input_width=808)

    start, end = model(torch.zeros((2, 808), dtype=torch.float32))

    assert model.config.hidden_size >= model.config.free_dof_count
    assert start.shape == (2, 400)
    assert end.shape == (2, 400)
    assert start.dtype == torch.float32
    assert end.dtype == torch.float32


def test_loss_and_execution_policy_are_fixed_before_test_access() -> None:
    """Require the selected precision, loss, sampling, and validation schedule."""
    config = load_inc_elastic_multimode_config(CONFIG_PATH)

    assert config.data.sample_stride == 2
    assert config.model.dtype == "float32"
    assert config.execution.fem_dtype == "float64"
    assert config.loss.label_weight == 1.0
    assert config.loss.step_weight == 1.0
    assert config.loss.energy_weight == 0.01
    assert config.loss.displacement_component_weight == 1.5
    assert config.loss.velocity_component_weight == 0.5
    assert config.training.epochs == 200
    assert config.training.full_validation_interval == 10


def test_checkpoint_score_remains_finite_for_zero_baseline_channel_errors() -> None:
    """Keep hard channel gates separate from continuous checkpoint selection."""
    config = load_inc_elastic_multimode_config(CONFIG_PATH)
    record = {
        "label_mass_dual_nrmse": 0.1,
        "displacement_error_ratio": 0.35,
        "velocity_error_ratio": 0.35,
        "energy_error_ratio": 0.4,
        "channels": {
            "right_transverse_displacement": {
                "baseline_arrival_error": 0.0,
                "baseline_peak_error": 0.0,
                "baseline_phase_error_steps": 0,
                "inc_arrival_error": 1.0,
                "inc_peak_error": 1.0,
                "inc_phase_error_steps": 1,
            }
        },
    }

    score = validation_score(record, config)

    assert np.isfinite(score)
    assert score == 0.5


def test_kinematic_score_uses_only_displacement_and_velocity_ratios() -> None:
    """Select a response checkpoint without weakening full acceptance."""
    config = load_inc_elastic_multimode_config(CONFIG_PATH)
    record = {
        "displacement_error_ratio": 0.35,
        "velocity_error_ratio": 0.21,
        "energy_error_ratio": 100.0,
    }

    score = kinematic_validation_score(record, config)

    assert score == 0.5


def test_delivery_requires_both_kinematic_ratios_below_coarse_baseline() -> None:
    """Require strict displacement and velocity improvement on independent paths."""
    assert _passes_kinematic_delivery(
        {"displacement_error_ratio": 0.99, "velocity_error_ratio": 0.75}
    )
    assert not _passes_kinematic_delivery(
        {"displacement_error_ratio": 1.0, "velocity_error_ratio": 0.75}
    )
    assert not _passes_kinematic_delivery(
        {"displacement_error_ratio": 0.75, "velocity_error_ratio": 1.01}
    )
