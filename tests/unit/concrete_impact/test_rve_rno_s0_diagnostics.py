"""Unit tests for deterministic RVE-RNO S0 diagnostic primitives.

Author: Zhen Hao.
Created: 2026-07-19.
"""

from __future__ import annotations

import numpy as np

from concrete_impact.nn.datasets.rve import RNOPathSample
from concrete_impact.nn.evaluation.rve_rno_s0 import (
    coarsen_rno_path,
    whole_history_relative_stress_error,
)


def test_whole_history_relative_stress_error_uses_training_scale_floor() -> None:
    """Verify the zero-reference error remains finite and scale normalized."""
    reference = np.zeros((2, 6), dtype=np.float64)
    predicted = np.ones((2, 6), dtype=np.float64)
    scale = np.full(6, 2.0, dtype=np.float64)
    assert np.isclose(
        whole_history_relative_stress_error(reference, predicted, scale),
        0.5,
    )


def test_coarsen_rno_path_sums_time_steps_and_selects_endpoints() -> None:
    """Verify blockwise Forward Euler coarsening preserves endpoint labels."""
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    sample = RNOPathSample(
        path_id="response.h5:path",
        time=np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float64),
        time_step=np.asarray((0.1, 0.1, 0.1, 0.1), dtype=np.float64),
        macro_strain=values,
        macro_stress=2.0 * values,
        material_parameters=np.asarray((), dtype=np.float64),
        microstructure_features=np.asarray((), dtype=np.float64),
        dissipation_density=np.arange(4, dtype=np.float64),
    )
    coarse = coarsen_rno_path(sample, factor=2)
    np.testing.assert_allclose(coarse.time, (0.2, 0.4))
    np.testing.assert_allclose(coarse.time_step, (0.2, 0.2))
    np.testing.assert_allclose(coarse.macro_strain, values[[1, 3]])
    np.testing.assert_allclose(coarse.macro_stress, 2.0 * values[[1, 3]])
    np.testing.assert_allclose(coarse.dissipation_density, (1.0, 3.0))
