"""Tests for Armijo and convergence-driven time-step controls.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

import numpy as np
import pytest

from fem.solvers import (
    AdaptiveTimeStepSettings,
    ArmijoLineSearchError,
    ArmijoSettings,
    cut_back_time_step,
    grow_time_step,
    perform_armijo_search,
)


def test_armijo_accepts_full_newton_step() -> None:
    """Verify a sufficient full step is accepted without backtracking."""
    result = perform_armijo_search(
        np.asarray([1.0]),
        lambda step_length: np.asarray([1.0 - step_length]),
        ArmijoSettings(enabled=True),
    )

    assert result.step_length == 1.0
    assert result.backtracks == 0
    assert result.residual_norm == 0.0


def test_armijo_backtracks_to_first_sufficient_candidate() -> None:
    """Verify rejected candidates are not accepted as a minimum-step fallback."""
    result = perform_armijo_search(
        np.asarray([1.0]),
        lambda step_length: np.asarray([1.0 - 4.0 * step_length]),
        ArmijoSettings(enabled=True),
    )

    assert result.step_length == 0.25
    assert result.backtracks == 2
    assert result.residual_norm == 0.0


def test_armijo_exhaustion_raises_with_diagnostics() -> None:
    """Verify an exhausted line search terminates instead of accepting a candidate."""
    with pytest.raises(ArmijoLineSearchError) as error_info:
        perform_armijo_search(
            np.asarray([1.0]),
            lambda step_length: np.asarray([1.0]),
            ArmijoSettings(enabled=True, max_backtracks=2),
        )

    assert error_info.value.diagnostics["max_backtracks"] == 2
    assert error_info.value.diagnostics["last_candidate_residual_norm"] == 1.0


def test_convergence_time_step_rules_respect_configured_bounds() -> None:
    """Verify deterministic shrink and growth rules use the initial-step bounds."""
    settings = AdaptiveTimeStepSettings(enabled=True)

    assert cut_back_time_step(1.0, 1.0, settings) == 0.5
    assert cut_back_time_step(0.0625, 1.0, settings) == 0.0625
    assert grow_time_step(1.0, 1.0, settings) == 2.0
    assert grow_time_step(2.0, 1.0, settings) == 2.0
