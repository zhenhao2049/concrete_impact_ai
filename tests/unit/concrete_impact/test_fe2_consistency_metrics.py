"""Tests for strict FE2 path, time-grid, and selection metrics.

Contents:
    Time-grid tolerance, response error, and selection-contract tests.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

import numpy as np
import pytest

from concrete_impact.benchmarks.fe2_consistency import (
    FE2PathTolerances,
    conservative_speed_ratio,
    evaluate_path_acceptance,
    exact_common_time_indices,
    fixed_scale_history_error,
    require_transverse_macro_resolution,
    select_accepted_snapshot_indices,
    select_implicit_candidate,
)


def test_exact_common_time_indices_reject_interpolation() -> None:
    """Verify only strictly nested accepted time grids may be compared."""
    fine = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4])
    coarse = np.asarray([0.0, 0.2, 0.4])
    np.testing.assert_array_equal(exact_common_time_indices(fine, coarse), [0, 2, 4])

    with pytest.raises(ValueError, match="exact nested"):
        exact_common_time_indices(fine, np.asarray([0.0, 0.25, 0.4]))


def test_exact_common_time_indices_accepts_machine_roundoff_from_accumulation() -> None:
    """Accept identical physical steps formed by multiplication and accumulation."""
    fine = np.arange(81, dtype=np.float64) * 1.0e-3
    coarse = np.empty(81, dtype=np.float64)
    coarse[0] = 0.0
    for index in range(1, coarse.size):
        coarse[index] = coarse[index - 1] + 1.0e-3

    indices = exact_common_time_indices(fine, coarse)

    np.testing.assert_array_equal(indices, np.arange(81, dtype=np.int64))


def test_fixed_scale_error_remains_finite_during_zero_response() -> None:
    """Verify unloading through zero cannot singularize the history metric."""
    reference = np.zeros((3, 2))
    candidate = np.asarray([[0.0, 0.0], [1.0e-3, 0.0], [0.0, 0.0]])

    assert fixed_scale_history_error(reference, candidate, 1.0e-2) == 0.1


def test_path_acceptance_includes_work_and_active_fraction_histories() -> None:
    """Verify loading diagnostics are mandatory path-consistency quantities."""
    metrics = {
        "displacement_error": 0.0,
        "velocity_error": 0.0,
        "strain_error": 0.0,
        "stress_error": 0.0,
        "equivalent_plastic_strain_error": 0.0,
        "dissipation_error": 0.0,
        "incremental_work_density_error": 0.051,
        "viscoplastic_active_volume_fraction_error": 0.049,
        "maximum_event_time_error": 0.0,
    }

    acceptance = evaluate_path_acceptance(metrics, FE2PathTolerances(), 1.0e-3)

    assert not acceptance["incremental_work_density"]
    assert acceptance["viscoplastic_active_volume_fraction"]


def test_high_fidelity_macro_mesh_requires_two_transverse_elements() -> None:
    """Verify both transverse directions retain spatial resolution."""
    require_transverse_macro_resolution((4, 2, 2))
    with pytest.raises(ValueError, match="transverse"):
        require_transverse_macro_resolution((4, 2, 1))


def test_implicit_selection_requires_path_acceptance_and_stable_speedup() -> None:
    """Verify accuracy precedes the conservative performance decision."""
    candidates = (
        {
            "name": "coarse",
            "time_step": 0.2,
            "path_passed": True,
            "conservative_speed_ratio": 1.3,
        },
        {
            "name": "fine",
            "time_step": 0.1,
            "path_passed": True,
            "conservative_speed_ratio": 2.0,
        },
    )

    decision = select_implicit_candidate(candidates, 1.2)

    assert decision["implicit_candidate"] == "coarse"
    assert decision["selected_scheme"] == "implicit"
    np.testing.assert_allclose(
        conservative_speed_ratio(np.asarray([10.0, 10.2, 9.8]), np.asarray([5.0, 5.1, 4.9])),
        9.8 / 5.1,
    )


def test_snapshot_indices_are_accepted_history_indices() -> None:
    """Verify four event states are selected without temporal interpolation."""
    events = select_accepted_snapshot_indices(
        np.asarray([0.0, 0.0, 0.1, 0.2]),
        np.asarray([0.0, 2.0, 5.0, 3.0]),
        np.asarray([0.0, 1.0, -2.0, -1.0]),
        1.0e-6,
    )

    assert events == {
        "first_yield": 2,
        "peak_stress": 2,
        "maximum_negative_work": 2,
        "final": 3,
    }
