"""High-fidelity FE2 path-comparison and selection metrics.

Contents:
    Common-time matching, response errors, physical checks, and scheme selection.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FE2PathScales:
    """Store fixed physical scales for zero-safe history comparisons."""

    displacement: float
    velocity: float
    strain: float
    stress: float
    equivalent_plastic_strain: float
    dissipation: float
    incremental_work_density: float
    viscoplastic_active_volume_fraction: float


@dataclass(frozen=True)
class FE2PathTolerances:
    """Store strict path and event-time acceptance thresholds."""

    displacement: float = 0.02
    velocity: float = 0.02
    strain: float = 0.02
    stress: float = 0.05
    equivalent_plastic_strain: float = 0.05
    dissipation: float = 0.05
    incremental_work_density: float = 0.05
    viscoplastic_active_volume_fraction: float = 0.05


def require_transverse_macro_resolution(divisions: tuple[int, int, int]) -> None:
    """Require at least two macro elements in both transverse directions."""
    if divisions[1] < 2 or divisions[2] < 2:
        raise ValueError(
            "High-fidelity FE2 macro mesh requires at least two elements in both "
            f"transverse directions: divisions={divisions}."
        )


def exact_common_time_indices(
    fine_times: NDArray[np.float64],
    coarse_times: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return nested-grid indices without interpolating accepted physical states."""
    if fine_times.ndim != 1 or coarse_times.ndim != 1:
        raise ValueError("FE2 time-grid comparison requires one-dimensional time arrays.")
    if fine_times.size < 2 or coarse_times.size < 2:
        raise ValueError("FE2 time-grid comparison requires at least two time points.")
    if np.any(np.diff(fine_times) <= 0.0) or np.any(np.diff(coarse_times) <= 0.0):
        raise ValueError("FE2 comparison time coordinates must be strictly increasing.")
    fine_intervals = fine_times.size - 1
    coarse_intervals = coarse_times.size - 1
    if fine_intervals % coarse_intervals != 0:
        raise ValueError("FE2 histories do not have an integer nested step ratio.")
    step_ratio = fine_intervals // coarse_intervals
    indices = np.arange(coarse_times.size, dtype=np.int64) * step_ratio
    scale = max(1.0, abs(float(fine_times[-1])), abs(float(coarse_times[-1])))
    tolerance = np.finfo(np.float64).eps * scale * 32.0
    if not np.allclose(fine_times[indices], coarse_times, rtol=0.0, atol=tolerance):
        raise ValueError("FE2 histories do not share an exact nested time grid.")
    return indices


def fixed_scale_history_error(
    values_a: NDArray[np.float64],
    values_b: NDArray[np.float64],
    physical_scale: float,
) -> float:
    """Compute a maximum-in-time norm error with a fixed physical lower scale."""
    if values_a.shape != values_b.shape:
        raise ValueError(
            "FE2 history comparison requires identical array shapes: "
            f"a={values_a.shape}, b={values_b.shape}."
        )
    if physical_scale <= 0.0:
        raise ValueError("FE2 history comparison scale must be strictly positive.")
    flattened_a = values_a.reshape(values_a.shape[0], -1)
    flattened_b = values_b.reshape(values_b.shape[0], -1)
    difference = np.linalg.norm(flattened_a - flattened_b, axis=1)
    norm_a = np.linalg.norm(flattened_a, axis=1)
    norm_b = np.linalg.norm(flattened_b, axis=1)
    denominator = max(physical_scale, float(np.max(norm_a)), float(np.max(norm_b)))
    return float(np.max(difference) / denominator)


def evaluate_path_acceptance(
    metrics: dict[str, float],
    tolerances: FE2PathTolerances,
    event_time_tolerance: float,
) -> dict[str, bool]:
    """Evaluate FE2 history metrics without modifying tolerances or data."""
    return {
        "displacement": metrics["displacement_error"] <= tolerances.displacement,
        "velocity": metrics["velocity_error"] <= tolerances.velocity,
        "strain": metrics["strain_error"] <= tolerances.strain,
        "stress": metrics["stress_error"] <= tolerances.stress,
        "equivalent_plastic_strain": (
            metrics["equivalent_plastic_strain_error"]
            <= tolerances.equivalent_plastic_strain
        ),
        "dissipation": metrics["dissipation_error"] <= tolerances.dissipation,
        "incremental_work_density": (
            metrics["incremental_work_density_error"]
            <= tolerances.incremental_work_density
        ),
        "viscoplastic_active_volume_fraction": (
            metrics["viscoplastic_active_volume_fraction_error"]
            <= tolerances.viscoplastic_active_volume_fraction
        ),
        "event_time": metrics["maximum_event_time_error"] <= event_time_tolerance,
    }


def conservative_speed_ratio(
    slow_samples: NDArray[np.float64],
    fast_samples: NDArray[np.float64],
) -> float:
    """Compute the median-MAD conservative speed ratio from repeated wall times."""
    if slow_samples.size == 0 or fast_samples.size == 0:
        raise ValueError("FE2 performance comparison requires nonempty timing samples.")
    if np.any(slow_samples <= 0.0) or np.any(fast_samples <= 0.0):
        raise ValueError("FE2 performance samples must be strictly positive.")
    slow_median = float(np.median(slow_samples))
    fast_median = float(np.median(fast_samples))
    slow_mad = float(np.median(np.abs(slow_samples - slow_median)))
    fast_mad = float(np.median(np.abs(fast_samples - fast_median)))
    return (slow_median - slow_mad) / (fast_median + fast_mad)


def select_implicit_candidate(
    candidates: tuple[dict[str, Any], ...],
    conservative_speedup_target: float,
) -> dict[str, Any]:
    """Select the largest accurate implicit step and classify production eligibility."""
    passing = [candidate for candidate in candidates if bool(candidate["path_passed"])]
    if not passing:
        return {
            "selected_scheme": "explicit_fine",
            "implicit_candidate": None,
            "implicit_production_eligible": False,
            "reason": "no_implicit_candidate_passed_path_acceptance",
        }
    selected = max(passing, key=lambda candidate: float(candidate["time_step"]))
    eligible = float(selected["conservative_speed_ratio"]) > conservative_speedup_target
    return {
        "selected_scheme": "implicit" if eligible else "explicit_fine",
        "implicit_candidate": str(selected["name"]),
        "implicit_production_eligible": eligible,
        "reason": (
            "implicit_path_consistent_and_stably_faster"
            if eligible
            else "implicit_path_consistent_without_stable_speedup"
        ),
    }


def select_accepted_snapshot_indices(
    equivalent_plastic_strain: NDArray[np.float64],
    equivalent_stress: NDArray[np.float64],
    incremental_work: NDArray[np.float64],
    plastic_threshold: float,
) -> dict[str, int]:
    """Select accepted event states without interpolation."""
    if not (
        equivalent_plastic_strain.shape
        == equivalent_stress.shape
        == incremental_work.shape
    ):
        raise ValueError("FE2 snapshot event histories must have identical shapes.")
    yield_ids = np.flatnonzero(equivalent_plastic_strain >= plastic_threshold)
    if yield_ids.size == 0:
        raise ValueError("FE2 snapshot history contains no accepted first-yield state.")
    return {
        "first_yield": int(yield_ids[0]),
        "peak_stress": int(np.argmax(equivalent_stress)),
        "maximum_negative_work": int(np.argmin(incremental_work)),
        "final": int(equivalent_stress.size - 1),
    }
