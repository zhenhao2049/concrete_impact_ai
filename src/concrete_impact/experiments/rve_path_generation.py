"""Deterministic control-path generators for RVE-RNO data sources.

Contents:
    Legacy paths and fixed-cell production strain histories with analytic timing.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from concrete_impact.experiments.rve_data_plan import DataSourcePlan, RVEDataTask

TENSOR_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class GeneratedControlPath:
    """Store one analytic-time-grid control history for a data task."""

    task_id: str
    control_kind: str
    times: NDArray[np.float64]
    values: NDArray[np.float64]
    family: str


def generate_data_control_path(
    source: DataSourcePlan,
    task: RVEDataTask,
) -> GeneratedControlPath:
    """Generate one deterministic strain or pressure history from its task seed."""
    load_case = next(case for case in source.load_cases if case.name == task.load_case_name)
    if source.production_design is not None:
        return _generate_fixed_cell_production_path(source, task)
    family = load_case.path_families[task.path_index % len(load_case.path_families)]
    times = np.linspace(0.0, load_case.duration, load_case.time_points, dtype=np.float64)
    if source.kind == "impact_path":
        normalized_time = times / load_case.duration
        values = load_case.peak_scale * np.sin(np.pi * normalized_time)
        return GeneratedControlPath(task.task_id, "pressure", times, values[:, None], family)
    rng = np.random.default_rng(task.seed)
    direction = rng.normal(size=6)
    direction /= np.linalg.norm(direction)
    strain = _generate_strain_family(times, load_case.peak_scale, direction, family, rng)
    return GeneratedControlPath(task.task_id, "macro_strain", times, strain, family)


def _generate_fixed_cell_production_path(
    source: DataSourcePlan,
    task: RVEDataTask,
) -> GeneratedControlPath:
    """Generate one analytically timed path from the frozen factorial design."""
    design = source.production_design
    required = (
        task.family,
        task.tensor_direction,
        task.amplitude,
        task.target_strain_rate,
        task.accepted_increment_count,
    )
    if design is None or any(value is None for value in required):
        raise ValueError(
            "Fixed-cell production task is missing design fields: "
            f"task_id={task.task_id}, fields={required}."
        )
    family = str(task.family)
    rng = np.random.default_rng(task.seed)
    direction = _production_tensor_direction(str(task.tensor_direction))
    direction = _perturb_tensor_direction(
        direction,
        design.direction_perturbation,
        rng,
    )
    knots = _production_path_knots(
        direction,
        family,
        task.turning_angle,
        rng,
    )
    knot_norms = _tensor_norm(knots)
    normalized_knots = knots / float(np.max(knot_norms))
    amplitude = float(task.amplitude)
    target_rate = float(task.target_strain_rate)
    maximum_normalized_rate = _maximum_smoothstep_rate(normalized_knots)
    duration = amplitude * maximum_normalized_rate / target_rate
    times = np.linspace(
        0.0,
        duration,
        int(task.accepted_increment_count) + 1,
        dtype=np.float64,
    )
    normalized_time = times / duration
    strain = amplitude * _smooth_vector_interpolation(normalized_time, normalized_knots)
    return GeneratedControlPath(task.task_id, "macro_strain", times, strain, family)


def _production_tensor_direction(name: str) -> NDArray[np.float64]:
    """Return one canonical engineering-Voigt tensor direction."""
    directions = {
        "deviatoric_uniaxial": np.asarray([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
        "deviatoric_biaxial": np.asarray([0.5, 0.5, -1.0, 0.0, 0.0, 0.0]),
        "shear_dominant": np.asarray([0.1, -0.1, 0.0, 1.0, 0.25, 0.0]),
        "positive_triaxial": np.asarray([1.0, 0.35, 0.35, 0.15, 0.0, 0.0]),
        "negative_triaxial": np.asarray([-1.0, -0.35, -0.35, 0.15, 0.0, 0.0]),
    }
    return _normalize_tensor(directions[name])


def _perturb_tensor_direction(
    direction: NDArray[np.float64],
    magnitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply a deterministic weighted-orthogonal direction perturbation."""
    perturbation = rng.normal(size=6)
    perturbation -= _tensor_inner(perturbation, direction) * direction
    perturbation = _normalize_tensor(perturbation)
    return _normalize_tensor(direction + magnitude * perturbation)


def _production_path_knots(
    direction: NDArray[np.float64],
    family: str,
    turning_angle: float | None,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Construct vector knots for one production loading family."""
    if family == "nonproportional":
        if turning_angle is None:
            raise ValueError("Nonproportional production path requires a turning angle.")
        orthogonal = rng.normal(size=6)
        orthogonal -= _tensor_inner(orthogonal, direction) * direction
        orthogonal = _normalize_tensor(orthogonal)
        rotated = (
            np.cos(turning_angle) * direction
            + np.sin(turning_angle) * orthogonal
        )
        return np.stack(
            (
                np.zeros(6, dtype=np.float64),
                direction,
                direction + 0.75 * rotated,
            )
        )
    controls = {
        "monotonic": np.asarray([0.0, 1.0]),
        "load_unload_reload": np.asarray([0.0, 1.0, 0.25, 1.2]),
        "reverse": np.asarray([0.0, 1.0, -1.0, 0.5]),
        "hold": np.asarray([0.0, 1.0, 1.0]),
    }[family]
    return controls[:, None] * direction[None, :]


def _maximum_smoothstep_rate(knots: NDArray[np.float64]) -> float:
    """Compute the exact maximum normalized-time rate of piecewise smoothstep."""
    segment_count = knots.shape[0] - 1
    increments = np.diff(knots, axis=0)
    return float(1.5 * segment_count * np.max(_tensor_norm(increments)))


def _smooth_vector_interpolation(
    normalized_time: NDArray[np.float64],
    knots: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate vector knots with a cubic smoothstep on each segment."""
    segment_coordinate = normalized_time * (knots.shape[0] - 1)
    segment = np.minimum(segment_coordinate.astype(np.int64), knots.shape[0] - 2)
    local = segment_coordinate - segment
    smooth = local**2 * (3.0 - 2.0 * local)
    return knots[segment] + smooth[:, None] * (knots[segment + 1] - knots[segment])


def _tensor_inner(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> float:
    """Evaluate the engineering-tensor weighted inner product."""
    return float(np.sum(first * second * TENSOR_WEIGHTS))


def _tensor_norm(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate engineering-tensor norms along the final component axis."""
    return np.sqrt(np.sum(values**2 * TENSOR_WEIGHTS, axis=-1))


def _normalize_tensor(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize one engineering-Voigt tensor direction."""
    return values / float(_tensor_norm(values))


def _generate_strain_family(
    times: NDArray[np.float64],
    amplitude: float,
    direction: NDArray[np.float64],
    family: str,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Evaluate one explicitly selected smooth piecewise strain family."""
    controls = {
        "monotonic": np.asarray([0.0, 1.0]),
        "load_unload_reload": np.asarray([0.0, 1.0, 0.25, 1.2]),
        "reverse": np.asarray([0.0, 1.0, -1.0, 0.5]),
        "hold": np.asarray([0.0, 1.0, 1.0]),
        "nonproportional": np.asarray([0.0, 1.0, 0.5, 1.1]),
    }[family]
    normalized_time = (times - times[0]) / (times[-1] - times[0])
    scalar = _smooth_control_interpolation(normalized_time, controls)
    result = amplitude * scalar[:, None] * direction[None, :]
    if family == "nonproportional":
        second_direction = rng.normal(size=6)
        second_direction -= direction * float(second_direction @ direction)
        second_direction /= np.linalg.norm(second_direction)
        rotation = np.sin(np.pi * normalized_time)[:, None]
        result += 0.5 * amplitude * rotation * second_direction[None, :]
    return result


def _smooth_control_interpolation(
    normalized_time: NDArray[np.float64],
    controls: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate controls with a fixed cubic smoothstep on every segment."""
    segment_coordinate = normalized_time * (controls.size - 1)
    segment = np.minimum(segment_coordinate.astype(np.int64), controls.size - 2)
    local = segment_coordinate - segment
    smooth = local**2 * (3.0 - 2.0 * local)
    return controls[segment] + smooth * (controls[segment + 1] - controls[segment])
