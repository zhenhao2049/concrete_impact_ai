"""Accepted-step drivers for prescribed quasi-static RVE strain paths.

Contents:
    Increment iteration, accepted-step solution, and path-history assembly.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import MaterialUpdateSettings
from fem.rve.data import (
    RVEPathResult,
    RVERequest,
    RVEResponse,
    RVEState,
    StructuredHex8RVE,
)
from fem.rve.solver import (
    build_rve_operator_cache,
    initialize_rve_state,
    solve_rve_microequilibrium,
)
from fem.solvers.progress import ProgressCallback, SolverProgressEvent


@dataclass(frozen=True)
class AcceptedRVEPathStep:
    """Store one accepted prescribed-path step for streaming consumers."""

    step_id: int
    time: float
    time_step: float
    previous_state: RVEState
    response: RVEResponse


def iterate_prescribed_rve_path(
    model: StructuredHex8RVE,
    times: NDArray[np.float64],
    macro_strains: NDArray[np.float64],
    update_settings: MaterialUpdateSettings,
    progress_callback: ProgressCallback | None = None,
) -> Iterator[AcceptedRVEPathStep]:
    """Yield accepted RVE states without retaining the complete response history."""
    if times.ndim != 1 or macro_strains.shape != (times.size, 6):
        raise ValueError("Prescribed RVE path requires time (n,) and strain (n, 6).")
    time_steps = np.diff(times)
    if times.size < 2 or np.any(time_steps <= 0.0):
        raise ValueError("Prescribed RVE path requires at least two increasing times.")
    cache = build_rve_operator_cache(model)
    start_time = perf_counter()
    state = initialize_rve_state(model)
    for step_id, time_step in enumerate(time_steps, start=1):
        previous_state = state
        response = solve_rve_microequilibrium(
            model,
            RVERequest(
                macro_strain=macro_strains[step_id],
                time_step=float(time_step),
                material_update_settings=update_settings,
            ),
            state,
            cache,
        )
        state = response.state
        if progress_callback is not None:
            progress_callback(
                SolverProgressEvent(
                    stage="rve_path",
                    scheme="quasi_static_microequilibrium",
                    accepted_step=step_id,
                    nominal_total_steps=time_steps.size,
                    physical_time=float(times[step_id]),
                    final_time=float(times[-1]),
                    newton_iterations=int(response.diagnostics["newton_iterations"]),
                    armijo_backtracks=int(response.diagnostics["armijo_backtracks"]),
                    elapsed_seconds=perf_counter() - start_time,
                    residual_norm=float(response.diagnostics["micro_residual_norm"]),
                    residual_tolerance=float(
                        response.diagnostics["micro_residual_tolerance"]
                    ),
                )
            )
        yield AcceptedRVEPathStep(
            step_id=step_id,
            time=float(times[step_id]),
            time_step=float(time_step),
            previous_state=previous_state,
            response=response,
        )


def solve_prescribed_rve_path(
    model: StructuredHex8RVE,
    times: NDArray[np.float64],
    macro_strains: NDArray[np.float64],
    update_settings: MaterialUpdateSettings,
    progress_callback: ProgressCallback | None = None,
) -> RVEPathResult:
    """Solve and commit every accepted point of one prescribed strain path."""
    accepted = tuple(
        iterate_prescribed_rve_path(
            model,
            times,
            macro_strains,
            update_settings,
            progress_callback,
        )
    )
    return RVEPathResult(
        times=np.asarray([step.time for step in accepted], dtype=np.float64),
        responses=tuple(step.response for step in accepted),
    )
