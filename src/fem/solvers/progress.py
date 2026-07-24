"""Accepted-step progress events for long finite-element calculations.

Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SolverProgressEvent:
    """Describe one accepted physical step without exposing trial states."""

    stage: str
    scheme: str
    accepted_step: int
    nominal_total_steps: int
    physical_time: float
    final_time: float
    newton_iterations: int
    armijo_backtracks: int
    elapsed_seconds: float
    residual_norm: float | None = None
    residual_tolerance: float | None = None


ProgressCallback = Callable[[SolverProgressEvent], None]
