"""Residual-merit Armijo line search for nonlinear solvers.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fem.solvers.data import ArmijoSettings

ResidualEvaluator = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True)
class ArmijoResult:
    """Store one accepted line-search step."""

    step_length: float
    backtracks: int
    residual_norm: float


class ArmijoLineSearchError(RuntimeError):
    """Report exhaustion of the configured Armijo sequence."""

    def __init__(self, diagnostics: dict[str, float | int]) -> None:
        """Initialize one line-search failure."""
        super().__init__("Armijo line search exhausted all configured candidates.")
        self.diagnostics = diagnostics


def perform_armijo_search(
    current_residual: NDArray[np.float64],
    evaluate_residual: ResidualEvaluator,
    settings: ArmijoSettings,
) -> ArmijoResult:
    """Select the first candidate satisfying residual-merit sufficient decrease."""
    residual_norm = float(np.linalg.norm(current_residual))
    current_merit = 0.5 * residual_norm**2
    last_candidate_norm = np.inf
    for backtrack in range(settings.max_backtracks + 1):
        step_length = settings.reduction_factor**backtrack
        candidate_residual = evaluate_residual(step_length)
        last_candidate_norm = float(np.linalg.norm(candidate_residual))
        candidate_merit = 0.5 * last_candidate_norm**2
        armijo_bound = current_merit - (
            settings.sufficient_decrease * step_length * residual_norm**2
        )
        if candidate_merit <= armijo_bound:
            return ArmijoResult(step_length, backtrack, last_candidate_norm)

    raise ArmijoLineSearchError(
        {
            "current_residual_norm": residual_norm,
            "current_merit": current_merit,
            "last_candidate_residual_norm": last_candidate_norm,
            "max_backtracks": settings.max_backtracks,
        }
    )
