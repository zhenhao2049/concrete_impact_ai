"""Newton nonlinear iteration framework.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from fem.solvers.data import NewtonSettings
from fem.solvers.linear import solve_sparse_direct

ResidualFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]
TangentFunction = Callable[[NDArray[np.float64]], csr_matrix]


@dataclass(frozen=True)
class NewtonResult:
    """Store one Newton solve result."""

    solution: NDArray[np.float64]
    iterations: int
    residual_norm: float
    increment_norm: float


def solve_newton(
    initial_solution: NDArray[np.float64],
    residual_function: ResidualFunction,
    tangent_function: TangentFunction,
    settings: NewtonSettings,
) -> NewtonResult:
    """Solve a nonlinear residual equation with Newton iteration."""
    solution = initial_solution.copy()
    residual = residual_function(solution)
    residual_norm = float(np.linalg.norm(residual))
    increment_norm = np.inf

    for iteration in range(1, settings.max_iterations + 1):
        tangent = tangent_function(solution)
        increment = solve_sparse_direct(tangent, -residual)
        solution += increment

        residual = residual_function(solution)
        residual_norm = float(np.linalg.norm(residual))
        increment_norm = float(np.linalg.norm(increment))

        if (
            residual_norm <= settings.residual_tolerance
            and increment_norm <= settings.increment_tolerance
        ):
            return NewtonResult(
                solution=solution,
                iterations=iteration,
                residual_norm=residual_norm,
                increment_norm=increment_norm,
            )

    message = (
        "Newton iteration failed to converge: "
        f"residual={residual_norm:.6e}, increment={increment_norm:.6e}."
    )
    raise RuntimeError(message)
