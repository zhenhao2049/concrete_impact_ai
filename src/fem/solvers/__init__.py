"""Finite element solver modules.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.solvers.data import (
    AdaptiveTimeStepSettings,
    ArmijoSettings,
    DirichletDofSet,
    DynamicSolution,
    FEMRunDef,
    LinearSolverSettings,
    MaterialDynamicSolution,
    NewtonSettings,
    NonlinearNewtonSettings,
    StaticSolution,
    SurrogateSettings,
    TimeIntegrationSettings,
)
from fem.solvers.line_search import (
    ArmijoLineSearchError,
    ArmijoResult,
    perform_armijo_search,
)
from fem.solvers.linear import solve_sparse_direct
from fem.solvers.newton import NewtonResult, solve_newton
from fem.solvers.progress import ProgressCallback, SolverProgressEvent
from fem.solvers.static import solve_linear_static
from fem.solvers.time_step import cut_back_time_step, grow_time_step

__all__ = [
    "AdaptiveTimeStepSettings",
    "ArmijoSettings",
    "ArmijoLineSearchError",
    "ArmijoResult",
    "DirichletDofSet",
    "DynamicSolution",
    "MaterialDynamicSolution",
    "FEMRunDef",
    "LinearSolverSettings",
    "NewtonResult",
    "NewtonSettings",
    "NonlinearNewtonSettings",
    "ProgressCallback",
    "SolverProgressEvent",
    "StaticSolution",
    "SurrogateSettings",
    "TimeIntegrationSettings",
    "solve_linear_static",
    "cut_back_time_step",
    "grow_time_step",
    "perform_armijo_search",
    "solve_newton",
    "solve_sparse_direct",
]
