"""Dynamic time integration modules.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.dynamics.material import solve_material_explicit, solve_material_implicit_newmark
from fem.dynamics.newmark import solve_explicit_central_difference, solve_implicit_newmark
from fem.dynamics.nonlinear import (
    build_newmark_effective_residual,
    build_newmark_effective_tangent,
    compute_newmark_acceleration,
    compute_newmark_velocity,
)
from fem.dynamics.stability import (
    StabilityReport,
    check_explicit_stability,
    compute_explicit_stability_report,
    compute_linear_elastic_pressure_wave_speed,
    estimate_minimum_edge_length,
)

__all__ = [
    "StabilityReport",
    "build_newmark_effective_residual",
    "build_newmark_effective_tangent",
    "check_explicit_stability",
    "compute_newmark_acceleration",
    "compute_newmark_velocity",
    "compute_explicit_stability_report",
    "compute_linear_elastic_pressure_wave_speed",
    "estimate_minimum_edge_length",
    "solve_explicit_central_difference",
    "solve_implicit_newmark",
    "solve_material_explicit",
    "solve_material_implicit_newmark",
]
