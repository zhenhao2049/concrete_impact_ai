"""Convergence-driven implicit time-step update rules.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from fem.solvers.data import AdaptiveTimeStepSettings


def cut_back_time_step(
    current_time_step: float,
    initial_time_step: float,
    settings: AdaptiveTimeStepSettings,
) -> float:
    """Reduce a rejected step without crossing the configured minimum."""
    minimum = initial_time_step * settings.minimum_factor

    return max(minimum, current_time_step * settings.cutback_factor)


def grow_time_step(
    current_time_step: float,
    initial_time_step: float,
    settings: AdaptiveTimeStepSettings,
) -> float:
    """Increase an easy accepted step without crossing the configured maximum."""
    maximum = initial_time_step * settings.maximum_factor

    return min(maximum, current_time_step * settings.growth_factor)
