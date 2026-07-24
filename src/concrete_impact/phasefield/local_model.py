"""PF-CZM local response interface.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PFLocalState:
    """Local PF-CZM state at one quadrature point."""

    strain: tuple[float, ...]
    strain_rate: tuple[float, ...]
    phasefield: float
    history: float
    time_step: float


@dataclass(frozen=True)
class PFLocalResponse:
    """Local PF-CZM response at one quadrature point."""

    stress: tuple[float, ...]
    crack_driving_force: float
    updated_history: float


def evaluate_pfczm_local_response(
    state: PFLocalState,
    material_parameters: dict[str, float],
) -> PFLocalResponse:
    """Evaluate local PF-CZM stress, crack driving force, and history field."""
    raise NotImplementedError("PF-CZM local response is not implemented yet.")

