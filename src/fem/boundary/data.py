"""Boundary condition data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundaryCondition:
    """One boundary condition entry."""

    name: str
    kind: str
    group: str
    components: tuple[int, ...]
    values: NDArray[np.float64]
    nodes: NDArray[np.int64]
    dofs: NDArray[np.int64]


@dataclass(frozen=True)
class BoundaryConditionSet:
    """Boundary condition groups for one finite element model."""

    dirichlet: tuple[BoundaryCondition, ...]
    velocity: tuple[BoundaryCondition, ...]
    traction: tuple[BoundaryCondition, ...]
    pressure: tuple[BoundaryCondition, ...]
