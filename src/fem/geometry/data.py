"""Geometry data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryDef:
    """Geometry definition used by the mesh generator."""

    name: str
    dimension: int
    kind: str
    spec: dict[str, Any]
