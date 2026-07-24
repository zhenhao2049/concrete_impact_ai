"""Geometry definition builders.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.geometry.data import GeometryDef
from fem.preprocess.data import ModelDef


def build_geometry(model_def: ModelDef) -> GeometryDef:
    """Build a geometry definition from the raw model definition."""
    return GeometryDef(
        name=model_def.name,
        dimension=model_def.dimension,
        kind=model_def.geometry["type"],
        spec=model_def.geometry,
    )
