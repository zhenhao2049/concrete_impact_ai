"""Post-processing and field-output modules.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.post.fields import (
    CellFieldData,
    QuadratureFieldData,
    average_quadrature_fields,
    compute_linear_elastic_quadrature_fields,
)
from fem.post.response import (
    average_boundary_component,
    average_gauge_component,
    sum_boundary_component,
)
from fem.post.vtk import (
    write_mesh_vtk,
    write_named_solution_vtk,
    write_quadrature_points_vtk,
    write_solution_vtk,
)

__all__ = [
    "CellFieldData",
    "QuadratureFieldData",
    "average_boundary_component",
    "average_gauge_component",
    "average_quadrature_fields",
    "compute_linear_elastic_quadrature_fields",
    "sum_boundary_component",
    "write_mesh_vtk",
    "write_named_solution_vtk",
    "write_quadrature_points_vtk",
    "write_solution_vtk",
]
