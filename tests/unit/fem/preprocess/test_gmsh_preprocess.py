"""Tests for Gmsh-based finite element preprocessing.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path

import numpy as np

from fem.preprocess import build_preprocess_data, load_preprocess_defs


def test_plate_2d_gmsh_preprocess_generates_valid_mesh() -> None:
    """Verify two-dimensional Gmsh preprocessing and VTK output."""
    model_def, output_def = load_preprocess_defs("configs/preprocess/plate_2d_gmsh.yaml")
    bundle = build_preprocess_data(model_def, output_def)

    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache

    assert mesh_info.nodes.shape == (45, 2)
    assert mesh_info.elements.shape == (32, 4)
    assert mesh_info.element_ordering == "tensor-product-xi-eta"
    assert set(mesh_info.boundary_groups) == {"left", "right", "bottom", "top"}

    assert np.allclose(cache.shape_values.sum(axis=1), 1.0)
    assert np.allclose(cache.shape_gradients_reference.sum(axis=1), 0.0)
    assert np.all(cache.jacobian_determinants > 0.0)

    assert bundle.boundary_conditions.dirichlet[0].dofs.shape == (10,)
    assert bundle.boundary_conditions.velocity[0].dofs.shape == (5,)
    assert Path("results/tests/preprocess/plate_2d_gmsh.vtu").is_file()


def test_block_3d_gmsh_preprocess_generates_valid_mesh() -> None:
    """Verify three-dimensional Gmsh preprocessing and VTK output."""
    model_def, output_def = load_preprocess_defs("configs/preprocess/block_3d_gmsh.yaml")
    bundle = build_preprocess_data(model_def, output_def)

    mesh_info = bundle.mesh_info
    cache = bundle.shape_function_cache

    assert mesh_info.nodes.shape == (45, 3)
    assert mesh_info.elements.shape == (16, 8)
    assert mesh_info.element_ordering == "tensor-product-xi-eta-zeta"
    assert set(mesh_info.boundary_groups) == {"left", "right", "front", "back", "bottom", "top"}

    assert np.allclose(cache.shape_values.sum(axis=1), 1.0)
    assert np.allclose(cache.shape_gradients_reference.sum(axis=1), 0.0)
    assert np.all(cache.jacobian_determinants > 0.0)

    assert bundle.boundary_conditions.dirichlet[0].dofs.shape == (27,)
    assert bundle.boundary_conditions.velocity[0].dofs.shape == (9,)
    assert Path("results/tests/preprocess/block_3d_gmsh.vtu").is_file()
