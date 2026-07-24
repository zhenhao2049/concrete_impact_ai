"""Tests for the report-specific c48 explicit FE2 plastic-impact workflow.

Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

import numpy as np
import pytest

from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.benchmarks.report_fe2_plastic_impact import (
    _coordinate_plane_triangles,
    _lumped_project_hex8_scalar_to_nodes,
    _shared_triangle_boundary_edges,
    select_report_event_steps,
)
from concrete_impact.core.config import load_yaml_config
from fem.assembly.nonlinear_solid import build_hex8_material_assembly_cache
from fem.mesh.structured import build_cylindrical_ogrid_hex8, build_structured_hex8_box
from fem.rve import RVEExecutionSettings


def test_report_impact_config_fixes_requested_discretization() -> None:
    """Verify the report case uses only the requested explicit c48 discretization."""
    config = load_yaml_config("configs/benchmarks/rve/report_fe2_plastic_impact_c48.yaml")

    assert config["model"]["mesh"]["divisions"] == [8, 2, 2]
    assert config["rve"]["circumferential_divisions"] == 48
    assert config["rve"]["matrix_radial_divisions"] == 6
    assert config["rve"]["axial_divisions"] == 6
    assert config["rve"]["execution"] == {
        "backend": "process_pool",
        "workers": 12,
        "threads_per_worker": 1,
    }
    assert config["analysis"]["explicit"]["time_step"] == 1.0e-3
    assert config["analysis"]["explicit"]["num_steps"] == 80
    assert "implicit" not in config["analysis"]
    assert get_benchmark_runner(config["case"]["name"]) is not None


def test_rve_execution_rejects_excess_logical_cpu_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject process and numerical-thread products above half the machine CPUs."""
    monkeypatch.setattr("fem.rve.data.os.cpu_count", lambda: 32)

    with pytest.raises(ValueError, match="CPU budget exceeds half"):
        RVEExecutionSettings(
            backend="process_pool",
            workers=12,
            threads_per_worker=2,
        )


def test_report_event_selection_uses_five_distinct_accepted_states() -> None:
    """Verify event-based montage frames remain distinct and reproducible."""
    times = np.arange(6, dtype=np.float64) * 1.0e-3
    arrival_stress = np.asarray([0.0, 0.2, 1.0, 0.8, 0.4, 0.2])
    maximum_q = np.asarray([0.0, 0.0, 2.0e-6, 3.0e-6, 4.0e-6, 4.0e-6])
    active_fraction = np.asarray([0.0, 0.0, 0.2, 0.8, 0.1, 0.0])
    negative_work = np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, -0.2])

    steps = select_report_event_steps(
        times,
        arrival_stress,
        maximum_q,
        active_fraction,
        negative_work,
        1.0e-6,
    )

    assert steps == {
        "wave_arrival": 1,
        "first_yield": 2,
        "peak_activity": 3,
        "maximum_negative_work": 4,
        "final_residual": 5,
    }


def test_report_event_selection_rejects_duplicate_frames() -> None:
    """Reject a montage that would conceal coincident physical events."""
    times = np.arange(5, dtype=np.float64) * 1.0e-3
    arrival_stress = np.asarray([0.0, 1.0, 0.8, 0.5, 0.2])
    maximum_q = np.asarray([0.0, 2.0e-6, 3.0e-6, 3.0e-6, 3.0e-6])
    active_fraction = np.asarray([0.0, 1.0, 0.2, 0.0, 0.0])
    negative_work = np.asarray([0.0, 0.0, -1.0, -0.5, -0.1])

    with pytest.raises(ValueError, match="event states are not distinct"):
        select_report_event_steps(
            times,
            arrival_stress,
            maximum_q,
            active_fraction,
            negative_work,
            1.0e-6,
        )


def test_lumped_projection_preserves_a_constant_hex8_quadrature_field() -> None:
    """Verify the visualization projection is continuous and constant preserving."""
    nodes, elements = build_structured_hex8_box((1.0, 1.0, 1.0), (1, 1, 1))
    cache = build_hex8_material_assembly_cache(nodes, elements, quadrature_order=2)
    values = np.full((2, 1, 8), 3.25, dtype=np.float64)

    projected = _lumped_project_hex8_scalar_to_nodes(
        elements,
        values,
        cache.jacobian_weights,
        nodes.shape[0],
        quadrature_order=2,
    )

    assert np.allclose(projected, 3.25, rtol=0.0, atol=1.0e-14)


def test_c48_xz_section_keeps_matrix_and_inclusion_triangulations_distinct() -> None:
    """Verify phase-aware x-z projection retains the material interface."""
    mesh = build_cylindrical_ogrid_hex8((1.0, 1.0, 1.0), 0.05, 48, 6, 6)
    matrix_triangles = _coordinate_plane_triangles(
        mesh.nodes,
        mesh.elements,
        axis=1,
        coordinate=0.0,
        element_mask=mesh.element_phase_ids == 0,
    )
    inclusion_triangles = _coordinate_plane_triangles(
        mesh.nodes,
        mesh.elements,
        axis=1,
        coordinate=0.0,
        element_mask=mesh.element_phase_ids == 1,
    )

    assert matrix_triangles.shape[0] > 0
    assert inclusion_triangles.shape[0] > 0
    assert len(_shared_triangle_boundary_edges(matrix_triangles, inclusion_triangles)) > 0
