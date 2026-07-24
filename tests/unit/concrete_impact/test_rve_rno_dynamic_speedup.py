"""Tests for the matched dynamic FE2 and RVE-RNO speed benchmark.

Author:
    Zhen Hao.
Created:
    2026-07-18.
"""

from __future__ import annotations

import numpy as np

from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.benchmarks.rve_rno_dynamic_speedup import TimedMaterialAdapter
from concrete_impact.core.config import load_yaml_config
from fem.materials.data import MaterialPointRequest, MaterialPointResponse, MaterialState


class _FakeMaterial:
    """Provide one deterministic material response for timer tests."""

    name = "fake"
    density = 1.0
    maximum_wave_speed = 1.0

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize a scalar state at every point."""
        return MaterialState({"value": np.zeros((n_points, 1))})

    def update(self, request, state, requirements):
        """Return zero stress without changing the state."""
        del requirements
        return MaterialPointResponse(
            stresses=np.zeros_like(request.strains),
            state=state,
            free_energy=np.zeros(request.strains.shape[0]),
            dissipation=np.zeros(request.strains.shape[0]),
        )


def test_speed_config_uses_matched_low_cost_dynamic_case() -> None:
    """Verify the engineering smoke fixes one shared dynamic discretization."""
    config = load_yaml_config(
        "configs/benchmarks/rve/rve_rno_dynamic_material_speedup.yaml"
    )

    assert config["model"]["mesh"]["divisions"] == [1, 1, 1]
    assert config["rve"]["resolution_label"] == "c16_r2_z1"
    assert config["rve"]["execution"] == {
        "backend": "process_pool",
        "workers": 8,
        "threads_per_worker": 1,
    }
    assert config["analysis"]["explicit"]["num_steps"] == 8
    assert config["artifact"]["model"]["device"] == "cpu"
    assert config["verification"]["maximum_axial_stress_error"] == 0.15
    assert get_benchmark_runner(config["case"]["name"]) is not None


def test_timed_material_adapter_counts_complete_batched_updates() -> None:
    """Verify material-boundary timing retains calls and point counts."""
    adapter = TimedMaterialAdapter(_FakeMaterial())
    state = adapter.initialize_state(3)
    request = MaterialPointRequest(
        strains=np.zeros((3, 6)),
        strain_rates=np.zeros((3, 6)),
        time_step=1.0e-3,
        kinematics="three_dimensional",
        update_settings=None,
    )

    response = adapter.update(request, state)

    assert response.stresses.shape == (3, 6)
    assert adapter.point_counts == [3]
    assert len(adapter.update_seconds) == 1
    assert adapter.update_seconds[0] >= 0.0
    adapter.reset_timings()
    assert adapter.point_counts == []
    assert adapter.update_seconds == []
