"""Tests for project-level failed material-update records.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

import json
from pathlib import Path

import pytest

from concrete_impact.core.failure_records import record_material_point_failures
from fem.materials import MaterialPointConvergenceError


def test_material_failure_record_is_written_before_exception_propagation(tmp_path: Path) -> None:
    """Verify a benchmark failure writes strict JSON and remains an exception."""
    config = {
        "case": {"name": "failure_record_test"},
        "output": {"root": str(tmp_path)},
    }

    @record_material_point_failures
    def failing_runner(config: dict) -> None:
        del config
        raise MaterialPointConvergenceError(
            "J2 viscoplastic local Newton failed: maximum_iterations_exceeded.",
            "maximum_iterations_exceeded",
            {
                "algorithm": "j2_viscoplastic_local_newton",
                "iteration": 2,
                "residual": float("inf"),
            },
        )

    with pytest.raises(MaterialPointConvergenceError):
        failing_runner(config)

    record_path = tmp_path / "failure_diagnostics.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_type"] == "j2_viscoplastic_local_newton"
    assert payload["reason"] == "maximum_iterations_exceeded"
    assert payload["diagnostics"]["residual"] == "inf"
    assert not (tmp_path / "metrics.json").exists()
