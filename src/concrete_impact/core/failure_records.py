"""Project-level recording for failed material-point updates.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from fem.cases import RunResult
from fem.dynamics.material import NonlinearStepConvergenceError
from fem.materials import MaterialPointConvergenceError

BenchmarkRunner = Callable[[dict[str, Any]], RunResult]


def record_material_point_failures(runner: BenchmarkRunner) -> BenchmarkRunner:
    """Record a structured material-point failure and re-raise it."""

    @wraps(runner)
    def wrapped(config: dict[str, Any]) -> RunResult:
        try:
            return runner(config)
        except (MaterialPointConvergenceError, NonlinearStepConvergenceError) as error:
            _write_numerical_failure_record(config, error)
            raise

    return wrapped


def _write_numerical_failure_record(
    config: dict[str, Any],
    error: MaterialPointConvergenceError | NonlinearStepConvergenceError,
) -> Path:
    """Write one failed local or global nonlinear-solve record."""
    output_root = Path(str(config["output"]["root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "failure_diagnostics.json"
    payload = {
        "status": "failed",
        "failure_type": error.diagnostics["algorithm"],
        "case_name": str(config["case"]["name"]),
        "created_utc": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "message": str(error),
        "reason": error.reason,
        "diagnostics": error.diagnostics,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    return output_path
