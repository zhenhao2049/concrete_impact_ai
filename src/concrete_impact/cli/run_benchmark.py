"""Benchmark command-line entrypoint.

Contents:
    Benchmark selection, execution, and JSON-safe failure reporting.
Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from concrete_impact.benchmarks.errors import BenchmarkAcceptanceError
from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.core.config import load_yaml_config
from fem.dynamics.material import NonlinearStepConvergenceError
from fem.materials import MaterialPointConvergenceError
from fem.rve import RVEEquilibriumError


def main() -> int:
    """Run a configured benchmark."""
    parser = argparse.ArgumentParser(description="Run a concrete impact benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML configuration.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    runner = get_benchmark_runner(config["case"]["name"])
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = runner(config)
    except (
        MaterialPointConvergenceError,
        NonlinearStepConvergenceError,
        RVEEquilibriumError,
    ) as error:
        (output_root / "numerical_failure.json").write_text(
            json.dumps(
                {
                    "case_name": str(config["case"]["name"]),
                    "error_type": type(error).__name__,
                    "failure_reason": error.reason,
                    "message": str(error),
                    "diagnostics": error.diagnostics,
                    "resolved_config": config,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=_json_diagnostic_value,
            ),
            encoding="utf-8",
        )
        raise
    (output_root / "benchmark_result.json").write_text(
        json.dumps(
            {
                "case_name": result.name,
                "passed": result.passed,
                "failure_reason": result.failure_reason,
                "metrics": result.metrics,
                "output_paths": {
                    name: str(path) for name, path in result.output_paths.items()
                },
                "resolved_config": result.resolved_config,
                "metadata": result.metadata,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    if not result.passed:
        (output_root / "acceptance_failure.json").write_text(
            json.dumps(
                {
                    "case_name": result.name,
                    "failure_reason": result.failure_reason,
                    "metrics": result.metrics,
                    "resolved_config": result.resolved_config,
                    "metadata": result.metadata,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        raise BenchmarkAcceptanceError(
            result.name,
            str(result.failure_reason),
            {
                "metrics": result.metrics,
                "resolved_config": result.resolved_config,
                "metadata": result.metadata,
            },
        )
    return 0


def _json_diagnostic_value(value: Any) -> Any:
    """Convert numerical diagnostic scalars and arrays for strict JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported numerical diagnostic value: {type(value).__name__}.")


if __name__ == "__main__":
    raise SystemExit(main())
