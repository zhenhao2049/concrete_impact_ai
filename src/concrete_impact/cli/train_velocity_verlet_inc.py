"""Velocity-Verlet INC training command-line entrypoint.

Contents:
    Strict configuration loading, training dispatch, and structured failure output.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concrete_impact.nn.config import load_velocity_verlet_inc_training_config
from concrete_impact.nn.training.velocity_verlet_inc import (
    INCTrainingNumericalError,
    train_velocity_verlet_inc,
)


def main() -> int:
    """Train one configured fixed-system velocity-Verlet INC."""
    parser = argparse.ArgumentParser(description="Train velocity-Verlet INC.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_velocity_verlet_inc_training_config(args.config)
    try:
        train_velocity_verlet_inc(config)
    except Exception as error:
        output = Path(config.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        failure: dict[str, object] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "config": str(args.config),
        }
        if isinstance(error, INCTrainingNumericalError):
            failure["reason"] = error.reason
            failure["diagnostics"] = error.diagnostics
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
