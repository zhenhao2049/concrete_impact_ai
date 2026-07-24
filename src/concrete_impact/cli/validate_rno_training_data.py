"""RNO training-data acceptance command-line entrypoint.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import argparse

from concrete_impact.nn.training.acceptance import (
    TrainingDataAcceptanceConfig,
    TrainingDataAcceptanceError,
    validate_training_data_acceptance,
)
from fem.io.config import load_yaml_config


def main() -> int:
    """Validate RNO training data and reject every failed acceptance item."""
    parser = argparse.ArgumentParser(description="Validate RNO training data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = validate_training_data_acceptance(
        TrainingDataAcceptanceConfig.model_validate(load_yaml_config(args.config))
    )
    if not bool(report["passed"]):
        raise TrainingDataAcceptanceError(
            "RNO training data did not pass the configured acceptance criteria."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
