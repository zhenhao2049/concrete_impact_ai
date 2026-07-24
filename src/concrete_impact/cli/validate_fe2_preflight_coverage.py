"""Command-line validation of pilot J2 and direct-FE2 state coverage.

Contents:
    Coverage-validation CLI and structured preflight result output.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import argparse

from concrete_impact.core.config import load_yaml_config
from concrete_impact.nn.training.acceptance import (
    CoverageBins,
    TrainingDataAcceptanceError,
    compare_j2_fe2_preflight_coverage,
)


def main() -> int:
    """Validate direct-FE2 preflight states against fixed pilot J2 coverage."""
    parser = argparse.ArgumentParser(description="Validate FE2 preflight coverage.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    acceptance_config = load_yaml_config(config["coverage_reference"])
    report = compare_j2_fe2_preflight_coverage(
        config["j2_response_shard_manifest"],
        config["fe2_response_shard_manifest"],
        CoverageBins.model_validate(acceptance_config["coverage"]),
        float(config["fe2_out_of_coverage_fraction"]),
        config["output_path"],
    )
    if not bool(report["passed"]):
        raise TrainingDataAcceptanceError(
            "Direct FE2 preflight contains states outside the fixed pilot J2 coverage."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
