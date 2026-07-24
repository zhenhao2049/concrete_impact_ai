"""Command-line launcher for formal Direct-RNO artifact acceptance.

Contents:
    Strict configuration loading, validation execution, and exit status.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse

from concrete_impact.nn.training.rve_rno_artifact_acceptance import (
    load_direct_rno_artifact_acceptance_config,
    validate_direct_rno_artifact,
)


def main() -> int:
    """Run formal Direct-RNO artifact acceptance."""
    parser = argparse.ArgumentParser(description="Validate one c48 Direct-RNO artifact.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = validate_direct_rno_artifact(
        load_direct_rno_artifact_acceptance_config(args.config)
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
