"""Command-line entry point for complete RVE-RNO artifact evaluation.

Contents:
    Strict configuration loading and accuracy, tangent, timing, and figure dispatch.
Author:
    Zhen Hao.
Created:
    2026-07-17.
"""

from __future__ import annotations

import argparse
import json

from concrete_impact.nn.evaluation.rve_rno import (
    evaluate_rve_rno_artifact,
    load_rve_rno_evaluation_config,
)


def main() -> int:
    """Run one strict trained RVE-RNO evaluation request."""
    parser = argparse.ArgumentParser(description="Evaluate one trained RVE-RNO artifact.")
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_rve_rno_evaluation_config(arguments.config)
    report = evaluate_rve_rno_artifact(config, arguments.config)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
