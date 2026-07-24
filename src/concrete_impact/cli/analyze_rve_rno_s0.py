"""Command-line entry point for post-training RVE-RNO S0 diagnostics.

Contents:
    Explicit argument parsing and delegation to the evaluation project layer.
Author:
    Zhen Hao.
Created:
    2026-07-19.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from concrete_impact.nn.evaluation.rve_rno_s0 import (
    RVERNOS0DiagnosticsConfig,
    generate_rve_rno_s0_diagnostics,
)


def main() -> int:
    """Parse one S0 request and generate immutable diagnostic outputs."""
    parser = argparse.ArgumentParser(
        description="Generate post-training RVE-RNO state-rate and time-grid diagnostics."
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--response-shard-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    config = RVERNOS0DiagnosticsConfig(
        run_directory=Path(args.run_directory),
        response_shard_manifest=Path(args.response_shard_manifest),
        split_manifest=Path(args.split_manifest),
        output_directory=Path(args.output_directory),
        device=cast(Literal["cpu", "cuda"], args.device),
        batch_size=args.batch_size,
        dpi=args.dpi,
    )
    generate_rve_rno_s0_diagnostics(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
