"""Guarded J2-VP-RNO training command-line entrypoint.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import argparse

from concrete_impact.nn.config import load_j2_vp_rno_config
from concrete_impact.nn.training.acceptance import (
    require_current_training_data_acceptance,
)


def main() -> int:
    """Require current data-acceptance hashes before future model training."""
    parser = argparse.ArgumentParser(description="Train the structured J2-VP-RNO model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-acceptance", required=True)
    parser.add_argument("--data-plan", required=True)
    parser.add_argument("--response-shard-manifest", required=True)
    args = parser.parse_args()
    load_j2_vp_rno_config(args.config)
    require_current_training_data_acceptance(
        args.data_acceptance,
        args.data_plan,
        args.response_shard_manifest,
    )
    raise NotImplementedError(
        "J2-VP-RNO optimization is outside the current training-data acceptance implementation."
    )


if __name__ == "__main__":
    raise SystemExit(main())
