"""Velocity-Verlet INC dataset command-line entrypoint.

Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse

from concrete_impact.experiments.inc_data_generation import (
    generate_velocity_verlet_inc_dataset,
)
from concrete_impact.nn.config import load_velocity_verlet_inc_data_config


def main() -> int:
    """Generate one configured fixed-system INC dataset."""
    parser = argparse.ArgumentParser(description="Generate velocity-Verlet INC data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_velocity_verlet_inc_data_config(args.config)
    generate_velocity_verlet_inc_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
