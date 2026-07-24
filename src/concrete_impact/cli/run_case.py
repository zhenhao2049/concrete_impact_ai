"""Case command-line entrypoint.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import argparse

from concrete_impact.benchmarks.registry import get_benchmark_runner
from concrete_impact.core.config import load_yaml_config


def main() -> int:
    """Run a configured concrete impact case."""
    parser = argparse.ArgumentParser(description="Run a concrete impact case.")
    parser.add_argument("--config", required=True, help="Path to case YAML configuration.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    runner = get_benchmark_runner(config["case"]["name"])
    runner(config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
