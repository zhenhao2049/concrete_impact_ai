"""Script wrapper for complete RVE-RNO artifact evaluation.

Contents:
    Project-local entry point for evaluation, timing, and scientific figures.
Author:
    Zhen Hao.
Created:
    2026-07-17.
"""

from concrete_impact.cli.evaluate_rve_rno import main

if __name__ == "__main__":
    raise SystemExit(main())
