"""Command-line launcher for deterministic RVE snapshot VTK export.

Contents:
    Argument parsing and compact-shard snapshot export.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse

from concrete_impact.experiments.rve_snapshot_export import export_rve_snapshots


def main() -> int:
    """Export all stored events for one accepted RVE path."""
    parser = argparse.ArgumentParser(description="Export compact RVE snapshots to VTK.")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_rve_snapshots(args.shard, args.path_id, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
