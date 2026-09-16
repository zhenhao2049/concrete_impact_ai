"""Command-line interface for linear-elastic INC data production.

Author:
    Zhen Hao.
Created:
    2026-09-15.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from concrete_impact.cluster.inc_elastic_production import (
    analyze_elastic_inc_pilot,
    finalize_elastic_inc_production,
    prepare_elastic_inc_production,
    run_elastic_inc_bundle,
    run_elastic_inc_preflight,
    summarize_elastic_inc_status,
)


def main() -> int:
    """Dispatch one linear-elastic INC production command."""
    parser = argparse.ArgumentParser(description="Generate fixed-system elastic INC data.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--bundle-count", type=int, required=True)
    prepare.add_argument("--parallel-paths", type=int, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--manifest", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--bundle", required=True)
    worker.add_argument("--allow-local-p1", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--manifest", required=True)
    status.add_argument("--compact", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    analyze = commands.add_parser("analyze-pilot")
    analyze.add_argument("--manifest", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result: object = str(
            prepare_elastic_inc_production(
                args.config,
                args.bundle_count,
                args.parallel_paths,
            )
        )
    elif args.command == "preflight":
        result = run_elastic_inc_preflight(args.manifest)
    elif args.command == "worker":
        result = run_elastic_inc_bundle(
            args.manifest,
            args.bundle,
            allow_local_p1=args.allow_local_p1,
        )
    elif args.command == "status":
        result = summarize_elastic_inc_status(args.manifest)
        if args.compact:
            _print_compact_status(result)
            return 0
    elif args.command == "finalize":
        result = finalize_elastic_inc_production(args.manifest)
    else:
        result = analyze_elastic_inc_pilot(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if isinstance(result, dict) and result.get("failure_count", 0) != 0:
        return 1
    return 0


def _print_compact_status(status: dict[str, Any]) -> None:
    """Print only the key login-node production status fields."""
    print(
        f"轨迹进度：{status['succeeded']}/{status['task_count']} 完成，"
        f"{status['running']} 运行，{status['pending']} 等待，"
        f"{status['failed']} 失败；已完成轨迹累计耗时 "
        f"{float(status['completed_path_wall_seconds']):.1f} s"
    )
    for active in list(status["active_examples"])[:2]:
        print(
            f"运行示例：{active['task_id']} {active['progress_percent']:.1f}% "
            f"已耗时 {active['elapsed_seconds']:.1f} s"
        )
    for failure in list(status["failure_examples"])[:2]:
        print(f"失败示例：{failure['task_id']} {failure['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
