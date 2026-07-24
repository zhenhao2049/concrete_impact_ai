"""Command-line launcher for deterministic RVE-RNO data tasks.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from concrete_impact.experiments.rve_data_execution import (
    audit_response_shards,
    generate_control_path_shards,
    generate_response_shards,
    profile_fixed_cell_path_workers,
    write_rve_data_manifest,
)
from concrete_impact.experiments.rve_data_plan import (
    load_rve_data_plan,
    resolve_fixed_cell_mesh,
)


def main() -> int:
    """Validate a plan and generate its manifest or process-owned control shards."""
    parser = argparse.ArgumentParser(description="Prepare RVE-RNO data generation tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--profile-path-workers", action="store_true")
    args = parser.parse_args()
    plan = resolve_fixed_cell_mesh(load_rve_data_plan(args.config))
    plan.output_directory.mkdir(parents=True, exist_ok=True)
    original_config = Path(args.config).read_text(encoding="utf-8")
    existing_config_path = plan.output_directory / "original_config.yaml"
    if existing_config_path.exists():
        existing_config = existing_config_path.read_text(encoding="utf-8")
        if existing_config != original_config:
            raise ValueError(
                "RVE data output directory belongs to a different configuration: "
                f"directory={plan.output_directory}."
            )
    (plan.output_directory / "original_config.yaml").write_text(
        original_config, encoding="utf-8"
    )
    (plan.output_directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = Path(plan.output_directory) / "manifest.json"
    write_rve_data_manifest(plan, manifest_path)
    if args.profile_path_workers:
        profile_fixed_cell_path_workers(plan, os.cpu_count() or 1)
        return 0
    if args.manifest_only:
        return 0
    if args.audit_existing:
        shard_manifest = json.loads(
            (plan.output_directory / "response_shards.json").read_text(encoding="utf-8")
        )
        response_paths = tuple(
            Path(path) for path in shard_manifest["response_shards"]
        )
        audit = audit_response_shards(plan, response_paths)
        (plan.output_directory / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0
    if args.controls_only:
        generate_control_path_shards(plan, os.cpu_count() or 1)
    else:
        response_paths = generate_response_shards(plan, os.cpu_count() or 1)
        (plan.output_directory / "response_shards.json").write_text(
            json.dumps(
                {"response_shards": [str(path) for path in response_paths]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        audit = audit_response_shards(plan, response_paths)
        (plan.output_directory / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
