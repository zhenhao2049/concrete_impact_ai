"""Case result recording helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fem.cases.data import CaseDef, RunResult
from fem.io.config import write_yaml_config


def write_run_records(
    case_def: CaseDef,
    run_result: RunResult,
    original_config: dict[str, Any],
    resolved_config: dict[str, Any],
) -> dict[str, Path]:
    """Write configuration, metadata, and metric records for one run."""
    case_def.output.root.mkdir(parents=True, exist_ok=True)

    original_config_path = case_def.output.root / "original_config.yaml"
    resolved_config_path = case_def.output.root / "resolved_config.yaml"
    metadata_path = case_def.output.root / "run_metadata.json"
    metrics_path = case_def.output.root / "metrics.json"

    write_yaml_config(original_config, original_config_path)
    write_yaml_config(resolved_config, resolved_config_path)
    _write_json(metadata_path, _build_metadata_record(case_def))
    _write_json(metrics_path, _build_metrics_record(run_result))

    return {
        "original_config": original_config_path,
        "resolved_config": resolved_config_path,
        "metadata": metadata_path,
        "metrics": metrics_path,
    }


def _build_metadata_record(case_def: CaseDef) -> dict[str, Any]:
    """Build a JSON-serializable metadata record."""
    return {
        "case_name": case_def.name,
        "analysis_type": case_def.run.analysis_type,
        "dimension": case_def.model.dimension,
        "mesh_generator": case_def.model.mesh["generator"],
        "element_type": case_def.model.mesh["element_type"],
        "created_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "metadata": case_def.metadata,
    }


def _build_metrics_record(run_result: RunResult) -> dict[str, Any]:
    """Build a JSON-serializable metric record."""
    return {
        "case_name": run_result.name,
        "passed": run_result.passed,
        "failure_reason": run_result.failure_reason,
        "metrics": run_result.metrics,
        "execution_metadata": run_result.metadata,
        "output_paths": {
            key: str(path)
            for key, path in run_result.output_paths.items()
        },
    }


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable indentation."""
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
