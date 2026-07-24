"""Controlled comparison of direct-rate and mobility-gradient RVE-RNO runs.

Contents:
    CLI parsing, completed-run loading, control checks, and comparison reporting.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from concrete_impact.core.progress import write_json_atomic


def main() -> int:
    """Validate controlled variables and write one two-model comparison."""
    parser = argparse.ArgumentParser(description="Compare two RVE-RNO v2 evolutions.")
    parser.add_argument("--direct-run", required=True)
    parser.add_argument("--mobility-run", required=True)
    parser.add_argument("--direct-smoke", required=True)
    parser.add_argument("--mobility-smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    direct = _load_run(Path(args.direct_run))
    mobility = _load_run(Path(args.mobility_run))
    direct_smoke = _read_json(Path(args.direct_smoke))
    mobility_smoke = _read_json(Path(args.mobility_smoke))
    _require_controlled_pair(direct, mobility, direct_smoke, mobility_smoke)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": True,
        "comparison": "direct_rate_vs_mobility_gradient",
        "controlled_fields": _controlled_fields(direct),
        "direct_rate": _comparison_record(direct, direct_smoke),
        "mobility_gradient": _comparison_record(mobility, mobility_smoke),
        "interpretation_rule": (
            "Compare frozen-test stress and dissipation errors first, thermodynamic "
            "violation second, then throughput and memory; one seed is not a "
            "statistical architecture ranking."
        ),
    }
    write_json_atomic(output / "model_comparison.json", report)
    return 0


def _load_run(path: Path) -> dict[str, Any]:
    """Load all required records from one completed run directory."""
    return {
        "path": str(path),
        "metadata": _read_json(path / "run_metadata.json"),
        "artifact": _read_json(path / "artifact_metadata.json"),
        "summary": _read_json(path / "training_summary.json"),
        "performance": _read_json(path / "performance_report.json"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required JSON object without search or fallback."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RVE-RNO comparison requires a JSON object: {path}.")
    return payload


def _require_controlled_pair(
    direct: dict[str, Any],
    mobility: dict[str, Any],
    direct_smoke: dict[str, Any],
    mobility_smoke: dict[str, Any],
) -> None:
    """Reject comparison when evolution labels or controlled fields differ."""
    if direct["metadata"]["model"]["evolution"] != "direct_rate":
        raise ValueError("The direct run does not use direct_rate evolution.")
    if mobility["metadata"]["model"]["evolution"] != "mobility_gradient":
        raise ValueError("The mobility run does not use mobility_gradient evolution.")
    if direct_smoke["evolution"] != "direct_rate":
        raise ValueError("The direct smoke does not use direct_rate evolution.")
    if mobility_smoke["evolution"] != "mobility_gradient":
        raise ValueError("The mobility smoke does not use mobility_gradient evolution.")
    direct_fields = _controlled_fields(direct)
    mobility_fields = _controlled_fields(mobility)
    if direct_fields != mobility_fields:
        raise ValueError(
            "RVE-RNO runs do not form a controlled evolution comparison: "
            f"direct={direct_fields}; mobility={mobility_fields}."
        )
    if not direct_smoke["passed"] or not mobility_smoke["passed"]:
        raise ValueError("Both batching performance smoke reports must pass.")


def _controlled_fields(run: dict[str, Any]) -> dict[str, Any]:
    """Extract fields that must be identical across the two evolutions."""
    metadata = run["metadata"]
    model = dict(metadata["model"])
    model.pop("evolution")
    return {
        "model_except_evolution": model,
        "loss": metadata["loss"],
        "optimizer": metadata["optimizer"],
        "random_seed": metadata["random_seed"],
        "data_plan_sha256": metadata["data_plan_sha256"],
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "accepted_time_step_range": run["artifact"]["accepted_time_step_range"],
    }


def _comparison_record(
    run: dict[str, Any], smoke: dict[str, Any]
) -> dict[str, Any]:
    """Collect frozen-test accuracy, physics, throughput, and memory metrics."""
    test = run["summary"]["test_metrics"]
    batching = smoke["batching_benchmark"]
    return {
        "run_directory": run["path"],
        "best_epoch": run["summary"]["best_epoch"],
        "best_validation_loss": run["summary"]["best_validation_loss"],
        "test_total_loss": test["total_loss"],
        "test_stress_loss": test["stress_loss"],
        "test_maximum_path_stress_error": test["maximum_path_stress_error"],
        "test_dissipation_loss": test["dissipation_loss"],
        "test_maximum_dissipation_error": test["maximum_dissipation_error"],
        "test_thermodynamic_violation_loss": test[
            "thermodynamic_violation_loss"
        ],
        "test_maximum_thermodynamic_violation": test[
            "maximum_thermodynamic_violation"
        ],
        "maximum_training_steps_per_second": run["performance"][
            "maximum_training_steps_per_second"
        ],
        "maximum_cuda_memory_allocated_bytes": run["performance"][
            "maximum_cuda_memory_allocated_bytes"
        ],
        "scalar_reference_speedup": batching["speedup"],
        "batching_accuracy_difference": batching["maximum_absolute_difference"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
