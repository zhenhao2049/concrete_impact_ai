#!/bin/bash
# Purpose: Launch the accepted local RTX 5060 Ti RVE-RNO training configuration.
# Contents: Device, data-acceptance, preflight-performance, and unique-output gates.
# Author: Zhen Hao.
# Created: 2026-07-19.

set -euo pipefail

CHECK_ONLY=false
if [[ $# -eq 1 ]] && [[ $1 == "--check-only" ]]; then
  CHECK_ONLY=true
elif [[ $# -ne 0 ]]; then
  printf 'Usage: %s [--check-only]\n' "$0" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${RVE_RNO_PYTHON_BIN:-python}"
TRAIN_CONFIG="configs/nn/rve_rno_c48_v2_k6_gpu_liu_direct.yaml"
OUTPUT_ROOT="results/rve_rno_training/c48_v2_liu_direct_gpu_optimized"
PREFLIGHT_ROOT="results/rve_rno_training/c48_v2_liu_direct_gpu_optimized_preflight"
ACCEPTANCE_REPORT="results/rve_rno_training/rve_rno_c48_v2_2000/validation_staging/acceptance/training_data_acceptance.json"

test -x "${PYTHON_BIN}"
test -f "${TRAIN_CONFIG}"
test -f "${ACCEPTANCE_REPORT}"
test ! -e "${OUTPUT_ROOT}"

"${PYTHON_BIN}" - "${ACCEPTANCE_REPORT}" "${PREFLIGHT_ROOT}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

import torch

acceptance_path = Path(sys.argv[1])
preflight_root = Path(sys.argv[2])
acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
assert acceptance["passed"] is True
paths = acceptance["split_manifest"]["paths"]
split_counts = {
    split: sum(path["split"] == split for path in paths)
    for split in ("test", "train", "validation")
}
assert len(paths) == 2000
assert split_counts == {"test": 200, "train": 1600, "validation": 200}

assert torch.cuda.is_available()
properties = torch.cuda.get_device_properties(0)
assert "RTX 5060 Ti" in properties.name
assert properties.total_memory >= 15 * 1024**3

smoke = json.loads(
    (preflight_root / "gpu_smoke_final/performance_smoke.json").read_text(encoding="utf-8")
)
assert smoke["passed"] is True
assert smoke["batching_benchmark"]["accuracy_equivalent"] is True
assert smoke["batching_benchmark"]["performance_gate_passed"] is True

epoch_metrics = {}
for stage in ("two_epoch", "ten_epoch"):
    active = json.loads((preflight_root / stage / "active_run.json").read_text(encoding="utf-8"))
    run_directory = Path(active["run_directory"])
    summary = json.loads((run_directory / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    epoch_metrics[stage] = records

ten_epoch = epoch_metrics["ten_epoch"]
train_by_epoch = {record["epoch"]: record for record in ten_epoch if record["split"] == "train"}
validation_by_epoch = {
    record["epoch"]: record for record in ten_epoch if record["split"] == "validation"
}
steady_epochs = sorted(epoch for epoch in train_by_epoch if epoch >= 2)
steady_epoch_seconds = [
    train_by_epoch[epoch]["elapsed_seconds"] + validation_by_epoch[epoch]["elapsed_seconds"]
    for epoch in steady_epochs
]
steady_train_throughput = [train_by_epoch[epoch]["steps_per_second"] for epoch in steady_epochs]
projected_seconds = 500 * statistics.median(steady_epoch_seconds)

assert len(epoch_metrics["two_epoch"]) == 5
assert projected_seconds <= 4500.0
assert statistics.median(steady_train_throughput) >= 4.0 * 5823.240298333358

print(
    "Preflight accepted: "
    f"gpu={properties.name}, "
    f"steady_train_steps_per_second={statistics.median(steady_train_throughput):.1f}, "
    f"projected_500_epoch_minutes={projected_seconds / 60.0:.1f}"
)
PY

if [[ "${CHECK_ONLY}" == true ]]; then
  printf 'Preflight checks completed; formal training was not submitted.\n'
  exit 0
fi

"${PYTHON_BIN}" -m concrete_impact.cli.run_background \
  --detach \
  --poll-seconds 10 \
  --inactivity-seconds 1200 \
  --output "${OUTPUT_ROOT}" \
  -- "${PYTHON_BIN}" -m concrete_impact.cli.train_rve_rno \
  --config "${TRAIN_CONFIG}"

printf 'Submitted local RVE-RNO training.\nProgress: %s/progress.json\nLog: %s/summary.log\n' \
  "${OUTPUT_ROOT}" "${OUTPUT_ROOT}"
