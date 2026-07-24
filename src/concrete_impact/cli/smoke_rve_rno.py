"""CPU/CUDA smoke and batching-performance validation for RVE-RNO v2.

Contents:
    Smoke CLI, repeated training-step timing, and CUDA metadata collection.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import argparse
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from concrete_impact.core.progress import write_json_atomic
from concrete_impact.core.response_manifest import load_response_shard_paths
from concrete_impact.nn.config import load_rve_rno_training_config
from concrete_impact.nn.datasets.rve import (
    HDF5RNOPathDataset,
    PreloadedRNOPathDataset,
    RNOConditioningSpec,
    collate_rno_paths,
)
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO
from concrete_impact.nn.training.acceptance import (
    require_current_training_data_acceptance,
)
from concrete_impact.nn.training.rve_rno import (
    RVERNOPathRollout,
    _batch_loss,
    _compile_model_networks,
    benchmark_rve_rno_batching,
)


def main() -> int:
    """Run one strict device smoke and write its structured performance report."""
    parser = argparse.ArgumentParser(description="Smoke-test explicit RVE-RNO batching.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.training_repetitions <= 0:
        raise ValueError("RVE-RNO smoke training repetitions must be positive.")
    config = load_rve_rno_training_config(args.config)
    if config.model.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RVE-RNO smoke requested CUDA, but CUDA is unavailable.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    acceptance = require_current_training_data_acceptance(
        config.data_acceptance_report,
        config.data_plan,
        config.response_shard_manifest,
    )
    shard_paths = load_response_shard_paths(config.response_shard_manifest)
    dataset = PreloadedRNOPathDataset(HDF5RNOPathDataset(shard_paths, RNOConditioningSpec((), ())))
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collate_rno_paths,
        pin_memory=config.execution.pin_memory and config.model.device == "cuda",
        num_workers=config.execution.num_workers,
    )
    batch = next(iter(loader))
    model = EnergyDissipationRVERNO(config.model)
    if config.execution.compile.enabled:
        _compile_model_networks(model, config)
    benchmark = benchmark_rve_rno_batching(model, batch)
    training = _training_step_smoke(
        model,
        batch,
        config,
        acceptance["normalization"],
        args.training_repetitions,
    )
    passed = (
        benchmark["accuracy_equivalent"]
        and benchmark["performance_gate_passed"]
        and training["all_gradients_finite"]
    )
    report = {
        "passed": passed,
        "config": str(args.config),
        "device": config.model.device,
        "dtype": config.model.dtype,
        "evolution": config.model.evolution,
        "integrator": config.model.integrator,
        "preloaded_path_count": len(dataset),
        "batching_benchmark": benchmark,
        "training_step": training,
        "cuda": _cuda_metadata(config.model.device),
    }
    write_json_atomic(output / "performance_smoke.json", report)
    if not passed:
        raise RuntimeError("RVE-RNO performance smoke failed; inspect performance_smoke.json.")
    return 0


def _training_step_smoke(
    model: EnergyDissipationRVERNO,
    batch: Any,
    config: Any,
    normalization: dict[str, Any],
    repetitions: int,
) -> dict[str, Any]:
    """Time one full path-batched forward and backward training step."""
    rollout = RVERNOPathRollout(model)
    model.zero_grad(set_to_none=True)
    warmup_loss, _, _, _ = _batch_loss(rollout, batch, config, normalization)
    warmup_loss.backward()
    if config.model.device == "cuda":
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    if config.model.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    start = perf_counter()
    loss = torch.zeros((), dtype=model.dtype, device=model.device)
    for _ in range(repetitions):
        model.zero_grad(set_to_none=True)
        loss, _, _, _ = _batch_loss(rollout, batch, config, normalization)
        loss.backward()
    if config.model.device == "cuda":
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = (
        usage_end.ru_utime + usage_end.ru_stime - usage_start.ru_utime - usage_start.ru_stime
    )
    finite = torch.stack(
        [
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    ).all()
    valid_steps = int(batch.valid_mask.sum()) * repetitions
    return {
        "loss": float(loss.detach().cpu()),
        "repetitions": repetitions,
        "elapsed_seconds": elapsed,
        "valid_material_steps": valid_steps,
        "steps_per_second": valid_steps / elapsed,
        "process_cpu_seconds": cpu_seconds,
        "process_average_cpu_percent": 100.0 * cpu_seconds / elapsed,
        "process_maximum_resident_set_kibibytes": usage_end.ru_maxrss,
        "all_gradients_finite": bool(finite.detach().cpu()),
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if config.model.device == "cuda" else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if config.model.device == "cuda" else 0
        ),
    }


def _cuda_metadata(device: str) -> dict[str, Any] | None:
    """Collect CUDA identity without inferring utilization from memory use."""
    if device != "cuda":
        return None
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
