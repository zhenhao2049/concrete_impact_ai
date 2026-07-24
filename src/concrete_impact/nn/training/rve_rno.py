"""Batched complete-path training for the explicit energy-based RVE-RNO.

Contents:
    Path rollout, losses, epochs, numerical checks, checkpoints, and performance reports.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
from typing import Any, Literal, cast

import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from concrete_impact.core.progress import JsonProgressRecorder
from concrete_impact.experiments.rve_data_plan import load_rve_data_plan
from concrete_impact.nn.config import RVERNOArtifactMetadata, RVERNOTrainingConfig
from concrete_impact.nn.datasets.rve import (
    DeviceRNOPathBatchLoader,
    HDF5RNOPathDataset,
    PreloadedRNOPathDataset,
    RNOConditioningSpec,
    RNOPathBatch,
    RNOPathSample,
    collate_rno_paths,
)
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO
from concrete_impact.nn.training.acceptance import (
    require_current_training_data_acceptance,
)


@dataclass(frozen=True)
class EpochMetrics:
    """Store losses, physical diagnostics, throughput, and memory for one epoch."""

    epoch: int
    split: str
    total_loss: float
    stress_loss: float
    thermodynamic_violation_loss: float
    dissipation_loss: float
    weighted_thermodynamic_contribution: float
    weighted_dissipation_contribution: float
    maximum_path_stress_error: float
    maximum_thermodynamic_violation: float
    thermodynamic_violation_count: int
    maximum_dissipation_error: float
    maximum_nonequilibrium_energy_increase: float
    latent_norm_median: float
    latent_norm_p95: float
    latent_norm_p99: float
    latent_norm_maximum: float
    dimensionless_state_rate_norm_median: float
    dimensionless_state_rate_norm_p95: float
    dimensionless_state_rate_norm_p99: float
    dimensionless_state_rate_norm_maximum: float
    latent_increment_norm_median: float
    latent_increment_norm_p95: float
    latent_increment_norm_p99: float
    latent_increment_norm_maximum: float
    batch_count: int
    material_steps: int
    executed_material_slots: int
    padding_efficiency: float
    steps_per_second: float
    elapsed_seconds: float
    peak_cuda_memory_allocated_bytes: int
    peak_cuda_memory_reserved_bytes: int


class TrainingNumericalError(RuntimeError):
    """Report a confirmed non-finite training loss, gradient, or parameter."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        """Store exact finite-value failure diagnostics."""
        super().__init__(f"RVE-RNO training numerical failure: {reason}; {diagnostics}")
        self.reason = reason
        self.diagnostics = diagnostics


class IndexedPathSubset(Dataset[RNOPathSample]):
    """Select complete preloaded paths by stable acceptance-report identifiers."""

    def __init__(self, dataset: Dataset[RNOPathSample], indices: tuple[int, ...]) -> None:
        """Store immutable parent-dataset indices for one split."""
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Return the complete-path count in this split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> RNOPathSample:
        """Read one preloaded complete path."""
        return self.dataset[self.indices[index]]


class RVERNOPathRollout(nn.Module):
    """Advance time sequentially while batching all independent paths."""

    def __init__(self, model: EnergyDissipationRVERNO) -> None:
        """Store the constitutive model used by the rollout."""
        super().__init__()
        self.model = model

    def forward(
        self,
        strains: Tensor,
        time_steps: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Roll out responses and dimensionless latent-state diagnostics."""
        batch_size, time_count, _ = strains.shape
        latent = self.model.initial_state_batch(batch_size)
        anchors = self.model.prepare_energy_anchors()
        stress_history: list[Tensor] = []
        dissipation_history: list[Tensor] = []
        violation_history: list[Tensor] = []
        energy_increment_history: list[Tensor] = []
        latent_norm_history: list[Tensor] = []
        dimensionless_rate_norm_history: list[Tensor] = []
        latent_increment_norm_history: list[Tensor] = []
        for step_id in range(time_count):
            active = valid_mask[:, step_id]
            update = self.model.update_batch(
                strains[:, step_id],
                latent,
                torch.where(
                    active,
                    time_steps[:, step_id],
                    torch.ones_like(time_steps[:, step_id]),
                ),
                anchors,
            )
            latent_increment = update.latent_state - latent
            latent = torch.where(active.unsqueeze(-1), update.latent_state, latent)
            stress_history.append(
                torch.where(active.unsqueeze(-1), update.stress, torch.zeros_like(update.stress))
            )
            dissipation_history.append(
                torch.where(active, update.dissipation, torch.zeros_like(update.dissipation))
            )
            violation_history.append(
                torch.where(
                    active,
                    update.thermodynamic_violation,
                    torch.zeros_like(update.thermodynamic_violation),
                )
            )
            energy_increment_history.append(
                torch.where(
                    active,
                    update.nonequilibrium_energy_increment,
                    torch.zeros_like(update.nonequilibrium_energy_increment),
                )
            )
            latent_norm_history.append(
                torch.where(active, torch.linalg.vector_norm(latent, dim=-1), 0.0)
            )
            dimensionless_rate_norm_history.append(
                torch.where(
                    active,
                    torch.linalg.vector_norm(update.dimensionless_state_rate, dim=-1),
                    0.0,
                )
            )
            latent_increment_norm_history.append(
                torch.where(
                    active,
                    torch.linalg.vector_norm(latent_increment, dim=-1),
                    0.0,
                )
            )
        return (
            torch.stack(stress_history, dim=1),
            torch.stack(dissipation_history, dim=1),
            torch.stack(violation_history, dim=1),
            torch.stack(energy_increment_history, dim=1),
            torch.stack(latent_norm_history, dim=1),
            torch.stack(dimensionless_rate_norm_history, dim=1),
            torch.stack(latent_increment_norm_history, dim=1),
        )


def train_rve_rno(
    config: RVERNOTrainingConfig,
    progress_recorder: JsonProgressRecorder | None = None,
) -> dict[str, Any]:
    """Train, select, test, benchmark, and export one explicit RVE-RNO artifact."""
    _require_device(config.model.device)
    torch.set_float32_matmul_precision(config.execution.matmul_precision)
    torch.manual_seed(config.random_seed)
    if config.model.device == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    logger = _build_logger(output / "train.log")
    warnings_path = output / "warnings.jsonl"
    warnings_path.touch(exist_ok=False)
    recorder = progress_recorder or JsonProgressRecorder(
        output / "progress.json",
        {"calculation": "rve_rno_training", "run_directory": str(output)},
        output / "summary.log",
    )
    acceptance = require_current_training_data_acceptance(
        config.data_acceptance_report,
        config.data_plan,
        config.response_shard_manifest,
    )
    acceptance_scope = str(acceptance.get("acceptance_scope", "strict"))
    recorder.publish_summary(
        "rve_rno_training_initialized",
        {"completed": 0, "total": config.epochs, "acceptance_scope": acceptance_scope},
    )
    _write_run_inputs(output, config, acceptance)
    normalization = acceptance["normalization"]
    metadata = _run_metadata(config, acceptance)
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Starting explicit RVE-RNO training: %s", json.dumps(metadata, sort_keys=True))
    lazy_dataset = HDF5RNOPathDataset(
        _load_shards(config.response_shard_manifest), RNOConditioningSpec((), ())
    )
    dataset = PreloadedRNOPathDataset(lazy_dataset)
    subsets = _build_subsets(dataset, acceptance["split_manifest"])
    loaders = _build_path_loaders(subsets, config)
    model = EnergyDissipationRVERNO(config.model)
    if config.execution.compile.enabled:
        _compile_model_networks(model, config)
    rollout: nn.Module = RVERNOPathRollout(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        fused=config.execution.fused_adam,
    )
    metrics_path = output / "metrics.jsonl"
    state_diagnostics_path = output / "state_diagnostics.jsonl"
    state_diagnostics_path.touch(exist_ok=False)
    path_groups = {
        str(item["path_id"]): (str(item["regime"]), str(item["family"]))
        for item in acceptance["split_manifest"]["paths"]
    }
    best_validation = float("inf")
    best_epoch = 0
    all_metrics: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        training_loader = loaders["train"]
        if isinstance(training_loader, DeviceRNOPathBatchLoader):
            training_loader.set_epoch(epoch)
        train_metrics = _run_epoch(
            model,
            rollout,
            loaders["train"],
            config,
            normalization,
            optimizer,
            epoch,
            "train",
            warnings_path,
            logger,
            path_groups,
            state_diagnostics_path,
        )
        validation_metrics = _run_epoch(
            model,
            rollout,
            loaders["validation"],
            config,
            normalization,
            None,
            epoch,
            "validation",
            warnings_path,
            logger,
            path_groups,
            state_diagnostics_path,
        )
        all_metrics.extend((train_metrics, validation_metrics))
        _append_metrics(metrics_path, train_metrics)
        _append_metrics(metrics_path, validation_metrics)
        if epoch % config.checkpoint_interval == 0:
            _save_checkpoint(
                output / "checkpoints" / f"epoch_{epoch:05d}.pt", model, optimizer, epoch
            )
        if validation_metrics.total_loss < best_validation:
            best_validation = validation_metrics.total_loss
            best_epoch = epoch
            _save_checkpoint(output / "best_model.pt", model, optimizer, epoch)
        epoch_elapsed = train_metrics.elapsed_seconds + validation_metrics.elapsed_seconds
        peak_allocated = max(
            train_metrics.peak_cuda_memory_allocated_bytes,
            validation_metrics.peak_cuda_memory_allocated_bytes,
        )
        logger.info(
            "epoch=%d train=%.8e validation=%.8e best_epoch=%d "
            "best_validation=%.8e stress=%.8e thermo=%.8e dissipation=%.8e "
            "latent_p99=%.8e dimensionless_rate_p99=%.8e increment_p99=%.8e "
            "train_steps_per_second=%.3f elapsed_seconds=%.6f peak_cuda_bytes=%d",
            epoch,
            train_metrics.total_loss,
            validation_metrics.total_loss,
            best_epoch,
            best_validation,
            train_metrics.stress_loss,
            train_metrics.thermodynamic_violation_loss,
            train_metrics.dissipation_loss,
            train_metrics.latent_norm_p99,
            train_metrics.dimensionless_state_rate_norm_p99,
            train_metrics.latent_increment_norm_p99,
            train_metrics.steps_per_second,
            epoch_elapsed,
            peak_allocated,
        )
        recorder.publish_summary(
            "rve_rno_epoch_complete",
            {
                "completed": epoch,
                "total": config.epochs,
                "acceptance_scope": acceptance_scope,
                "train_loss": train_metrics.total_loss,
                "validation_loss": validation_metrics.total_loss,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation,
                "stress_loss": train_metrics.stress_loss,
                "thermodynamic_violation_loss": (train_metrics.thermodynamic_violation_loss),
                "dissipation_loss": train_metrics.dissipation_loss,
                "latent_norm_p99": train_metrics.latent_norm_p99,
                "dimensionless_state_rate_norm_p99": (
                    train_metrics.dimensionless_state_rate_norm_p99
                ),
                "latent_increment_norm_p99": train_metrics.latent_increment_norm_p99,
                "steps_per_second": train_metrics.steps_per_second,
                "elapsed_seconds": epoch_elapsed,
                "peak_cuda_memory_bytes": peak_allocated,
            },
        )
    checkpoint = torch.load(
        output / "best_model.pt", map_location=config.model.device, weights_only=True
    )
    _load_canonical_state_dict(model, checkpoint["model_state_dict"])
    test_metrics = _run_epoch(
        model,
        rollout,
        loaders["test"],
        config,
        normalization,
        None,
        best_epoch,
        "test",
        warnings_path,
        logger,
        path_groups,
        state_diagnostics_path,
    )
    all_metrics.append(test_metrics)
    _append_metrics(metrics_path, test_metrics)
    artifact = _artifact_metadata(config, acceptance, normalization, dataset)
    (output / "artifact_metadata.json").write_text(
        artifact.model_dump_json(indent=2), encoding="utf-8"
    )
    performance = _performance_report(config, all_metrics)
    (output / "performance_report.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "passed": True,
        "acceptance_scope": acceptance_scope,
        "selection_split": "validation",
        "frozen_test_evaluation_count": 1,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "test_metrics": asdict(test_metrics),
        "artifact": str(output / "best_model.pt"),
        "artifact_metadata": str(output / "artifact_metadata.json"),
        "performance_report": str(output / "performance_report.json"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Completed explicit RVE-RNO training: %s", json.dumps(summary, sort_keys=True))
    recorder.publish_summary(
        "rve_rno_training_complete",
        {
            "passed": True,
            "acceptance_scope": acceptance_scope,
            "completed": config.epochs,
            "total": config.epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "test_loss": test_metrics.total_loss,
            "maximum_path_stress_error": test_metrics.maximum_path_stress_error,
            "steps_per_second": test_metrics.steps_per_second,
            "elapsed_seconds": test_metrics.elapsed_seconds,
            "peak_cuda_memory_bytes": test_metrics.peak_cuda_memory_allocated_bytes,
        },
    )
    return summary


def _run_epoch(
    model: EnergyDissipationRVERNO,
    rollout: nn.Module,
    loader: Iterable[RNOPathBatch],
    config: RVERNOTrainingConfig,
    normalization: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    split: str,
    warnings_path: Path,
    logger: logging.Logger,
    path_groups: dict[str, tuple[str, str]],
    state_diagnostics_path: Path,
) -> EpochMetrics:
    """Run one complete-path training or evaluation epoch."""
    model.train(optimizer is not None)
    start = perf_counter()
    totals = [0.0] * 6
    path_count = 0
    batch_count = 0
    material_steps = 0
    executed_material_slots = 0
    maximum_path_error = 0.0
    maximum_violation = 0.0
    violation_count = 0
    maximum_dissipation_error = 0.0
    maximum_energy_increase = 0.0
    state_chunks: dict[str, dict[str, list[Tensor]]] = {}
    if config.model.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in enumerate(loader):
        batch_count += 1
        executed_material_slots += batch.valid_mask.numel()
        context = {
            "epoch": epoch,
            "split": split,
            "batch_index": batch_index,
            "path_ids": list(batch.path_ids),
            "learning_rate": config.learning_rate,
        }
        with _capture_training_warnings(warnings_path, logger, context):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, components, diagnostic_tensors, state_tensors = _batch_loss(
                rollout, batch, config, normalization
            )
            synchronized = _synchronize_batch_scalars(loss, components, diagnostic_tensors, context)
            if optimizer is not None:
                loss.backward()
                _require_finite_gradients(model, {**context, **synchronized["losses"]})
                optimizer.step()
                _require_finite_parameters(model, {**context, **synchronized["losses"]})
            _collect_state_diagnostics(
                state_chunks,
                state_tensors,
                batch.valid_mask,
                batch.path_ids,
                path_groups,
            )
        count = len(batch.path_ids)
        loss_values = synchronized["loss_values"]
        for index, value in enumerate(loss_values):
            totals[index] += value * count
        diagnostics = synchronized["diagnostics"]
        path_count += count
        material_steps += int(diagnostics[5])
        maximum_path_error = max(maximum_path_error, diagnostics[0])
        maximum_violation = max(maximum_violation, diagnostics[1])
        violation_count += int(diagnostics[2])
        maximum_dissipation_error = max(maximum_dissipation_error, diagnostics[3])
        maximum_energy_increase = max(maximum_energy_increase, diagnostics[4])
    elapsed = perf_counter() - start
    averages = [value / path_count for value in totals]
    peak_allocated = int(torch.cuda.max_memory_allocated()) if config.model.device == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if config.model.device == "cuda" else 0
    state_summary = _summarize_state_diagnostics(
        state_chunks,
        epoch,
        split,
        state_diagnostics_path,
    )
    overall_state = state_summary["groups"]["all"]
    return EpochMetrics(
        epoch=epoch,
        split=split,
        total_loss=averages[0],
        stress_loss=averages[1],
        thermodynamic_violation_loss=averages[2],
        dissipation_loss=averages[3],
        weighted_thermodynamic_contribution=averages[4],
        weighted_dissipation_contribution=averages[5],
        maximum_path_stress_error=maximum_path_error,
        maximum_thermodynamic_violation=maximum_violation,
        thermodynamic_violation_count=violation_count,
        maximum_dissipation_error=maximum_dissipation_error,
        maximum_nonequilibrium_energy_increase=maximum_energy_increase,
        latent_norm_median=overall_state["latent_norm"]["median"],
        latent_norm_p95=overall_state["latent_norm"]["p95"],
        latent_norm_p99=overall_state["latent_norm"]["p99"],
        latent_norm_maximum=overall_state["latent_norm"]["maximum"],
        dimensionless_state_rate_norm_median=overall_state["dimensionless_state_rate_norm"][
            "median"
        ],
        dimensionless_state_rate_norm_p95=overall_state["dimensionless_state_rate_norm"]["p95"],
        dimensionless_state_rate_norm_p99=overall_state["dimensionless_state_rate_norm"]["p99"],
        dimensionless_state_rate_norm_maximum=overall_state["dimensionless_state_rate_norm"][
            "maximum"
        ],
        latent_increment_norm_median=overall_state["latent_increment_norm"]["median"],
        latent_increment_norm_p95=overall_state["latent_increment_norm"]["p95"],
        latent_increment_norm_p99=overall_state["latent_increment_norm"]["p99"],
        latent_increment_norm_maximum=overall_state["latent_increment_norm"]["maximum"],
        batch_count=batch_count,
        material_steps=material_steps,
        executed_material_slots=executed_material_slots,
        padding_efficiency=material_steps / executed_material_slots,
        steps_per_second=material_steps / elapsed,
        elapsed_seconds=elapsed,
        peak_cuda_memory_allocated_bytes=peak_allocated,
        peak_cuda_memory_reserved_bytes=peak_reserved,
    )


def _batch_loss(
    rollout: nn.Module,
    batch: RNOPathBatch,
    config: RVERNOTrainingConfig,
    normalization: dict[str, Any],
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Evaluate path-normalized stress and auxiliary thermodynamic losses."""
    device = torch.device(config.model.device)
    dtype = _torch_dtype(config.model.dtype)
    non_blocking = config.execution.non_blocking_transfer and device.type == "cuda"
    strains = batch.network_inputs[..., :6].to(
        device=device, dtype=dtype, non_blocking=non_blocking
    )
    time_steps = batch.time_step.to(device=device, dtype=dtype, non_blocking=non_blocking)
    target_stress = batch.macro_stress.to(device=device, dtype=dtype, non_blocking=non_blocking)
    target_dissipation = batch.dissipation_density.to(
        device=device, dtype=dtype, non_blocking=non_blocking
    )
    mask = batch.valid_mask.to(device=device, non_blocking=non_blocking)
    (
        stress,
        dissipation,
        violation,
        energy_increment,
        latent_norm,
        dimensionless_rate_norm,
        latent_increment_norm,
    ) = rollout(strains, time_steps, mask)
    stress_scale = _normalization_scale(normalization, "macro_stress", device, dtype)
    dissipation_scale = _normalization_scale(
        normalization, "dissipation_density", device, dtype
    ).reshape(())
    mask_float = mask.to(dtype)
    stress_difference = stress - target_stress
    component_stress_numerator = (
        (stress_difference * stress_difference) * mask_float.unsqueeze(-1)
    ).sum(dim=1)
    stress_numerator = component_stress_numerator.sum(dim=-1)
    target_energy = ((target_stress * target_stress).sum(dim=-1) * mask_float).sum(dim=-1)
    valid_steps = mask_float.sum(dim=-1)
    stress_floor = valid_steps * (stress_scale * stress_scale).sum()
    stress_denominator = torch.maximum(target_energy, stress_floor)
    path_stress_diagnostic_losses = stress_numerator / stress_denominator
    if config.loss.stress == "path_normalized_mse":
        path_stress_losses = path_stress_diagnostic_losses
    else:
        component_target_energy = (
            (target_stress * target_stress) * mask_float.unsqueeze(-1)
        ).sum(dim=1)
        component_stress_floor = valid_steps.unsqueeze(-1) * (stress_scale * stress_scale)
        component_stress_denominator = torch.maximum(
            component_target_energy,
            component_stress_floor,
        )
        path_stress_losses = (
            component_stress_numerator / component_stress_denominator
        ).mean(dim=-1)
    stress_loss = path_stress_losses.mean()
    valid_count = mask_float.sum()
    thermodynamic_loss = (((violation / dissipation_scale) ** 2) * mask_float).sum() / valid_count
    dissipation_difference = (dissipation - target_dissipation) / dissipation_scale
    dissipation_loss = ((dissipation_difference**2) * mask_float).sum() / valid_count
    weighted_thermodynamic = config.loss.thermodynamic_violation_weight * thermodynamic_loss
    weighted_dissipation = config.loss.dissipation_supervision_weight * dissipation_loss
    total = stress_loss + weighted_thermodynamic + weighted_dissipation
    masked_violation = torch.where(mask, violation, torch.zeros_like(violation))
    physical_dissipation_error = torch.where(
        mask,
        torch.abs(dissipation - target_dissipation),
        torch.zeros_like(dissipation),
    )
    positive_energy_increment = torch.where(
        mask, torch.relu(energy_increment), torch.zeros_like(energy_increment)
    )
    diagnostics = (
        torch.sqrt(path_stress_diagnostic_losses.max()),
        masked_violation.max(),
        (masked_violation > 0.0).sum().to(dtype),
        physical_dissipation_error.max(),
        positive_energy_increment.max(),
        valid_count,
        latent_norm.max(),
        dimensionless_rate_norm.max(),
        latent_increment_norm.max(),
    )
    return (
        total,
        (
            stress_loss,
            thermodynamic_loss,
            dissipation_loss,
            weighted_thermodynamic,
            weighted_dissipation,
        ),
        diagnostics,
        (latent_norm, dimensionless_rate_norm, latent_increment_norm),
    )


def _collect_state_diagnostics(
    chunks: dict[str, dict[str, list[Tensor]]],
    state_tensors: tuple[Tensor, ...],
    valid_mask: Tensor,
    path_ids: tuple[str, ...],
    path_groups: dict[str, tuple[str, str]],
) -> None:
    """Collect valid state norms by response regime and loading family."""
    names = (
        "latent_norm",
        "dimensionless_state_rate_norm",
        "latent_increment_norm",
    )
    values = torch.stack(state_tensors).detach()
    device_mask = valid_mask.detach().to(device=values.device)
    for path_index, path_id in enumerate(path_ids):
        regime, family = path_groups[path_id]
        keys = (
            "all",
            f"regime:{regime}",
            f"family:{family}",
            f"regime:{regime}|family:{family}",
        )
        mask = device_mask[path_index]
        for key in keys:
            group = chunks.setdefault(key, {name: [] for name in names})
            for metric_index, name in enumerate(names):
                group[name].append(values[metric_index, path_index, mask])


def _summarize_state_diagnostics(
    chunks: dict[str, dict[str, list[Tensor]]],
    epoch: int,
    split: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write median, p95, p99, and maximum state norms for one epoch split."""
    groups: dict[str, dict[str, dict[str, float | int]]] = {}
    for group_name, metrics in sorted(chunks.items()):
        groups[group_name] = {}
        for metric_name, arrays in metrics.items():
            values = torch.cat(arrays).to(torch.float64)
            quantile_levels = torch.tensor(
                [0.5, 0.95, 0.99], dtype=torch.float64, device=values.device
            )
            if not bool(torch.isfinite(values).all()):
                raise TrainingNumericalError(
                    "non_finite_latent_state",
                    {
                        "epoch": epoch,
                        "split": split,
                        "group": group_name,
                        "metric": metric_name,
                        "non_finite_count": int((~torch.isfinite(values)).sum()),
                    },
                )
            quantiles = torch.quantile(values, quantile_levels)
            groups[group_name][metric_name] = {
                "sample_count": int(values.numel()),
                "median": float(quantiles[0]),
                "p95": float(quantiles[1]),
                "p99": float(quantiles[2]),
                "maximum": float(torch.max(values)),
            }
    summary = {"epoch": epoch, "split": split, "groups": groups}
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")
    return summary


def benchmark_rve_rno_batching(
    model: EnergyDissipationRVERNO,
    batch: RNOPathBatch,
    maximum_time_steps: int = 16,
) -> dict[str, Any]:
    """Compare batched and scalar explicit rollouts without changing the algorithm."""
    device = model.device
    dtype = model.dtype
    time_count = min(batch.time_step.shape[1], maximum_time_steps)
    strains = batch.network_inputs[:, :time_count, :6].to(device=device, dtype=dtype)
    time_steps = batch.time_step[:, :time_count].to(device=device, dtype=dtype)
    mask = batch.valid_mask[:, :time_count].to(device=device)
    rollout = RVERNOPathRollout(model)
    rollout(strains, time_steps, mask)
    _scalar_reference_rollout(model, strains, time_steps, mask)
    _synchronize_device(device)
    batched_start = perf_counter()
    batched = rollout(strains, time_steps, mask)
    _synchronize_device(device)
    batched_seconds = perf_counter() - batched_start
    _synchronize_device(device)
    scalar_start = perf_counter()
    scalar = _scalar_reference_rollout(model, strains, time_steps, mask)
    _synchronize_device(device)
    scalar_seconds = perf_counter() - scalar_start
    valid_steps = int(mask.sum().detach().cpu())
    maximum_difference = max(
        float(torch.max(torch.abs(batched_value - scalar_value)).detach().cpu())
        for batched_value, scalar_value in zip(batched, scalar, strict=True)
    )
    tolerance = 1.0e-10 if dtype == torch.float64 else 2.0e-5
    speedup = scalar_seconds / batched_seconds
    return {
        "reference": "same_explicit_algorithm_scalar_path_loop",
        "batch_size": strains.shape[0],
        "time_steps": time_count,
        "valid_material_steps": valid_steps,
        "maximum_absolute_difference": maximum_difference,
        "equivalence_tolerance": tolerance,
        "accuracy_equivalent": maximum_difference <= tolerance,
        "batched_seconds": batched_seconds,
        "scalar_seconds": scalar_seconds,
        "batched_steps_per_second": valid_steps / batched_seconds,
        "scalar_steps_per_second": valid_steps / scalar_seconds,
        "speedup": speedup,
        "required_speedup": 10.0,
        "performance_gate_passed": speedup >= 10.0,
    }


def _compile_model_networks(
    model: EnergyDissipationRVERNO,
    config: RVERNOTrainingConfig,
) -> None:
    """Compile the explicitly selected RVE-RNO tensor scope."""
    compile_config = config.execution.compile
    if compile_config.scope == "state_evolution":
        model.state_evolution = cast(
            nn.Module,
            torch.compile(
                model.state_evolution,
                backend=compile_config.backend,
                fullgraph=compile_config.fullgraph,
                dynamic=compile_config.dynamic,
                mode=compile_config.mode,
            ),
        )
    else:
        model.compile_constitutive_step(
            compile_config.backend,
            compile_config.fullgraph,
            compile_config.dynamic,
            compile_config.mode,
        )


def _scalar_reference_rollout(
    model: EnergyDissipationRVERNO,
    strains: Tensor,
    time_steps: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluate the same explicit update through independent scalar calls."""
    stress = torch.zeros_like(strains)
    dissipation = torch.zeros_like(time_steps)
    violation = torch.zeros_like(time_steps)
    energy_increment = torch.zeros_like(time_steps)
    latent_norm = torch.zeros_like(time_steps)
    dimensionless_rate_norm = torch.zeros_like(time_steps)
    latent_increment_norm = torch.zeros_like(time_steps)
    anchors = model.prepare_energy_anchors()
    for path_id in range(strains.shape[0]):
        latent = model.initial_state_batch(1)
        for step_id in range(strains.shape[1]):
            if bool(mask[path_id, step_id].detach().cpu()):
                update = model.update_batch(
                    strains[path_id, step_id].unsqueeze(0),
                    latent,
                    time_steps[path_id, step_id].reshape(1),
                    anchors,
                )
                stress[path_id, step_id] = update.stress[0]
                dissipation[path_id, step_id] = update.dissipation[0]
                violation[path_id, step_id] = update.thermodynamic_violation[0]
                energy_increment[path_id, step_id] = update.nonequilibrium_energy_increment[0]
                latent_norm[path_id, step_id] = torch.linalg.vector_norm(update.latent_state[0])
                dimensionless_rate_norm[path_id, step_id] = torch.linalg.vector_norm(
                    update.dimensionless_state_rate[0]
                )
                latent_increment_norm[path_id, step_id] = torch.linalg.vector_norm(
                    update.latent_state[0] - latent[0]
                )
                latent = update.latent_state
    return (
        stress,
        dissipation,
        violation,
        energy_increment,
        latent_norm,
        dimensionless_rate_norm,
        latent_increment_norm,
    )


def _synchronize_batch_scalars(
    loss: Tensor,
    components: tuple[Tensor, ...],
    diagnostics: tuple[Tensor, ...],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Perform one device synchronization for all batch loss and diagnostic scalars."""
    packed = torch.stack((loss, *components, *diagnostics)).detach().cpu()
    values = [float(value) for value in packed]
    loss_values = [values[0], *values[1:6]]
    names = (
        "total_loss",
        "stress_loss",
        "thermodynamic_violation_loss",
        "dissipation_loss",
        "weighted_thermodynamic_contribution",
        "weighted_dissipation_contribution",
    )
    non_finite = [
        name for name, value in zip(names, loss_values, strict=True) if not math.isfinite(value)
    ]
    if non_finite:
        raise TrainingNumericalError(
            "non_finite_loss",
            {
                **context,
                "non_finite_fields": non_finite,
                "losses": {
                    name: _diagnostic_scalar(value)
                    for name, value in zip(names, loss_values, strict=True)
                },
            },
        )
    diagnostic_names = (
        "maximum_path_stress_error",
        "maximum_thermodynamic_violation",
        "thermodynamic_violation_count",
        "maximum_dissipation_error",
        "maximum_nonequilibrium_energy_increase",
        "valid_material_step_count",
        "maximum_latent_norm",
        "maximum_dimensionless_state_rate_norm",
        "maximum_latent_increment_norm",
    )
    diagnostic_values = values[6:]
    non_finite_diagnostics = [
        name
        for name, value in zip(diagnostic_names, diagnostic_values, strict=True)
        if not math.isfinite(value)
    ]
    if non_finite_diagnostics:
        raise TrainingNumericalError(
            "non_finite_batch_diagnostic",
            {
                **context,
                "non_finite_fields": non_finite_diagnostics,
                "diagnostics": {
                    name: _diagnostic_scalar(value)
                    for name, value in zip(diagnostic_names, diagnostic_values, strict=True)
                },
            },
        )
    return {
        "loss_values": loss_values,
        "diagnostics": diagnostic_values,
        "losses": dict(zip(names, loss_values, strict=True)),
    }


def _normalization_scale(
    normalization: dict[str, Any],
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Load one accepted training-only RMS scale tensor."""
    return torch.tensor(normalization[name]["rms_scale"], dtype=dtype, device=device)


def _build_subsets(
    dataset: PreloadedRNOPathDataset,
    split_manifest: dict[str, Any],
) -> dict[str, IndexedPathSubset]:
    """Map acceptance-report path identifiers to deterministic in-memory subsets."""
    split_by_id = {item["path_id"]: item["split"] for item in split_manifest["paths"]}
    indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for index in range(len(dataset)):
        path_id = dataset[index].path_id
        indices[split_by_id[path_id]].append(index)
    if any(not values for values in indices.values()):
        raise ValueError(f"RVE-RNO training requires nonempty data splits: {indices}.")
    return {name: IndexedPathSubset(dataset, tuple(values)) for name, values in indices.items()}


def _build_path_loaders(
    subsets: dict[str, IndexedPathSubset],
    config: RVERNOTrainingConfig,
) -> dict[str, Iterable[RNOPathBatch]]:
    """Build legacy padded or optimized device-resident path loaders."""
    if config.execution.batching_strategy == "random_padded":
        pin_memory = config.execution.pin_memory and config.model.device == "cuda"
        return {
            name: DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=name == "train",
                collate_fn=collate_rno_paths,
                pin_memory=pin_memory,
                num_workers=config.execution.num_workers,
            )
            for name, subset in subsets.items()
        }
    device = torch.device(config.execution.preload_device)
    dtype = _torch_dtype(config.model.dtype)
    loaders: dict[str, Iterable[RNOPathBatch]] = {}
    for name, subset in subsets.items():
        if name != "train" and config.execution.validation_batching_strategy == "padded_batches":
            loaders[name] = DataLoader(
                subset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_rno_paths,
                pin_memory=False,
                num_workers=config.execution.num_workers,
            )
            continue
        samples = tuple(subset[index] for index in range(len(subset)))
        strategy = (
            "length_bucket" if name == "train" else config.execution.validation_batching_strategy
        )
        loaders[name] = DeviceRNOPathBatchLoader(
            samples,
            config.batch_size,
            device,
            dtype,
            strategy,
            shuffle=name == "train",
            random_seed=config.random_seed,
        )
    return loaders


def _load_shards(path: Path) -> tuple[Path, ...]:
    """Load the exact legacy or portable shard list."""
    from concrete_impact.core.response_manifest import load_response_shard_paths

    return load_response_shard_paths(path)


def _write_run_inputs(
    output: Path,
    config: RVERNOTrainingConfig,
    acceptance: dict[str, Any],
) -> None:
    """Write resolved immutable training inputs into the per-run directory."""
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    (output / "normalization.json").write_text(
        json.dumps(acceptance["normalization"], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_checkpoint(
    path: Path,
    model: EnergyDissipationRVERNO,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Write one restartable state-dict checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _canonical_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _canonical_state_dict(model: EnergyDissipationRVERNO) -> dict[str, Tensor]:
    """Remove compilation wrapper names from a deployment-compatible state dict."""
    return {
        name.replace("state_evolution._orig_mod.", "state_evolution."): value
        for name, value in model.state_dict().items()
    }


def _load_canonical_state_dict(
    model: EnergyDissipationRVERNO,
    canonical: dict[str, Tensor],
) -> None:
    """Load canonical weights into an eager or compiled training model."""
    target = {
        name: canonical[name.replace("state_evolution._orig_mod.", "state_evolution.")]
        for name in model.state_dict()
    }
    model.load_state_dict(target, strict=True)


def _append_metrics(path: Path, metrics: EpochMetrics) -> None:
    """Append one structured epoch record to the JSON-lines log."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True, allow_nan=False) + "\n")


def _require_finite_gradients(
    model: EnergyDissipationRVERNO,
    context: dict[str, Any],
) -> None:
    """Reject non-finite gradients after one aggregate device check."""
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    aggregate = torch.stack(
        [torch.isfinite(gradient).all() for _, gradient in named_gradients]
    ).all()
    if bool(aggregate.detach().cpu()):
        return
    for name, gradient in named_gradients:
        finite = torch.isfinite(gradient)
        if not bool(finite.all().detach().cpu()):
            raise TrainingNumericalError(
                "non_finite_gradient",
                {
                    **context,
                    "parameter_name": name,
                    "non_finite_count": int((~finite).sum().detach().cpu()),
                    "gradient_element_count": gradient.numel(),
                },
            )
    raise RuntimeError("Aggregate gradient check failed without a non-finite parameter.")


def _require_finite_parameters(
    model: EnergyDissipationRVERNO,
    context: dict[str, Any],
) -> None:
    """Reject non-finite parameters after one aggregate device check."""
    named_parameters = list(model.named_parameters())
    aggregate = torch.stack(
        [torch.isfinite(parameter).all() for _, parameter in named_parameters]
    ).all()
    if bool(aggregate.detach().cpu()):
        return
    for name, parameter in named_parameters:
        finite = torch.isfinite(parameter)
        if not bool(finite.all().detach().cpu()):
            raise TrainingNumericalError(
                "non_finite_parameter",
                {
                    **context,
                    "parameter_name": name,
                    "non_finite_count": int((~finite).sum().detach().cpu()),
                    "parameter_element_count": parameter.numel(),
                },
            )
    raise RuntimeError("Aggregate parameter check failed without a non-finite parameter.")


@contextmanager
def _capture_training_warnings(
    path: Path,
    logger: logging.Logger,
    context: dict[str, Any],
) -> Iterator[None]:
    """Record Python and PyTorch warnings with their exact training context."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            yield
        finally:
            for warning in captured:
                record = {
                    **context,
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "line_number": warning.lineno,
                    "recorded_unix_time": time(),
                }
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                logger.warning("training_warning=%s", json.dumps(record, sort_keys=True))


def _diagnostic_scalar(value: float) -> float | str:
    """Convert one scalar into a JSON-safe finite value or non-finite label."""
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_inf"
    if value == -math.inf:
        return "negative_inf"
    return value


def _require_device(device: str) -> None:
    """Reject unavailable CUDA explicitly without CPU substitution."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RVE-RNO training requested CUDA, but CUDA is unavailable.")


def _build_logger(path: Path) -> logging.Logger:
    """Create one file-only training logger for the run directory."""
    logger = logging.getLogger(f"rve_rno_training::{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _run_metadata(config: RVERNOTrainingConfig, acceptance: dict[str, Any]) -> dict[str, Any]:
    """Collect reproducible software, device, data, loss, and execution metadata."""
    cuda = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "device": config.model.device,
        "model": config.model.model_dump(mode="json"),
        "loss": config.loss.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "optimizer": {
            "name": "Adam",
            "learning_rate": config.learning_rate,
            "fused": config.execution.fused_adam,
        },
        "random_seed": config.random_seed,
        "acceptance_scope": acceptance.get("acceptance_scope", "strict"),
        "acceptance_limitation_note": acceptance.get("coverage_policy", {}).get("limitation_note"),
        "data_plan_sha256": acceptance["data_plan_sha256"],
        "split_manifest_sha256": acceptance["split_manifest_sha256"],
        "normalization_sha256": acceptance["normalization_sha256"],
    }


def _artifact_metadata(
    config: RVERNOTrainingConfig,
    acceptance: dict[str, Any],
    normalization: dict[str, Any],
    dataset: PreloadedRNOPathDataset,
) -> RVERNOArtifactMetadata:
    """Build the explicit-only artifact contract and fixed-cell applicability record."""
    plan = load_rve_data_plan(config.data_plan)
    report_path = plan.mesh_selection_report
    if report_path is None:
        raise ValueError("RVE-RNO artifact requires a mesh-selection report.")
    mesh_report = json.loads(report_path.read_text(encoding="utf-8"))
    minimum_time_step = min(
        float(sample.time_step.min())
        for sample in (dataset[index] for index in range(len(dataset)))
    )
    maximum_time_step = max(
        float(sample.time_step.max())
        for sample in (dataset[index] for index in range(len(dataset)))
    )
    return RVERNOArtifactMetadata(
        model_family="energy_rve_rno",
        integrator=config.model.integrator,
        evolution=config.model.evolution,
        voigt_order=("xx", "yy", "zz", "yz", "xz", "xy"),
        strain_convention="engineering_shear",
        stress_components=("xx", "yy", "zz", "yz", "xz", "xy"),
        latent_dimension=config.model.latent_dimension,
        hidden_size=config.model.hidden_size,
        hidden_layers=config.model.hidden_layers,
        activation=config.model.activation,
        strain_input_scale=config.model.strain_input_scale,
        time_representation="continuous",
        reference_time_scale=config.model.reference_time_scale,
        time_step_role="nondimensional_forward_euler_increment",
        deployment_integrator="explicit_only",
        accepted_time_step_range=(minimum_time_step, maximum_time_step),
        dtype=config.model.dtype,
        normalization=normalization,
        capabilities={
            "stress": True,
            "dissipation": True,
            "free_energy": True,
            "algorithmic_tangent": True,
            "thermodynamic_constraint_exact": config.model.evolution == "mobility_gradient",
            "microscopic_peak_stress": False,
            "parameterized_microstructure": False,
            "dynamic_rve": False,
        },
        data_plan_sha256=acceptance["data_plan_sha256"],
        response_shard_manifest_sha256=acceptance["response_shard_manifest_sha256"],
        split_manifest_sha256=acceptance["split_manifest_sha256"],
        normalization_sha256=acceptance["normalization_sha256"],
        mesh_quality=str(mesh_report["reason"]),
        acceptance_scope=cast(
            Literal["strict", "pilot"], acceptance.get("acceptance_scope", "strict")
        ),
        acceptance_limitation_note=acceptance.get("coverage_policy", {}).get("limitation_note"),
    )


def _performance_report(
    config: RVERNOTrainingConfig,
    metrics: list[EpochMetrics],
) -> dict[str, Any]:
    """Summarize observed training throughput and GPU memory without inference."""
    training = [record for record in metrics if record.split == "train"]
    return {
        "device": config.model.device,
        "dtype": config.model.dtype,
        "integrator": config.model.integrator,
        "evolution": config.model.evolution,
        "path_batch_size": config.batch_size,
        "preload_to_memory": config.execution.preload_to_memory,
        "preload_device": config.execution.preload_device,
        "batching_strategy": config.execution.batching_strategy,
        "validation_batching_strategy": config.execution.validation_batching_strategy,
        "pin_memory": config.execution.pin_memory,
        "non_blocking_transfer": config.execution.non_blocking_transfer,
        "fused_adam": config.execution.fused_adam,
        "matmul_precision": config.execution.matmul_precision,
        "compile": config.execution.compile.model_dump(mode="json"),
        "compiled_scope": (
            config.execution.compile.scope if config.execution.compile.enabled else "disabled"
        ),
        "training_epoch_steps_per_second": [record.steps_per_second for record in training],
        "training_optimizer_updates_per_epoch": [record.batch_count for record in training],
        "training_padding_efficiency": [record.padding_efficiency for record in training],
        "maximum_training_steps_per_second": max(record.steps_per_second for record in training),
        "maximum_cuda_memory_allocated_bytes": max(
            record.peak_cuda_memory_allocated_bytes for record in metrics
        ),
        "maximum_cuda_memory_reserved_bytes": max(
            record.peak_cuda_memory_reserved_bytes for record in metrics
        ),
        "batching_gate": (
            "Run benchmark_rve_rno_batching in the CPU/GPU smoke suite; "
            "the required speedup is 10x over the identical scalar explicit update."
        ),
    }


def _synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA only at explicit benchmark timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _torch_dtype(name: str) -> torch.dtype:
    """Map a validated precision name to a PyTorch dtype."""
    return {"float64": torch.float64, "float32": torch.float32}[name]
