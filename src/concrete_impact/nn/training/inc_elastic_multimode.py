"""CUDA training for the fixed-system elastic multimode INC.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from concrete_impact.nn.config_inc_elastic import INCElasticMultimodeConfig
from concrete_impact.nn.datasets.inc_elastic_multimode import (
    INCElasticNormalization,
    elastic_inc_feature_layout,
    elastic_inc_input_width,
    load_inc_elastic_cache,
    prepare_inc_elastic_training_cache,
)
from concrete_impact.nn.evaluation.inc_elastic_multimode import (
    evaluate_inc_elastic_paths,
    kinematic_validation_score,
    plot_inc_elastic_kinematic_selection,
    plot_inc_elastic_training_history,
    validation_score,
)
from concrete_impact.nn.models.inc_elastic_multimode import INCElasticMultimodeMLP


class INCElasticTrainingError(RuntimeError):
    """Report one explicit training or complete-validation failure."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        """Store reproducible failure diagnostics."""
        super().__init__(f"Elastic multimode INC training failed: {reason}.")
        self.reason = reason
        self.diagnostics = diagnostics


def inspect_inc_elastic_cuda(config: INCElasticMultimodeConfig) -> dict[str, Any]:
    """Require the configured CUDA device and minimum memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; elastic INC training will not use a CPU fallback.")
    device = torch.device(config.model.device)
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = float(1024**3)
    report = {
        "cuda_available": True,
        "device_name": properties.name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "total_vram_gib": total_bytes / gib,
        "free_vram_gib": free_bytes / gib,
    }
    total_vram_gib = total_bytes / gib
    free_vram_gib = free_bytes / gib
    if total_vram_gib < config.execution.minimum_total_vram_gib:
        raise RuntimeError(f"CUDA total memory is below the configured minimum: {report}.")
    if free_vram_gib < config.execution.minimum_free_vram_gib:
        raise RuntimeError(f"CUDA free memory is below the configured minimum: {report}.")
    probe = torch.ones((16, 16), dtype=torch.float32, device=device)
    probe_result = probe @ probe
    torch.cuda.synchronize(device)
    if not bool(torch.all(torch.isfinite(probe_result))):
        raise FloatingPointError("CUDA float32 matrix probe produced non-finite values.")
    return report


def train_inc_elastic_multimode(config: INCElasticMultimodeConfig) -> dict[str, Any]:
    """Train and validate one state-independent two-stage INC model."""
    cuda_report = inspect_inc_elastic_cuda(config)
    prepared = prepare_inc_elastic_training_cache(config)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    logger = _build_logger(output / "train.log")
    _write_run_inputs(config, prepared.normalization, cuda_report, output)

    torch.manual_seed(config.training.random_seed)
    torch.cuda.manual_seed_all(config.training.random_seed)
    torch.set_float32_matmul_precision(config.execution.matmul_precision)
    train_cpu, train_metadata = load_inc_elastic_cache(config, "train")
    validation_cpu, validation_metadata = load_inc_elastic_cache(config, "validation")
    if train_metadata["normalization"] != validation_metadata["normalization"]:
        raise ValueError("Train and validation caches use different normalization values.")
    train = {name: value.to(device="cuda") for name, value in train_cpu.items()}
    validation = {name: value.to(device="cuda") for name, value in validation_cpu.items()}
    del train_cpu, validation_cpu

    model = INCElasticMultimodeMLP(
        config.model,
        elastic_inc_input_width(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    physical = _build_physical_tensors(config, prepared, device="cuda")
    zero_corrector_metrics = _evaluate_zero_corrector_losses(train, physical, config)
    _validate_zero_corrector_losses(zero_corrector_metrics, config)
    (output / "zero_corrector_loss_check.json").write_text(
        json.dumps(zero_corrector_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = output / "metrics.jsonl"
    best_score = float("inf")
    best_epoch = 0
    started = perf_counter()

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train,
            physical,
            config,
            optimizer,
            epoch,
        )
        validation_metrics = _run_epoch(
            model,
            validation,
            physical,
            config,
            None,
            epoch,
        )
        full_score = None
        full_records = None
        if epoch % config.training.full_validation_interval == 0:
            full_records, _ = evaluate_inc_elastic_paths(
                config,
                model,
                prepared.system,
                prepared.normalization,
                prepared.angular_frequencies,
                prepared.final_time,
                config.data.validation_paths,
            )
            full_score = max(validation_score(record, config) for record in full_records)
            if not np.isfinite(full_score):
                raise INCElasticTrainingError(
                    "non_finite_full_validation_score",
                    {"epoch": epoch, "records": full_records},
                )
            if full_score < best_score:
                best_score = full_score
                best_epoch = epoch
                _save_best_model(output / "best_model.pt", model, epoch, full_score)
                (output / "best_validation_metrics.json").write_text(
                    json.dumps(full_records, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        record = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
            "full_validation_score": full_score,
            "elapsed_seconds": perf_counter() - started,
        }
        _append_json_line(metrics_path, record)
        if epoch % config.training.checkpoint_interval == 0:
            _save_checkpoint(
                output / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                full_score,
            )
        logger.info(
            "epoch=%d train=%.8e validation=%.8e full_score=%s elapsed=%.1f",
            epoch,
            train_metrics["total_loss"],
            validation_metrics["total_loss"],
            "pending" if full_score is None else f"{full_score:.8e}",
            record["elapsed_seconds"],
        )

    if best_epoch == 0:
        raise INCElasticTrainingError(
            "no_complete_validation_checkpoint",
            {"epochs": config.training.epochs},
        )
    _load_model_state(output / "best_model.pt", model)
    final_records, _ = evaluate_inc_elastic_paths(
        config,
        model,
        prepared.system,
        prepared.normalization,
        prepared.angular_frequencies,
        prepared.final_time,
        config.data.validation_paths,
    )
    passed = all(record["passed"] for record in final_records)
    artifact = _build_artifact_metadata(
        config,
        prepared.normalization,
        prepared.angular_frequencies,
        prepared.final_time,
        prepared.system.time_step,
        best_epoch,
        best_score,
    )
    (output / "artifact_metadata.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_inc_elastic_training_history(metrics_path, output / "training_convergence.png")
    summary = {
        "schema_version": "1.0",
        "passed": passed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "elapsed_seconds": perf_counter() - started,
        "model_path": str(output / "best_model.pt"),
        "metadata_path": str(output / "artifact_metadata.json"),
        "validation_paths": list(config.data.validation_paths),
        "test_paths_opened": [],
        "zero_corrector_loss_check": zero_corrector_metrics,
        "validation_metrics": final_records,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise INCElasticTrainingError(
            "complete_validation_acceptance_failed",
            {"best_epoch": best_epoch, "validation_metrics": final_records},
        )
    return summary


def load_inc_elastic_artifact(
    config: INCElasticMultimodeConfig,
    checkpoint_name: str = "best_model.pt",
) -> tuple[
    INCElasticMultimodeMLP,
    dict[str, Any],
    INCElasticNormalization,
    NDArray[np.float64],
    float,
]:
    """Load one selected checkpoint with its fixed deployment metadata."""
    metadata_path = config.output_directory / "artifact_metadata.json"
    model_path = config.output_directory / checkpoint_name
    if checkpoint_name not in {"best_model.pt", "response_model.pt"}:
        raise ValueError(f"Unsupported elastic INC checkpoint: {checkpoint_name}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model"] != config.model.model_dump(mode="json"):
        raise ValueError("Elastic INC artifact model configuration differs from the request.")
    expected_input_width = elastic_inc_input_width(
        config.features.modal_count,
        config.features.include_duration_modal_phase,
    )
    if metadata["input_width"] != expected_input_width:
        raise ValueError("Elastic INC artifact input width differs from the request.")
    normalization = INCElasticNormalization.from_mapping(metadata["normalization"])
    angular_frequencies = np.asarray(metadata["angular_frequencies"], dtype=np.float64)
    if angular_frequencies.shape != (config.features.modal_count,):
        raise ValueError("Elastic INC artifact modal-frequency count differs from the request.")
    if normalization.residual_start_scale.shape != (config.model.free_dof_count,):
        raise ValueError("Elastic INC artifact start-residual scale has the wrong width.")
    if normalization.residual_end_scale.shape != (config.model.free_dof_count,):
        raise ValueError("Elastic INC artifact end-residual scale has the wrong width.")
    model = INCElasticMultimodeMLP(config.model, int(metadata["input_width"]))
    _load_model_state(model_path, model)
    model.eval()
    return model, metadata, normalization, angular_frequencies, float(metadata["final_time"])


def select_inc_elastic_kinematic_checkpoint(
    config: INCElasticMultimodeConfig,
) -> dict[str, Any]:
    """Select a response candidate from completed checkpoints without opening test paths."""
    inspect_inc_elastic_cuda(config)
    prepared = prepare_inc_elastic_training_cache(config)
    output = config.output_directory
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    if summary["test_paths_opened"] != []:
        raise ValueError("Kinematic checkpoint selection requires an untouched test split.")

    model = INCElasticMultimodeMLP(
        config.model,
        elastic_inc_input_width(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
    )
    candidate_records = []
    best_score = float("inf")
    best_epoch = 0
    best_metrics: list[dict[str, Any]] = []
    response_model_path = output / "response_model.pt"
    for checkpoint_path in sorted((output / "checkpoints").glob("epoch_*.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        records, _ = evaluate_inc_elastic_paths(
            config,
            model,
            prepared.system,
            prepared.normalization,
            prepared.angular_frequencies,
            prepared.final_time,
            config.data.validation_paths,
        )
        score = max(kinematic_validation_score(record, config) for record in records)
        candidate_records.append({"epoch": checkpoint["epoch"], "score": score})
        if score < best_score:
            best_score = score
            best_epoch = int(checkpoint["epoch"])
            best_metrics = records
            partial = response_model_path.with_suffix(".pt.partial")
            torch.save(
                {
                    "model_state_dict": checkpoint["model_state_dict"],
                    "epoch": best_epoch,
                    "kinematic_validation_score": best_score,
                },
                partial,
            )
            partial.replace(response_model_path)

    if best_epoch == 0:
        raise INCElasticTrainingError(
            "no_checkpoint_for_kinematic_selection",
            {"checkpoint_directory": str(output / "checkpoints")},
        )
    response_summary = {
        "schema_version": "1.0",
        "selection_objective": "worst_normalized_displacement_velocity_ratio",
        "kinematic_acceptance_passed": best_score <= 1.0,
        "full_acceptance_passed": all(record["passed"] for record in best_metrics),
        "best_epoch": best_epoch,
        "best_kinematic_validation_score": best_score,
        "model_path": str(response_model_path),
        "validation_paths": list(config.data.validation_paths),
        "test_paths_opened": [],
        "validation_metrics": best_metrics,
        "candidate_scores": candidate_records,
    }
    (output / "response_candidate_summary.json").write_text(
        json.dumps(response_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_inc_elastic_kinematic_selection(
        response_summary,
        output / "response_checkpoint_selection.png",
    )
    return response_summary


def _run_epoch(
    model: INCElasticMultimodeMLP,
    tensors: dict[str, torch.Tensor],
    physical: dict[str, torch.Tensor | float],
    config: INCElasticMultimodeConfig,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> dict[str, float]:
    """Run one deterministic full-tensor train or validation epoch."""
    model.train(optimizer is not None)
    sample_count = tensors["features"].shape[0]
    if optimizer is None:
        order = torch.arange(sample_count, device="cuda")
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(config.training.random_seed + epoch)
        order = torch.randperm(sample_count, generator=generator, device="cuda")
    totals = np.zeros(4, dtype=np.float64)
    for first in range(0, sample_count, config.execution.batch_size):
        indices = order[first : first + config.execution.batch_size]
        batch = {name: values[indices] for name, values in tensors.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if optimizer is None:
            with torch.no_grad():
                losses = _batch_losses(model, batch, physical, config)
        else:
            losses = _batch_losses(model, batch, physical, config)
        if not bool(torch.isfinite(losses["total_loss"])):
            raise INCElasticTrainingError(
                "non_finite_loss",
                {
                    "epoch": epoch,
                    "batch_first": first,
                    "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
                },
            )
        if optimizer is not None:
            losses["total_loss"].backward()
            optimizer.step()
        weight = indices.numel()
        totals += weight * np.asarray(
            [
                float(losses["total_loss"].detach().cpu()),
                float(losses["label_loss"].detach().cpu()),
                float(losses["step_loss"].detach().cpu()),
                float(losses["energy_loss"].detach().cpu()),
            ]
        )
    totals /= sample_count
    return {
        "total_loss": float(totals[0]),
        "label_loss": float(totals[1]),
        "step_loss": float(totals[2]),
        "energy_loss": float(totals[3]),
    }


def _batch_losses(
    model: INCElasticMultimodeMLP,
    batch: dict[str, torch.Tensor],
    physical: dict[str, torch.Tensor | float],
    config: INCElasticMultimodeConfig,
) -> dict[str, torch.Tensor]:
    """Compute normalized label, one-step state, and mechanical-energy losses."""
    predicted_start, predicted_end = model(batch["features"])
    return _loss_terms(predicted_start, predicted_end, batch, physical, config)


def _loss_terms(
    predicted_start: torch.Tensor,
    predicted_end: torch.Tensor,
    batch: dict[str, torch.Tensor],
    physical: dict[str, torch.Tensor | float],
    config: INCElasticMultimodeConfig,
) -> dict[str, torch.Tensor]:
    """Compute loss terms from explicit normalized two-stage predictions."""
    delta_start_normalized = predicted_start - batch["target_start"]
    delta_end_normalized = predicted_end - batch["target_end"]
    label_loss = 0.5 * (
        torch.mean(delta_start_normalized**2) + torch.mean(delta_end_normalized**2)
    )
    mass = cast(torch.Tensor, physical["mass"])
    stiffness = cast(torch.Tensor, physical["stiffness"])
    start_scale = cast(torch.Tensor, physical["residual_start_scale"])
    end_scale = cast(torch.Tensor, physical["residual_end_scale"])
    time_step = float(physical["time_step"])
    delta_start = delta_start_normalized * start_scale
    delta_end = delta_end_normalized * end_scale
    displacement_error = -0.5 * time_step**2 * delta_start / mass
    stiffness_error = displacement_error @ stiffness.T
    velocity_error = 0.5 * time_step * (
        -delta_start / mass - stiffness_error / mass - delta_end / mass
    )
    mass_sum = torch.sum(mass)
    displacement_scale = float(physical["displacement_scale"])
    velocity_scale = float(physical["velocity_scale"])
    displacement_component = torch.sum(mass * displacement_error**2, dim=1) / (
        mass_sum * displacement_scale**2
    )
    velocity_component = torch.sum(mass * velocity_error**2, dim=1) / (
        mass_sum * velocity_scale**2
    )
    step_loss = torch.mean(
        config.loss.displacement_component_weight * displacement_component
        + config.loss.velocity_component_weight * velocity_component
    )
    predicted_displacement = batch["reference_displacement_next"] + displacement_error
    predicted_velocity = batch["reference_velocity_next"] + velocity_error
    predicted_energy = 0.5 * (
        torch.sum(mass * predicted_velocity**2, dim=1)
        + torch.sum((predicted_displacement @ stiffness.T) * predicted_displacement, dim=1)
    )
    energy_scale = float(physical["energy_scale"])
    energy_loss = torch.mean(
        ((predicted_energy - batch["reference_energy_next"]) / energy_scale) ** 2
    )
    total_loss = (
        config.loss.label_weight * label_loss
        + config.loss.step_weight * step_loss
        + config.loss.energy_weight * energy_loss
    )
    return {
        "total_loss": total_loss,
        "label_loss": label_loss,
        "step_loss": step_loss,
        "energy_loss": energy_loss,
    }


def _evaluate_zero_corrector_losses(
    tensors: dict[str, torch.Tensor],
    physical: dict[str, torch.Tensor | float],
    config: INCElasticMultimodeConfig,
) -> dict[str, float]:
    """Evaluate exact training-normalization identities for a zero corrector."""
    sample_count = tensors["features"].shape[0]
    totals = np.zeros(4, dtype=np.float64)
    with torch.no_grad():
        for first in range(0, sample_count, config.execution.batch_size):
            last = min(first + config.execution.batch_size, sample_count)
            batch = {name: values[first:last] for name, values in tensors.items()}
            zero_start = torch.zeros_like(batch["target_start"])
            zero_end = torch.zeros_like(batch["target_end"])
            losses = _loss_terms(zero_start, zero_end, batch, physical, config)
            weight = last - first
            totals += weight * np.asarray(
                [
                    float(losses["total_loss"].detach().cpu()),
                    float(losses["label_loss"].detach().cpu()),
                    float(losses["step_loss"].detach().cpu()),
                    float(losses["energy_loss"].detach().cpu()),
                ]
            )
    totals /= sample_count
    return {
        "total_loss": float(totals[0]),
        "label_loss": float(totals[1]),
        "step_loss": float(totals[2]),
        "energy_loss": float(totals[3]),
    }


def _validate_zero_corrector_losses(
    metrics: dict[str, float],
    config: INCElasticMultimodeConfig,
) -> None:
    """Require the analytic normalization identities before optimization."""
    expected = {
        "label_loss": 1.0,
        "step_loss": (
            config.loss.displacement_component_weight + config.loss.velocity_component_weight
        ),
        "energy_loss": 1.0,
    }
    tolerance = 5.0e-6
    failures = {
        name: {"value": metrics[name], "expected": value, "absolute_tolerance": tolerance}
        for name, value in expected.items()
        if abs(metrics[name] - value) > tolerance
    }
    if failures:
        raise INCElasticTrainingError(
            "zero_corrector_normalization_identity_failed",
            {"metrics": metrics, "failures": failures},
        )


def _build_physical_tensors(config, prepared, device: str) -> dict[str, torch.Tensor | float]:
    """Build fixed float64 tensors used by the physical loss terms."""
    return {
        "mass": torch.as_tensor(prepared.system.mass, dtype=torch.float64, device=device),
        "stiffness": torch.as_tensor(
            prepared.system.stiffness.toarray(), dtype=torch.float64, device=device
        ),
        "residual_start_scale": torch.as_tensor(
            prepared.normalization.residual_start_scale, dtype=torch.float64, device=device
        ),
        "residual_end_scale": torch.as_tensor(
            prepared.normalization.residual_end_scale, dtype=torch.float64, device=device
        ),
        "time_step": prepared.system.time_step,
        "displacement_scale": prepared.normalization.displacement_scale,
        "velocity_scale": prepared.normalization.velocity_scale,
        "energy_scale": prepared.normalization.energy_scale,
    }


def _write_run_inputs(
    config: INCElasticMultimodeConfig,
    normalization: INCElasticNormalization,
    cuda_report: dict[str, Any],
    output: Path,
) -> None:
    """Write resolved configuration, split ownership, normalization, and environment."""
    (output / "resolved_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "normalization.json").write_text(
        json.dumps(normalization.to_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_manifest = {
        "train": list(config.data.train_paths),
        "validation": list(config.data.validation_paths),
        "test": list(config.data.test_paths),
        "test_paths_opened_during_training": [],
    }
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda": cuda_report,
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_artifact_metadata(
    config: INCElasticMultimodeConfig,
    normalization: INCElasticNormalization,
    angular_frequencies: NDArray[np.float64],
    final_time: float,
    time_step: float,
    best_epoch: int,
    best_score: float,
) -> dict[str, Any]:
    """Build the fixed-system deployment contract."""
    return {
        "schema_version": "1.0",
        "name": config.name,
        "model": config.model.model_dump(mode="json"),
        "input_width": elastic_inc_input_width(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
        "feature_layout": elastic_inc_feature_layout(
            config.features.modal_count,
            config.features.include_duration_modal_phase,
        ),
        "output_semantics": "additive_left_residual_force",
        "stage_order": ["start_kick", "end_kick"],
        "normalization": normalization.to_mapping(),
        "angular_frequencies": angular_frequencies.tolist(),
        "final_time": final_time,
        "time_step": time_step,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "train_paths": list(config.data.train_paths),
        "validation_paths": list(config.data.validation_paths),
        "test_paths": list(config.data.test_paths),
        "pressure_scaling": "linear_force_residual_divided_by_pressure_amplitude",
        "deployment_precision": {
            "network": config.model.dtype,
            "finite_element_integration": config.execution.fem_dtype,
        },
    }


def _save_best_model(path: Path, model: INCElasticMultimodeMLP, epoch: int, score: float) -> None:
    """Atomically save the current best state dictionary."""
    partial = path.with_suffix(".pt.partial")
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "validation_score": score},
        partial,
    )
    partial.replace(path)


def _save_checkpoint(
    path: Path,
    model: INCElasticMultimodeMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    full_score: float | None,
) -> None:
    """Save one scheduled optimizer checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "full_validation_score": full_score,
        },
        path,
    )


def _load_model_state(path: Path, model: INCElasticMultimodeMLP) -> None:
    """Load one selected state dictionary strictly."""
    checkpoint = torch.load(path, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _append_json_line(path: Path, values: dict[str, Any]) -> None:
    """Append one epoch record to the structured training history."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, sort_keys=True) + "\n")


def _build_logger(path: Path) -> logging.Logger:
    """Build one file-only training logger."""
    logger = logging.getLogger(f"inc_elastic_multimode:{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
