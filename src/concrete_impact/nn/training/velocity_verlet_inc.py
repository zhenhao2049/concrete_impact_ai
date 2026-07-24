"""Two-stage training for fixed-system velocity-Verlet INC.

Contents:
    Teacher-forced label pretraining, differentiable rollout training, and artifact export.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from concrete_impact.nn.config import (
    VelocityVerletINCArtifactMetadata,
    VelocityVerletINCTrainingConfig,
)
from concrete_impact.nn.datasets.velocity_verlet_inc import (
    INC_INPUT_FIELD_ORDER,
    VelocityVerletINCNormalization,
    VelocityVerletINCPath,
    VelocityVerletINCSystemData,
    compute_inc_normalization,
    load_velocity_verlet_inc_dataset,
    pack_inc_step_features,
    select_inc_split,
)
from concrete_impact.nn.models.velocity_verlet_inc import VelocityVerletINCMLP
from concrete_impact.nn.registry import build_velocity_verlet_inc_model


@dataclass(frozen=True)
class INCStageMetrics:
    """Store one epoch or final split of INC training metrics."""

    stage: str
    split: str
    epoch: int
    total_loss: float
    label_rmse: float
    trajectory_loss: float
    energy_loss: float
    elapsed_seconds: float


class INCTrainingNumericalError(RuntimeError):
    """Report one non-finite or acceptance failure with structured diagnostics."""

    def __init__(self, reason: str, diagnostics: dict[str, object]) -> None:
        """Initialize one reproducible INC training failure."""
        super().__init__(f"Velocity-Verlet INC training failed: {reason}.")
        self.reason = reason
        self.diagnostics = diagnostics


class _TeacherStepDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Store normalized teacher-forced steps for label pretraining."""

    def __init__(
        self,
        paths: tuple[VelocityVerletINCPath, ...],
        normalization: VelocityVerletINCNormalization,
    ) -> None:
        """Pack every complete-path step exactly once."""
        features = []
        starts = []
        ends = []
        start_statistics = normalization.statistics["residual_force_start"]
        end_statistics = normalization.statistics["residual_force_end"]
        for path in paths:
            for step_id in range(path.time.size - 1):
                features.append(pack_inc_step_features(path, step_id, normalization))
                starts.append(
                    (path.residual_force_start[step_id] - start_statistics["mean"])
                    / start_statistics["scale"]
                )
                ends.append(
                    (path.residual_force_end[step_id] - end_statistics["mean"])
                    / end_statistics["scale"]
                )
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float64))
        self.starts = torch.from_numpy(np.asarray(starts, dtype=np.float64))
        self.ends = torch.from_numpy(np.asarray(ends, dtype=np.float64))

    def __len__(self) -> int:
        """Return the number of teacher-forced steps."""
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one normalized teacher-forced sample."""
        return self.features[index], self.starts[index], self.ends[index]


class _RolloutWindowDataset(Dataset[tuple[int, int]]):
    """Index fixed-length windows without splitting complete path ownership."""

    def __init__(self, paths: tuple[VelocityVerletINCPath, ...], rollout_steps: int) -> None:
        """Build all valid path and start-step pairs."""
        self.index = tuple(
            (path_id, start)
            for path_id, path in enumerate(paths)
            for start in range(path.time.size - rollout_steps)
        )
        if not self.index:
            raise ValueError("INC rollout window exceeds every selected path.")

    def __len__(self) -> int:
        """Return the number of rollout windows."""
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Return one path and start-step pair."""
        return self.index[index]


def train_velocity_verlet_inc(config: VelocityVerletINCTrainingConfig) -> dict[str, object]:
    """Train, test, and export one fixed-system two-stage INC artifact."""
    torch.manual_seed(config.random_seed)
    if config.model.device == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    logger = _build_logger(output / "train.log")
    system, paths = load_velocity_verlet_inc_dataset(config.dataset_path)
    split_paths = {
        name: select_inc_split(paths, name) for name in ("train", "validation", "test")
    }
    normalization = compute_inc_normalization(
        split_paths["train"], system.mass_lumped_free
    )
    input_width = 6 * system.free_dofs.size + 1 + len(system.control_parameter_order)
    model = build_velocity_verlet_inc_model(
        config.model, input_width, system.free_dofs.size
    )
    _write_training_inputs(output, config, system, paths, normalization)
    metrics_path = output / "metrics.jsonl"

    teacher_loaders = {
        name: DataLoader(
            _TeacherStepDataset(selected, normalization),
            batch_size=config.pretrain_batch_size,
            shuffle=name == "train",
        )
        for name, selected in split_paths.items()
    }
    pretrain_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.pretrain_learning_rate
    )
    best_pretrain = float("inf")
    for epoch in range(1, config.pretrain_epochs + 1):
        train_metrics = _run_label_epoch(
            model, teacher_loaders["train"], pretrain_optimizer, epoch, "train"
        )
        validation_metrics = _run_label_epoch(
            model, teacher_loaders["validation"], None, epoch, "validation"
        )
        _append_metrics(metrics_path, train_metrics, validation_metrics)
        if validation_metrics.total_loss < best_pretrain:
            best_pretrain = validation_metrics.total_loss
            _save_checkpoint(output / "best_pretrain.pt", model, pretrain_optimizer, epoch)
        if epoch % config.checkpoint_interval == 0:
            _save_checkpoint(
                output / "checkpoints" / f"pretrain_{epoch:05d}.pt",
                model,
                pretrain_optimizer,
                epoch,
            )
        logger.info(
            "stage=pretrain epoch=%d train=%.8e validation=%.8e",
            epoch,
            train_metrics.total_loss,
            validation_metrics.total_loss,
        )
    _load_checkpoint(output / "best_pretrain.pt", model, config.model.device)
    pretrain_validation = _run_label_epoch(
        model,
        teacher_loaders["validation"],
        None,
        config.pretrain_epochs,
        "validation",
    )
    if pretrain_validation.label_rmse > config.label_acceptance:
        raise INCTrainingNumericalError(
            "label_pretraining_acceptance_failed",
            {
                "label_rmse": pretrain_validation.label_rmse,
                "acceptance": config.label_acceptance,
            },
        )

    rollout_loaders = {
        name: DataLoader(
            _RolloutWindowDataset(selected, config.rollout_steps),
            batch_size=config.rollout_batch_size,
            shuffle=name == "train",
        )
        for name, selected in split_paths.items()
    }
    rollout_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.rollout_learning_rate
    )
    energy_scale = _training_energy_scale(split_paths["train"])
    best_rollout = float("inf")
    for epoch in range(1, config.rollout_epochs + 1):
        train_metrics = _run_rollout_epoch(
            model,
            rollout_loaders["train"],
            split_paths["train"],
            system,
            normalization,
            config,
            energy_scale,
            rollout_optimizer,
            epoch,
            "train",
        )
        validation_metrics = _run_rollout_epoch(
            model,
            rollout_loaders["validation"],
            split_paths["validation"],
            system,
            normalization,
            config,
            energy_scale,
            None,
            epoch,
            "validation",
        )
        _append_metrics(metrics_path, train_metrics, validation_metrics)
        if validation_metrics.total_loss < best_rollout:
            best_rollout = validation_metrics.total_loss
            _save_checkpoint(output / "best_model.pt", model, rollout_optimizer, epoch)
        if epoch % config.checkpoint_interval == 0:
            _save_checkpoint(
                output / "checkpoints" / f"rollout_{epoch:05d}.pt",
                model,
                rollout_optimizer,
                epoch,
            )
        logger.info(
            "stage=rollout epoch=%d train=%.8e validation=%.8e",
            epoch,
            train_metrics.total_loss,
            validation_metrics.total_loss,
        )
    _load_checkpoint(output / "best_model.pt", model, config.model.device)
    test_metrics = _run_rollout_epoch(
        model,
        rollout_loaders["test"],
        split_paths["test"],
        system,
        normalization,
        config,
        energy_scale,
        None,
        config.rollout_epochs,
        "test",
    )
    _append_metrics(metrics_path, test_metrics)
    artifact = _build_artifact_metadata(config, system, paths, normalization)
    (output / "artifact_metadata.json").write_text(
        artifact.model_dump_json(indent=2), encoding="utf-8"
    )
    summary = {
        "passed": True,
        "best_pretrain_validation_loss": best_pretrain,
        "best_rollout_validation_loss": best_rollout,
        "test_metrics": asdict(test_metrics),
        "model_path": str(output / "best_model.pt"),
        "metadata_path": str(output / "artifact_metadata.json"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    return summary


def _run_label_epoch(
    model: VelocityVerletINCMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    split: str,
) -> INCStageMetrics:
    """Run one teacher-forced label epoch."""
    started = perf_counter()
    model.train(optimizer is not None)
    total = 0.0
    sample_count = 0
    for features, target_start, target_end in loader:
        features = features.to(device=model.config.device, dtype=torch.float64)
        target_start = target_start.to(device=model.config.device, dtype=torch.float64)
        target_end = target_end.to(device=model.config.device, dtype=torch.float64)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        predicted_start, predicted_end = model(features)
        loss = 0.5 * (
            torch.mean((predicted_start - target_start) ** 2)
            + torch.mean((predicted_end - target_end) ** 2)
        )
        _require_finite_loss(loss, "pretrain", split, epoch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        batch_size = features.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    mean_loss = total / sample_count

    return INCStageMetrics(
        stage="pretrain",
        split=split,
        epoch=epoch,
        total_loss=mean_loss,
        label_rmse=float(np.sqrt(mean_loss)),
        trajectory_loss=0.0,
        energy_loss=0.0,
        elapsed_seconds=perf_counter() - started,
    )


def _run_rollout_epoch(
    model: VelocityVerletINCMLP,
    loader: DataLoader,
    paths: tuple[VelocityVerletINCPath, ...],
    system: VelocityVerletINCSystemData,
    normalization: VelocityVerletINCNormalization,
    config: VelocityVerletINCTrainingConfig,
    energy_scale: float,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    split: str,
) -> INCStageMetrics:
    """Run one differentiable multi-step rollout epoch."""
    started = perf_counter()
    model.train(optimizer is not None)
    totals = np.zeros(4, dtype=np.float64)
    batch_count = 0
    for path_ids, starts in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        losses = _rollout_batch_losses(
            model,
            paths,
            path_ids.tolist(),
            starts.tolist(),
            system,
            normalization,
            config.rollout_steps,
            energy_scale,
        )
        total_loss = (
            losses[0]
            + config.rollout_label_weight * losses[1]
            + config.rollout_energy_weight * losses[2]
        )
        _require_finite_loss(total_loss, "rollout", split, epoch)
        if optimizer is not None:
            total_loss.backward()
            optimizer.step()
        totals += np.asarray(
            [
                float(total_loss.detach().cpu()),
                float(losses[0].detach().cpu()),
                float(losses[1].detach().cpu()),
                float(losses[2].detach().cpu()),
            ]
        )
        batch_count += 1
    totals /= batch_count

    return INCStageMetrics(
        stage="rollout",
        split=split,
        epoch=epoch,
        total_loss=float(totals[0]),
        label_rmse=float(np.sqrt(totals[2])),
        trajectory_loss=float(totals[1]),
        energy_loss=float(totals[3]),
        elapsed_seconds=perf_counter() - started,
    )


def _rollout_batch_losses(
    model: VelocityVerletINCMLP,
    paths: tuple[VelocityVerletINCPath, ...],
    path_ids: list[int],
    starts: list[int],
    system: VelocityVerletINCSystemData,
    normalization: VelocityVerletINCNormalization,
    rollout_steps: int,
    energy_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory, label, and energy losses for one window batch."""
    device = model.config.device
    mass = torch.as_tensor(system.mass_lumped_free, dtype=torch.float64, device=device)
    stiffness = torch.as_tensor(system.stiffness_free, dtype=torch.float64, device=device)
    displacement = _stack_path_values(paths, path_ids, starts, "displacement", device)
    velocity = _stack_path_values(paths, path_ids, starts, "velocity", device)
    trajectory_loss = torch.zeros((), dtype=torch.float64, device=device)
    label_loss = torch.zeros_like(trajectory_loss)
    energy_loss = torch.zeros_like(trajectory_loss)
    displacement_scale = normalization.statistics["displacement"]["scale"][0]
    velocity_scale = normalization.statistics["velocity"]["scale"][0]
    mass_sum = torch.sum(mass)
    for offset in range(rollout_steps):
        step_ids = [start + offset for start in starts]
        external_start = _stack_path_values(
            paths, path_ids, step_ids, "external_force", device
        )
        external_end = _stack_path_values(
            paths, path_ids, [step + 1 for step in step_ids], "external_force", device
        )
        controls = torch.as_tensor(
            np.vstack([paths[path_id].control_parameters for path_id in path_ids]),
            dtype=torch.float64,
            device=device,
        )
        time = torch.as_tensor(
            [
                paths[path_id].time[step]
                for path_id, step in zip(path_ids, step_ids, strict=True)
            ],
            dtype=torch.float64,
            device=device,
        )[:, None]
        internal_start = displacement @ stiffness.T
        baseline_acceleration = (external_start - internal_start) / mass
        features = _pack_torch_features(
            displacement,
            velocity,
            baseline_acceleration,
            internal_start,
            external_start,
            external_end,
            time,
            controls,
            normalization,
            device,
        )
        predicted_start_normalized, predicted_end_normalized = model(features)
        residual_start = _denormalize_torch_output(
            predicted_start_normalized,
            normalization.statistics["residual_force_start"],
            device,
        )
        residual_end = _denormalize_torch_output(
            predicted_end_normalized,
            normalization.statistics["residual_force_end"],
            device,
        )
        start_acceleration = baseline_acceleration - residual_start / mass
        next_displacement = (
            displacement
            + system.time_step * velocity
            + 0.5 * system.time_step**2 * start_acceleration
        )
        internal_end = next_displacement @ stiffness.T
        end_acceleration = (external_end - internal_end - residual_end) / mass
        next_velocity = velocity + 0.5 * system.time_step * (
            start_acceleration + end_acceleration
        )
        reference_displacement = _stack_path_values(
            paths, path_ids, [step + 1 for step in step_ids], "displacement", device
        )
        reference_velocity = _stack_path_values(
            paths, path_ids, [step + 1 for step in step_ids], "velocity", device
        )
        trajectory_loss = trajectory_loss + torch.mean(
            torch.sum(mass * (next_displacement - reference_displacement) ** 2, dim=1)
            / (mass_sum * displacement_scale**2)
            + torch.sum(mass * (next_velocity - reference_velocity) ** 2, dim=1)
            / (mass_sum * velocity_scale**2)
        )
        target_start = _normalized_target(
            paths, path_ids, step_ids, "residual_force_start", normalization, device
        )
        target_end = _normalized_target(
            paths, path_ids, step_ids, "residual_force_end", normalization, device
        )
        label_loss = label_loss + 0.5 * (
            torch.mean((predicted_start_normalized - target_start) ** 2)
            + torch.mean((predicted_end_normalized - target_end) ** 2)
        )
        predicted_energy = 0.5 * (
            torch.sum(mass * next_velocity**2, dim=1)
            + torch.sum((next_displacement @ stiffness) * next_displacement, dim=1)
        )
        reference_energy = torch.as_tensor(
            [
                paths[path_id].mechanical_energy[step + 1]
                for path_id, step in zip(path_ids, step_ids, strict=True)
            ],
            dtype=torch.float64,
            device=device,
        )
        energy_loss = energy_loss + torch.mean(
            ((predicted_energy - reference_energy) / energy_scale) ** 2
        )
        displacement = next_displacement
        velocity = next_velocity

    divisor = float(rollout_steps)
    return trajectory_loss / divisor, label_loss / divisor, energy_loss / divisor


def _pack_torch_features(
    displacement: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    internal_force: torch.Tensor,
    external_start: torch.Tensor,
    external_end: torch.Tensor,
    time: torch.Tensor,
    controls: torch.Tensor,
    normalization: VelocityVerletINCNormalization,
    device: str,
) -> torch.Tensor:
    """Normalize and concatenate one batch in the deployment field order."""
    fields = {
        "displacement": displacement,
        "velocity": velocity,
        "baseline_acceleration": acceleration,
        "internal_force_start": internal_force,
        "external_force_start": external_start,
        "external_force_end": external_end,
        "time": time,
        "control_parameters": controls,
    }
    packed = []
    for name in INC_INPUT_FIELD_ORDER:
        statistics = normalization.statistics[name]
        mean = torch.as_tensor(statistics["mean"], dtype=torch.float64, device=device)
        scale = torch.as_tensor(statistics["scale"], dtype=torch.float64, device=device)
        packed.append((fields[name] - mean) / scale)
    return torch.cat(packed, dim=1)


def _stack_path_values(
    paths: tuple[VelocityVerletINCPath, ...],
    path_ids: list[int],
    step_ids: list[int],
    field: str,
    device: str,
) -> torch.Tensor:
    """Stack one path field at independently selected time indices."""
    values = np.vstack(
        [
            getattr(paths[path_id], field)[step]
            for path_id, step in zip(path_ids, step_ids, strict=True)
        ]
    )
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _normalized_target(
    paths: tuple[VelocityVerletINCPath, ...],
    path_ids: list[int],
    step_ids: list[int],
    field: str,
    normalization: VelocityVerletINCNormalization,
    device: str,
) -> torch.Tensor:
    """Stack and normalize one stage target."""
    values = np.vstack(
        [
            getattr(paths[path_id], field)[step]
            for path_id, step in zip(path_ids, step_ids, strict=True)
        ]
    )
    statistics = normalization.statistics[field]
    normalized = (values - statistics["mean"]) / statistics["scale"]
    return torch.as_tensor(normalized, dtype=torch.float64, device=device)


def _denormalize_torch_output(
    values: torch.Tensor,
    statistics: dict[str, np.ndarray],
    device: str,
) -> torch.Tensor:
    """Restore one stage residual in physical force coordinates."""
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float64, device=device)
    scale = torch.as_tensor(statistics["scale"], dtype=torch.float64, device=device)
    return values * scale + mean


def _training_energy_scale(paths: tuple[VelocityVerletINCPath, ...]) -> float:
    """Compute the nonzero training-only RMS mechanical energy scale."""
    values = np.concatenate([path.mechanical_energy for path in paths])
    scale = float(np.sqrt(np.mean(values**2)))
    if scale == 0.0:
        raise ValueError("INC training paths have zero mechanical energy scale.")
    return scale


def _require_finite_loss(loss: torch.Tensor, stage: str, split: str, epoch: int) -> None:
    """Terminate at the first non-finite training objective."""
    if not bool(torch.isfinite(loss)):
        raise INCTrainingNumericalError(
            "non_finite_loss",
            {"stage": stage, "split": split, "epoch": epoch, "loss": float(loss)},
        )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Write one explicit training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def _load_checkpoint(path: Path, model: nn.Module, device: str) -> None:
    """Load one exact model state from a selected checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _append_metrics(path: Path, *metrics: INCStageMetrics) -> None:
    """Append strict JSON-lines training metrics."""
    with path.open("a", encoding="utf-8") as stream:
        for values in metrics:
            stream.write(json.dumps(asdict(values), sort_keys=True) + "\n")


def _write_training_inputs(
    output: Path,
    config: VelocityVerletINCTrainingConfig,
    system: VelocityVerletINCSystemData,
    paths: tuple[VelocityVerletINCPath, ...],
    normalization: VelocityVerletINCNormalization,
) -> None:
    """Write resolved inputs and immutable split and normalization records."""
    (output / "resolved_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    split_manifest = {name: [path.path_id for path in paths if path.split == name] for name in (
        "train",
        "validation",
        "test",
    )}
    (output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "normalization.json").write_text(
        json.dumps(normalization.to_json_mapping(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "random_seed": config.random_seed,
                "device": config.model.device,
                "dtype": config.model.dtype,
                "path_count": len(paths),
                "free_dof_count": system.free_dofs.size,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _build_artifact_metadata(
    config: VelocityVerletINCTrainingConfig,
    system: VelocityVerletINCSystemData,
    paths: tuple[VelocityVerletINCPath, ...],
    normalization: VelocityVerletINCNormalization,
) -> VelocityVerletINCArtifactMetadata:
    """Build one strict CPU deployment contract from completed training inputs."""
    split_manifest = {name: [path.path_id for path in paths if path.split == name] for name in (
        "train",
        "validation",
        "test",
    )}
    return VelocityVerletINCArtifactMetadata(
        model_name="velocity_verlet_inc_mlp",
        model_family=config.model.family,
        time_representation="discrete",
        hidden_size=config.model.hidden_size,
        residual_blocks=config.model.residual_blocks,
        activation=config.model.activation,
        free_dof_count=system.free_dofs.size,
        input_field_order=INC_INPUT_FIELD_ORDER,
        output_field_order=("residual_force_start", "residual_force_end"),
        output_semantics="additive_left_residual_force",
        stage_order=("start_kick", "end_kick"),
        control_parameter_order=system.control_parameter_order,
        time_step=system.time_step,
        cfl_ratio=system.cfl_ratio,
        dtype="float64",
        device="cpu",
        mesh_sha256=system.mesh_sha256,
        lumped_mass_sha256=system.lumped_mass_sha256,
        material_config_sha256=system.material_config_sha256,
        boundary_config_sha256=system.boundary_config_sha256,
        free_dof_order_sha256=system.free_dof_order_sha256,
        dataset_sha256=_sha256_file(config.dataset_path),
        split_manifest_sha256=_sha256_payload(split_manifest),
        normalization=normalization.to_json_mapping(),
    )


def _sha256_file(path: Path) -> str:
    """Hash one complete file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: object) -> str:
    """Hash one canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_logger(path: Path) -> logging.Logger:
    """Build one file-only training logger."""
    logger = logging.getLogger(f"velocity_verlet_inc.{path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
