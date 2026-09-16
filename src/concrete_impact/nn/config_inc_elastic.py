"""Strict configuration for the elastic multimode INC delivery workflow.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from fem.io.config import load_yaml_config


class _StrictModel(BaseModel):
    """Reject undeclared elastic INC configuration fields."""

    model_config = ConfigDict(extra="forbid")


class INCElasticDataConfig(_StrictModel):
    """Define the accepted source data and locked path ownership."""

    root: Path
    system_path: Path
    manifest_path: Path
    acceptance_path: Path
    cache_directory: Path
    source_split_policy: Literal["locked_manifest", "training_pool_resplit"] = "locked_manifest"
    train_paths: tuple[str, ...]
    validation_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    sample_stride: PositiveInt

    @model_validator(mode="after")
    def require_disjoint_selected_path_split(self) -> INCElasticDataConfig:
        """Require one reviewed path count for the selected ownership policy."""
        path_counts = tuple(map(len, (self.train_paths, self.validation_paths, self.test_paths)))
        if self.source_split_policy == "locked_manifest":
            if path_counts not in ((12, 3, 3), (32, 3, 3)):
                raise ValueError("Locked elastic INC data require a 12/3/3 or 32/3/3 split.")
        elif path_counts != (12, 2, 2):
            raise ValueError("Training-pool resplitting requires a 12/2/2 path split.")
        all_paths = self.train_paths + self.validation_paths + self.test_paths
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("Elastic multimode INC path ownership must be disjoint.")
        if self.sample_stride != 2:
            raise ValueError("Elastic multimode INC requires the reviewed stride of two.")
        return self


class INCElasticFeatureConfig(_StrictModel):
    """Define the state-independent time and loading feature map."""

    modal_count: PositiveInt
    include_time: Literal[True]
    include_duration: Literal[True]
    include_spatial_coefficients: Literal[True]
    include_pulse_endpoints: Literal[True]
    include_modal_phase: Literal[True]
    include_duration_modal_phase: bool = False


class INCElasticModelConfig(_StrictModel):
    """Define the direct two-stage residual network."""

    family: Literal["inc_elastic_multimode_mlp"]
    hidden_size: PositiveInt
    residual_blocks: PositiveInt
    activation: Literal["silu"]
    free_dof_count: PositiveInt
    dtype: Literal["float32"]
    device: Literal["cuda"]


class INCElasticLossConfig(_StrictModel):
    """Store fixed one-step training loss weights."""

    label_weight: PositiveFloat
    step_weight: PositiveFloat
    energy_weight: PositiveFloat
    displacement_component_weight: PositiveFloat = 1.0
    velocity_component_weight: PositiveFloat = 1.0


class INCElasticExecutionConfig(_StrictModel):
    """Store deterministic CUDA and batching controls."""

    preload_device: Literal["cuda"]
    fem_dtype: Literal["float64"]
    batch_size: PositiveInt
    prediction_batch_size: PositiveInt
    matmul_precision: Literal["highest"]
    minimum_total_vram_gib: PositiveFloat
    minimum_free_vram_gib: PositiveFloat


class INCElasticTrainingControls(_StrictModel):
    """Store the fixed optimizer and validation schedule."""

    random_seed: int = Field(ge=0)
    epochs: PositiveInt
    learning_rate: PositiveFloat
    checkpoint_interval: PositiveInt
    full_validation_interval: PositiveInt
    optimizer: Literal["adam"]

    @model_validator(mode="after")
    def require_intervals_within_training(self) -> INCElasticTrainingControls:
        """Require aligned checkpoint and full-validation epochs."""
        if self.checkpoint_interval > self.epochs:
            raise ValueError("Checkpoint interval exceeds the training horizon.")
        if self.full_validation_interval > self.epochs:
            raise ValueError("Full-validation interval exceeds the training horizon.")
        if self.epochs % self.checkpoint_interval != 0:
            raise ValueError("Training epochs must be divisible by the checkpoint interval.")
        if self.epochs % self.full_validation_interval != 0:
            raise ValueError("Training epochs must be divisible by the validation interval.")
        return self


class INCElasticAcceptanceConfig(_StrictModel):
    """Store complete-path acceptance limits."""

    oracle_reconstruction_error: PositiveFloat
    label_mass_dual_nrmse: PositiveFloat
    displacement_error_ratio: PositiveFloat
    velocity_error_ratio: PositiveFloat
    energy_error_ratio: PositiveFloat
    normalized_impulse_balance_error: PositiveFloat
    arrival_threshold_ratio: float = Field(gt=0.0, lt=1.0)
    require_arrival_not_worse: Literal[True]
    require_peak_not_worse: Literal[True]
    require_phase_not_worse: Literal[True]


class INCElasticBenchmarkConfig(_StrictModel):
    """Store repeated timing controls for the delivery test cases."""

    fine_ratio: Literal[64]
    fine_warmup: PositiveInt
    fine_repeats: PositiveInt
    coarse_warmup: PositiveInt
    coarse_repeats: PositiveInt
    inc_warmup: PositiveInt
    inc_repeats: PositiveInt


class INCElasticMultimodeConfig(_StrictModel):
    """Define the complete fixed-system elastic INC workflow."""

    schema_version: Literal["1.0"]
    name: str
    data: INCElasticDataConfig
    features: INCElasticFeatureConfig
    model: INCElasticModelConfig
    loss: INCElasticLossConfig
    execution: INCElasticExecutionConfig
    training: INCElasticTrainingControls
    acceptance: INCElasticAcceptanceConfig
    benchmark: INCElasticBenchmarkConfig
    output_directory: Path

    @model_validator(mode="after")
    def require_reviewed_architecture(self) -> INCElasticMultimodeConfig:
        """Lock the reviewed fixed-system architecture and training policy."""
        if self.features.modal_count != self.model.free_dof_count:
            raise ValueError("Every free structural mode requires one phase feature pair.")
        if self.model.hidden_size < self.model.free_dof_count:
            raise ValueError("INC hidden width must not impose an output-rank bottleneck.")
        if self.model.hidden_size != 512 or self.model.residual_blocks != 3:
            raise ValueError("Elastic multimode INC requires the reviewed 512-by-3 MLP.")
        if self.training.epochs != 200 or self.execution.batch_size != 2048:
            raise ValueError("Elastic multimode INC requires 200 epochs and batch size 2048.")
        if self.loss.label_weight != 1.0 or self.loss.step_weight != 1.0:
            raise ValueError("Elastic multimode INC requires unit label and one-step weights.")
        if self.loss.energy_weight != 0.01:
            raise ValueError("Elastic multimode INC requires energy weight 0.01.")
        return self


def load_inc_elastic_multimode_config(path: str | Path) -> INCElasticMultimodeConfig:
    """Load one strict elastic multimode INC configuration."""
    return INCElasticMultimodeConfig.model_validate(load_yaml_config(path))
