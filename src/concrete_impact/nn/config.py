"""Validated project configuration for surrogate deployment and RNO models.

Contents:
    Strict configuration schemas, artifact contracts, loaders, and solver conversion.
Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from fem.io.config import load_yaml_config
from fem.solvers.data import SurrogateSettings


class StrictConfigModel(BaseModel):
    """Forbid unknown configuration fields in surrogate schemas."""

    model_config = ConfigDict(extra="forbid")


class DisabledSurrogateConfig(StrictConfigModel):
    """Represent an explicitly disabled surrogate selection."""

    enabled: Literal[False]


class SurrogateArtifactConfig(StrictConfigModel):
    """Store a deployed model artifact and metadata pair."""

    format: Literal["state_dict", "torch_export"]
    model_path: Path
    metadata_path: Path


class SurrogateRuntimeConfig(StrictConfigModel):
    """Store the initial deterministic surrogate runtime policy."""

    device: Literal["cpu"]
    dtype: Literal["float64"]


class EnabledSurrogateConfig(StrictConfigModel):
    """Represent one fully specified project surrogate selection."""

    enabled: Literal[True]
    kind: Literal["material_point", "residual_corrector", "response", "field"]
    model: str
    backend: Literal["pytorch"]
    artifact: SurrogateArtifactConfig
    runtime: SurrogateRuntimeConfig


SurrogateDeploymentConfig = Annotated[
    DisabledSurrogateConfig | EnabledSurrogateConfig,
    Field(discriminator="enabled"),
]


class SurrogateConfigEnvelope(StrictConfigModel):
    """Store one top-level surrogate deployment section."""

    surrogate: SurrogateDeploymentConfig


class J2VPRNOModelConfig(StrictConfigModel):
    """Store the first continuous-time structured J2-VP-RNO architecture."""

    family: Literal["j2_vp_rno"]
    time_representation: Literal["continuous"]
    hidden_size: PositiveInt
    hidden_layers: PositiveInt
    activation: Literal["tanh"]
    correction_scale: PositiveFloat
    dtype: Literal["float64"]
    device: Literal["cpu"]


class J2VPRNOConfigEnvelope(StrictConfigModel):
    """Store the structured J2-VP-RNO model section."""

    model: J2VPRNOModelConfig


class RVERNOModelConfig(StrictConfigModel):
    """Store the explicit fixed-cell energy-based RVE-RNO architecture."""

    family: Literal["energy_rve_rno"]
    time_representation: Literal["continuous"]
    integrator: Literal[
        "cs_semi_implicit_forward_euler",
        "current_state_stress_forward_euler",
    ]
    evolution: Literal["direct_rate", "mobility_gradient"]
    reference_time_scale: PositiveFloat
    latent_dimension: PositiveInt
    hidden_size: PositiveInt
    hidden_layers: PositiveInt
    activation: Literal["silu"]
    strain_input_scale: tuple[PositiveFloat, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    dtype: Literal["float64", "float32"]
    device: Literal["cpu", "cuda"]

    @model_validator(mode="after")
    def require_six_strain_input_scales(self) -> RVERNOModelConfig:
        """Require one positive zero-preserving scale per engineering strain component."""
        if len(self.strain_input_scale) != 6:
            raise ValueError("RVE-RNO strain input scale must contain six values.")
        return self


class RVERNOLossConfig(StrictConfigModel):
    """Store the stress-primary and dissipation-auxiliary loss policy."""

    stress: Literal[
        "path_normalized_mse",
        "component_balanced_path_normalized_mse",
    ]
    thermodynamic_violation_weight: float = Field(default=1.0e-3, ge=0.0)
    dissipation_supervision_weight: float = Field(default=1.0e-2, ge=0.0)


class RVERNOCompileConfig(StrictConfigModel):
    """Store an explicit PyTorch compilation request without fallback."""

    enabled: bool = False
    backend: Literal["inductor"] = "inductor"
    fullgraph: bool = True
    dynamic: bool = False
    scope: Literal["state_evolution", "constitutive_step"] = "state_evolution"
    mode: Literal["default", "reduce-overhead"] = "default"


class RVERNOExecutionConfig(StrictConfigModel):
    """Store the in-memory input and device-transfer execution policy."""

    preload_to_memory: Literal[True] = True
    preload_device: Literal["cpu", "cuda"] = "cpu"
    batching_strategy: Literal["random_padded", "length_bucket"] = "random_padded"
    validation_batching_strategy: Literal["padded_batches", "single_padded_batch"] = (
        "padded_batches"
    )
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    num_workers: Literal[0] = 0
    fused_adam: bool = False
    matmul_precision: Literal["highest", "high"] = "highest"
    compile: RVERNOCompileConfig = Field(default_factory=RVERNOCompileConfig)

    @model_validator(mode="after")
    def require_cuda_residency_policy(self) -> RVERNOExecutionConfig:
        """Require length buckets and CUDA execution for device-resident data."""
        if self.preload_device == "cuda" and self.batching_strategy != "length_bucket":
            raise ValueError("CUDA-resident RVE-RNO data requires length_bucket batching.")
        if (
            self.preload_device == "cuda"
            and self.validation_batching_strategy != "single_padded_batch"
        ):
            raise ValueError("CUDA-resident RVE-RNO data requires single_padded_batch validation.")
        return self


class RVERNOTrainingConfig(StrictConfigModel):
    """Store one deterministic batched-path RVE-RNO training request."""

    schema_version: Literal["2.0"]
    model: RVERNOModelConfig
    loss: RVERNOLossConfig
    execution: RVERNOExecutionConfig
    data_plan: Path
    response_shard_manifest: Path
    data_acceptance_report: Path
    output_directory: Path
    random_seed: int = Field(ge=0)
    epochs: PositiveInt
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    checkpoint_interval: PositiveInt

    @model_validator(mode="after")
    def require_checkpoint_interval(self) -> RVERNOTrainingConfig:
        """Reject a checkpoint interval beyond the fixed training horizon."""
        if self.checkpoint_interval > self.epochs:
            raise ValueError("RVE-RNO checkpoint interval cannot exceed epoch count.")
        if self.execution.preload_device == "cuda" and self.model.device != "cuda":
            raise ValueError("CUDA-resident RVE-RNO data requires a CUDA model.")
        if self.execution.fused_adam and self.model.device != "cuda":
            raise ValueError("Fused RVE-RNO Adam requires a CUDA model.")
        return self


class RVERNOArtifactMetadata(StrictConfigModel):
    """Store the explicit fixed-cell RVE-RNO state-dict artifact contract."""

    schema_version: Literal["2.0"] = "2.0"
    model_family: Literal["energy_rve_rno"]
    integrator: Literal[
        "cs_semi_implicit_forward_euler",
        "current_state_stress_forward_euler",
    ]
    evolution: Literal["direct_rate", "mobility_gradient"]
    voigt_order: tuple[str, ...]
    strain_convention: Literal["engineering_shear"]
    stress_components: tuple[str, ...]
    latent_dimension: PositiveInt
    hidden_size: PositiveInt
    hidden_layers: PositiveInt
    activation: Literal["silu"]
    strain_input_scale: tuple[PositiveFloat, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    time_representation: Literal["continuous"]
    reference_time_scale: PositiveFloat
    time_step_role: Literal[
        "explicit_forward_euler_increment",
        "nondimensional_forward_euler_increment",
    ]
    deployment_integrator: Literal["explicit_only"]
    accepted_time_step_range: tuple[PositiveFloat, PositiveFloat]
    dtype: Literal["float64", "float32"]
    normalization: dict[str, dict[str, list[float]]]
    capabilities: dict[str, bool]
    data_plan_sha256: str
    response_shard_manifest_sha256: str
    split_manifest_sha256: str
    normalization_sha256: str
    mesh_quality: str
    acceptance_scope: Literal["strict", "pilot"]
    acceptance_limitation_note: str | None = None


class VelocityVerletINCModelConfig(StrictConfigModel):
    """Store the fixed-mesh two-stage velocity-Verlet INC architecture."""

    family: Literal["velocity_verlet_inc_mlp"]
    time_representation: Literal["discrete"]
    hidden_size: PositiveInt
    residual_blocks: PositiveInt
    activation: Literal["silu"]
    dtype: Literal["float64"]
    device: Literal["cpu", "cuda"]


class VelocityVerletINCTrainingConfig(StrictConfigModel):
    """Store deterministic two-stage INC training controls."""

    schema_version: Literal["1.0"]
    model: VelocityVerletINCModelConfig
    dataset_path: Path
    output_directory: Path
    random_seed: int = Field(ge=0)
    pretrain_epochs: PositiveInt
    rollout_epochs: PositiveInt
    pretrain_batch_size: PositiveInt
    rollout_batch_size: PositiveInt
    rollout_steps: PositiveInt
    pretrain_learning_rate: PositiveFloat
    rollout_learning_rate: PositiveFloat
    rollout_label_weight: float = Field(ge=0.0)
    rollout_energy_weight: float = Field(ge=0.0)
    label_acceptance: PositiveFloat
    checkpoint_interval: PositiveInt

    @model_validator(mode="after")
    def require_checkpoint_interval_within_each_stage(
        self,
    ) -> VelocityVerletINCTrainingConfig:
        """Reject checkpoints beyond either configured training stage."""
        if self.checkpoint_interval > min(self.pretrain_epochs, self.rollout_epochs):
            raise ValueError("INC checkpoint interval cannot exceed either stage length.")
        return self


class VelocityVerletINCArtifactMetadata(StrictConfigModel):
    """Store one fixed-system two-stage INC deployment contract."""

    schema_version: Literal["1.0"] = "1.0"
    model_name: str
    model_family: Literal["velocity_verlet_inc_mlp"]
    time_representation: Literal["discrete"]
    hidden_size: PositiveInt
    residual_blocks: PositiveInt
    activation: Literal["silu"]
    free_dof_count: PositiveInt
    input_field_order: tuple[str, ...]
    output_field_order: tuple[Literal["residual_force_start", "residual_force_end"], ...]
    output_semantics: Literal["additive_left_residual_force"]
    stage_order: tuple[Literal["start_kick", "end_kick"], ...]
    control_parameter_order: tuple[str, ...]
    time_step: PositiveFloat
    cfl_ratio: float = Field(gt=0.0, le=1.0)
    dtype: Literal["float64"]
    device: Literal["cpu"]
    mesh_sha256: str
    lumped_mass_sha256: str
    material_config_sha256: str
    boundary_config_sha256: str
    free_dof_order_sha256: str
    dataset_sha256: str
    split_manifest_sha256: str
    normalization: dict[str, dict[str, list[float]]]

    @model_validator(mode="after")
    def require_two_stage_contract(self) -> VelocityVerletINCArtifactMetadata:
        """Reject artifact metadata that changes fixed two-stage semantics."""
        expected_inputs = (
            "displacement",
            "velocity",
            "baseline_acceleration",
            "internal_force_start",
            "external_force_start",
            "external_force_end",
            "time",
            "control_parameters",
        )
        if self.input_field_order != expected_inputs:
            raise ValueError("INC artifact input fields do not match the fixed contract.")
        if self.stage_order != ("start_kick", "end_kick"):
            raise ValueError("INC artifact requires start_kick followed by end_kick.")
        if self.output_field_order != (
            "residual_force_start",
            "residual_force_end",
        ):
            raise ValueError("INC artifact requires start and end residual-force outputs.")
        return self


class VelocityVerletINCControlConfig(StrictConfigModel):
    """Store one pressure-pulse path and its complete-path split."""

    path_id: str
    split: Literal["train", "validation", "test"]
    pressure_amplitude: PositiveFloat
    pulse_duration: PositiveFloat


class VelocityVerletINCDataGenerationConfig(StrictConfigModel):
    """Store one fixed linear-rod INC dataset request."""

    schema_version: Literal["1.0"]
    model: dict[str, Any]
    plane_state: Literal["plane_stress"]
    thickness: PositiveFloat
    time_step: PositiveFloat
    num_steps: PositiveInt
    cfl_safety_factor: PositiveFloat
    fine_ratio: PositiveInt
    verification_ratio: PositiveInt
    pressure_boundary: str
    arrival_coordinate: float
    arrival_threshold: float = Field(gt=0.0, lt=1.0)
    displacement_reference_tolerance: PositiveFloat
    velocity_reference_tolerance: PositiveFloat
    arrival_reference_tolerance: PositiveFloat
    energy_reference_tolerance: PositiveFloat
    controls: tuple[VelocityVerletINCControlConfig, ...]
    output_path: Path

    @model_validator(mode="after")
    def require_nested_reference_ratios(self) -> VelocityVerletINCDataGenerationConfig:
        """Require nested fine and verification time grids plus all splits."""
        if self.fine_ratio <= 1:
            raise ValueError("INC fine ratio must exceed one.")
        if self.verification_ratio <= self.fine_ratio:
            raise ValueError("INC verification ratio must exceed the fine ratio.")
        if self.verification_ratio % self.fine_ratio != 0:
            raise ValueError("INC verification ratio must be divisible by the fine ratio.")
        if {control.split for control in self.controls} != {"train", "validation", "test"}:
            raise ValueError("INC controls must cover train, validation, and test splits.")
        path_ids = tuple(control.path_id for control in self.controls)
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("INC control path IDs must be unique.")
        return self


def load_rve_rno_training_config(path: str | Path) -> RVERNOTrainingConfig:
    """Load one strict fixed-cell RVE-RNO training configuration."""
    return RVERNOTrainingConfig.model_validate(load_yaml_config(path))


def load_velocity_verlet_inc_training_config(
    path: str | Path,
) -> VelocityVerletINCTrainingConfig:
    """Load one strict fixed-system INC training configuration."""
    return VelocityVerletINCTrainingConfig.model_validate(load_yaml_config(path))


def load_velocity_verlet_inc_data_config(
    path: str | Path,
) -> VelocityVerletINCDataGenerationConfig:
    """Load one strict fixed-system INC data-generation configuration."""
    return VelocityVerletINCDataGenerationConfig.model_validate(load_yaml_config(path))


def load_surrogate_deployment_config(path: str | Path) -> SurrogateDeploymentConfig:
    """Load and validate one deployment configuration file."""
    payload = load_yaml_config(path)

    return SurrogateConfigEnvelope.model_validate(payload).surrogate


def build_surrogate_settings(config: SurrogateDeploymentConfig) -> SurrogateSettings:
    """Convert validated project configuration into solver-level settings."""
    if isinstance(config, DisabledSurrogateConfig):
        return SurrogateSettings(enabled=False)
    if not config.artifact.model_path.is_file():
        raise FileNotFoundError(config.artifact.model_path)
    if not config.artifact.metadata_path.is_file():
        raise FileNotFoundError(config.artifact.metadata_path)

    return SurrogateSettings(
        enabled=True,
        kind=config.kind,
        model=config.model,
        backend=config.backend,
        artifact_format=config.artifact.format,
        model_path=config.artifact.model_path,
        metadata_path=config.artifact.metadata_path,
        device=config.runtime.device,
        dtype=config.runtime.dtype,
    )


def load_j2_vp_rno_config(path: str | Path) -> J2VPRNOModelConfig:
    """Load and validate one structured J2-VP-RNO architecture file."""
    payload = load_yaml_config(path)

    return J2VPRNOConfigEnvelope.model_validate(payload).model
