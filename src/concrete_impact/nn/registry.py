"""Project registry for concrete-impact surrogate model families.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from concrete_impact.nn.config import J2VPRNOModelConfig, VelocityVerletINCModelConfig
from concrete_impact.nn.models.j2_vp_rno import StructuredJ2VPRNO
from concrete_impact.nn.models.velocity_verlet_inc import VelocityVerletINCMLP
from fem.surrogates import SurrogateConfigError, load_surrogate_metadata

SurrogateModelBuilder = Callable[[dict[str, Any], J2VPRNOModelConfig], Any]
VelocityVerletINCModelBuilder = Callable[
    [VelocityVerletINCModelConfig, int, int],
    VelocityVerletINCMLP,
]


def build_j2_vp_rno(
    material_spec: dict[str, Any],
    model_config: J2VPRNOModelConfig,
) -> StructuredJ2VPRNO:
    """Build the structured J2-VP-RNO from physical and architecture configs."""
    return StructuredJ2VPRNO(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
        yield_stress=float(material_spec["yield_stress"]),
        hardening_modulus=float(material_spec["hardening_modulus"]),
        time_scale=float(material_spec["time_scale"]),
        reference_stress=float(material_spec["reference_stress"]),
        rate_exponent=float(material_spec["rate_exponent"]),
        equivalent_plastic_strain_scale=float(
            material_spec["equivalent_plastic_strain_scale"]
        ),
        model_config=model_config,
    )


SURROGATE_MODEL_BUILDERS: dict[str, SurrogateModelBuilder] = {
    "j2_vp_rno": build_j2_vp_rno,
}


def _build_velocity_verlet_inc_mlp(
    model_config: VelocityVerletINCModelConfig,
    input_width: int,
    free_dof_count: int,
) -> VelocityVerletINCMLP:
    """Build one fixed-mesh two-stage residual MLP."""
    return VelocityVerletINCMLP(model_config, input_width, free_dof_count)


VELOCITY_VERLET_INC_MODEL_BUILDERS: dict[str, VelocityVerletINCModelBuilder] = {
    "velocity_verlet_inc_mlp": _build_velocity_verlet_inc_mlp,
}


def build_velocity_verlet_inc_model(
    model_config: VelocityVerletINCModelConfig,
    input_width: int,
    free_dof_count: int,
) -> VelocityVerletINCMLP:
    """Build one residual corrector without entering the material-model registry."""
    try:
        builder = VELOCITY_VERLET_INC_MODEL_BUILDERS[model_config.family]
    except KeyError as error:
        raise SurrogateConfigError(
            f"Unknown velocity-Verlet INC model family: {model_config.family}."
        ) from error
    return builder(model_config, input_width, free_dof_count)


def build_surrogate_model(
    model_name: str,
    material_spec: dict[str, Any],
    model_config: J2VPRNOModelConfig,
) -> Any:
    """Build one project surrogate model from an explicit registry key."""
    try:
        builder = SURROGATE_MODEL_BUILDERS[model_name]
    except KeyError as error:
        raise SurrogateConfigError(
            f"Unknown concrete-impact surrogate model: {model_name}."
        ) from error

    return builder(material_spec, model_config)


def load_j2_vp_rno_state_dict(
    model: StructuredJ2VPRNO,
    model_path: str | Path,
    metadata_path: str | Path,
) -> StructuredJ2VPRNO:
    """Load a tangent-capable structured RNO after strict metadata matching."""
    metadata = load_surrogate_metadata(metadata_path)
    if metadata.kind != "material_point" or metadata.model_family != "j2_vp_rno":
        raise SurrogateConfigError("J2-VP-RNO artifact has an incompatible model contract.")
    if metadata.backend != "pytorch" or metadata.artifact_format != "state_dict":
        raise SurrogateConfigError("J2-VP-RNO deployment requires a PyTorch state_dict artifact.")
    if metadata.input_fields != model.metadata.input_fields:
        raise SurrogateConfigError("J2-VP-RNO artifact input fields do not match the model.")
    if metadata.output_fields != model.metadata.output_fields:
        raise SurrogateConfigError("J2-VP-RNO artifact output fields do not match the model.")
    if metadata.state_fields != model.metadata.state_fields:
        raise SurrogateConfigError("J2-VP-RNO artifact state fields do not match the model.")
    if metadata.normalization != model.metadata.normalization:
        raise SurrogateConfigError("J2-VP-RNO artifact normalization does not match the model.")
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.metadata = metadata

    return model
