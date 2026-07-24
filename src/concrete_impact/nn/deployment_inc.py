"""Deployment and fixed-system signatures for velocity-Verlet INC.

Contents:
    System hashing, strict artifact loading, request packing, and protocol adaptation.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from concrete_impact.nn.config import (
    VelocityVerletINCArtifactMetadata,
    VelocityVerletINCModelConfig,
)
from concrete_impact.nn.datasets.velocity_verlet_inc import (
    INC_INPUT_FIELD_ORDER,
    VelocityVerletINCNormalization,
    VelocityVerletINCSystemData,
    denormalize_inc_field,
    normalize_inc_field,
)
from concrete_impact.nn.models.velocity_verlet_inc import VelocityVerletINCMLP
from concrete_impact.nn.registry import build_velocity_verlet_inc_model
from fem.assembly.elasticity import assemble_lumped_mass_vector, assemble_stiffness_matrix
from fem.dynamics.stability import compute_explicit_stability_report
from fem.preprocess.data import PreprocessBundle
from fem.solvers.data import DirichletDofSet, TimeIntegrationSettings
from fem.surrogates.data import (
    VelocityVerletCorrectionRequest,
    VelocityVerletCorrectionResponse,
    VelocityVerletCorrectorMetadata,
)


class VelocityVerletINCAdapter:
    """Expose one frozen fixed-system MLP through the generic corrector protocol."""

    def __init__(
        self,
        model: VelocityVerletINCMLP,
        artifact: VelocityVerletINCArtifactMetadata,
    ) -> None:
        """Store the model, normalized fields, and generic runtime metadata."""
        self.model = model
        self.model.eval()
        self.artifact = artifact
        self.name = artifact.model_name
        self.normalization = VelocityVerletINCNormalization(
            {
                field: {
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in statistics.items()
                }
                for field, statistics in artifact.normalization.items()
            }
        )
        self.metadata = VelocityVerletCorrectorMetadata(
            schema_version=artifact.schema_version,
            model_name=artifact.model_name,
            model_family=artifact.model_family,
            time_step=artifact.time_step,
            cfl_ratio=artifact.cfl_ratio,
            output_semantics=artifact.output_semantics,
            stage_order=("start_kick", "end_kick"),
            dtype=artifact.dtype,
            device=artifact.device,
            control_parameter_order=artifact.control_parameter_order,
            mesh_sha256=artifact.mesh_sha256,
            lumped_mass_sha256=artifact.lumped_mass_sha256,
            material_config_sha256=artifact.material_config_sha256,
            boundary_config_sha256=artifact.boundary_config_sha256,
            free_dof_order_sha256=artifact.free_dof_order_sha256,
        )

    def evaluate(
        self,
        request: VelocityVerletCorrectionRequest,
    ) -> VelocityVerletCorrectionResponse:
        """Predict and denormalize both free-DOF stage residual forces."""
        _validate_inc_request(request, self.artifact)
        free = request.free_dofs
        field_values = {
            "displacement": request.displacement[free],
            "velocity": request.velocity[free],
            "baseline_acceleration": request.baseline_acceleration[free],
            "internal_force_start": request.internal_force_start[free],
            "external_force_start": request.external_force_start[free],
            "external_force_end": request.external_force_end[free],
            "time": np.asarray([request.time], dtype=np.float64),
            "control_parameters": request.control_parameters,
        }
        features = np.concatenate(
            [
                normalize_inc_field(name, field_values[name], self.normalization)
                for name in INC_INPUT_FIELD_ORDER
            ]
        )
        tensor = torch.as_tensor(features[None, :], dtype=torch.float64, device="cpu")
        with torch.no_grad():
            normalized_start, normalized_end = self.model(tensor)
        start_values = normalized_start[0].detach().cpu().numpy()
        end_values = normalized_end[0].detach().cpu().numpy()
        residual_start = denormalize_inc_field(
            "residual_force_start", start_values, self.normalization
        )
        residual_end = denormalize_inc_field(
            "residual_force_end", end_values, self.normalization
        )

        return VelocityVerletCorrectionResponse(
            residual_force_start_free=residual_start,
            residual_force_end_free=residual_end,
            diagnostics={
                "normalized_start_l2": np.asarray(
                    np.linalg.norm(start_values), dtype=np.float64
                ),
                "normalized_end_l2": np.asarray(
                    np.linalg.norm(end_values), dtype=np.float64
                ),
            },
        )


def build_velocity_verlet_inc_system_data(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
    dirichlet: DirichletDofSet,
    plane_state: str,
    control_parameter_order: tuple[str, ...],
    material_config: dict[str, Any],
) -> VelocityVerletINCSystemData:
    """Build free operators and deterministic compatibility hashes."""
    dof_count = bundle.mesh_info.dof_map.size
    free_dofs = np.setdiff1d(
        np.arange(dof_count, dtype=np.int64),
        dirichlet.dofs,
    )
    mass_lumped = assemble_lumped_mass_vector(bundle)
    stiffness = assemble_stiffness_matrix(bundle, plane_state)
    report = compute_explicit_stability_report(bundle, time_settings)
    mesh_sha256 = _sha256_arrays(
        bundle.mesh_info.nodes,
        bundle.mesh_info.elements,
        bundle.mesh_info.dof_map,
    )
    boundary_sha256 = _sha256_arrays(dirichlet.dofs, dirichlet.values)

    return VelocityVerletINCSystemData(
        free_dofs=free_dofs,
        mass_lumped_free=mass_lumped[free_dofs],
        stiffness_free=stiffness[free_dofs, :][:, free_dofs].toarray(),
        control_parameter_order=control_parameter_order,
        time_step=time_settings.time_step,
        cfl_ratio=time_settings.time_step / report.stable_time_step,
        mesh_sha256=mesh_sha256,
        lumped_mass_sha256=_sha256_arrays(mass_lumped),
        material_config_sha256=_sha256_mapping(material_config),
        boundary_config_sha256=boundary_sha256,
        free_dof_order_sha256=_sha256_arrays(free_dofs),
    )


def load_velocity_verlet_inc_corrector(
    model_path: str | Path,
    metadata_path: str | Path,
    system: VelocityVerletINCSystemData,
) -> VelocityVerletINCAdapter:
    """Load one CPU float64 INC artifact after exact system matching."""
    artifact = VelocityVerletINCArtifactMetadata.model_validate_json(
        Path(metadata_path).read_text(encoding="utf-8")
    )
    _validate_artifact_system(artifact, system)
    model_config = VelocityVerletINCModelConfig(
        family=artifact.model_family,
        time_representation="discrete",
        hidden_size=artifact.hidden_size,
        residual_blocks=artifact.residual_blocks,
        activation=artifact.activation,
        dtype="float64",
        device="cpu",
    )
    input_width = 6 * artifact.free_dof_count + 1 + len(artifact.control_parameter_order)
    model = build_velocity_verlet_inc_model(
        model_config, input_width, artifact.free_dof_count
    )
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return VelocityVerletINCAdapter(model, artifact)


def _validate_inc_request(
    request: VelocityVerletCorrectionRequest,
    artifact: VelocityVerletINCArtifactMetadata,
) -> None:
    """Validate fixed widths, field finiteness, and free-DOF identity."""
    if request.time_step != artifact.time_step:
        raise ValueError("INC request time step differs from artifact metadata.")
    if _sha256_arrays(request.free_dofs) != artifact.free_dof_order_sha256:
        raise ValueError("INC request free-DOF order differs from artifact metadata.")
    if request.free_dofs.size != artifact.free_dof_count:
        raise ValueError("INC request free-DOF count differs from artifact metadata.")
    if request.control_parameters.shape != (len(artifact.control_parameter_order),):
        raise ValueError("INC request control width differs from artifact metadata.")
    full_shape = request.displacement.shape
    fields = (
        request.displacement,
        request.velocity,
        request.baseline_acceleration,
        request.internal_force_start,
        request.external_force_start,
        request.external_force_end,
    )
    if any(values.shape != full_shape for values in fields):
        raise ValueError("INC request full-field shapes are inconsistent.")
    if any(not np.all(np.isfinite(values)) for values in fields):
        raise FloatingPointError("INC request contains non-finite field values.")


def _validate_artifact_system(
    artifact: VelocityVerletINCArtifactMetadata,
    system: VelocityVerletINCSystemData,
) -> None:
    """Require exact fixed-system compatibility before loading model weights."""
    expected = {
        "free_dof_count": system.free_dofs.size,
        "control_parameter_order": system.control_parameter_order,
        "time_step": system.time_step,
        "cfl_ratio": system.cfl_ratio,
        "mesh_sha256": system.mesh_sha256,
        "lumped_mass_sha256": system.lumped_mass_sha256,
        "material_config_sha256": system.material_config_sha256,
        "boundary_config_sha256": system.boundary_config_sha256,
        "free_dof_order_sha256": system.free_dof_order_sha256,
    }
    received = {name: getattr(artifact, name) for name in expected}
    if received != expected:
        raise ValueError(
            "INC artifact does not match the requested fixed system: "
            f"expected={expected}; received={received}."
        )


def _sha256_arrays(*arrays: NDArray[Any]) -> str:
    """Hash ordered array shapes, dtypes, and contiguous values."""
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_mapping(mapping: dict[str, Any]) -> str:
    """Hash one canonical JSON mapping."""
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
