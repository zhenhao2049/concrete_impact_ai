"""Macroscopic material adapter for a trained fixed-cell RVE-RNO.

Contents:
    Strict artifact loading and batched material-point deployment adaptation.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from concrete_impact.nn.config import RVERNOArtifactMetadata, RVERNOModelConfig
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO
from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)
from fem.surrogates import SurrogateCapabilityError


def load_rve_rno_state_dict(
    model_config: RVERNOModelConfig,
    model_path: str | Path,
    metadata_path: str | Path,
) -> EnergyDissipationRVERNO:
    """Load one explicit RVE-RNO after strict architecture-contract matching."""
    metadata = RVERNOArtifactMetadata.model_validate_json(
        Path(metadata_path).read_text(encoding="utf-8")
    )
    expected = {
        "model_family": model_config.family,
        "integrator": model_config.integrator,
        "evolution": model_config.evolution,
        "reference_time_scale": model_config.reference_time_scale,
        "latent_dimension": model_config.latent_dimension,
        "hidden_size": model_config.hidden_size,
        "hidden_layers": model_config.hidden_layers,
        "activation": model_config.activation,
        "strain_input_scale": model_config.strain_input_scale,
        "dtype": model_config.dtype,
        "deployment_integrator": "explicit_only",
    }
    received = {name: getattr(metadata, name) for name in expected}
    if received != expected:
        raise ValueError(
            "RVE-RNO artifact metadata does not match the requested architecture: "
            f"expected={expected}; received={received}."
        )
    checkpoint = torch.load(
        Path(model_path), map_location=model_config.device, weights_only=True
    )
    state_dict = checkpoint["model_state_dict"]
    model = EnergyDissipationRVERNO(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class RVERNOMaterialAdapter:
    """Expose one trained RVE-RNO through the common material-point protocol."""

    def __init__(
        self,
        model: EnergyDissipationRVERNO,
        name: str,
        density: float,
        maximum_wave_speed: float,
    ) -> None:
        """Store fixed-cell deployment properties and a frozen PyTorch model."""
        if density <= 0.0 or maximum_wave_speed <= 0.0:
            raise ValueError("RVE-RNO density and wave-speed bound must be positive.")
        self.model = model
        self.model.eval()
        self.name = name
        self.density = density
        self.maximum_wave_speed = maximum_wave_speed

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize one zero latent vector at every macroscopic integration point."""
        return MaterialState(
            {
                "latent_state": np.zeros(
                    (n_points, self.model.config.latent_dimension), dtype=np.float64
                )
            }
        )

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update independent macroscopic integration points from committed states."""
        if request.kinematics != "three_dimensional":
            raise ValueError("Fixed-cell RVE-RNO currently requires three-dimensional strain.")
        latent_values = state.variables["latent_state"]
        if request.strains.shape[0] != latent_values.shape[0]:
            raise ValueError("RVE-RNO strain and latent-state point counts differ.")
        device = self.model.device
        dtype = self.model.dtype
        batch_size = request.strains.shape[0]
        anchors = self.model.prepare_energy_anchors()
        response = self.model.update_batch(
            torch.as_tensor(request.strains, dtype=dtype, device=device),
            torch.as_tensor(latent_values, dtype=dtype, device=device),
            torch.full(
                (batch_size,), request.time_step, dtype=dtype, device=device
            ),
            anchors,
            compute_tangent=requirements.tangent,
        )
        if requirements.tangent and response.tangent is None:
            raise SurrogateCapabilityError(
                "RVE-RNO tangent was requested but not returned."
            )
        stresses = response.stress.detach().cpu().numpy()
        latent_states = response.latent_state.detach().cpu().numpy()
        dissipations = response.dissipation.detach().cpu().numpy()
        free_energy = response.free_energy.detach().cpu().numpy()
        thermodynamic_violation = (
            response.thermodynamic_violation.detach().cpu().numpy()
        )
        energy_increment = (
            response.nonequilibrium_energy_increment.detach().cpu().numpy()
        )
        tangents = (
            None if response.tangent is None else response.tangent.detach().cpu().numpy()
        )
        return MaterialPointResponse(
            stresses=np.asarray(stresses, dtype=np.float64),
            state=MaterialState(
                {"latent_state": np.asarray(latent_states, dtype=np.float64)}
            ),
            tangents=(
                np.asarray(tangents, dtype=np.float64) if requirements.tangent else None
            ),
            dissipation=(
                np.asarray(dissipations, dtype=np.float64)
                if requirements.dissipation
                else None
            ),
            free_energy=(
                np.asarray(free_energy, dtype=np.float64)
                if requirements.free_energy
                else None
            ),
            diagnostics={
                "thermodynamic_violation": np.asarray(
                    thermodynamic_violation, dtype=np.float64
                ),
                "nonequilibrium_energy_increment": np.asarray(
                    energy_increment, dtype=np.float64
                ),
            },
        )
