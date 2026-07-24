"""Batched explicit energy-based RVE recurrent neural operator.

Contents:
    Energy networks, direct or mobility evolution, batched updates, and tangents.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from concrete_impact.nn.config import RVERNOModelConfig


@dataclass(frozen=True)
class EnergyAnchors:
    """Store zero-state energy values and strain gradients for one rollout."""

    equilibrium_value: Tensor
    equilibrium_strain_gradient: Tensor
    nonequilibrium_value: Tensor
    nonequilibrium_strain_gradient: Tensor


@dataclass(frozen=True)
class RVERNOUpdateBatch:
    """Store one batched constitutive update at a common path time index."""

    stress: Tensor
    latent_state: Tensor
    state_rate: Tensor
    dimensionless_state_rate: Tensor
    dissipation: Tensor
    free_energy: Tensor
    thermodynamic_violation: Tensor
    nonequilibrium_energy_increment: Tensor
    tangent: Tensor | None = None


@dataclass(frozen=True)
class RVERNOUpdate:
    """Store one single-point constitutive update for deployment compatibility."""

    stress: Tensor
    latent_state: Tensor
    state_rate: Tensor
    dimensionless_state_rate: Tensor
    dissipation: Tensor
    free_energy: Tensor
    thermodynamic_violation: Tensor
    nonequilibrium_energy_increment: Tensor
    tangent: Tensor | None = None


class MultilayerPerceptron(nn.Module):
    """Map state vectors to scalar or vector outputs with SiLU activation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        hidden_layers: int,
        output_bias: bool = True,
    ) -> None:
        """Build a dense network with Xavier initialization."""
        super().__init__()
        layers: list[nn.Module] = []
        width = input_size
        for _ in range(hidden_layers):
            linear = nn.Linear(width, hidden_size)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.SiLU()))
            width = hidden_size
        output = nn.Linear(width, output_size, bias=output_bias)
        nn.init.xavier_uniform_(output.weight)
        if output.bias is not None:
            nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        """Evaluate all leading batch indices in one dense network call."""
        return self.network(values)


class ScalarMLP(nn.Module):
    """Map a state vector to one scalar energy potential."""

    def __init__(self, input_size: int, hidden_size: int, hidden_layers: int) -> None:
        """Build the scalar energy network."""
        super().__init__()
        self.network = MultilayerPerceptron(
            input_size, 1, hidden_size, hidden_layers, output_bias=False
        )

    def forward(self, values: Tensor) -> Tensor:
        """Evaluate one scalar potential for every leading batch index."""
        return self.network(values).squeeze(-1)

    def value_and_input_gradient(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate one scalar potential and its exact input gradient."""
        hidden = values
        linears: list[nn.Linear] = []
        preactivations: list[Tensor] = []
        modules = tuple(self.network.network)
        for index in range(0, len(modules) - 1, 2):
            linear = cast(nn.Linear, modules[index])
            preactivation = linear(hidden)
            hidden = torch.nn.functional.silu(preactivation)
            linears.append(linear)
            preactivations.append(preactivation)
        output = cast(nn.Linear, modules[-1])
        value = output(hidden).squeeze(-1)
        gradient = output.weight[0].expand(*hidden.shape[:-1], output.in_features)
        for linear, preactivation in reversed(tuple(zip(linears, preactivations, strict=True))):
            sigmoid = torch.sigmoid(preactivation)
            silu_derivative = sigmoid * (1.0 + preactivation * (1.0 - sigmoid))
            gradient = torch.matmul(gradient * silu_derivative, linear.weight)
        return value, gradient


class LowerTriangularMLP(nn.Module):
    """Predict a batched lower-triangular mobility factor."""

    def __init__(
        self,
        input_size: int,
        latent_dimension: int,
        hidden_size: int,
        hidden_layers: int,
    ) -> None:
        """Build the independent lower-triangular factor network."""
        super().__init__()
        self.latent_dimension = latent_dimension
        tril_size = latent_dimension * (latent_dimension + 1) // 2
        self.network = MultilayerPerceptron(input_size, tril_size, hidden_size, hidden_layers)
        indices = torch.tril_indices(latent_dimension, latent_dimension)
        self.register_buffer("row_indices", indices[0])
        self.register_buffer("column_indices", indices[1])

    def forward(self, values: Tensor) -> Tensor:
        """Assemble the lower-triangular factor for every batch entry."""
        entries = self.network(values)
        shape = (*entries.shape[:-1], self.latent_dimension, self.latent_dimension)
        lower = torch.zeros(shape, dtype=entries.dtype, device=entries.device)
        row_indices = cast(Tensor, self.row_indices)
        column_indices = cast(Tensor, self.column_indices)
        lower[..., row_indices, column_indices] = entries
        return lower


class EnergyDissipationRVERNO(nn.Module):
    """Predict energy-derived stress with an explicit configured state ordering."""

    def __init__(self, config: RVERNOModelConfig) -> None:
        """Build energy and selected state-evolution networks."""
        super().__init__()
        self.config = config
        latent = config.latent_dimension
        state_width = 6 + latent
        self.equilibrium_energy = ScalarMLP(6, config.hidden_size, config.hidden_layers)
        self.nonequilibrium_energy = ScalarMLP(
            state_width, config.hidden_size, config.hidden_layers
        )
        if config.evolution == "direct_rate":
            self.state_evolution: nn.Module = MultilayerPerceptron(
                state_width,
                latent,
                config.hidden_size,
                config.hidden_layers,
            )
        else:
            self.state_evolution = LowerTriangularMLP(
                state_width,
                latent,
                config.hidden_size,
                config.hidden_layers,
            )
        self.register_buffer(
            "strain_input_scale",
            torch.tensor(config.strain_input_scale),
            persistent=False,
        )
        self._compiled_constitutive_step: Callable[..., tuple[Tensor, ...]] | None = None
        self.to(device=config.device, dtype=_torch_dtype(config.dtype))

    @property
    def dtype(self) -> torch.dtype:
        """Return the parameter precision used by the model."""
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return the parameter device used by the model."""
        return next(self.parameters()).device

    def initial_state_batch(self, batch_size: int) -> Tensor:
        """Return zero latent states for a complete path batch."""
        return torch.zeros(
            (batch_size, self.config.latent_dimension),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

    def initial_state(self, device: torch.device | str | None = None) -> Tensor:
        """Return one zero latent state for the scalar compatibility interface."""
        target = self.device if device is None else torch.device(device)
        return torch.zeros(
            self.config.latent_dimension,
            dtype=self.dtype,
            device=target,
            requires_grad=True,
        )

    def prepare_energy_anchors(self) -> EnergyAnchors:
        """Evaluate zero-state energy anchors once for an entire rollout."""
        zero_strain = torch.zeros(6, dtype=self.dtype, device=self.device)
        zero_latent = torch.zeros(
            self.config.latent_dimension,
            dtype=self.dtype,
            device=self.device,
        )
        equilibrium_value, equilibrium_gradient = self.equilibrium_energy.value_and_input_gradient(
            zero_strain
        )
        nonequilibrium_value, nonequilibrium_gradient = (
            self.nonequilibrium_energy.value_and_input_gradient(
                torch.cat((zero_strain, zero_latent))
            )
        )
        return EnergyAnchors(
            equilibrium_value=equilibrium_value,
            equilibrium_strain_gradient=equilibrium_gradient,
            nonequilibrium_value=nonequilibrium_value,
            nonequilibrium_strain_gradient=nonequilibrium_gradient[:6],
        )

    def update_batch(
        self,
        strain: Tensor,
        previous_latent: Tensor,
        time_step: Tensor,
        anchors: EnergyAnchors,
        compute_tangent: bool = False,
    ) -> RVERNOUpdateBatch:
        """Update all paths or material points at one time index."""
        batch_size = strain.shape[0]
        if strain.shape != (batch_size, 6):
            raise ValueError(f"RVE-RNO strain must have shape [B, 6], received {strain.shape}.")
        if previous_latent.shape != (batch_size, self.config.latent_dimension):
            raise ValueError("RVE-RNO latent state must have shape [B, latent_dimension].")
        if time_step.shape != (batch_size,):
            raise ValueError("RVE-RNO time step must have shape [B].")
        working_strain = strain.detach().requires_grad_(True) if compute_tangent else strain
        step = (
            self._constitutive_tensors
            if compute_tangent or self._compiled_constitutive_step is None
            else self._compiled_constitutive_step
        )
        tensors = step(
            working_strain,
            previous_latent,
            time_step,
            anchors.equilibrium_value,
            anchors.equilibrium_strain_gradient,
            anchors.nonequilibrium_value,
            anchors.nonequilibrium_strain_gradient,
        )
        (
            stress,
            candidate,
            state_rate,
            dimensionless_state_rate,
            dissipation,
            free_energy,
            thermodynamic_violation,
            energy_increment,
        ) = tensors
        tangent = None
        if compute_tangent:
            tangent_rows = []
            for component in range(6):
                tangent_rows.append(
                    torch.autograd.grad(
                        stress[:, component].sum(),
                        working_strain,
                        retain_graph=True,
                        create_graph=True,
                    )[0]
                )
            tangent = torch.stack(tangent_rows, dim=1)
        return RVERNOUpdateBatch(
            stress=stress,
            latent_state=candidate,
            state_rate=state_rate,
            dimensionless_state_rate=dimensionless_state_rate,
            dissipation=dissipation,
            free_energy=free_energy,
            thermodynamic_violation=thermodynamic_violation,
            nonequilibrium_energy_increment=energy_increment,
            tangent=tangent,
        )

    def compile_constitutive_step(
        self,
        backend: str,
        fullgraph: bool,
        dynamic: bool,
        mode: str,
    ) -> None:
        """Compile the pure-tensor constitutive step without an eager fallback."""
        self._compiled_constitutive_step = cast(
            Callable[..., tuple[Tensor, ...]],
            torch.compile(
                self._constitutive_tensors,
                backend=backend,
                fullgraph=fullgraph,
                dynamic=dynamic,
                mode=mode,
            ),
        )

    def _constitutive_tensors(
        self,
        strain: Tensor,
        previous_latent: Tensor,
        time_step: Tensor,
        equilibrium_anchor_value: Tensor,
        equilibrium_anchor_gradient: Tensor,
        nonequilibrium_anchor_value: Tensor,
        nonequilibrium_anchor_gradient: Tensor,
    ) -> tuple[Tensor, ...]:
        """Evaluate one energy-derived update through differentiable tensor operations."""
        strain_scale = cast(Tensor, self.strain_input_scale)
        network_strain = strain / strain_scale
        equilibrium_raw, equilibrium_raw_gradient = (
            self.equilibrium_energy.value_and_input_gradient(network_strain)
        )
        equilibrium = (
            equilibrium_raw
            - equilibrium_anchor_value
            - (network_strain * equilibrium_anchor_gradient).sum(dim=-1)
        )
        equilibrium_stress = (
            equilibrium_raw_gradient - equilibrium_anchor_gradient
        ) / strain_scale
        previous_state = torch.cat((network_strain, previous_latent), dim=-1)
        previous_raw, previous_raw_gradient = self.nonequilibrium_energy.value_and_input_gradient(
            previous_state
        )
        previous_nonequilibrium = (
            previous_raw
            - nonequilibrium_anchor_value
            - (network_strain * nonequilibrium_anchor_gradient).sum(dim=-1)
        )
        previous_beta = previous_raw_gradient[..., 6:]
        if self.config.evolution == "direct_rate":
            dimensionless_state_rate = self.state_evolution(previous_state)
            state_rate = dimensionless_state_rate / self.config.reference_time_scale
            dissipation = -(previous_beta * state_rate).sum(dim=-1)
        else:
            force = -previous_beta
            lower = self.state_evolution(previous_state)
            projected_force = torch.matmul(lower.transpose(-1, -2), force.unsqueeze(-1)).squeeze(-1)
            dimensionless_state_rate = torch.matmul(lower, projected_force.unsqueeze(-1)).squeeze(
                -1
            )
            state_rate = dimensionless_state_rate / self.config.reference_time_scale
            dissipation = (projected_force * projected_force).sum(
                dim=-1
            ) / self.config.reference_time_scale
        candidate = (
            previous_latent
            + (time_step / self.config.reference_time_scale).unsqueeze(-1)
            * dimensionless_state_rate
        )
        candidate_state = torch.cat((network_strain, candidate), dim=-1)
        candidate_raw, candidate_raw_gradient = self.nonequilibrium_energy.value_and_input_gradient(
            candidate_state
        )
        candidate_nonequilibrium = (
            candidate_raw
            - nonequilibrium_anchor_value
            - (network_strain * nonequilibrium_anchor_gradient).sum(dim=-1)
        )
        if self.config.integrator == "current_state_stress_forward_euler":
            response_nonequilibrium = candidate_nonequilibrium
            response_stress = (
                candidate_raw_gradient[..., :6] - nonequilibrium_anchor_gradient
            ) / strain_scale
        else:
            response_nonequilibrium = previous_nonequilibrium
            response_stress = (
                previous_raw_gradient[..., :6] - nonequilibrium_anchor_gradient
            ) / strain_scale
        stress = equilibrium_stress + response_stress
        thermodynamic_violation = torch.relu(-dissipation)
        return (
            stress,
            candidate,
            state_rate,
            dimensionless_state_rate,
            dissipation,
            equilibrium + response_nonequilibrium,
            thermodynamic_violation,
            candidate_nonequilibrium - previous_nonequilibrium,
        )

    def update(
        self,
        strain: Tensor,
        previous_latent: Tensor,
        time_step: Tensor,
        compute_tangent: bool = False,
        context: dict[str, object] | None = None,
    ) -> RVERNOUpdate:
        """Run the batched implementation with a batch size of one."""
        del context
        if strain.shape != (6,):
            raise ValueError(f"RVE-RNO strain must have shape [6], received {strain.shape}.")
        if previous_latent.shape != (self.config.latent_dimension,):
            raise ValueError("RVE-RNO latent state has an incompatible shape.")
        anchors = self.prepare_energy_anchors()
        result = self.update_batch(
            strain.unsqueeze(0),
            previous_latent.unsqueeze(0),
            time_step.reshape(1),
            anchors,
            compute_tangent,
        )
        return RVERNOUpdate(
            stress=result.stress[0],
            latent_state=result.latent_state[0],
            state_rate=result.state_rate[0],
            dimensionless_state_rate=result.dimensionless_state_rate[0],
            dissipation=result.dissipation[0],
            free_energy=result.free_energy[0],
            thermodynamic_violation=result.thermodynamic_violation[0],
            nonequilibrium_energy_increment=result.nonequilibrium_energy_increment[0],
            tangent=None if result.tangent is None else result.tangent[0],
        )

    def _anchored_equilibrium(self, strain: Tensor, anchors: EnergyAnchors) -> Tensor:
        """Evaluate equilibrium energy with value and stress anchoring at zero."""
        network_strain = strain / cast(Tensor, self.strain_input_scale)
        return (
            self.equilibrium_energy(network_strain)
            - anchors.equilibrium_value
            - (network_strain * anchors.equilibrium_strain_gradient).sum(dim=-1)
        )

    def _anchored_nonequilibrium(
        self,
        state: Tensor,
        strain: Tensor,
        anchors: EnergyAnchors,
    ) -> Tensor:
        """Evaluate nonequilibrium energy with zero-state stress anchoring."""
        network_strain = strain / cast(Tensor, self.strain_input_scale)
        network_state = torch.cat((network_strain, state[..., 6:]), dim=-1)
        return (
            self.nonequilibrium_energy(network_state)
            - anchors.nonequilibrium_value
            - (network_strain * anchors.nonequilibrium_strain_gradient).sum(dim=-1)
        )


def _torch_dtype(name: str) -> torch.dtype:
    """Map a validated model precision name to a PyTorch dtype."""
    return {"float64": torch.float64, "float32": torch.float32}[name]
