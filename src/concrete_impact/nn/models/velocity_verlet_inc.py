"""Fixed-mesh residual MLP for two-stage velocity-Verlet INC.

Contents:
    Residual blocks, shared trunk, and independent start/end force heads.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import torch
from torch import nn

from concrete_impact.nn.config import VelocityVerletINCModelConfig


class _ResidualMLPBlock(nn.Module):
    """Apply one width-preserving residual MLP transformation."""

    def __init__(self, width: int) -> None:
        """Build one two-layer SiLU residual block."""
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return the residual block output."""
        increment = self.second(self.activation(self.first(values)))
        return values + increment


class VelocityVerletINCMLP(nn.Module):
    """Predict two free-DOF left residual-force vectors from one committed state."""

    def __init__(
        self,
        config: VelocityVerletINCModelConfig,
        input_width: int,
        free_dof_count: int,
    ) -> None:
        """Build the fixed-width shared trunk and two output heads."""
        super().__init__()
        self.config = config
        self.input_width = input_width
        self.free_dof_count = free_dof_count
        self.input_projection = nn.Linear(input_width, config.hidden_size)
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            _ResidualMLPBlock(config.hidden_size) for _ in range(config.residual_blocks)
        )
        self.start_head = nn.Linear(config.hidden_size, free_dof_count)
        self.end_head = nn.Linear(config.hidden_size, free_dof_count)
        self.to(dtype=torch.float64, device=config.device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized start- and end-stage residual-force coordinates."""
        hidden = self.activation(self.input_projection(features))
        for block in self.blocks:
            hidden = block(hidden)

        return self.start_head(hidden), self.end_head(hidden)
