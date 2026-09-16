"""Direct two-stage residual network for the elastic multimode INC.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import torch
from torch import nn

from concrete_impact.nn.config_inc_elastic import INCElasticModelConfig


class _SiLUResidualBlock(nn.Module):
    """Apply one width-preserving two-layer residual transformation."""

    def __init__(self, width: int) -> None:
        """Build the residual block."""
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return the residual-block output."""
        return values + self.second(self.activation(self.first(values)))


class INCElasticMultimodeMLP(nn.Module):
    """Predict normalized start- and end-stage residual forces."""

    def __init__(self, config: INCElasticModelConfig, input_width: int) -> None:
        """Build the shared trunk and two full-rank output heads."""
        super().__init__()
        self.config = config
        self.input_width = input_width
        self.input_projection = nn.Linear(input_width, config.hidden_size)
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            _SiLUResidualBlock(config.hidden_size) for _ in range(config.residual_blocks)
        )
        self.start_head = nn.Linear(config.hidden_size, config.free_dof_count)
        self.end_head = nn.Linear(config.hidden_size, config.free_dof_count)
        self.to(device=config.device, dtype=torch.float32)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both normalized residual-force fields."""
        hidden = self.activation(self.input_projection(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.start_head(hidden), self.end_head(hidden)
