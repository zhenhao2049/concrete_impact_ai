"""Automatic differentiation backends for FEM material kernels.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.ad.torch_backend import torch_batch_jacobian, torch_jacobian

__all__ = [
    "torch_batch_jacobian",
    "torch_jacobian",
]
