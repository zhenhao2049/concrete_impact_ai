"""PyTorch automatic differentiation helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


def torch_jacobian(function: Callable[[Any], Any], inputs: Any) -> Any:
    """Evaluate one PyTorch Jacobian with forward-mode AD."""
    import torch

    with _suppress_torch_internal_jit_warning():
        return torch.func.jacfwd(function)(inputs)


def torch_batch_jacobian(function: Callable[[Any], Any], inputs: Any) -> Any:
    """Evaluate batched PyTorch Jacobians with vmap and forward-mode AD."""
    import torch

    with _suppress_torch_internal_jit_warning():
        return torch.func.vmap(torch.func.jacfwd(function))(inputs)


@contextmanager
def _suppress_torch_internal_jit_warning():
    """Suppress the PyTorch internal deprecation warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*deprecated.*",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        yield
