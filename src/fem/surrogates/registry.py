"""Explicit registries for surrogate artifact backends.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fem.surrogates.errors import SurrogateConfigError

SurrogateBackendBuilder = Callable[..., Any]
SURROGATE_BACKEND_BUILDERS: dict[str, SurrogateBackendBuilder] = {}


def register_surrogate_backend(name: str, builder: SurrogateBackendBuilder) -> None:
    """Register one artifact backend under an explicit unique key."""
    if name in SURROGATE_BACKEND_BUILDERS:
        raise SurrogateConfigError(f"Surrogate backend is already registered: {name}.")
    SURROGATE_BACKEND_BUILDERS[name] = builder


def build_surrogate_backend(
    name: str,
    model_path: str | Path,
    metadata_path: str | Path,
    **kwargs: Any,
) -> Any:
    """Build one deployed surrogate backend from registered configuration."""
    try:
        builder = SURROGATE_BACKEND_BUILDERS[name]
    except KeyError as error:
        raise SurrogateConfigError(f"Unknown surrogate backend: {name}.") from error

    return builder(model_path, metadata_path, **kwargs)
