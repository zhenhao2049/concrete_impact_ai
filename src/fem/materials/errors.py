"""Material-update exceptions with structured numerical diagnostics.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

import math
from typing import Any


class MaterialPointConvergenceError(RuntimeError):
    """Report one failed material-point nonlinear update."""

    def __init__(
        self,
        message: str,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> None:
        """Initialize a material-point failure and its serializable diagnostics."""
        super().__init__(message)
        self.reason = reason
        self.diagnostics = _to_serializable(diagnostics)

    def with_context(self, **context: Any) -> MaterialPointConvergenceError:
        """Return the same failure enriched with outer-layer context."""
        diagnostics = {**self.diagnostics, **context}

        return MaterialPointConvergenceError(str(self), self.reason, diagnostics)


def _to_serializable(value: Any) -> Any:
    """Convert numerical diagnostic values to strict JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        return _to_serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    return value
