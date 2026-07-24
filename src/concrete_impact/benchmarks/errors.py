"""Structured benchmark acceptance errors.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

from typing import Any


class BenchmarkAcceptanceError(RuntimeError):
    """Report a completed benchmark whose acceptance criteria were not met."""

    def __init__(
        self,
        case_name: str,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> None:
        """Store benchmark failure context for CLI and tests."""
        super().__init__(f"Benchmark acceptance failed for {case_name}: {reason}.")
        self.case_name = case_name
        self.reason = reason
        self.diagnostics = diagnostics
