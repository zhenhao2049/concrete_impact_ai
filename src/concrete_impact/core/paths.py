"""Project path utilities.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[3]


def default_results_root() -> Path:
    """Return the default run-results directory."""
    return project_root() / "results" / "runs"

