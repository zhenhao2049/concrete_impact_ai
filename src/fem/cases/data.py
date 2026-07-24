"""Generic FEM case data containers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fem.preprocess.data import ModelDef
from fem.solvers.data import FEMRunDef


@dataclass(frozen=True)
class OutputDef:
    """Store result-output settings for one case."""

    root: Path
    mesh_path: Path
    vtk_path: Path
    save_vtk: bool
    save_history: bool
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CaseDef:
    """Store one complete FEM case definition."""

    name: str
    model: ModelDef
    run: FEMRunDef
    output: OutputDef
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunResult:
    """Store one complete FEM case result."""

    name: str
    metrics: dict[str, float]
    output_paths: dict[str, Path]
    passed: bool
    resolved_config: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
