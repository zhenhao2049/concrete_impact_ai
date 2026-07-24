"""Portable response-shard manifest utilities.

Contents:
    Relative-path publication and path resolution for movable datasets.
Author:
    Zhen Hao.
Created:
    2026-07-19.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_response_shard_paths(manifest_path: str | Path) -> tuple[Path, ...]:
    """Resolve legacy or portable response-shard paths."""
    return load_portable_manifest_paths(manifest_path, "response_shards")


def load_portable_manifest_paths(
    manifest_path: str | Path,
    field_name: str,
) -> tuple[Path, ...]:
    """Resolve one explicit path list from a legacy or portable manifest."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent if payload.get("path_base") == "manifest_directory" else Path(".")
    return tuple((base / Path(value)).resolve() for value in payload[field_name])


def portable_response_manifest(
    manifest_sha256: str,
    dataset_root: Path,
    response_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Build a dataset-root-relative response manifest."""
    return {
        "schema_version": "1.1",
        "manifest_sha256": manifest_sha256,
        "path_base": "manifest_directory",
        "response_shards": [
            str(Path(os.path.relpath(path, dataset_root))) for path in response_paths
        ],
    }
