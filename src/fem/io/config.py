"""YAML configuration input-output helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    return data


def write_yaml_config(data: dict[str, Any], path: str | Path) -> Path:
    """Write a YAML configuration file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)

    return output_path
