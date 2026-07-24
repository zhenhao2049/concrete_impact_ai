"""Project configuration loading helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path
from typing import Any

from fem.io.config import load_yaml_config as _load_yaml_config


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    return _load_yaml_config(path)
