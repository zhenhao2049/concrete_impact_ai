"""Smoke tests for the initial project scaffold.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from pathlib import Path

import yaml

import concrete_impact


def test_package_version_available() -> None:
    """Check that the package exposes a version string."""
    assert concrete_impact.__version__


def test_environment_file_exists() -> None:
    """Check that the Conda environment specification exists."""
    assert Path("environment.yml").is_file()


def test_environment_file_declares_the_unified_cuda_environment() -> None:
    """Check the unified environment name and exact CUDA PyTorch wheel contract."""
    payload = yaml.safe_load(Path("environment.yml").read_text(encoding="utf-8"))
    assert payload["name"] == "surrogate-model"
    dependencies = payload["dependencies"]
    pip_section = next(item["pip"] for item in dependencies if isinstance(item, dict))
    assert "torch==2.12.1+cu130" in pip_section
    assert not any(
        isinstance(item, str) and item.startswith(("pytorch-cpu", "pytorch-gpu"))
        for item in dependencies
    )
