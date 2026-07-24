"""Model definition I/O helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fem.io.config import load_yaml_config
from fem.preprocess.data import ModelDef

if TYPE_CHECKING:
    from fem.cases.data import OutputDef


def load_model_def(config_path: str | Path) -> ModelDef:
    """Load a model definition from a YAML file."""
    config = load_yaml_config(config_path)
    return ModelDef(**config)


def load_preprocess_defs(config_path: str | Path) -> tuple[ModelDef, OutputDef]:
    """Load model and output definitions from one preprocess YAML file."""
    from fem.cases.data import OutputDef

    config = load_yaml_config(config_path)
    output = config.pop("output")

    model_def = ModelDef(**config)
    output_def = OutputDef(
        root=Path(output["root"]),
        mesh_path=Path(output["mesh_path"]),
        vtk_path=Path(output["vtk_path"]),
        save_vtk=bool(output["save_vtk"]),
        save_history=bool(output["save_history"]),
        fields=tuple(output["fields"]),
    )

    return model_def, output_def
