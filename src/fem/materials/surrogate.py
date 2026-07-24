"""Compatibility imports for material-point surrogate deployment.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from fem.surrogates import (
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    TensorFieldSpec,
    build_surrogate_backend,
    load_surrogate_metadata,
    save_surrogate_metadata,
)
from fem.surrogates.torch_backend import (
    TorchExportMaterialSurrogate,
    TorchStateDictMaterialSurrogate,
    export_torch_material_surrogate,
)

__all__ = [
    "SurrogateArtifactMetadata",
    "SurrogateCapabilities",
    "TensorFieldSpec",
    "TorchExportMaterialSurrogate",
    "TorchStateDictMaterialSurrogate",
    "build_surrogate_backend",
    "export_torch_material_surrogate",
    "load_surrogate_metadata",
    "save_surrogate_metadata",
]
