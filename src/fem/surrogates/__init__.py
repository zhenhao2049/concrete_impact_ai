"""Project-neutral surrogate interfaces and artifact backends.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from fem.surrogates.data import (
    MaterialPointSurrogate,
    ResidualCorrectionRequest,
    ResidualCorrectionResponse,
    ResidualCorrectorSurrogate,
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    SurrogateKind,
    TensorFieldSpec,
    TimeRepresentation,
    VelocityVerletCorrectionRequest,
    VelocityVerletCorrectionResponse,
    VelocityVerletCorrectorMetadata,
    VelocityVerletResidualCorrector,
)
from fem.surrogates.errors import SurrogateCapabilityError, SurrogateConfigError
from fem.surrogates.metadata import (
    load_surrogate_metadata,
    save_surrogate_metadata,
    validate_surrogate_metadata,
)
from fem.surrogates.registry import (
    build_surrogate_backend,
    register_surrogate_backend,
)
from fem.surrogates.torch_backend import (
    TorchExportMaterialSurrogate,
    TorchStateDictMaterialSurrogate,
    export_torch_material_surrogate,
)

__all__ = [
    "MaterialPointSurrogate",
    "ResidualCorrectionRequest",
    "ResidualCorrectionResponse",
    "ResidualCorrectorSurrogate",
    "SurrogateArtifactMetadata",
    "SurrogateCapabilities",
    "SurrogateCapabilityError",
    "SurrogateConfigError",
    "SurrogateKind",
    "TensorFieldSpec",
    "TimeRepresentation",
    "VelocityVerletCorrectionRequest",
    "VelocityVerletCorrectionResponse",
    "VelocityVerletCorrectorMetadata",
    "VelocityVerletResidualCorrector",
    "TorchExportMaterialSurrogate",
    "TorchStateDictMaterialSurrogate",
    "build_surrogate_backend",
    "export_torch_material_surrogate",
    "load_surrogate_metadata",
    "register_surrogate_backend",
    "save_surrogate_metadata",
    "validate_surrogate_metadata",
]
