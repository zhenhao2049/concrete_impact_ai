"""Material models and constitutive utilities.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.materials.data import (
    MaterialModel,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
)
from fem.materials.errors import MaterialPointConvergenceError
from fem.materials.frozen import FrozenHistoryMaterial, build_frozen_history_material
from fem.materials.linear_elastic import (
    LinearElasticMaterial,
    build_elasticity_matrix,
    build_linear_elastic_material,
    evaluate_linear_elastic_points,
)
from fem.materials.phased import (
    PhasedMaterialModel,
    RVEPhase,
    decode_phase_state_key,
    encode_phase_state_key,
)
from fem.materials.plasticity import (
    DruckerPragerCapMaterial,
    DruckerPragerMaterial,
    J2PlasticMaterial,
    J2ViscoplasticMaterial,
    build_drucker_prager_cap_material,
    build_drucker_prager_material,
    build_j2_plastic_material,
    build_j2_viscoplastic_material,
    compute_stress_invariants,
    expand_plane_strain_to_3d,
    reduce_3d_stress_to_plane_strain,
)
from fem.materials.registry import build_material
from fem.materials.surrogate import (
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    TensorFieldSpec,
    TorchExportMaterialSurrogate,
    TorchStateDictMaterialSurrogate,
    build_surrogate_backend,
    export_torch_material_surrogate,
    load_surrogate_metadata,
    save_surrogate_metadata,
)

__all__ = [
    "DruckerPragerCapMaterial",
    "DruckerPragerMaterial",
    "FrozenHistoryMaterial",
    "J2PlasticMaterial",
    "J2ViscoplasticMaterial",
    "LinearElasticMaterial",
    "MaterialModel",
    "SurrogateArtifactMetadata",
    "SurrogateCapabilities",
    "TensorFieldSpec",
    "MaterialPointRequest",
    "MaterialPointResponse",
    "MaterialResponseRequirements",
    "MaterialPointConvergenceError",
    "MaterialState",
    "MaterialUpdateSettings",
    "PhasedMaterialModel",
    "RVEPhase",
    "decode_phase_state_key",
    "encode_phase_state_key",
    "TorchExportMaterialSurrogate",
    "TorchStateDictMaterialSurrogate",
    "build_drucker_prager_cap_material",
    "build_drucker_prager_material",
    "build_elasticity_matrix",
    "build_frozen_history_material",
    "build_j2_plastic_material",
    "build_j2_viscoplastic_material",
    "build_linear_elastic_material",
    "build_material",
    "build_surrogate_backend",
    "compute_stress_invariants",
    "evaluate_linear_elastic_points",
    "expand_plane_strain_to_3d",
    "export_torch_material_surrogate",
    "load_surrogate_metadata",
    "reduce_3d_stress_to_plane_strain",
    "save_surrogate_metadata",
]
