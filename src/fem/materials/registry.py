"""Material builder registry.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.materials.linear_elastic import (
    LinearElasticMaterial,
    build_linear_elastic_material,
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
)

MATERIAL_BUILDERS = {
    "linear_elastic": build_linear_elastic_material,
    "j2_plastic": build_j2_plastic_material,
    "j2_viscoplastic": build_j2_viscoplastic_material,
    "drucker_prager": build_drucker_prager_material,
    "drucker_prager_cap": build_drucker_prager_cap_material,
}


def build_material(
    material_spec: dict[str, object],
) -> (
    LinearElasticMaterial
    | J2PlasticMaterial
    | J2ViscoplasticMaterial
    | DruckerPragerMaterial
    | DruckerPragerCapMaterial
):
    """Build a material definition from the material registry."""
    return MATERIAL_BUILDERS[str(material_spec["type"])](material_spec)
