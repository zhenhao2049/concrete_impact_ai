"""Linear elastic material data and kernels.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    compute_isotropic_pressure_wave_speed,
)


@dataclass(frozen=True)
class LinearElasticMaterial:
    """Linear elastic material parameters."""

    name: str
    density: float
    young_modulus: float
    poisson_ratio: float

    @property
    def maximum_wave_speed(self) -> float:
        """Return the elastic longitudinal wave speed."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize an empty state for linear elasticity."""
        return MaterialState(variables={})

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Update linear-elastic material points."""
        return evaluate_linear_elastic_points(request, self, state, requirements)


def build_linear_elastic_material(material_spec: dict[str, Any]) -> LinearElasticMaterial:
    """Build a linear elastic material definition."""
    return LinearElasticMaterial(
        name=str(material_spec["name"]),
        density=float(material_spec["density"]),
        young_modulus=float(material_spec["young_modulus"]),
        poisson_ratio=float(material_spec["poisson_ratio"]),
    )


def build_elasticity_matrix(
    material: LinearElasticMaterial,
    dimension: int,
    plane_state: str,
) -> NDArray[np.float64]:
    """Build the small-strain isotropic elasticity matrix."""
    return ELASTICITY_MATRIX_BUILDERS[(dimension, plane_state)](material)


def evaluate_linear_elastic_points(
    request: MaterialPointRequest,
    material: LinearElasticMaterial,
    state: MaterialState,
    requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
) -> MaterialPointResponse:
    """Evaluate stress, tangent and energy at linear elastic material points."""
    dimension = 3 if request.kinematics == "three_dimensional" else 2
    elasticity_matrix = build_elasticity_matrix(material, dimension, request.kinematics)
    stresses = request.strains @ elasticity_matrix.T
    tangents = None
    if requirements.tangent:
        tangents = np.repeat(elasticity_matrix[None, :, :], request.strains.shape[0], axis=0)
    free_energy = None
    if requirements.free_energy:
        free_energy = 0.5 * np.einsum("qi,qi->q", request.strains, stresses)
    dissipation = None
    if requirements.dissipation:
        dissipation = np.zeros(request.strains.shape[0], dtype=np.float64)

    return MaterialPointResponse(
        stresses=stresses,
        state=state,
        tangents=tangents,
        free_energy=free_energy,
        dissipation=dissipation,
    )


def _build_plane_strain_matrix(material: LinearElasticMaterial) -> NDArray[np.float64]:
    """Build the plane-strain elasticity matrix."""
    young_modulus = material.young_modulus
    poisson_ratio = material.poisson_ratio
    coefficient = young_modulus / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

    matrix = coefficient * np.asarray(
        [
            [1.0 - poisson_ratio, poisson_ratio, 0.0],
            [poisson_ratio, 1.0 - poisson_ratio, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * poisson_ratio)],
        ],
        dtype=np.float64,
    )

    return matrix


def _build_plane_stress_matrix(material: LinearElasticMaterial) -> NDArray[np.float64]:
    """Build the plane-stress elasticity matrix."""
    young_modulus = material.young_modulus
    poisson_ratio = material.poisson_ratio
    coefficient = young_modulus / (1.0 - poisson_ratio**2)

    matrix = coefficient * np.asarray(
        [
            [1.0, poisson_ratio, 0.0],
            [poisson_ratio, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson_ratio)],
        ],
        dtype=np.float64,
    )

    return matrix


def _build_three_dimensional_matrix(material: LinearElasticMaterial) -> NDArray[np.float64]:
    """Build the three-dimensional elasticity matrix."""
    young_modulus = material.young_modulus
    poisson_ratio = material.poisson_ratio
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus
        * poisson_ratio
        / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lame_lambda
    matrix[0, 0] += 2.0 * shear_modulus
    matrix[1, 1] += 2.0 * shear_modulus
    matrix[2, 2] += 2.0 * shear_modulus
    matrix[3, 3] = shear_modulus
    matrix[4, 4] = shear_modulus
    matrix[5, 5] = shear_modulus

    return matrix


ELASTICITY_MATRIX_BUILDERS = {
    (2, "plane_strain"): _build_plane_strain_matrix,
    (2, "plane_stress"): _build_plane_stress_matrix,
    (3, "three_dimensional"): _build_three_dimensional_matrix,
}
