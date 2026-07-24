"""Continuous-time structured J2 viscoplastic RNO material model.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from concrete_impact.nn.config import J2VPRNOModelConfig
from fem.ad import torch_jacobian
from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
    compute_isotropic_pressure_wave_speed,
)
from fem.materials.errors import MaterialPointConvergenceError
from fem.materials.linear_elastic import LinearElasticMaterial, build_elasticity_matrix
from fem.surrogates import (
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    TensorFieldSpec,
)

VOIGT_DIM = 6
IDENTITY = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
STRESS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float64)
TENSOR_TO_ENGINEERING_STRAIN = torch.diag(
    torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float64)
)


class FlowRateCorrectionNetwork(torch.nn.Module):
    """Map isotropic dimensionless state features to a log-rate multiplier."""

    def __init__(self, config: J2VPRNOModelConfig) -> None:
        """Build a tanh multilayer perceptron with exact zero baseline output."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        input_width = 2
        for _ in range(config.hidden_layers):
            layers.append(torch.nn.Linear(input_width, config.hidden_size, dtype=torch.float64))
            layers.append(torch.nn.Tanh())
            input_width = config.hidden_size
        output_layer = torch.nn.Linear(input_width, 1, dtype=torch.float64)
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        self.network = torch.nn.Sequential(*layers)
        self.correction_scale = float(config.correction_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Evaluate the dimensionless logarithmic flow-rate correction."""
        return self.correction_scale * self.network(features).squeeze(-1)


class StructuredJ2VPRNO(torch.nn.Module):
    """Preserve J2-Perzyna mechanics while learning a scalar rate correction."""

    def __init__(
        self,
        *,
        name: str,
        density: float,
        young_modulus: float,
        poisson_ratio: float,
        yield_stress: float,
        hardening_modulus: float,
        time_scale: float,
        reference_stress: float,
        rate_exponent: float,
        equivalent_plastic_strain_scale: float,
        model_config: J2VPRNOModelConfig,
    ) -> None:
        """Initialize physical parameters and the continuous flow-rate network."""
        super().__init__()
        self.name = name
        self.density = density
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_stress = yield_stress
        self.hardening_modulus = hardening_modulus
        self.time_scale = time_scale
        self.reference_stress = reference_stress
        self.rate_exponent = rate_exponent
        self.equivalent_plastic_strain_scale = equivalent_plastic_strain_scale
        self.flow_correction = FlowRateCorrectionNetwork(model_config)
        self.metadata = _build_metadata(
            name,
            reference_stress,
            equivalent_plastic_strain_scale,
        )
        self._validate_parameters()

    @property
    def maximum_wave_speed(self) -> float:
        """Return the retained elastic longitudinal wave-speed bound."""
        return compute_isotropic_pressure_wave_speed(
            self.young_modulus, self.poisson_ratio, self.density
        )

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize plastic strain, accumulated plastic strain, and multiplier."""
        return MaterialState(
            variables={
                "plastic_strain": np.zeros((n_points, VOIGT_DIM), dtype=np.float64),
                "equivalent_plastic_strain": np.zeros(n_points, dtype=np.float64),
                "viscoplastic_multiplier": np.zeros(n_points, dtype=np.float64),
            }
        )

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Integrate the continuous RNO flow law by strict local backward Euler."""
        if request.kinematics != "three_dimensional":
            raise ValueError("Structured J2-VP-RNO requires three_dimensional kinematics.")
        if request.strains.ndim != 2 or request.strains.shape[1] != VOIGT_DIM:
            raise ValueError("Structured J2-VP-RNO requires strain shape (n_points, 6).")
        point_count = request.strains.shape[0]
        stresses = np.zeros((point_count, VOIGT_DIM), dtype=np.float64)
        tangents = (
            np.zeros((point_count, VOIGT_DIM, VOIGT_DIM), dtype=np.float64)
            if requirements.tangent
            else None
        )
        free_energy = np.zeros(point_count, dtype=np.float64) if requirements.free_energy else None
        dissipation = np.zeros(point_count, dtype=np.float64) if requirements.dissipation else None
        next_variables = {name: values.copy() for name, values in state.variables.items()}
        diagnostics = {
            "trial_yield_value": np.zeros(point_count, dtype=np.float64),
            "yield_value": np.zeros(point_count, dtype=np.float64),
            "plastic_multiplier": np.zeros(point_count, dtype=np.float64),
            "equivalent_plastic_strain": np.zeros(point_count, dtype=np.float64),
            "local_iterations": np.zeros(point_count, dtype=np.float64),
            "local_residual": np.zeros(point_count, dtype=np.float64),
            "log_rate_correction": np.zeros(point_count, dtype=np.float64),
            "yield_scale": np.zeros(point_count, dtype=np.float64),
            "yield_activation_threshold": np.zeros(point_count, dtype=np.float64),
            "yield_activation_margin": np.zeros(point_count, dtype=np.float64),
            "viscoplastic_active": np.zeros(point_count, dtype=np.float64),
        }

        for point_id in range(point_count):
            try:
                result = self._update_point(
                    request.strains[point_id],
                    state,
                    point_id,
                    request.time_step,
                    request.update_settings,
                    requirements.tangent,
                )
            except MaterialPointConvergenceError as error:
                raise error.with_context(
                    material_name=self.name,
                    material_type=type(self).__name__,
                    point_id=point_id,
                    time_step=request.time_step,
                    strain=request.strains[point_id].tolist(),
                ) from error
            stresses[point_id] = result["stress"]
            if tangents is not None:
                tangents[point_id] = result["tangent"]
            if free_energy is not None:
                free_energy[point_id] = result["free_energy"]
            if dissipation is not None:
                dissipation[point_id] = result["dissipation"]
            for state_name, value in result["state"].items():
                next_variables[state_name][point_id] = value
            for diagnostic_name in diagnostics:
                diagnostics[diagnostic_name][point_id] = result["diagnostics"][diagnostic_name]

        return MaterialPointResponse(
            stresses=stresses,
            state=MaterialState(variables=next_variables),
            tangents=tangents,
            free_energy=free_energy,
            dissipation=dissipation,
            diagnostics=diagnostics,
        )

    def _update_point(
        self,
        strain_numpy: NDArray[np.float64],
        state: MaterialState,
        point_id: int,
        time_step: float,
        update_settings: Any,
        need_tangent: bool,
    ) -> dict[str, Any]:
        """Update one point and compute an implicit-function tangent when requested."""
        strain = torch.tensor(strain_numpy, dtype=torch.float64)
        plastic_strain = torch.tensor(
            state.variables["plastic_strain"][point_id], dtype=torch.float64
        )
        q_old = torch.tensor(
            float(state.variables["equivalent_plastic_strain"][point_id]),
            dtype=torch.float64,
        )
        elasticity = self._elasticity_tensor()
        trial_stress = elasticity @ (strain - plastic_strain)
        trial_deviatoric = _deviatoric(trial_stress)
        trial_equivalent = _strict_equivalent_stress(trial_deviatoric)
        trial_yield = trial_equivalent - self.yield_stress - self.hardening_modulus * q_old
        yield_scale = max(
            self.yield_stress,
            self.reference_stress,
            float(trial_equivalent),
        )
        yield_relative_tolerance = float(update_settings.yield_relative_tolerance)
        yield_activation_threshold = yield_relative_tolerance * yield_scale
        if (
            not np.isfinite(yield_relative_tolerance)
            or yield_relative_tolerance < 0.0
            or not np.isfinite(yield_activation_threshold)
        ):
            raise MaterialPointConvergenceError(
                "Structured J2-VP-RNO received an invalid yield activation tolerance.",
                "invalid_yield_activation_tolerance",
                {
                    "algorithm": "structured_j2_vp_rno_yield_activation",
                    "yield_relative_tolerance": yield_relative_tolerance,
                    "yield_scale": yield_scale,
                    "yield_activation_threshold": yield_activation_threshold,
                    "trial_yield_value": float(trial_yield),
                },
            )
        plastic_active = float(trial_yield) > yield_activation_threshold
        yield_activation_margin = (
            float(trial_yield) - yield_activation_threshold
        ) / yield_scale
        flow_direction = torch.zeros(VOIGT_DIM, dtype=torch.float64)
        multiplier = torch.tensor(0.0, dtype=torch.float64)
        iterations = 0
        residual_value = 0.0

        if plastic_active:
            flow_direction = 1.5 * trial_deviatoric / trial_equivalent
            multiplier, iterations, residual_value = self._solve_multiplier(
                trial_yield,
                q_old,
                time_step,
                update_settings,
            )

        stress = self._stress_from_multiplier(
            strain,
            plastic_strain,
            flow_direction,
            multiplier,
            elasticity,
        )
        plastic_increment = (
            multiplier * TENSOR_TO_ENGINEERING_STRAIN @ flow_direction
        )
        next_plastic_strain = plastic_strain + plastic_increment
        next_q = q_old + multiplier
        final_deviatoric = _deviatoric(stress)
        final_equivalent = _strict_equivalent_stress(final_deviatoric)
        final_yield = final_equivalent - self.yield_stress - self.hardening_modulus * next_q
        log_correction = (
            self._log_correction(final_yield, next_q)
            if plastic_active
            else torch.tensor(0.0)
        )
        elastic_strain = strain - next_plastic_strain
        energy = 0.5 * torch.dot(stress, elastic_strain) + 0.5 * self.hardening_modulus * next_q**2
        plastic_work = torch.dot(stress, plastic_increment)
        dissipation = (
            plastic_work / time_step
            - self.hardening_modulus * next_q * multiplier / time_step
        )
        if float(dissipation) < 0.0:
            raise MaterialPointConvergenceError(
                "Structured J2-VP-RNO produced negative dissipation.",
                "negative_dissipation",
                {
                    "algorithm": "structured_j2_vp_rno_backward_euler",
                    "dissipation": float(dissipation),
                    "plastic_multiplier": float(multiplier),
                    "yield_value": float(final_yield),
                },
            )
        tangent = None
        if need_tangent:
            tangent = self._implicit_tangent(
                strain,
                plastic_strain,
                q_old,
                flow_direction,
                multiplier,
                elasticity,
                time_step,
                plastic_active,
            )

        return {
            "stress": stress.detach().numpy(),
            "tangent": None if tangent is None else tangent.detach().numpy(),
            "free_energy": float(energy),
            "dissipation": float(dissipation),
            "state": {
                "plastic_strain": next_plastic_strain.detach().numpy(),
                "equivalent_plastic_strain": float(next_q),
                "viscoplastic_multiplier": float(
                    state.variables["viscoplastic_multiplier"][point_id] + float(multiplier)
                ),
            },
            "diagnostics": {
                "trial_yield_value": float(trial_yield),
                "yield_value": float(final_yield),
                "plastic_multiplier": float(multiplier),
                "equivalent_plastic_strain": float(next_q),
                "local_iterations": float(iterations),
                "local_residual": residual_value,
                "log_rate_correction": float(log_correction.detach()),
                "yield_scale": yield_scale,
                "yield_activation_threshold": yield_activation_threshold,
                "yield_activation_margin": yield_activation_margin,
                "viscoplastic_active": float(plastic_active),
            },
        }

    def _solve_multiplier(
        self,
        trial_yield: torch.Tensor,
        q_old: torch.Tensor,
        time_step: float,
        settings: Any,
    ) -> tuple[torch.Tensor, int, float]:
        """Solve the scalar backward-Euler residual by unprotected Newton."""
        hardening_slope = 3.0 * self._shear_modulus() + self.hardening_modulus
        upper_bound = trial_yield / hardening_slope
        multiplier = 0.5 * upper_bound.detach()
        tolerance = max(
            settings.residual_absolute_tolerance,
            settings.residual_relative_tolerance * float(upper_bound),
        )
        for iteration in range(1, settings.max_iterations + 1):
            candidate = multiplier.detach().requires_grad_(True)
            residual = self._residual(candidate, trial_yield, q_old, time_step)
            derivative = torch_jacobian(
                lambda value: self._residual(value, trial_yield, q_old, time_step),
                candidate,
            )
            diagnostics = {
                "algorithm": "structured_j2_vp_rno_backward_euler",
                "iteration": iteration,
                "max_iterations": settings.max_iterations,
                "residual": float(residual.detach()),
                "residual_tolerance": tolerance,
                "derivative": float(derivative.detach()),
                "candidate_multiplier": float(candidate.detach()),
                "physical_upper_bound": float(upper_bound),
                "trial_yield_value": float(trial_yield),
                "time_step": time_step,
            }
            if not torch.isfinite(residual) or not torch.isfinite(derivative):
                raise MaterialPointConvergenceError(
                    "Structured J2-VP-RNO local Newton produced non-finite values.",
                    "non_finite_iteration_value",
                    diagnostics,
                )
            if float(derivative.detach()) <= 0.0:
                raise MaterialPointConvergenceError(
                    "Structured J2-VP-RNO local residual derivative is non-positive.",
                    "non_positive_derivative",
                    diagnostics,
                )
            if abs(float(residual.detach())) <= tolerance:
                if not 0.0 < float(candidate.detach()) < float(upper_bound):
                    raise MaterialPointConvergenceError(
                        "Structured J2-VP-RNO converged outside its physical domain.",
                        "converged_solution_out_of_bounds",
                        diagnostics,
                    )

                return candidate.detach(), iteration, float(residual.detach())

            newton_candidate = candidate - residual / derivative
            diagnostics["newton_candidate"] = float(newton_candidate.detach())
            if not torch.isfinite(newton_candidate):
                raise MaterialPointConvergenceError(
                    "Structured J2-VP-RNO Newton candidate is non-finite.",
                    "non_finite_newton_candidate",
                    diagnostics,
                )
            if not 0.0 < float(newton_candidate.detach()) < float(upper_bound):
                raise MaterialPointConvergenceError(
                    "Structured J2-VP-RNO Newton candidate left its physical domain.",
                    "newton_candidate_out_of_bounds",
                    diagnostics,
                )
            multiplier = newton_candidate.detach()

        raise MaterialPointConvergenceError(
            "Structured J2-VP-RNO local Newton exceeded its iteration limit.",
            "maximum_iterations_exceeded",
            {
                "algorithm": "structured_j2_vp_rno_backward_euler",
                "iteration": settings.max_iterations,
                "residual": float(residual.detach()),
                "residual_tolerance": tolerance,
                "candidate_multiplier": float(multiplier),
                "physical_upper_bound": float(upper_bound),
            },
        )

    def _residual(
        self,
        multiplier: torch.Tensor,
        trial_yield: torch.Tensor,
        q_old: torch.Tensor,
        time_step: float,
    ) -> torch.Tensor:
        """Evaluate the continuous flow law after backward-Euler discretization."""
        hardening_slope = 3.0 * self._shear_modulus() + self.hardening_modulus
        overstress = trial_yield - hardening_slope * multiplier
        q_next = q_old + multiplier
        log_correction = self._log_correction(overstress, q_next)
        base_rate = (
            (overstress / self.reference_stress) ** self.rate_exponent / self.time_scale
        )

        return multiplier - time_step * base_rate * torch.exp(log_correction)

    def _log_correction(
        self,
        overstress: torch.Tensor,
        equivalent_plastic_strain: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the invariant logarithmic rate correction at a candidate state."""
        features = torch.stack(
            (
                overstress / self.reference_stress,
                equivalent_plastic_strain / self.equivalent_plastic_strain_scale,
            )
        )

        return self.flow_correction(features)

    def _implicit_tangent(
        self,
        strain: torch.Tensor,
        plastic_strain: torch.Tensor,
        q_old: torch.Tensor,
        flow_direction: torch.Tensor,
        multiplier: torch.Tensor,
        elasticity: torch.Tensor,
        time_step: float,
        plastic: bool,
    ) -> torch.Tensor:
        """Differentiate the converged local residual by the implicit function theorem."""
        if not plastic:
            return elasticity
        gamma = multiplier.detach()
        def trial_yield_function(value: torch.Tensor) -> torch.Tensor:
            """Evaluate trial overstress as a differentiable strain function."""
            equivalent_stress = _equivalent_stress(
                _deviatoric(elasticity @ (value - plastic_strain))
            )

            return equivalent_stress - self.yield_stress - self.hardening_modulus * q_old

        def residual_strain(value: torch.Tensor) -> torch.Tensor:
            """Evaluate the local residual as a strain function."""
            return self._residual(
                gamma,
                trial_yield_function(value),
                q_old,
                time_step,
            )

        def residual_gamma(value: torch.Tensor) -> torch.Tensor:
            """Evaluate the local residual as a multiplier function."""
            return self._residual(
                value,
                trial_yield_function(strain),
                q_old,
                time_step,
            )
        dr_dstrain = torch_jacobian(residual_strain, strain)
        dr_dgamma = torch_jacobian(residual_gamma, gamma)
        if not torch.isfinite(dr_dgamma) or float(dr_dgamma.detach()) == 0.0:
            raise MaterialPointConvergenceError(
                "Structured J2-VP-RNO implicit tangent has a singular local Jacobian.",
                "singular_residual_jacobian",
                {
                    "algorithm": "structured_j2_vp_rno_implicit_tangent",
                    "residual_state_derivative": float(dr_dgamma.detach()),
                    "plastic_multiplier": float(gamma),
                },
            )
        dgamma_dstrain = -dr_dstrain / dr_dgamma
        def stress_strain(value: torch.Tensor) -> torch.Tensor:
            """Evaluate converged stress as a strain function."""
            return self._stress_candidate(value, plastic_strain, gamma, elasticity)

        def stress_gamma(value: torch.Tensor) -> torch.Tensor:
            """Evaluate converged stress as a multiplier function."""
            return self._stress_candidate(strain, plastic_strain, value, elasticity)
        partial_stress_strain = torch_jacobian(stress_strain, strain)
        partial_stress_gamma = torch_jacobian(stress_gamma, gamma)

        return partial_stress_strain + torch.outer(partial_stress_gamma, dgamma_dstrain)

    def _stress_candidate(
        self,
        strain: torch.Tensor,
        plastic_strain: torch.Tensor,
        multiplier: torch.Tensor,
        elasticity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stress while retaining the trial-direction strain dependence."""
        trial_stress = elasticity @ (strain - plastic_strain)
        trial_deviatoric = _deviatoric(trial_stress)
        flow_direction = 1.5 * trial_deviatoric / _equivalent_stress(trial_deviatoric)

        return trial_stress - 2.0 * self._shear_modulus() * multiplier * flow_direction

    def _stress_from_multiplier(
        self,
        strain: torch.Tensor,
        plastic_strain: torch.Tensor,
        flow_direction: torch.Tensor,
        multiplier: torch.Tensor,
        elasticity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute stress from one candidate plastic multiplier."""
        trial_stress = elasticity @ (strain - plastic_strain)

        return trial_stress - 2.0 * self._shear_modulus() * multiplier * flow_direction

    def _elasticity_tensor(self) -> torch.Tensor:
        """Build the three-dimensional elastic matrix as a torch tensor."""
        material = LinearElasticMaterial(
            name=self.name,
            density=self.density,
            young_modulus=self.young_modulus,
            poisson_ratio=self.poisson_ratio,
        )

        return torch.as_tensor(
            build_elasticity_matrix(material, 3, "three_dimensional"),
            dtype=torch.float64,
        )

    def _shear_modulus(self) -> float:
        """Compute the elastic shear modulus."""
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def _validate_parameters(self) -> None:
        """Validate the physical domain of the structured material."""
        if self.density <= 0.0 or self.young_modulus <= 0.0:
            raise ValueError("Structured J2-VP-RNO requires positive density and modulus.")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Structured J2-VP-RNO requires -1 < poisson_ratio < 0.5.")
        if self.yield_stress <= 0.0 or self.hardening_modulus < 0.0:
            raise ValueError("Structured J2-VP-RNO has invalid yield or hardening parameters.")
        if self.time_scale <= 0.0 or self.reference_stress <= 0.0:
            raise ValueError("Structured J2-VP-RNO requires positive rate scales.")
        if self.rate_exponent < 1.0 or self.equivalent_plastic_strain_scale <= 0.0:
            raise ValueError("Structured J2-VP-RNO has invalid rate exponent or state scale.")


def _deviatoric(stress: torch.Tensor) -> torch.Tensor:
    """Compute the deviatoric part of a stress-like Voigt vector."""
    return stress - torch.sum(stress[:3]) * IDENTITY / 3.0


def _equivalent_stress(deviatoric: torch.Tensor) -> torch.Tensor:
    """Compute the differentiable J2 equivalent-stress expression."""
    radicand = 1.5 * torch.sum(STRESS_WEIGHTS * deviatoric * deviatoric)

    return torch.sqrt(radicand)


def _strict_equivalent_stress(deviatoric: torch.Tensor) -> torch.Tensor:
    """Validate the J2 invariant before evaluating its square root."""
    radicand = 1.5 * torch.sum(STRESS_WEIGHTS * deviatoric * deviatoric)
    radicand_value = float(radicand)
    if not np.isfinite(radicand_value) or radicand_value < 0.0:
        raise MaterialPointConvergenceError(
            "Structured J2-VP-RNO received an invalid equivalent-stress invariant.",
            "invalid_j2_invariant",
            {
                "algorithm": "structured_j2_vp_rno_equivalent_stress",
                "radicand": radicand_value,
                "deviatoric_stress": deviatoric.detach().tolist(),
            },
        )

    return torch.sqrt(radicand)


def _build_metadata(
    name: str,
    reference_stress: float,
    equivalent_plastic_strain_scale: float,
) -> SurrogateArtifactMetadata:
    """Build the fixed semantic contract for the first structured RNO."""
    return SurrogateArtifactMetadata(
        schema_version="1.0",
        model_name=name,
        model_family="j2_vp_rno",
        kind="material_point",
        backend="pytorch",
        artifact_format="state_dict",
        dtype="float64",
        device="cpu",
        kinematics=("three_dimensional",),
        mandel_convention="voigt_engineering_shear_11_22_33_12_23_13",
        input_fields=(
            TensorFieldSpec("strain", 6, "dimensionless"),
            TensorFieldSpec("state.plastic_strain", 6, "dimensionless"),
            TensorFieldSpec("state.equivalent_plastic_strain", 1, "dimensionless"),
        ),
        output_fields=(TensorFieldSpec("stress", 6, "material_stress_unit"),),
        state_fields=(
            TensorFieldSpec("plastic_strain", 6, "dimensionless"),
            TensorFieldSpec("equivalent_plastic_strain", 1, "dimensionless"),
            TensorFieldSpec("viscoplastic_multiplier", 1, "dimensionless"),
        ),
        capabilities=SurrogateCapabilities(
            provides_tangent=True,
            provides_free_energy=True,
            provides_dissipation=True,
            supports_autodiff_tangent=True,
            uses_strain_rate=False,
            time_representation="continuous",
        ),
        normalization={
            "overstress": {"scale": (reference_stress,)},
            "equivalent_plastic_strain": {
                "scale": (equivalent_plastic_strain_scale,)
            },
        },
        framework_version=torch.__version__,
    )
