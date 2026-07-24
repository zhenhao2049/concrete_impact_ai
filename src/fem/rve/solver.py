"""Quasi-static microequilibrium and homogenization for Hex8 RVEs.

Contents:
    Microequilibrium solution, homogenized response, and convergence diagnostics.
Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem.assembly.nonlinear_solid import (
    NonlinearSolidAssemblyResult,
    assemble_hex8_material_response_cached,
    build_hex8_material_assembly_cache,
    initialize_hex8_material_state,
)
from fem.rve.data import (
    RVEOperatorCache,
    RVERequest,
    RVEResponse,
    RVEState,
    StructuredHex8RVE,
)
from fem.rve.periodic import compute_periodic_fluctuation_error
from fem.solvers.line_search import ArmijoLineSearchError, perform_armijo_search
from fem.solvers.linear import solve_sparse_direct


class RVEEquilibriumError(RuntimeError):
    """Report a failed periodic microequilibrium solve with diagnostics."""

    def __init__(self, reason: str, diagnostics: dict[str, Any]) -> None:
        """Initialize one RVE equilibrium failure."""
        super().__init__(f"RVE microequilibrium failed: {reason}.")
        self.reason = reason
        self.diagnostics = diagnostics


def initialize_rve_state(model: StructuredHex8RVE) -> RVEState:
    """Initialize the complete committed state of one RVE."""
    material_state = initialize_hex8_material_state(
        model.material,
        model.elements.shape[0],
        model.quadrature_order,
    )
    independent_dofs = model.constraint.transformation.shape[1]

    return RVEState(
        material_state=material_state,
        macro_strain=np.zeros(6, dtype=np.float64),
        fluctuation_dofs=np.zeros(independent_dofs, dtype=np.float64),
    )


def solve_rve_microequilibrium(
    model: StructuredHex8RVE,
    request: RVERequest,
    committed_state: RVEState,
    operator_cache: RVEOperatorCache | None = None,
) -> RVEResponse:
    """Solve one strain-driven periodic RVE update from a committed state."""
    if request.macro_strain.shape != (6,):
        raise ValueError("RVE macro strain must have shape (6,).")
    if request.time_step <= 0.0:
        raise ValueError("RVE time step must be positive.")
    if operator_cache is None:
        operator_cache = build_rve_operator_cache(model)

    transformation = model.constraint.transformation
    affine = model.constraint.affine_matrix
    previous_displacement = (
        transformation @ committed_state.fluctuation_dofs
        + affine @ committed_state.macro_strain
    )
    fluctuation = committed_state.fluctuation_dofs.copy()
    initial_residual_norm: float | None = None
    last_increment_norm = np.inf
    total_backtracks = 0

    for iteration in range(model.newton_settings.max_iterations + 1):
        displacement = transformation @ fluctuation + affine @ request.macro_strain
        assembly = _assemble_rve_trial(
            model,
            request,
            committed_state,
            displacement,
            previous_displacement,
            operator_cache,
        )
        reduced_residual = np.asarray(
            transformation.T @ assembly.internal_force,
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(reduced_residual))
        if initial_residual_norm is None:
            initial_residual_norm = residual_norm
        residual_tolerance = max(
            model.newton_settings.residual_absolute_tolerance,
            model.newton_settings.residual_relative_tolerance * initial_residual_norm,
        )
        increment_scale = max(float(np.linalg.norm(fluctuation)), 1.0)
        increment_tolerance = max(
            model.newton_settings.increment_absolute_tolerance,
            model.newton_settings.increment_relative_tolerance * increment_scale,
        )
        if residual_norm <= residual_tolerance and (
            iteration == 0 or last_increment_norm <= increment_tolerance
        ):
            return _build_rve_response(
                model,
                request,
                committed_state,
                fluctuation,
                displacement,
                assembly,
                iteration,
                total_backtracks,
                residual_norm,
                residual_tolerance,
                operator_cache,
            )
        if iteration == model.newton_settings.max_iterations:
            raise RVEEquilibriumError(
                "maximum_iterations_exceeded",
                _build_failure_diagnostics(
                    request,
                    iteration,
                    residual_norm,
                    residual_tolerance,
                    last_increment_norm,
                    increment_tolerance,
                    assembly,
                ),
            )

        reduced_tangent = (transformation.T @ assembly.tangent @ transformation).tocsr()
        increment = solve_sparse_direct(reduced_tangent, -reduced_residual)
        if not np.all(np.isfinite(increment)):
            raise RVEEquilibriumError(
                "non_finite_increment",
                _build_failure_diagnostics(
                    request,
                    iteration,
                    residual_norm,
                    residual_tolerance,
                    float(np.linalg.norm(increment)),
                    increment_tolerance,
                    assembly,
                ),
            )
        increment_norm = float(np.linalg.norm(increment))
        if residual_norm <= residual_tolerance and increment_norm <= increment_tolerance:
            return _build_rve_response(
                model,
                request,
                committed_state,
                fluctuation,
                displacement,
                assembly,
                iteration,
                total_backtracks,
                residual_norm,
                residual_tolerance,
                operator_cache,
            )

        accepted_increment = increment
        if model.newton_settings.armijo.enabled:
            def evaluate_candidate(
                step_length: float,
                base_fluctuation: NDArray[np.float64] = fluctuation,
                newton_increment: NDArray[np.float64] = increment,
            ) -> NDArray[np.float64]:
                """Evaluate one RVE candidate from the committed microstate."""
                candidate_fluctuation = base_fluctuation + step_length * newton_increment
                candidate_displacement = (
                    transformation @ candidate_fluctuation
                    + affine @ request.macro_strain
                )
                candidate_assembly = _assemble_rve_trial(
                    model,
                    request,
                    committed_state,
                    candidate_displacement,
                    previous_displacement,
                    operator_cache,
                )
                candidate_residual = np.asarray(
                    transformation.T @ candidate_assembly.internal_force,
                    dtype=np.float64,
                )

                return candidate_residual

            try:
                line_search = perform_armijo_search(
                    reduced_residual,
                    evaluate_candidate,
                    model.newton_settings.armijo,
                )
            except ArmijoLineSearchError as error:
                diagnostics = _build_failure_diagnostics(
                    request,
                    iteration,
                    residual_norm,
                    residual_tolerance,
                    float(np.linalg.norm(increment)),
                    increment_tolerance,
                    assembly,
                )
                diagnostics["armijo"] = error.diagnostics
                raise RVEEquilibriumError("armijo_exhausted", diagnostics) from error
            fluctuation += line_search.step_length * increment
            accepted_increment = line_search.step_length * increment
            total_backtracks += line_search.backtracks
        else:
            fluctuation += increment
        last_increment_norm = float(np.linalg.norm(accepted_increment))

    raise AssertionError("RVE Newton loop ended without a convergence decision.")


def _assemble_rve_trial(
    model: StructuredHex8RVE,
    request: RVERequest,
    committed_state: RVEState,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    operator_cache: RVEOperatorCache,
) -> NonlinearSolidAssemblyResult:
    """Assemble one RVE candidate from the unchanged committed microstate."""
    return assemble_hex8_material_response_cached(
        model.nodes,
        model.elements,
        model.material,
        committed_state.material_state,
        displacement,
        previous_displacement,
        request.time_step,
        request.material_update_settings,
        operator_cache.assembly,
        need_tangent=True,
    )


def _build_rve_response(
    model: StructuredHex8RVE,
    request: RVERequest,
    committed_state: RVEState,
    fluctuation: NDArray[np.float64],
    displacement: NDArray[np.float64],
    assembly: NonlinearSolidAssemblyResult,
    newton_iterations: int,
    armijo_backtracks: int,
    residual_norm: float,
    residual_tolerance: float,
    operator_cache: RVEOperatorCache,
) -> RVEResponse:
    """Homogenize one converged periodic microstate."""
    weights = operator_cache.quadrature_weights
    volume = operator_cache.volume
    macro_stress = np.sum(assembly.stresses * weights[:, None], axis=0) / volume
    virtual_stress = model.constraint.affine_matrix.T @ assembly.internal_force / volume
    effective_tangent = None
    if request.requirements.tangent:
        effective_tangent = _compute_effective_tangent(model, assembly, volume)

    previous_micro_strains = assembly.strains - request.time_step * assembly.strain_rates
    strain_increment = assembly.strains - previous_micro_strains
    micro_work = float(
        np.sum(np.einsum("ij,ij->i", assembly.stresses, strain_increment) * weights)
        / volume
    )
    macro_work = float(macro_stress @ (request.macro_strain - committed_state.macro_strain))
    hill_mandel_residual = macro_work - micro_work
    hill_mandel_error = abs(hill_mandel_residual) / (
        abs(macro_work)
        + abs(micro_work)
        + model.hill_mandel_power_scale * request.time_step
    )
    point_work = assembly.diagnostics.get("incremental_work_density")
    if point_work is None:
        raise ValueError("RVE J2 diagnostics require incremental_work_density.")
    active = assembly.diagnostics.get("viscoplastic_active")
    if active is None:
        raise ValueError("RVE J2 diagnostics require viscoplastic_active.")
    positive_work = float(np.sum(np.maximum(point_work, 0.0) * weights) / volume)
    negative_work = float(np.sum(np.minimum(point_work, 0.0) * weights) / volume)
    unloading_fraction = float(np.sum(weights[point_work < 0.0]) / volume)
    active_fraction = float(np.sum(weights[active > 0.5]) / volume)
    periodic_error = compute_periodic_fluctuation_error(
        displacement,
        request.macro_strain,
        model.constraint,
    )
    reaction_error = _compute_face_reaction_error(model, assembly.internal_force)
    class_reaction_error = _compute_periodic_class_reaction_error(
        model, assembly.internal_force
    )
    next_state = RVEState(
        material_state=assembly.state,
        macro_strain=request.macro_strain.copy(),
        fluctuation_dofs=fluctuation.copy(),
    )
    free_energy_density = float(np.sum(assembly.free_energy * weights) / volume)
    dissipation_density = float(np.sum(assembly.dissipation * weights) / volume)
    phase_diagnostics = _compute_phase_diagnostics(model, assembly, weights, volume)
    activation_margin = assembly.diagnostics.get(
        "yield_activation_margin",
        np.full(weights.size, np.nan),
    )
    finite_activation_margin = activation_margin[np.isfinite(activation_margin)]
    minimum_activation_margin = (
        float(np.min(finite_activation_margin))
        if finite_activation_margin.size > 0
        else np.nan
    )
    maximum_activation_margin = (
        float(np.max(finite_activation_margin))
        if finite_activation_margin.size > 0
        else np.nan
    )

    return RVEResponse(
        macro_stress=macro_stress,
        effective_tangent=effective_tangent,
        state=next_state,
        displacement=displacement,
        micro_strains=assembly.strains,
        micro_stresses=assembly.stresses,
        micro_diagnostics=assembly.diagnostics,
        free_energy_density=free_energy_density,
        dissipation_density=dissipation_density,
        phase_diagnostics=phase_diagnostics,
        diagnostics={
            "micro_residual_norm": residual_norm,
            "micro_residual_tolerance": residual_tolerance,
            "newton_iterations": newton_iterations,
            "armijo_backtracks": armijo_backtracks,
            "volume": volume,
            "virtual_work_stress_error": float(np.linalg.norm(virtual_stress - macro_stress)),
            "periodic_fluctuation_error": periodic_error,
            "reaction_antiperiodicity_error": reaction_error,
            "periodic_class_reaction_error": class_reaction_error,
            "macro_work_increment": macro_work,
            "micro_work_increment": micro_work,
            "hill_mandel_residual": hill_mandel_residual,
            "hill_mandel_error": hill_mandel_error,
            "positive_work_density": positive_work,
            "negative_work_density": negative_work,
            "unloading_volume_fraction": unloading_fraction,
            "viscoplastic_active_volume_fraction": active_fraction,
            "yield_activation_margin_available": int(finite_activation_margin.size > 0),
            "minimum_yield_activation_margin": minimum_activation_margin,
            "maximum_yield_activation_margin": maximum_activation_margin,
        },
    )


def build_rve_operator_cache(model: StructuredHex8RVE) -> RVEOperatorCache:
    """Build immutable geometry, quadrature, phase, and volume data once."""
    assembly = build_hex8_material_assembly_cache(
        model.nodes,
        model.elements,
        model.quadrature_order,
    )
    weights = assembly.jacobian_weights.reshape(-1)
    if np.any(weights <= 0.0):
        raise ValueError("RVE operator cache requires strictly positive quadrature weights.")
    quadrature_count = assembly.jacobian_weights.shape[1]
    element_phase_ids = (
        np.zeros(model.elements.shape[0], dtype=np.int64)
        if model.element_phase_ids is None
        else model.element_phase_ids
    )
    return RVEOperatorCache(
        assembly=assembly,
        quadrature_weights=weights,
        volume=float(np.sum(weights)),
        point_phase_ids=np.repeat(element_phase_ids, quadrature_count),
    )


def _compute_phase_diagnostics(
    model: StructuredHex8RVE,
    assembly: NonlinearSolidAssemblyResult,
    weights: NDArray[np.float64],
    volume: float,
) -> dict[str, dict[str, Any]]:
    """Compute phase volumes, stresses, energies, and real state-field names."""
    if model.element_phase_ids is None:
        point_phase_ids = np.zeros(weights.size, dtype=np.int64)
        phase_names: tuple[str, ...] = ("domain",)
    else:
        quadrature_count = weights.size // model.elements.shape[0]
        point_phase_ids = np.repeat(model.element_phase_ids, quadrature_count)
        phase_names = model.phase_names
    phase_results: dict[str, dict[str, Any]] = {}
    for phase_id, phase_name in enumerate(phase_names):
        mask = point_phase_ids == phase_id
        phase_volume = float(np.sum(weights[mask]))
        state_fields = (
            tuple(assembly.state.variables)
            if model.element_phase_ids is None
            else tuple(
                name
                for name in assembly.state.variables
                if name.startswith(f"phase_{phase_name}__")
            )
        )
        phase_results[phase_name] = {
            "volume": phase_volume,
            "volume_fraction": phase_volume / volume,
            "average_stress": np.sum(
                assembly.stresses[mask] * weights[mask, None],
                axis=0,
            )
            / phase_volume,
            "free_energy_density": float(
                np.sum(assembly.free_energy[mask] * weights[mask]) / phase_volume
            ),
            "dissipation_density": float(
                np.sum(assembly.dissipation[mask] * weights[mask]) / phase_volume
            ),
            "state_fields": state_fields,
        }

    return phase_results


def _compute_effective_tangent(
    model: StructuredHex8RVE,
    assembly: NonlinearSolidAssemblyResult,
    volume: float,
) -> NDArray[np.float64]:
    """Compute the condensed RVE algorithmic tangent by a Schur complement."""
    transformation = model.constraint.transformation
    affine = model.constraint.affine_matrix
    stiffness_affine = np.asarray(assembly.tangent @ affine, dtype=np.float64)
    k_ee = affine.T @ stiffness_affine
    if transformation.shape[1] == 0:
        return k_ee / volume

    k_ww = (transformation.T @ assembly.tangent @ transformation).tocsr()
    k_we = np.asarray(transformation.T @ stiffness_affine, dtype=np.float64)
    sensitivity = solve_sparse_direct(k_ww, k_we)
    if not np.all(np.isfinite(sensitivity)):
        raise RVEEquilibriumError(
            "non_finite_effective_tangent_sensitivity",
            {"reduced_dof_count": transformation.shape[1]},
        )
    k_ew = np.asarray(affine.T @ assembly.tangent @ transformation, dtype=np.float64)

    return (k_ee - k_ew @ sensitivity) / volume


def _compute_face_reaction_error(
    model: StructuredHex8RVE,
    internal_force: NDArray[np.float64],
) -> float:
    """Compute opposite structured-face resultant imbalance."""
    nodal_force = internal_force.reshape(-1, 3)
    resultants: list[float] = []
    for minus_nodes, plus_nodes in model.constraint.opposite_face_nodes:
        minus = np.sum(nodal_force[minus_nodes], axis=0)
        plus = np.sum(nodal_force[plus_nodes], axis=0)
        denominator = max(float(np.linalg.norm(minus) + np.linalg.norm(plus)), 1.0)
        resultants.append(float(np.linalg.norm(minus + plus)) / denominator)

    return max(resultants)


def _compute_periodic_class_reaction_error(
    model: StructuredHex8RVE,
    internal_force: NDArray[np.float64],
) -> float:
    """Compute the maximum resultant over every periodic node equivalence class."""
    nodal_force = internal_force.reshape(-1, 3)
    force_scale = max(float(np.linalg.norm(internal_force)), 1.0)
    maximum = 0.0
    for class_id in range(model.constraint.representative_nodes.size):
        class_force = np.sum(
            nodal_force[model.constraint.equivalence_class_ids == class_id],
            axis=0,
        )
        maximum = max(maximum, float(np.linalg.norm(class_force)) / force_scale)
    return maximum


def _build_failure_diagnostics(
    request: RVERequest,
    iteration: int,
    residual_norm: float,
    residual_tolerance: float,
    increment_norm: float,
    increment_tolerance: float,
    assembly: NonlinearSolidAssemblyResult,
) -> dict[str, Any]:
    """Build one serializable RVE Newton failure snapshot."""
    return {
        "algorithm": "periodic_rve_microequilibrium_newton",
        "macro_strain": request.macro_strain.tolist(),
        "time_step": request.time_step,
        "iteration": iteration,
        "residual_norm": residual_norm,
        "residual_tolerance": residual_tolerance,
        "increment_norm": increment_norm,
        "increment_tolerance": increment_tolerance,
        "maximum_absolute_stress": float(np.max(np.abs(assembly.stresses))),
        "minimum_tangent_diagonal": float(np.min(assembly.tangent.diagonal())),
    }
