"""Material-point explicit and implicit solid dynamics.

Contents:
    Dynamic state integration, material assembly, energies, histories, and diagnostics.
Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from fem.assembly.elasticity import assemble_consistent_mass_matrix, assemble_lumped_mass_vector
from fem.assembly.nonlinear_solid import (
    NonlinearSolidAssemblyResult,
    assemble_bundle_material_response,
    build_material_assembly_cache,
    initialize_bundle_material_state,
    integrate_quadrature_density,
)
from fem.dynamics.stability import check_explicit_stability
from fem.materials.data import MaterialState, MaterialUpdateSettings
from fem.preprocess.data import PreprocessBundle
from fem.solvers.data import (
    DirichletDofSet,
    MaterialDynamicSolution,
    NonlinearNewtonSettings,
    TimeIntegrationSettings,
)
from fem.solvers.line_search import ArmijoLineSearchError, perform_armijo_search
from fem.solvers.linear import solve_sparse_direct
from fem.solvers.progress import ProgressCallback, SolverProgressEvent
from fem.solvers.time_step import cut_back_time_step, grow_time_step
from fem.surrogates.data import (
    VelocityVerletCorrectionRequest,
    VelocityVerletCorrectionResponse,
    VelocityVerletResidualCorrector,
)

LoadFunction = Callable[[float], NDArray[np.float64]]


@dataclass(frozen=True)
class _StepState:
    """Store one accepted implicit substep state."""

    time: float
    displacement: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    material_state: MaterialState
    assembly: NonlinearSolidAssemblyResult


@dataclass(frozen=True)
class _IntervalResult:
    """Store one accepted implicit Newmark interval."""

    state: _StepState
    dissipation_increment: float
    external_work_increment: float
    newton_iterations: int
    armijo_backtracks: int


class NonlinearStepConvergenceError(RuntimeError):
    """Report a failed global Newton step with structured diagnostics."""

    def __init__(self, reason: str, diagnostics: dict[str, object]) -> None:
        """Initialize a global Newton failure."""
        super().__init__(f"Implicit Newmark Newton failed: {reason}.")
        self.reason = reason
        self.diagnostics = diagnostics


class RecoverableNonlinearStepError(NonlinearStepConvergenceError):
    """Report a convergence failure eligible for configured time-step cutback."""


def solve_material_explicit(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
    initial_displacement: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
    dirichlet: DirichletDofSet,
    load_function: LoadFunction,
    kinematics: str,
    update_settings: MaterialUpdateSettings,
    progress_callback: ProgressCallback | None = None,
    residual_corrector: VelocityVerletResidualCorrector | None = None,
    correction_controls: NDArray[np.float64] | None = None,
    record_material_state_history: bool = False,
) -> MaterialDynamicSolution:
    """Solve material-point dynamics by explicit velocity Verlet integration."""
    _require_zero_dirichlet(dirichlet)
    stability = check_explicit_stability(bundle, time_settings)
    controls = _validate_explicit_corrector_configuration(
        residual_corrector,
        correction_controls,
        time_settings,
        stability.stable_time_step,
    )
    start_time = perf_counter()
    dt = time_settings.time_step
    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * dt
    dof_count = bundle.mesh_info.dof_map.size
    mass_lumped = assemble_lumped_mass_vector(bundle)
    free_dofs = _free_dofs(dof_count, dirichlet.dofs)
    state = initialize_bundle_material_state(bundle)
    assembly_cache = build_material_assembly_cache(bundle)
    displacement = np.zeros((times.size, dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)
    displacement[0] = initial_displacement
    velocity[0] = initial_velocity
    displacement[0, dirichlet.dofs] = 0.0
    velocity[0, dirichlet.dofs] = 0.0
    assembly = assemble_bundle_material_response(
        bundle,
        state,
        displacement[0],
        displacement[0],
        dt,
        kinematics,
        update_settings,
        assembly_cache,
        need_tangent=False,
    )
    state = assembly.state
    external = load_function(0.0)
    acceleration[0] = (external - assembly.internal_force) / mass_lumped
    acceleration[0, dirichlet.dofs] = 0.0
    histories = _initialize_histories(
        bundle,
        times.size,
        dof_count,
        assembly,
        state if record_material_state_history else None,
    )
    integrator_diagnostics = (
        _initialize_corrector_histories(time_settings.num_steps, dof_count, free_dofs.size)
        if residual_corrector is not None
        else {}
    )
    _record_step(
        histories,
        0,
        bundle,
        _compute_lumped_kinetic_energy(mass_lumped, velocity[0]),
        external,
        assembly,
        state,
    )

    for step_id in range(time_settings.num_steps):
        next_external = load_function(float(times[step_id + 1]))
        start_acceleration = acceleration[step_id]
        residual_start = np.zeros(dof_count, dtype=np.float64)
        residual_end = np.zeros(dof_count, dtype=np.float64)
        response: VelocityVerletCorrectionResponse | None = None
        if residual_corrector is not None:
            baseline_acceleration = (external - assembly.internal_force) / mass_lumped
            baseline_acceleration[dirichlet.dofs] = 0.0
            request = VelocityVerletCorrectionRequest(
                displacement=displacement[step_id],
                velocity=velocity[step_id],
                baseline_acceleration=baseline_acceleration,
                internal_force_start=assembly.internal_force,
                external_force_start=external,
                external_force_end=next_external,
                free_dofs=free_dofs,
                control_parameters=controls,
                time=float(times[step_id]),
                time_step=dt,
            )
            response = residual_corrector.evaluate(request)
            residual_start_free, residual_end_free = _validate_corrector_response(
                response,
                free_dofs.size,
            )
            residual_start[free_dofs] = residual_start_free
            residual_end[free_dofs] = residual_end_free
            start_acceleration = (external - assembly.internal_force - residual_start) / mass_lumped
            start_acceleration[dirichlet.dofs] = 0.0

        next_displacement = (
            displacement[step_id]
            + dt * velocity[step_id]
            + 0.5 * dt**2 * start_acceleration
        )
        next_displacement[dirichlet.dofs] = 0.0
        next_assembly = assemble_bundle_material_response(
            bundle,
            state,
            next_displacement,
            displacement[step_id],
            dt,
            kinematics,
            update_settings,
            assembly_cache,
            need_tangent=False,
        )
        next_acceleration = (
            next_external - next_assembly.internal_force - residual_end
        ) / mass_lumped
        next_acceleration[dirichlet.dofs] = 0.0
        next_velocity = velocity[step_id] + 0.5 * dt * (
            start_acceleration + next_acceleration
        )
        next_velocity[dirichlet.dofs] = 0.0

        displacement[step_id + 1] = next_displacement
        velocity[step_id + 1] = next_velocity
        acceleration[step_id + 1] = next_acceleration
        histories["dissipated_energy"][step_id + 1] = (
            histories["dissipated_energy"][step_id]
            + dt * integrate_quadrature_density(bundle, next_assembly.dissipation)
        )
        displacement_increment = next_displacement - displacement[step_id]
        histories["external_work"][step_id + 1] = (
            histories["external_work"][step_id]
            + 0.5 * (external + next_external) @ displacement_increment
        )
        if response is not None:
            _record_corrector_step(
                integrator_diagnostics,
                step_id,
                time_settings.num_steps,
                mass_lumped,
                free_dofs,
                displacement[step_id],
                next_displacement,
                velocity[step_id],
                next_velocity,
                start_acceleration,
                next_acceleration,
                external,
                next_external,
                assembly.internal_force,
                next_assembly.internal_force,
                residual_start,
                residual_end,
                response,
                dt,
            )
        state = next_assembly.state
        assembly = next_assembly
        external = next_external
        _record_step(
            histories,
            step_id + 1,
            bundle,
            _compute_lumped_kinetic_energy(mass_lumped, next_velocity),
            external,
            assembly,
            state,
        )
        if residual_corrector is not None:
            histories["energy_residual"][step_id + 1] -= integrator_diagnostics[
                "cumulative_correction_work"
            ][step_id + 1]
        if progress_callback is not None:
            progress_callback(
                SolverProgressEvent(
                    stage="material_dynamics",
                    scheme="explicit",
                    accepted_step=step_id + 1,
                    nominal_total_steps=time_settings.num_steps,
                    physical_time=float(times[step_id + 1]),
                    final_time=float(times[-1]),
                    newton_iterations=0,
                    armijo_backtracks=0,
                    elapsed_seconds=perf_counter() - start_time,
                )
            )

    solve_time = perf_counter() - start_time

    return _build_solution(
        times,
        displacement,
        velocity,
        acceleration,
        histories,
        state,
        solve_time,
        integrator_diagnostics=integrator_diagnostics,
    )


def solve_material_implicit_newmark(
    bundle: PreprocessBundle,
    time_settings: TimeIntegrationSettings,
    initial_displacement: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
    dirichlet: DirichletDofSet,
    load_function: LoadFunction,
    kinematics: str,
    update_settings: MaterialUpdateSettings,
    newton_settings: NonlinearNewtonSettings,
    progress_callback: ProgressCallback | None = None,
) -> MaterialDynamicSolution:
    """Solve material dynamics by implicit Newmark with configured nonlinear controls."""
    _require_zero_dirichlet(dirichlet)
    start_time = perf_counter()
    initial_time_step = time_settings.time_step
    final_time = initial_time_step * time_settings.num_steps
    dof_count = bundle.mesh_info.dof_map.size
    mass = assemble_consistent_mass_matrix(bundle)
    material_state = initialize_bundle_material_state(bundle)
    assembly_cache = build_material_assembly_cache(bundle)
    displacement = initial_displacement.copy()
    velocity = initial_velocity.copy()
    acceleration = np.zeros(dof_count, dtype=np.float64)
    displacement[dirichlet.dofs] = 0.0
    velocity[dirichlet.dofs] = 0.0
    assembly = assemble_bundle_material_response(
        bundle,
        material_state,
        displacement,
        displacement,
        initial_time_step,
        kinematics,
        update_settings,
        assembly_cache,
    )
    material_state = assembly.state
    external = load_function(0.0)
    free_dofs = _free_dofs(dof_count, dirichlet.dofs)
    acceleration[free_dofs] = solve_sparse_direct(
        mass[free_dofs, :][:, free_dofs].tocsr(),
        (external - assembly.internal_force)[free_dofs],
    )
    current = _StepState(
        time=0.0,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        material_state=material_state,
        assembly=assembly,
    )
    states = [current]
    intervals: list[_IntervalResult] = []
    rejected_counts = [0]
    rejected_reasons: list[tuple[str, ...]] = [()]
    pending_reasons: list[str] = []
    current_time_step = initial_time_step
    minimum_time_step = initial_time_step * time_settings.adaptive.minimum_factor
    consecutive_easy_steps = 0

    while current.time < final_time:
        trial_time_step = min(current_time_step, final_time - current.time)
        try:
            interval = _solve_implicit_step(
                bundle,
                mass,
                current,
                trial_time_step,
                dirichlet,
                load_function,
                kinematics,
                update_settings,
                newton_settings,
                assembly_cache,
                beta=time_settings.beta,
                gamma=time_settings.gamma,
            )
        except RecoverableNonlinearStepError as error:
            if not time_settings.adaptive.enabled:
                raise
            pending_reasons.append(error.reason)
            if trial_time_step <= minimum_time_step:
                error.diagnostics["minimum_time_step"] = minimum_time_step
                error.diagnostics["rejected_reasons"] = tuple(pending_reasons)
                raise
            current_time_step = cut_back_time_step(
                trial_time_step,
                initial_time_step,
                time_settings.adaptive,
            )
            consecutive_easy_steps = 0
            continue

        current = interval.state
        states.append(current)
        intervals.append(interval)
        rejected_counts.append(len(pending_reasons))
        rejected_reasons.append(tuple(pending_reasons))
        if progress_callback is not None:
            progress_callback(
                SolverProgressEvent(
                    stage="material_dynamics",
                    scheme="implicit_newmark",
                    accepted_step=len(intervals),
                    nominal_total_steps=time_settings.num_steps,
                    physical_time=current.time,
                    final_time=final_time,
                    newton_iterations=interval.newton_iterations,
                    armijo_backtracks=interval.armijo_backtracks,
                    elapsed_seconds=perf_counter() - start_time,
                )
            )
        easy_step = (
            not pending_reasons
            and interval.armijo_backtracks == 0
            and interval.newton_iterations <= time_settings.adaptive.easy_iteration_limit
        )
        consecutive_easy_steps = consecutive_easy_steps + 1 if easy_step else 0
        pending_reasons = []
        if (
            time_settings.adaptive.enabled
            and consecutive_easy_steps >= time_settings.adaptive.easy_step_count
        ):
            current_time_step = grow_time_step(
                trial_time_step,
                initial_time_step,
                time_settings.adaptive,
            )
            consecutive_easy_steps = 0
        else:
            current_time_step = trial_time_step

    solve_time = perf_counter() - start_time

    return _build_implicit_solution(
        bundle,
        mass,
        load_function,
        states,
        intervals,
        rejected_counts,
        rejected_reasons,
        solve_time,
    )


def _solve_implicit_step(
    bundle: PreprocessBundle,
    mass,
    start: _StepState,
    time_step: float,
    dirichlet: DirichletDofSet,
    load_function: LoadFunction,
    kinematics: str,
    update_settings: MaterialUpdateSettings,
    newton_settings: NonlinearNewtonSettings,
    assembly_cache,
    beta: float,
    gamma: float,
) -> _IntervalResult:
    """Solve one implicit Newmark substep from a committed state."""
    if newton_settings.max_iterations <= 0:
        raise ValueError("Implicit Newmark requires max_iterations > 0.")
    target_time = start.time + time_step
    displacement_predictor = (
        start.displacement
        + time_step * start.velocity
        + time_step**2 * (0.5 - beta) * start.acceleration
    )
    velocity_predictor = start.velocity + time_step * (1.0 - gamma) * start.acceleration
    displacement = displacement_predictor.copy()
    displacement[dirichlet.dofs] = 0.0
    free_dofs = _free_dofs(displacement.size, dirichlet.dofs)
    external = load_function(target_time)
    initial_residual_norm = None
    last_increment_norm = np.inf
    last_residual_norm = np.inf
    last_increment = np.zeros(free_dofs.size, dtype=np.float64)
    total_armijo_backtracks = 0

    for iteration in range(newton_settings.max_iterations + 1):
        acceleration, velocity, assembly, residual = _evaluate_implicit_trial(
            bundle,
            mass,
            start.material_state,
            displacement,
            start.displacement,
            displacement_predictor,
            velocity_predictor,
            time_step,
            external,
            kinematics,
            update_settings,
            assembly_cache,
            beta,
            gamma,
        )
        residual_norm = float(np.linalg.norm(residual[free_dofs]))
        last_residual_norm = residual_norm
        if initial_residual_norm is None:
            initial_residual_norm = residual_norm
        residual_tolerance = max(
            newton_settings.residual_absolute_tolerance,
            newton_settings.residual_relative_tolerance * initial_residual_norm,
        )
        displacement_scale = max(float(np.linalg.norm(displacement[free_dofs])), 1.0)
        increment_tolerance = max(
            newton_settings.increment_absolute_tolerance,
            newton_settings.increment_relative_tolerance * displacement_scale,
        )
        if residual_norm <= residual_tolerance and (
            iteration == 0 or last_increment_norm <= increment_tolerance
        ):
            dissipation_increment = time_step * integrate_quadrature_density(
                bundle,
                assembly.dissipation,
            )
            previous_external = load_function(start.time)
            external_work_increment = 0.5 * (previous_external + external) @ (
                displacement - start.displacement
            )
            accepted = _StepState(
                time=target_time,
                displacement=displacement,
                velocity=velocity,
                acceleration=acceleration,
                material_state=assembly.state,
                assembly=assembly,
            )

            return _IntervalResult(
                state=accepted,
                dissipation_increment=float(dissipation_increment),
                external_work_increment=float(external_work_increment),
                newton_iterations=iteration,
                armijo_backtracks=total_armijo_backtracks,
            )
        if iteration == newton_settings.max_iterations:
            break

        effective_tangent = assembly.tangent + mass / (beta * time_step**2)
        increment = solve_sparse_direct(
            effective_tangent[free_dofs, :][:, free_dofs].tocsr(),
            -residual[free_dofs],
        )
        if not np.all(np.isfinite(increment)):
            raise NonlinearStepConvergenceError(
                "non_finite_increment",
                _build_newton_failure_diagnostics(
                    target_time,
                    time_step,
                    iteration,
                    residual_norm,
                    residual_tolerance,
                    increment,
                    increment_tolerance,
                    displacement,
                    assembly,
                ),
            )
        increment_norm = float(np.linalg.norm(increment))
        if (
            residual_norm <= residual_tolerance
            and increment_norm <= increment_tolerance
        ):
            dissipation_increment = time_step * integrate_quadrature_density(
                bundle,
                assembly.dissipation,
            )
            previous_external = load_function(start.time)
            external_work_increment = 0.5 * (previous_external + external) @ (
                displacement - start.displacement
            )
            accepted = _StepState(
                time=target_time,
                displacement=displacement,
                velocity=velocity,
                acceleration=acceleration,
                material_state=assembly.state,
                assembly=assembly,
            )

            return _IntervalResult(
                state=accepted,
                dissipation_increment=float(dissipation_increment),
                external_work_increment=float(external_work_increment),
                newton_iterations=iteration,
                armijo_backtracks=total_armijo_backtracks,
            )
        accepted_increment = increment
        if newton_settings.armijo.enabled:
            def evaluate_candidate(
                step_length: float,
                base_displacement: NDArray[np.float64] = displacement,
                newton_increment: NDArray[np.float64] = increment,
            ) -> NDArray[np.float64]:
                """Evaluate one global candidate from the committed material state."""
                candidate = base_displacement.copy()
                candidate[free_dofs] += step_length * newton_increment
                _, _, _, candidate_residual = _evaluate_implicit_trial(
                    bundle,
                    mass,
                    start.material_state,
                    candidate,
                    start.displacement,
                    displacement_predictor,
                    velocity_predictor,
                    time_step,
                    external,
                    kinematics,
                    update_settings,
                    assembly_cache,
                    beta,
                    gamma,
                )

                return candidate_residual[free_dofs]

            try:
                line_search = perform_armijo_search(
                    residual[free_dofs],
                    evaluate_candidate,
                    newton_settings.armijo,
                )
            except ArmijoLineSearchError as error:
                diagnostics = _build_newton_failure_diagnostics(
                    target_time,
                    time_step,
                    iteration,
                    residual_norm,
                    residual_tolerance,
                    increment,
                    increment_tolerance,
                    displacement,
                    assembly,
                )
                diagnostics["armijo"] = error.diagnostics
                raise RecoverableNonlinearStepError(
                    "armijo_exhausted",
                    diagnostics,
                ) from error
            displacement[free_dofs] += line_search.step_length * increment
            accepted_increment = line_search.step_length * increment
            total_armijo_backtracks += line_search.backtracks
        else:
            displacement[free_dofs] += increment
        last_increment = accepted_increment
        last_increment_norm = float(np.linalg.norm(accepted_increment))

    raise RecoverableNonlinearStepError(
        "maximum_iterations_exceeded",
        _build_newton_failure_diagnostics(
            target_time,
            time_step,
            newton_settings.max_iterations,
            last_residual_norm,
            residual_tolerance,
            last_increment,
            increment_tolerance,
            displacement,
            assembly,
        ),
    )


def _evaluate_implicit_trial(
    bundle: PreprocessBundle,
    mass: csr_matrix,
    committed_state: MaterialState,
    displacement: NDArray[np.float64],
    previous_displacement: NDArray[np.float64],
    displacement_predictor: NDArray[np.float64],
    velocity_predictor: NDArray[np.float64],
    time_step: float,
    external: NDArray[np.float64],
    kinematics: str,
    update_settings: MaterialUpdateSettings,
    assembly_cache,
    beta: float,
    gamma: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NonlinearSolidAssemblyResult,
    NDArray[np.float64],
]:
    """Evaluate one Newmark residual from the unchanged committed state."""
    acceleration = (displacement - displacement_predictor) / (beta * time_step**2)
    velocity = velocity_predictor + gamma * time_step * acceleration
    assembly = assemble_bundle_material_response(
        bundle,
        committed_state,
        displacement,
        previous_displacement,
        time_step,
        kinematics,
        update_settings,
        assembly_cache,
    )
    residual = mass @ acceleration + assembly.internal_force - external

    return acceleration, velocity, assembly, residual


def _build_newton_failure_diagnostics(
    target_time: float,
    time_step: float,
    iteration: int,
    residual_norm: float,
    residual_tolerance: float,
    increment: NDArray[np.float64],
    increment_tolerance: float,
    displacement: NDArray[np.float64],
    assembly: NonlinearSolidAssemblyResult,
) -> dict[str, object]:
    """Build a serializable global Newton failure snapshot."""
    return {
        "algorithm": "implicit_newmark_full_step_newton",
        "target_time": target_time,
        "time_step": time_step,
        "iteration": iteration,
        "residual_norm": residual_norm,
        "residual_tolerance": residual_tolerance,
        "increment_norm": float(np.linalg.norm(increment)),
        "increment_tolerance": increment_tolerance,
        "increment": increment.tolist(),
        "displacement_norm": float(np.linalg.norm(displacement)),
        "maximum_absolute_stress": float(np.max(np.abs(assembly.stresses))),
        "minimum_tangent_diagonal": float(assembly.tangent.diagonal().min()),
    }


def _initialize_histories(
    bundle: PreprocessBundle,
    step_count: int,
    dof_count: int,
    assembly: NonlinearSolidAssemblyResult,
    material_state: MaterialState | None = None,
) -> dict[str, Any]:
    """Allocate dynamic diagnostic histories."""
    point_count, stress_count = assembly.stresses.shape

    material_diagnostics = {
        name: np.zeros((step_count, point_count), dtype=np.float64)
        for name, values in assembly.diagnostics.items()
        if values.shape == (point_count,)
    }
    material_diagnostics["free_energy_density"] = np.zeros(
        (step_count, point_count), dtype=np.float64
    )
    material_diagnostics["dissipation_density"] = np.zeros(
        (step_count, point_count), dtype=np.float64
    )
    histories = {
        "internal_force": np.zeros((step_count, dof_count)),
        "external_force": np.zeros((step_count, dof_count)),
        "stresses": np.zeros((step_count, point_count, stress_count)),
        "equivalent_plastic_strain": np.zeros((step_count, point_count)),
        "kinetic_energy": np.zeros(step_count),
        "free_energy": np.zeros(step_count),
        "dissipated_energy": np.zeros(step_count),
        "external_work": np.zeros(step_count),
        "energy_residual": np.zeros(step_count),
        "positive_incremental_work": np.zeros(step_count),
        "negative_incremental_work": np.zeros(step_count),
        "unloading_volume_fraction": np.zeros(step_count),
        "viscoplastic_active_volume_fraction": np.zeros(step_count),
        "minimum_yield_activation_margin": np.zeros(step_count),
        "maximum_yield_activation_margin": np.zeros(step_count),
        "newton_iterations": np.zeros(step_count, dtype=np.int64),
        "armijo_backtracks": np.zeros(step_count, dtype=np.int64),
        "material_diagnostics": material_diagnostics,
    }
    if material_state is not None:
        histories["material_state_history"] = {
            name: np.zeros((step_count, *values.shape), dtype=np.float64)
            for name, values in material_state.variables.items()
        }
    return histories


def _record_step(
    histories: dict[str, Any],
    step_id: int,
    bundle: PreprocessBundle,
    kinetic_energy: float,
    external_force: NDArray[np.float64],
    assembly: NonlinearSolidAssemblyResult,
    state: MaterialState,
) -> None:
    """Record one accepted dynamic state."""
    histories["internal_force"][step_id] = assembly.internal_force
    histories["external_force"][step_id] = external_force
    histories["stresses"][step_id] = assembly.stresses
    if "material_state_history" in histories:
        for name, values in histories["material_state_history"].items():
            values[step_id] = state.variables[name]
    for name, values in histories["material_diagnostics"].items():
        if name in assembly.diagnostics:
            values[step_id] = assembly.diagnostics[name]
    histories["material_diagnostics"]["free_energy_density"][step_id] = assembly.free_energy
    histories["material_diagnostics"]["dissipation_density"][step_id] = assembly.dissipation
    if "equivalent_plastic_strain" in assembly.diagnostics:
        histories["equivalent_plastic_strain"][step_id] = assembly.diagnostics[
            "equivalent_plastic_strain"
        ]
    elif "equivalent_plastic_strain" in state.variables:
        histories["equivalent_plastic_strain"][step_id] = state.variables[
            "equivalent_plastic_strain"
        ]
    histories["kinetic_energy"][step_id] = kinetic_energy
    histories["free_energy"][step_id] = integrate_quadrature_density(
        bundle,
        assembly.free_energy,
    )
    point_work = assembly.diagnostics.get("incremental_work_density")
    active = assembly.diagnostics.get("viscoplastic_active")
    activation_margin = assembly.diagnostics.get("yield_activation_margin")
    if point_work is not None and active is not None and activation_margin is not None:
        volume = integrate_quadrature_density(bundle, np.ones(point_work.size))
        histories["positive_incremental_work"][step_id] = integrate_quadrature_density(
            bundle,
            np.maximum(point_work, 0.0),
        )
        histories["negative_incremental_work"][step_id] = integrate_quadrature_density(
            bundle,
            np.minimum(point_work, 0.0),
        )
        histories["unloading_volume_fraction"][step_id] = integrate_quadrature_density(
            bundle,
            (point_work < 0.0).astype(np.float64),
        ) / volume
        histories["viscoplastic_active_volume_fraction"][step_id] = (
            integrate_quadrature_density(bundle, active) / volume
        )
        histories["minimum_yield_activation_margin"][step_id] = float(
            np.min(activation_margin)
        )
        histories["maximum_yield_activation_margin"][step_id] = float(
            np.max(activation_margin)
        )
    initial_energy = histories["kinetic_energy"][0] + histories["free_energy"][0]
    histories["energy_residual"][step_id] = (
        histories["kinetic_energy"][step_id]
        + histories["free_energy"][step_id]
        + histories["dissipated_energy"][step_id]
        - histories["external_work"][step_id]
        - initial_energy
    )


def _compute_lumped_kinetic_energy(
    mass_lumped: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> float:
    """Compute kinetic energy for a lumped-mass explicit system."""
    return 0.5 * float(np.sum(mass_lumped * velocity**2))


def _compute_consistent_kinetic_energy(
    mass: csr_matrix,
    velocity: NDArray[np.float64],
) -> float:
    """Compute kinetic energy for a consistent-mass implicit system."""
    return 0.5 * float(velocity @ (mass @ velocity))


def _build_implicit_solution(
    bundle: PreprocessBundle,
    mass: csr_matrix,
    load_function: LoadFunction,
    states: list[_StepState],
    intervals: list[_IntervalResult],
    rejected_counts: list[int],
    rejected_reasons: list[tuple[str, ...]],
    solve_time: float,
) -> MaterialDynamicSolution:
    """Build variable-step histories from accepted implicit states."""
    times = np.asarray([state.time for state in states], dtype=np.float64)
    displacement = np.stack([state.displacement for state in states])
    velocity = np.stack([state.velocity for state in states])
    acceleration = np.stack([state.acceleration for state in states])
    histories = _initialize_histories(
        bundle,
        len(states),
        displacement.shape[1],
        states[0].assembly,
    )

    for step_id, state in enumerate(states):
        if step_id > 0:
            interval = intervals[step_id - 1]
            histories["dissipated_energy"][step_id] = (
                histories["dissipated_energy"][step_id - 1]
                + interval.dissipation_increment
            )
            histories["external_work"][step_id] = (
                histories["external_work"][step_id - 1]
                + interval.external_work_increment
            )
            histories["newton_iterations"][step_id] = interval.newton_iterations
            histories["armijo_backtracks"][step_id] = interval.armijo_backtracks
        _record_step(
            histories,
            step_id,
            bundle,
            _compute_consistent_kinetic_energy(mass, state.velocity),
            load_function(state.time),
            state.assembly,
            state.material_state,
        )

    accepted_time_steps = np.zeros(times.size, dtype=np.float64)
    accepted_time_steps[1:] = np.diff(times)

    return _build_solution(
        times,
        displacement,
        velocity,
        acceleration,
        histories,
        states[-1].material_state,
        solve_time,
        accepted_time_steps=accepted_time_steps,
        rejected_step_counts=np.asarray(rejected_counts, dtype=np.int64),
        armijo_backtracks=histories["armijo_backtracks"],
        rejected_reasons=tuple(rejected_reasons),
    )


def _build_solution(
    times: NDArray[np.float64],
    displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    acceleration: NDArray[np.float64],
    histories: dict[str, Any],
    final_state: MaterialState,
    solve_time: float,
    accepted_time_steps: NDArray[np.float64] | None = None,
    rejected_step_counts: NDArray[np.int64] | None = None,
    armijo_backtracks: NDArray[np.int64] | None = None,
    rejected_reasons: tuple[tuple[str, ...], ...] | None = None,
    integrator_diagnostics: dict[str, NDArray[np.float64]] | None = None,
) -> MaterialDynamicSolution:
    """Build a nonlinear material-dynamics solution."""
    if accepted_time_steps is None:
        accepted_time_steps = np.zeros(times.size, dtype=np.float64)
        accepted_time_steps[1:] = np.diff(times)
    if rejected_step_counts is None:
        rejected_step_counts = np.zeros(times.size, dtype=np.int64)
    if armijo_backtracks is None:
        armijo_backtracks = np.zeros(times.size, dtype=np.int64)
    if rejected_reasons is None:
        rejected_reasons = tuple(() for _ in range(times.size))
    if integrator_diagnostics is None:
        integrator_diagnostics = {}

    return MaterialDynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
        internal_force=histories["internal_force"],
        external_force=histories["external_force"],
        stresses=histories["stresses"],
        equivalent_plastic_strain=histories["equivalent_plastic_strain"],
        kinetic_energy=histories["kinetic_energy"],
        free_energy=histories["free_energy"],
        dissipated_energy=histories["dissipated_energy"],
        external_work=histories["external_work"],
        energy_residual=histories["energy_residual"],
        positive_incremental_work=histories["positive_incremental_work"],
        negative_incremental_work=histories["negative_incremental_work"],
        unloading_volume_fraction=histories["unloading_volume_fraction"],
        viscoplastic_active_volume_fraction=histories[
            "viscoplastic_active_volume_fraction"
        ],
        minimum_yield_activation_margin=histories[
            "minimum_yield_activation_margin"
        ],
        maximum_yield_activation_margin=histories[
            "maximum_yield_activation_margin"
        ],
        newton_iterations=histories["newton_iterations"],
        accepted_time_steps=accepted_time_steps,
        rejected_step_counts=rejected_step_counts,
        armijo_backtracks=armijo_backtracks,
        rejected_reasons=rejected_reasons,
        material_diagnostics=histories["material_diagnostics"],
        integrator_diagnostics=integrator_diagnostics,
        final_state=final_state,
        solve_time=solve_time,
        material_state_history=histories.get("material_state_history"),
    )


def _validate_explicit_corrector_configuration(
    residual_corrector: VelocityVerletResidualCorrector | None,
    correction_controls: NDArray[np.float64] | None,
    time_settings: TimeIntegrationSettings,
    stable_time_step: float,
) -> NDArray[np.float64]:
    """Validate one explicit corrector selection and return immutable controls."""
    if residual_corrector is None:
        if correction_controls is not None:
            raise ValueError("Correction controls require an enabled residual corrector.")
        return np.empty(0, dtype=np.float64)
    if correction_controls is None:
        raise ValueError("An enabled residual corrector requires correction controls.")
    controls = np.asarray(correction_controls, dtype=np.float64)
    if controls.ndim != 1 or not np.all(np.isfinite(controls)):
        raise ValueError("Correction controls must be one finite one-dimensional vector.")
    metadata = residual_corrector.metadata
    if metadata.schema_version != "1.0":
        raise ValueError(
            f"Unsupported velocity-Verlet corrector schema: {metadata.schema_version}."
        )
    if metadata.output_semantics != "additive_left_residual_force":
        raise ValueError("Velocity-Verlet corrector output semantics are incompatible.")
    if metadata.stage_order != ("start_kick", "end_kick"):
        raise ValueError("Velocity-Verlet corrector stage order is incompatible.")
    if metadata.dtype != "float64" or metadata.device != "cpu":
        raise ValueError("Initial velocity-Verlet deployment requires CPU float64 metadata.")
    if metadata.time_step != time_settings.time_step:
        raise ValueError(
            "Velocity-Verlet corrector time step does not match the solver: "
            f"artifact={metadata.time_step:.16e}, solver={time_settings.time_step:.16e}."
        )
    cfl_ratio = time_settings.time_step / stable_time_step
    if not np.isclose(metadata.cfl_ratio, cfl_ratio, rtol=0.0, atol=1.0e-15):
        raise ValueError(
            "Velocity-Verlet corrector CFL ratio does not match the solver: "
            f"artifact={metadata.cfl_ratio:.16e}, solver={cfl_ratio:.16e}."
        )
    if controls.size != len(metadata.control_parameter_order):
        raise ValueError(
            "Velocity-Verlet control width does not match artifact metadata: "
            f"artifact={len(metadata.control_parameter_order)}, received={controls.size}."
        )

    return controls.copy()


def _validate_corrector_response(
    response: VelocityVerletCorrectionResponse,
    free_dof_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate two free-DOF stage residual vectors without altering them."""
    residual_start = np.asarray(response.residual_force_start_free, dtype=np.float64)
    residual_end = np.asarray(response.residual_force_end_free, dtype=np.float64)
    expected_shape = (free_dof_count,)
    if residual_start.shape != expected_shape or residual_end.shape != expected_shape:
        raise ValueError(
            "Velocity-Verlet corrector returned incompatible residual shapes: "
            f"expected={expected_shape}, start={residual_start.shape}, end={residual_end.shape}."
        )
    if not np.all(np.isfinite(residual_start)) or not np.all(np.isfinite(residual_end)):
        raise FloatingPointError("Velocity-Verlet corrector returned non-finite residual forces.")
    for name, raw_values in response.diagnostics.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"Velocity-Verlet corrector diagnostic is non-finite: {name}."
            )

    return residual_start.copy(), residual_end.copy()


def _initialize_corrector_histories(
    num_steps: int,
    dof_count: int,
    free_dof_count: int,
) -> dict[str, NDArray[np.float64]]:
    """Allocate fixed two-stage integrator diagnostics."""
    return {
        "start_acceleration": np.zeros((num_steps, dof_count), dtype=np.float64),
        "end_acceleration": np.zeros((num_steps, dof_count), dtype=np.float64),
        "residual_force_start_free": np.zeros(
            (num_steps, free_dof_count), dtype=np.float64
        ),
        "residual_force_end_free": np.zeros(
            (num_steps, free_dof_count), dtype=np.float64
        ),
        "residual_force_start_mass_dual_norm": np.zeros(num_steps, dtype=np.float64),
        "residual_force_end_mass_dual_norm": np.zeros(num_steps, dtype=np.float64),
        "correction_work_increment": np.zeros(num_steps, dtype=np.float64),
        "cumulative_correction_work": np.zeros(num_steps + 1, dtype=np.float64),
        "impulse_balance_norm": np.zeros(num_steps, dtype=np.float64),
    }


def _record_corrector_step(
    histories: dict[str, NDArray[np.float64]],
    step_id: int,
    num_steps: int,
    mass_lumped: NDArray[np.float64],
    free_dofs: NDArray[np.int64],
    displacement: NDArray[np.float64],
    next_displacement: NDArray[np.float64],
    velocity: NDArray[np.float64],
    next_velocity: NDArray[np.float64],
    start_acceleration: NDArray[np.float64],
    end_acceleration: NDArray[np.float64],
    external_start: NDArray[np.float64],
    external_end: NDArray[np.float64],
    internal_start: NDArray[np.float64],
    internal_end: NDArray[np.float64],
    residual_start: NDArray[np.float64],
    residual_end: NDArray[np.float64],
    response: VelocityVerletCorrectionResponse,
    time_step: float,
) -> None:
    """Record one accepted two-stage correction and exact impulse balance."""
    histories["start_acceleration"][step_id] = start_acceleration
    histories["end_acceleration"][step_id] = end_acceleration
    histories["residual_force_start_free"][step_id] = residual_start[free_dofs]
    histories["residual_force_end_free"][step_id] = residual_end[free_dofs]
    histories["residual_force_start_mass_dual_norm"][step_id] = np.sqrt(
        np.sum(residual_start[free_dofs] ** 2 / mass_lumped[free_dofs])
    )
    histories["residual_force_end_mass_dual_norm"][step_id] = np.sqrt(
        np.sum(residual_end[free_dofs] ** 2 / mass_lumped[free_dofs])
    )
    displacement_increment = next_displacement - displacement
    correction_work = -0.5 * float((residual_start + residual_end) @ displacement_increment)
    histories["correction_work_increment"][step_id] = correction_work
    histories["cumulative_correction_work"][step_id + 1] = (
        histories["cumulative_correction_work"][step_id] + correction_work
    )
    momentum_increment = mass_lumped * (next_velocity - velocity)
    force_impulse = 0.5 * time_step * (
        external_start
        + external_end
        - internal_start
        - internal_end
        - residual_start
        - residual_end
    )
    histories["impulse_balance_norm"][step_id] = np.linalg.norm(
        (momentum_increment - force_impulse)[free_dofs]
    )
    for name, raw_values in response.diagnostics.items():
        key = f"corrector.{name}"
        values = np.asarray(raw_values, dtype=np.float64)
        if key not in histories:
            histories[key] = np.zeros((num_steps, *values.shape), dtype=np.float64)
        if histories[key].shape[1:] != values.shape:
            raise ValueError(
                "Velocity-Verlet corrector diagnostic shape changed during rollout: "
                f"name={name}, expected={histories[key].shape[1:]}, received={values.shape}."
            )
        histories[key][step_id] = values


def _require_zero_dirichlet(dirichlet: DirichletDofSet) -> None:
    """Require fixed homogeneous constraints for the first material-dynamics solver."""
    if np.any(dirichlet.values != 0.0):
        raise ValueError("Material dynamics currently requires homogeneous Dirichlet values.")


def _free_dofs(dof_count: int, constrained_dofs: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return unconstrained global degrees of freedom."""
    return np.setdiff1d(np.arange(dof_count, dtype=np.int64), constrained_dofs)
