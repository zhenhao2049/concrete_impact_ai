"""Newmark time-integration solvers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

from fem.assembly.boundary import apply_dirichlet_to_linear_system, impose_values_on_vector
from fem.assembly.elasticity import (
    assemble_consistent_mass_matrix,
    assemble_lumped_mass_vector,
    assemble_stiffness_matrix,
)
from fem.preprocess.data import PreprocessBundle
from fem.solvers.data import (
    DirichletDofSet,
    DynamicSolution,
    FEMRunDef,
    TimeIntegrationSettings,
)
from fem.solvers.linear import solve_sparse_direct


def solve_explicit_central_difference(
    bundle: PreprocessBundle,
    run_def: FEMRunDef,
) -> DynamicSolution:
    """Solve linear dynamics with the explicit central-difference method."""
    stiffness = assemble_stiffness_matrix(bundle, run_def.plane_state)
    mass_lumped = assemble_lumped_mass_vector(bundle)
    time_settings = run_def.time
    time_step = time_settings.time_step
    dof_count = bundle.mesh_info.dof_map.size

    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step
    displacement = np.zeros((time_settings.num_steps + 1, dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)

    displacement[0, :] = impose_values_on_vector(
        run_def.initial_displacement,
        run_def.dirichlet.dofs,
        run_def.dirichlet.values,
    )
    velocity[0, :] = impose_values_on_vector(
        run_def.initial_velocity,
        run_def.dirichlet.dofs,
        np.zeros(run_def.dirichlet.dofs.shape[0], dtype=np.float64),
    )
    acceleration[0, :] = _compute_lumped_acceleration(
        stiffness,
        mass_lumped,
        displacement[0, :],
        run_def.external_force,
        run_def.dirichlet.dofs,
    )
    previous_displacement = (
        displacement[0, :]
        - time_step * velocity[0, :]
        + 0.5 * time_step**2 * acceleration[0, :]
    )

    for step_id in range(time_settings.num_steps):
        current_displacement = displacement[step_id, :]
        current_acceleration = _compute_lumped_acceleration(
            stiffness,
            mass_lumped,
            current_displacement,
            run_def.external_force,
            run_def.dirichlet.dofs,
        )
        next_displacement = (
            2.0 * current_displacement
            - previous_displacement
            + time_step**2 * current_acceleration
        )
        next_displacement = impose_values_on_vector(
            next_displacement,
            run_def.dirichlet.dofs,
            run_def.dirichlet.values,
        )

        displacement[step_id + 1, :] = next_displacement
        velocity[step_id + 1, :] = (next_displacement - previous_displacement) / (
            2.0 * time_step
        )
        acceleration[step_id + 1, :] = current_acceleration
        previous_displacement = current_displacement

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def solve_implicit_newmark(
    bundle: PreprocessBundle,
    run_def: FEMRunDef,
) -> DynamicSolution:
    """Solve linear dynamics with the implicit Newmark method."""
    stiffness = assemble_stiffness_matrix(bundle, run_def.plane_state)
    mass = assemble_consistent_mass_matrix(bundle)
    time_settings = run_def.time
    time_step = time_settings.time_step
    beta = time_settings.beta
    gamma = time_settings.gamma
    dof_count = bundle.mesh_info.dof_map.size

    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step
    displacement = np.zeros((time_settings.num_steps + 1, dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)

    displacement[0, :] = impose_values_on_vector(
        run_def.initial_displacement,
        run_def.dirichlet.dofs,
        run_def.dirichlet.values,
    )
    velocity[0, :] = impose_values_on_vector(
        run_def.initial_velocity,
        run_def.dirichlet.dofs,
        np.zeros(run_def.dirichlet.dofs.shape[0], dtype=np.float64),
    )
    acceleration[0, :] = _compute_consistent_acceleration(
        mass,
        stiffness,
        displacement[0, :],
        run_def.external_force,
        run_def.dirichlet.dofs,
    )

    a0 = 1.0 / (beta * time_step**2)
    a2 = 1.0 / (beta * time_step)
    a3 = 1.0 / (2.0 * beta) - 1.0
    effective_stiffness = stiffness + a0 * mass
    constrained_matrix, _ = apply_dirichlet_to_linear_system(
        effective_stiffness,
        np.zeros(dof_count, dtype=np.float64),
        run_def.dirichlet.dofs,
        run_def.dirichlet.values,
    )
    solve_effective_system = factorized(constrained_matrix.tocsc())

    for step_id in range(time_settings.num_steps):
        rhs = run_def.external_force + mass @ (
            a0 * displacement[step_id, :]
            + a2 * velocity[step_id, :]
            + a3 * acceleration[step_id, :]
        )
        constrained_rhs = _apply_dirichlet_to_right_hand_side(
            effective_stiffness,
            np.asarray(rhs, dtype=np.float64),
            run_def.dirichlet.dofs,
            run_def.dirichlet.values,
        )
        displacement[step_id + 1, :] = np.asarray(
            solve_effective_system(constrained_rhs),
            dtype=np.float64,
        )
        acceleration[step_id + 1, :] = (
            a0 * (displacement[step_id + 1, :] - displacement[step_id, :])
            - a2 * velocity[step_id, :]
            - a3 * acceleration[step_id, :]
        )
        velocity[step_id + 1, :] = velocity[step_id, :] + time_step * (
            (1.0 - gamma) * acceleration[step_id, :] + gamma * acceleration[step_id + 1, :]
        )

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def solve_implicit_newmark_matrices(
    stiffness: csr_matrix,
    mass: csr_matrix,
    dirichlet: DirichletDofSet,
    load_function,
    time_settings: TimeIntegrationSettings,
    initial_displacement: NDArray[np.float64],
    initial_velocity: NDArray[np.float64],
) -> DynamicSolution:
    """Solve a linear second-order system with implicit Newmark."""
    time_step = time_settings.time_step
    beta = time_settings.beta
    gamma = time_settings.gamma
    dof_count = initial_displacement.shape[0]
    times = np.arange(time_settings.num_steps + 1, dtype=np.float64) * time_step

    displacement = np.zeros((time_settings.num_steps + 1, dof_count), dtype=np.float64)
    velocity = np.zeros_like(displacement)
    acceleration = np.zeros_like(displacement)

    displacement[0, :] = impose_values_on_vector(
        initial_displacement,
        dirichlet.dofs,
        dirichlet.values,
    )
    velocity[0, :] = impose_values_on_vector(
        initial_velocity,
        dirichlet.dofs,
        np.zeros(dirichlet.dofs.shape[0], dtype=np.float64),
    )
    acceleration[0, :] = _compute_consistent_acceleration(
        mass,
        stiffness,
        displacement[0, :],
        load_function(0.0),
        dirichlet.dofs,
    )

    a0 = 1.0 / (beta * time_step**2)
    a2 = 1.0 / (beta * time_step)
    a3 = 1.0 / (2.0 * beta) - 1.0
    effective_stiffness = stiffness + a0 * mass
    constrained_matrix, _ = apply_dirichlet_to_linear_system(
        effective_stiffness,
        np.zeros(dof_count, dtype=np.float64),
        dirichlet.dofs,
        dirichlet.values,
    )
    solve_effective_system = factorized(constrained_matrix.tocsc())

    for step_id in range(time_settings.num_steps):
        rhs = load_function(float(times[step_id + 1])) + mass @ (
            a0 * displacement[step_id, :]
            + a2 * velocity[step_id, :]
            + a3 * acceleration[step_id, :]
        )
        constrained_rhs = _apply_dirichlet_to_right_hand_side(
            effective_stiffness,
            np.asarray(rhs, dtype=np.float64),
            dirichlet.dofs,
            dirichlet.values,
        )
        displacement[step_id + 1, :] = np.asarray(
            solve_effective_system(constrained_rhs),
            dtype=np.float64,
        )
        acceleration[step_id + 1, :] = (
            a0 * (displacement[step_id + 1, :] - displacement[step_id, :])
            - a2 * velocity[step_id, :]
            - a3 * acceleration[step_id, :]
        )
        velocity[step_id + 1, :] = velocity[step_id, :] + time_step * (
            (1.0 - gamma) * acceleration[step_id, :] + gamma * acceleration[step_id + 1, :]
        )

    return DynamicSolution(
        times=times,
        displacement=displacement,
        velocity=velocity,
        acceleration=acceleration,
    )


def _compute_lumped_acceleration(
    stiffness,
    mass_lumped: NDArray[np.float64],
    displacement: NDArray[np.float64],
    external_force: NDArray[np.float64],
    constrained_dofs: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute acceleration from a lumped mass vector."""
    acceleration = (external_force - stiffness @ displacement) / mass_lumped
    acceleration[constrained_dofs] = 0.0

    return np.asarray(acceleration, dtype=np.float64)


def _apply_dirichlet_to_right_hand_side(
    matrix: csr_matrix,
    right_hand_side: NDArray[np.float64],
    dofs: NDArray[np.int64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply prescribed DOF values to a right-hand side only."""
    modified_rhs = right_hand_side - matrix[:, dofs] @ values
    modified_rhs[dofs] = values

    return np.asarray(modified_rhs, dtype=np.float64)


def _compute_consistent_acceleration(
    mass,
    stiffness,
    displacement: NDArray[np.float64],
    external_force: NDArray[np.float64],
    constrained_dofs: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute acceleration from a consistent mass matrix."""
    rhs = external_force - stiffness @ displacement
    values = np.zeros(constrained_dofs.shape[0], dtype=np.float64)
    constrained_mass, constrained_rhs = apply_dirichlet_to_linear_system(
        mass,
        np.asarray(rhs, dtype=np.float64),
        constrained_dofs,
        values,
    )

    return solve_sparse_direct(constrained_mass, constrained_rhs)
