"""Macro material adapter assigning one independent RVE to each integration point.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)
from fem.rve.data import (
    RVEExecutionSettings,
    RVEOperatorCache,
    RVERequest,
    RVEState,
    StructuredHex8RVE,
)
from fem.rve.solver import (
    build_rve_operator_cache,
    initialize_rve_state,
    solve_rve_microequilibrium,
)


@dataclass(frozen=True)
class RVEHomogenizedMaterial:
    """Expose a fixed quasi-static RVE through the material-point protocol."""

    model: StructuredHex8RVE
    name: str
    execution_settings: RVEExecutionSettings = field(default_factory=RVEExecutionSettings)
    operator_cache: RVEOperatorCache = field(init=False, repr=False)
    _executor: ProcessPoolExecutor | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Build the immutable RVE operator cache once per material adapter."""
        object.__setattr__(self, "operator_cache", build_rve_operator_cache(self.model))
        if self.execution_settings.backend == "process_pool":
            context = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=self.execution_settings.workers,
                mp_context=context,
                initializer=_initialize_rve_worker,
                initargs=(
                    self.model,
                    self.operator_cache,
                    self.execution_settings.threads_per_worker,
                ),
            )
            object.__setattr__(self, "_executor", executor)

    @property
    def density(self) -> float:
        """Return the RVE volume-average density."""
        return float(self.model.material.density)

    @property
    def maximum_wave_speed(self) -> float:
        """Return the conservative wave-speed bound supplied by the RVE material."""
        return float(self.model.material.maximum_wave_speed)

    @property
    def young_modulus(self) -> float:
        """Return the homogeneous phase modulus used by the explicit CFL check."""
        return float(self.model.material.young_modulus)

    @property
    def poisson_ratio(self) -> float:
        """Return the homogeneous phase Poisson ratio used by the CFL check."""
        return float(self.model.material.poisson_ratio)

    def initialize_state(self, n_points: int) -> MaterialState:
        """Allocate one independent committed RVE state per macro point."""
        prototype = initialize_rve_state(self.model)
        variables = {
            "rve_macro_strain": np.repeat(
                prototype.macro_strain[None, :],
                n_points,
                axis=0,
            ),
            "rve_fluctuation_dofs": np.repeat(
                prototype.fluctuation_dofs[None, :],
                n_points,
                axis=0,
            ),
        }
        for name, values in prototype.material_state.variables.items():
            variables[f"rve_material__{name}"] = np.repeat(
                values[None, ...],
                n_points,
                axis=0,
            )

        return MaterialState(variables)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Solve independent RVEs for every macro integration-point strain."""
        point_count = request.strains.shape[0]
        stresses = np.zeros((point_count, 6), dtype=np.float64)
        tangents = np.zeros((point_count, 6, 6)) if requirements.tangent else None
        free_energy = np.zeros(point_count) if requirements.free_energy else None
        dissipation = np.zeros(point_count) if requirements.dissipation else None
        next_variables = {name: values.copy() for name, values in state.variables.items()}
        diagnostic_rows: list[dict[str, float]] = []

        committed_states = [
            decode_rve_point_state(state, point_id) for point_id in range(point_count)
        ]
        rve_requests = [
            RVERequest(
                macro_strain=request.strains[point_id],
                time_step=request.time_step,
                material_update_settings=request.update_settings,
                requirements=requirements,
            )
            for point_id in range(point_count)
        ]
        if self.execution_settings.backend == "serial":
            responses = [
                solve_rve_microequilibrium(
                    self.model,
                    rve_request,
                    committed,
                    self.operator_cache,
                )
                for rve_request, committed in zip(rve_requests, committed_states, strict=True)
            ]
        else:
            if self._executor is None:
                raise RuntimeError("Process-pool RVE material has no active executor.")
            responses = list(
                self._executor.map(
                    _solve_rve_worker,
                    zip(rve_requests, committed_states, strict=True),
                )
            )

        for point_id, response in enumerate(responses):
            stresses[point_id] = response.macro_stress
            if tangents is not None:
                if response.effective_tangent is None:
                    raise ValueError("Macro implicit RVE update requires an effective tangent.")
                tangents[point_id] = response.effective_tangent
            if free_energy is not None:
                free_energy[point_id] = response.free_energy_density
            if dissipation is not None:
                dissipation[point_id] = response.dissipation_density
            _encode_rve_point_state(next_variables, point_id, response.state)
            diagnostic_rows.append(_macro_diagnostics(response, self.model, self.operator_cache))

        diagnostic_names = sorted(set().union(*(row.keys() for row in diagnostic_rows)))
        diagnostics = {
            name: np.asarray([row[name] for row in diagnostic_rows], dtype=np.float64)
            for name in diagnostic_names
        }

        return MaterialPointResponse(
            stresses=stresses,
            state=MaterialState(next_variables),
            tangents=tangents,
            free_energy=free_energy,
            dissipation=dissipation,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        """Close the explicitly selected persistent process backend."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)


def decode_rve_point_state(state: MaterialState, point_id: int) -> RVEState:
    """Decode one macro point's RVE state from a fixed-topology batch."""
    material_variables = {
        name.removeprefix("rve_material__"): values[point_id].copy()
        for name, values in state.variables.items()
        if name.startswith("rve_material__")
    }
    return RVEState(
        material_state=MaterialState(material_variables),
        macro_strain=state.variables["rve_macro_strain"][point_id].copy(),
        fluctuation_dofs=state.variables["rve_fluctuation_dofs"][point_id].copy(),
    )


def _encode_rve_point_state(
    variables: dict[str, NDArray[np.float64]],
    point_id: int,
    state: RVEState,
) -> None:
    """Write one accepted candidate into a new batch-state allocation."""
    variables["rve_macro_strain"][point_id] = state.macro_strain
    variables["rve_fluctuation_dofs"][point_id] = state.fluctuation_dofs
    for name, values in state.material_state.variables.items():
        variables[f"rve_material__{name}"][point_id] = values


def _macro_diagnostics(
    response,
    model: StructuredHex8RVE,
    operator_cache: RVEOperatorCache,
) -> dict[str, float]:
    """Reduce RVE diagnostics to material-point-sized macro arrays."""
    diagnostics = response.diagnostics
    result = {
        "incremental_work_density": float(
            diagnostics["positive_work_density"] + diagnostics["negative_work_density"]
        ),
        "viscoplastic_active": float(diagnostics["viscoplastic_active_volume_fraction"]),
        "yield_activation_margin": float(diagnostics["minimum_yield_activation_margin"]),
        "micro_newton_iterations": float(diagnostics["newton_iterations"]),
        "micro_armijo_backtracks": float(diagnostics["armijo_backtracks"]),
        "hill_mandel_error": float(diagnostics["hill_mandel_error"]),
    }
    q_key = "equivalent_plastic_strain"
    if q_key in response.state.material_state.variables:
        weights = operator_cache.quadrature_weights
        q_values = response.state.material_state.variables[q_key]
        result[q_key] = float(np.sum(q_values * weights) / np.sum(weights))
    for phase_name in model.phase_names:
        phase_id = model.phase_names.index(phase_name)
        phase_mask = operator_cache.point_phase_ids == phase_id
        phase_weights = operator_cache.quadrature_weights[phase_mask]
        total_weights = operator_cache.quadrature_weights
        result[f"phase_{phase_name}__volume_fraction"] = float(
            np.sum(phase_weights) / np.sum(total_weights)
        )
        phase_stress = np.sum(
            response.micro_stresses[phase_mask] * phase_weights[:, None], axis=0
        ) / np.sum(phase_weights)
        for component, value in zip(
            ("xx", "yy", "zz", "yz", "xz", "xy"), phase_stress, strict=True
        ):
            result[f"phase_{phase_name}__stress_{component}"] = float(value)
        phase_q_key = f"phase_{phase_name}__equivalent_plastic_strain"
        if phase_q_key not in response.state.material_state.variables:
            continue
        q_values = response.state.material_state.variables[phase_q_key]
        result[phase_q_key] = float(np.sum(q_values * phase_weights) / np.sum(phase_weights))
        result[f"phase_{phase_name}__maximum_equivalent_plastic_strain"] = float(np.max(q_values))

    return result


_WORKER_MODEL: StructuredHex8RVE | None = None
_WORKER_CACHE: RVEOperatorCache | None = None
_WORKER_THREAD_LIMITER: Any = None


def _initialize_rve_worker(
    model: StructuredHex8RVE,
    operator_cache: RVEOperatorCache,
    threads_per_worker: int,
) -> None:
    """Initialize one persistent process with immutable RVE operators."""
    from threadpoolctl import threadpool_limits

    global _WORKER_MODEL, _WORKER_CACHE, _WORKER_THREAD_LIMITER
    _WORKER_MODEL = model
    _WORKER_CACHE = operator_cache
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=threads_per_worker)


def _solve_rve_worker(payload: tuple[RVERequest, RVEState]):
    """Solve one macro integration-point RVE in its persistent worker."""
    if _WORKER_MODEL is None or _WORKER_CACHE is None:
        raise RuntimeError("RVE process worker was not initialized.")
    request, state = payload
    return solve_rve_microequilibrium(_WORKER_MODEL, request, state, _WORKER_CACHE)
