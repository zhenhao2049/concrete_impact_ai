"""Benchmarks for fixed-system two-stage velocity-Verlet INC.

Contents:
    Oracle reconstruction, trained-artifact rollout, physical metrics, and acceptance.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from concrete_impact.experiments.inc_data_generation import (
    build_inc_linear_rod_bundle,
    build_inc_linear_rod_dirichlet,
    build_inc_pressure_load_function,
)
from concrete_impact.nn.config import load_velocity_verlet_inc_data_config
from concrete_impact.nn.datasets.velocity_verlet_inc import (
    VelocityVerletINCPath,
    load_velocity_verlet_inc_dataset,
)
from concrete_impact.nn.deployment_inc import (
    build_velocity_verlet_inc_system_data,
    load_velocity_verlet_inc_corrector,
)
from fem.cases import RunResult
from fem.dynamics.material import solve_material_explicit
from fem.materials.data import MaterialUpdateSettings
from fem.solvers.data import MaterialDynamicSolution, TimeIntegrationSettings
from fem.surrogates.data import (
    VelocityVerletCorrectionRequest,
    VelocityVerletCorrectionResponse,
    VelocityVerletCorrectorMetadata,
)


class _OraclePathCorrector:
    """Replay exact path labels to verify the embedded solver algebra."""

    def __init__(self, path: VelocityVerletINCPath, metadata) -> None:
        """Store one accepted path and its fixed-system metadata."""
        self.name = "velocity_verlet_inc_oracle"
        self.path = path
        self.metadata = VelocityVerletCorrectorMetadata(
            schema_version="1.0",
            model_name=self.name,
            model_family="velocity_verlet_inc_oracle",
            time_step=metadata.time_step,
            cfl_ratio=metadata.cfl_ratio,
            output_semantics="additive_left_residual_force",
            stage_order=("start_kick", "end_kick"),
            dtype="float64",
            device="cpu",
            control_parameter_order=metadata.control_parameter_order,
            mesh_sha256=metadata.mesh_sha256,
            lumped_mass_sha256=metadata.lumped_mass_sha256,
            material_config_sha256=metadata.material_config_sha256,
            boundary_config_sha256=metadata.boundary_config_sha256,
            free_dof_order_sha256=metadata.free_dof_order_sha256,
        )

    def evaluate(
        self,
        request: VelocityVerletCorrectionRequest,
    ) -> VelocityVerletCorrectionResponse:
        """Return exact labels at the request time node."""
        step_id = int(round(request.time / request.time_step))
        if not np.isclose(
            request.time,
            self.path.time[step_id],
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("Oracle INC request time does not match the reference path.")
        return VelocityVerletCorrectionResponse(
            residual_force_start_free=self.path.residual_force_start[step_id],
            residual_force_end_free=self.path.residual_force_end[step_id],
        )


def run_velocity_verlet_inc_linear_rod(config: dict[str, Any]) -> RunResult:
    """Run trained and oracle INC rollouts on one held-out linear-rod path."""
    data_config = load_velocity_verlet_inc_data_config(config["data_config"])
    system, paths = load_velocity_verlet_inc_dataset(config["dataset_path"])
    path = _select_path(paths, str(config["path_id"]))
    control = next(item for item in data_config.controls if item.path_id == path.path_id)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = build_inc_linear_rod_bundle(data_config, output_root)
    dirichlet = build_inc_linear_rod_dirichlet(bundle)
    load_function = build_inc_pressure_load_function(bundle, data_config, control)
    time_settings = _coarse_time_settings(data_config)
    current_system = build_velocity_verlet_inc_system_data(
        bundle,
        time_settings,
        dirichlet,
        data_config.plane_state,
        system.control_parameter_order,
        data_config.model["material"],
    )
    _require_system_identity(system, current_system)
    baseline = _solve_path(bundle, time_settings, dirichlet, load_function, data_config, None, None)
    oracle = _solve_path(
        bundle,
        time_settings,
        dirichlet,
        load_function,
        data_config,
        _OraclePathCorrector(path, system),
        path.control_parameters,
    )
    corrector = load_velocity_verlet_inc_corrector(
        config["artifact"]["model_path"],
        config["artifact"]["metadata_path"],
        current_system,
    )
    corrected = _solve_path(
        bundle,
        time_settings,
        dirichlet,
        load_function,
        data_config,
        corrector,
        path.control_parameters,
    )
    metrics = _compute_benchmark_metrics(
        path,
        system.mass_lumped_free,
        system.free_dofs,
        baseline,
        oracle,
        corrected,
        corrector,
        data_config.arrival_threshold,
    )
    passed = _passes_acceptance(metrics, config["acceptance"])
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    trajectories_path = output_root / "trajectories.npz"
    np.savez_compressed(
        trajectories_path,
        time=path.time,
        reference_displacement=path.displacement,
        reference_velocity=path.velocity,
        baseline_displacement=baseline.displacement[:, system.free_dofs],
        baseline_velocity=baseline.velocity[:, system.free_dofs],
        oracle_displacement=oracle.displacement[:, system.free_dofs],
        oracle_velocity=oracle.velocity[:, system.free_dofs],
        inc_displacement=corrected.displacement[:, system.free_dofs],
        inc_velocity=corrected.velocity[:, system.free_dofs],
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={"metrics": metrics_path, "trajectories": trajectories_path},
        passed=passed,
    )


def _solve_path(
    bundle,
    time_settings,
    dirichlet,
    load_function,
    data_config,
    corrector,
    controls,
) -> MaterialDynamicSolution:
    """Run one coarse path with an explicitly selected corrector."""
    dof_count = bundle.mesh_info.dof_map.size
    return solve_material_explicit(
        bundle,
        time_settings,
        np.zeros(dof_count, dtype=np.float64),
        np.zeros(dof_count, dtype=np.float64),
        dirichlet,
        load_function,
        data_config.plane_state,
        _linear_update_settings(),
        residual_corrector=corrector,
        correction_controls=controls,
    )


def _compute_benchmark_metrics(
    path: VelocityVerletINCPath,
    mass: NDArray[np.float64],
    free_dofs: NDArray[np.int64],
    baseline: MaterialDynamicSolution,
    oracle: MaterialDynamicSolution,
    corrected: MaterialDynamicSolution,
    corrector,
    arrival_threshold: float,
) -> dict[str, float]:
    """Compute reconstruction, trajectory, phase, energy, and impulse metrics."""
    baseline_u = baseline.displacement[:, free_dofs]
    baseline_v = baseline.velocity[:, free_dofs]
    oracle_u = oracle.displacement[:, free_dofs]
    oracle_v = oracle.velocity[:, free_dofs]
    corrected_u = corrected.displacement[:, free_dofs]
    corrected_v = corrected.velocity[:, free_dofs]
    baseline_u_error = _relative_mass_error(baseline_u, path.displacement, mass)
    baseline_v_error = _relative_mass_error(baseline_v, path.velocity, mass)
    inc_u_error = _relative_mass_error(corrected_u, path.displacement, mass)
    inc_v_error = _relative_mass_error(corrected_v, path.velocity, mass)
    oracle_error = max(
        _relative_mass_error(oracle_u, path.displacement, mass),
        _relative_mass_error(oracle_v, path.velocity, mass),
    )
    gauge = mass.size - 1
    threshold = arrival_threshold * float(np.max(np.abs(path.displacement[:, gauge])))
    reference_arrival = _arrival(path.time, path.displacement[:, gauge], threshold)
    baseline_arrival = _arrival(path.time, baseline_u[:, gauge], threshold)
    inc_arrival = _arrival(path.time, corrected_u[:, gauge], threshold)
    baseline_peak = _peak_error(baseline_u[:, gauge], path.displacement[:, gauge])
    inc_peak = _peak_error(corrected_u[:, gauge], path.displacement[:, gauge])
    baseline_phase = _phase_error(baseline_u[:, gauge], path.displacement[:, gauge])
    inc_phase = _phase_error(corrected_u[:, gauge], path.displacement[:, gauge])
    reference_energy = path.mechanical_energy
    baseline_energy = baseline.kinetic_energy + baseline.free_energy
    inc_energy = corrected.kinetic_energy + corrected.free_energy
    energy_scale = float(np.max(np.abs(reference_energy)))
    baseline_energy_error = float(np.max(np.abs(baseline_energy - reference_energy)) / energy_scale)
    inc_energy_error = float(np.max(np.abs(inc_energy - reference_energy)) / energy_scale)
    label_error = _teacher_forced_label_error(path, free_dofs, mass, corrector)
    momentum_scale = float(np.max(np.linalg.norm(path.velocity * mass[None, :], axis=1)))
    impulse_error = float(
        np.max(corrected.integrator_diagnostics["impulse_balance_norm"]) / momentum_scale
    )
    return {
        "oracle_reconstruction_error": oracle_error,
        "label_mass_dual_nrmse": label_error,
        "baseline_displacement_error": baseline_u_error,
        "inc_displacement_error": inc_u_error,
        "displacement_error_ratio": inc_u_error / baseline_u_error,
        "baseline_velocity_error": baseline_v_error,
        "inc_velocity_error": inc_v_error,
        "velocity_error_ratio": inc_v_error / baseline_v_error,
        "baseline_arrival_error": abs(baseline_arrival - reference_arrival),
        "inc_arrival_error": abs(inc_arrival - reference_arrival),
        "baseline_peak_error": baseline_peak,
        "inc_peak_error": inc_peak,
        "baseline_phase_error_steps": baseline_phase,
        "inc_phase_error_steps": inc_phase,
        "baseline_energy_error": baseline_energy_error,
        "inc_energy_error": inc_energy_error,
        "energy_error_ratio": inc_energy_error / baseline_energy_error,
        "normalized_impulse_balance_error": impulse_error,
    }


def _teacher_forced_label_error(path, free_dofs, mass, corrector) -> float:
    """Evaluate held-out stage labels in the mass-dual norm."""
    squared_error = 0.0
    squared_reference = 0.0
    full_size = int(free_dofs.max()) + 1
    for step_id in range(path.time.size - 1):
        fields = {}
        for name, values in (
            ("displacement", path.displacement[step_id]),
            ("velocity", path.velocity[step_id]),
            ("baseline_acceleration", path.baseline_acceleration[step_id]),
            ("internal_force_start", path.internal_force[step_id]),
            ("external_force_start", path.external_force[step_id]),
            ("external_force_end", path.external_force[step_id + 1]),
        ):
            field = np.zeros(full_size, dtype=np.float64)
            field[free_dofs] = values
            fields[name] = field
        response = corrector.evaluate(
            VelocityVerletCorrectionRequest(
                **fields,
                free_dofs=free_dofs,
                control_parameters=path.control_parameters,
                time=float(path.time[step_id]),
                time_step=corrector.metadata.time_step,
            )
        )
        error_start = response.residual_force_start_free - path.residual_force_start[step_id]
        error_end = response.residual_force_end_free - path.residual_force_end[step_id]
        squared_error += float(
            np.sum(error_start**2 / mass) + np.sum(error_end**2 / mass)
        )
        squared_reference += float(
            np.sum(path.residual_force_start[step_id] ** 2 / mass)
            + np.sum(path.residual_force_end[step_id] ** 2 / mass)
        )
    return float(np.sqrt(squared_error / squared_reference))


def _passes_acceptance(metrics: dict[str, float], acceptance: dict[str, Any]) -> bool:
    """Apply every configured correctness threshold without substitutions."""
    return bool(
        metrics["oracle_reconstruction_error"] <= float(acceptance["oracle_error"])
        and metrics["label_mass_dual_nrmse"] <= float(acceptance["label_nrmse"])
        and metrics["displacement_error_ratio"] <= float(acceptance["state_error_ratio"])
        and metrics["velocity_error_ratio"] <= float(acceptance["state_error_ratio"])
        and metrics["inc_arrival_error"] <= metrics["baseline_arrival_error"]
        and metrics["inc_peak_error"] <= metrics["baseline_peak_error"]
        and metrics["inc_phase_error_steps"] <= metrics["baseline_phase_error_steps"]
        and metrics["energy_error_ratio"] <= float(acceptance["energy_error_ratio"])
        and metrics["normalized_impulse_balance_error"]
        <= float(acceptance["impulse_balance_error"])
    )


def _coarse_time_settings(data_config) -> TimeIntegrationSettings:
    """Build the fixed coarse benchmark time integration settings."""
    return TimeIntegrationSettings(
        scheme="velocity_verlet",
        time_step=data_config.time_step,
        num_steps=data_config.num_steps,
        beta=0.0,
        gamma=0.5,
        cfl_safety_factor=data_config.cfl_safety_factor,
    )


def _linear_update_settings() -> MaterialUpdateSettings:
    """Build inactive material iteration settings for linear elasticity."""
    return MaterialUpdateSettings(1, 1.0e-12, 1.0e-12, 1.0e-12)


def _select_path(
    paths: tuple[VelocityVerletINCPath, ...],
    path_id: str,
) -> VelocityVerletINCPath:
    """Select exactly one held-out path by stable identifier."""
    selected = tuple(path for path in paths if path.path_id == path_id)
    if len(selected) != 1:
        raise ValueError(f"INC benchmark path selection is not unique: {path_id}.")
    if selected[0].split != "test":
        raise ValueError("INC benchmark requires a held-out test path.")
    return selected[0]


def _require_system_identity(expected, received) -> None:
    """Require dataset and rebuilt FEM system signatures to match exactly."""
    names = (
        "control_parameter_order",
        "time_step",
        "cfl_ratio",
        "mesh_sha256",
        "lumped_mass_sha256",
        "material_config_sha256",
        "boundary_config_sha256",
        "free_dof_order_sha256",
    )
    differences = {
        name: {"dataset": getattr(expected, name), "current": getattr(received, name)}
        for name in names
        if getattr(expected, name) != getattr(received, name)
    }
    if differences:
        raise ValueError(f"INC benchmark system differs from its dataset: {differences}.")


def _relative_mass_error(values, reference, mass) -> float:
    """Compute one relative trajectory mass norm."""
    numerator = np.sum((values - reference) ** 2 * mass[None, :])
    denominator = np.sum(reference**2 * mass[None, :])
    return float(np.sqrt(numerator / denominator))


def _arrival(times, values, threshold) -> float:
    """Return the first absolute threshold crossing."""
    matches = np.flatnonzero(np.abs(values) >= threshold)
    if matches.size == 0:
        raise ValueError("INC benchmark trajectory never reaches the arrival threshold.")
    return float(times[matches[0]])


def _peak_error(values, reference) -> float:
    """Compute one normalized absolute peak error."""
    reference_peak = float(np.max(np.abs(reference)))
    return abs(float(np.max(np.abs(values))) - reference_peak) / reference_peak


def _phase_error(values, reference) -> float:
    """Compute the absolute integer lag maximizing correlation."""
    correlation = np.correlate(values, reference, mode="full")
    lag = int(np.argmax(correlation) - (reference.size - 1))
    return float(abs(lag))
