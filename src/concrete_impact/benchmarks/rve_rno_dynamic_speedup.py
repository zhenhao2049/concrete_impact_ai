"""Compare direct FE2 and RVE-RNO material updates in one dynamic FEM solve.

Contents:
    Matched explicit dynamics, material-boundary timing, response errors, and figures.
Author:
    Zhen Hao.
Created:
    2026-07-18.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from concrete_impact.benchmarks.dynamic_viscoplastic import (
    _build_left_dirichlet,
    _build_material_update_settings,
    _build_model_def,
    _build_output_def,
    _build_pressure_load_function,
    _build_time_settings,
)
from concrete_impact.benchmarks.rve_heterogeneous import (
    _elastic_material,
    _newton_settings,
    _viscoplastic_material,
)
from concrete_impact.benchmarks.rve_plastic_impact import _build_rve_mesh
from concrete_impact.core.config import load_yaml_config
from concrete_impact.nn.config import RVERNOModelConfig
from concrete_impact.nn.deployment import RVERNOMaterialAdapter, load_rve_rno_state_dict
from fem.cases import RunResult
from fem.dynamics.material import solve_material_explicit
from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)
from fem.preprocess import build_preprocess_data
from fem.preprocess.data import PreprocessBundle
from fem.rve import (
    RVEExecutionSettings,
    RVEHomogenizedMaterial,
    build_cylindrical_two_phase_rve,
)
from fem.solvers.data import MaterialDynamicSolution

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass
class TimedMaterialAdapter:
    """Measure complete calls through one material-point update boundary."""

    material: Any
    update_seconds: list[float] = field(default_factory=list)
    point_counts: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Return the wrapped material name."""
        return str(self.material.name)

    @property
    def density(self) -> float:
        """Return the wrapped material density."""
        return float(self.material.density)

    @property
    def maximum_wave_speed(self) -> float:
        """Return the wrapped material wave-speed bound."""
        return float(self.material.maximum_wave_speed)

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize the wrapped material state."""
        return self.material.initialize_state(n_points)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Time one complete batched material update and retain its response."""
        start = perf_counter()
        response = self.material.update(request, state, requirements)
        self.update_seconds.append(perf_counter() - start)
        self.point_counts.append(int(request.strains.shape[0]))
        return response

    def reset_timings(self) -> None:
        """Clear warm-up or prior-repeat timing observations."""
        self.update_seconds.clear()
        self.point_counts.clear()


def run_rve_rno_dynamic_material_speedup(config: dict[str, Any]) -> RunResult:
    """Run matched FE2 and RNO-FEM dynamics and publish measured speedups."""
    output = _build_output_def(config["output"])
    output.root.mkdir(parents=True, exist_ok=True)
    base_bundle = build_preprocess_data(_build_model_def(config["model"]), output)
    update_settings = _build_material_update_settings(config)
    time_settings = _build_time_settings(config["analysis"]["explicit"])
    dirichlet = _build_left_dirichlet(base_bundle)
    load_function = _build_pressure_load_function(base_bundle, config)
    initial = np.zeros(base_bundle.mesh_info.dof_map.size, dtype=np.float64)
    performance = config["analysis"]["performance"]
    warmup_repeats = int(performance["warmup_repeats"])
    measured_repeats = int(performance["measured_repeats"])
    thread_count = int(performance["thread_count"])
    torch.set_num_threads(thread_count)

    fe2_material = _build_fe2_material(config)
    rno_material = _build_rno_material(config, fe2_material)
    fe2_timer = TimedMaterialAdapter(fe2_material)
    rno_timer = TimedMaterialAdapter(rno_material)
    fe2_bundle = replace(base_bundle, material=fe2_timer)
    rno_bundle = replace(base_bundle, material=rno_timer)

    try:
        with threadpool_limits(limits=thread_count):
            fe2_solution, fe2_timings = _run_repeated_dynamics(
                fe2_bundle,
                fe2_timer,
                time_settings,
                initial,
                dirichlet,
                load_function,
                update_settings,
                warmup_repeats,
                measured_repeats,
            )
            rno_solution, rno_timings = _run_repeated_dynamics(
                rno_bundle,
                rno_timer,
                time_settings,
                initial,
                dirichlet,
                load_function,
                update_settings,
                warmup_repeats,
                measured_repeats,
            )
    finally:
        fe2_material.close()

    response = _compute_response_histories(base_bundle, fe2_solution, rno_solution)
    metrics = _compute_metrics(
        response,
        fe2_timings,
        rno_timings,
        fe2_solution,
        rno_solution,
        config["verification"],
    )
    timing_path = output.root / "timing_observations.json"
    timing_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scope": "matched_dynamic_fem_material_update_boundary",
                "macro_quadrature_points": int(fe2_solution.stresses.shape[1]),
                "material_updates_per_solve": int(time_settings.num_steps + 1),
                "fe2": fe2_timings,
                "rno": rno_timings,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    response_path = _write_response_csv(output.root, response)
    figure_path = _write_response_figure(output.root, response, metrics)
    verification = config["verification"]
    passed = (
        metrics["material_update_speedup"]
        >= float(verification["minimum_material_update_speedup"])
        and metrics["displacement_history_relative_error"]
        <= float(verification["maximum_displacement_error"])
        and metrics["axial_stress_history_relative_error"]
        <= float(verification["maximum_axial_stress_error"])
        and metrics["peak_displacement_relative_error"]
        <= float(verification["maximum_peak_displacement_error"])
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths={
            "timing_observations": timing_path,
            "response_csv": response_path,
            "comparison_figure": figure_path,
        },
        passed=passed,
        metadata={
            "comparison_scope": "same_machine_same_macro_dynamics",
            "fe2_rve_resolution": str(config["rve"]["resolution_label"]),
            "rno_artifact_scope": str(config["artifact"]["scope"]),
            "thread_count": thread_count,
        },
    )


def _build_fe2_material(config: dict[str, Any]) -> RVEHomogenizedMaterial:
    """Build the configured direct two-phase RVE material."""
    reference = load_yaml_config(config["rve"]["reference_config"])
    mesh = _build_rve_mesh(config, reference)
    model = build_cylindrical_two_phase_rve(
        mesh,
        _viscoplastic_material(reference["materials"]["matrix_viscoplastic"]),
        _elastic_material("inclusion", reference["materials"]["inclusion_elastic"]),
        _newton_settings(reference),
        float(reference["verification"]["hill_mandel_power_scale"]),
    )
    execution = config["rve"]["execution"]
    return RVEHomogenizedMaterial(
        model,
        "direct_fe2_speed_reference",
        RVEExecutionSettings(
            backend=cast(
                Literal["serial", "process_pool"], str(execution["backend"])
            ),
            workers=int(execution["workers"]),
            threads_per_worker=int(execution["threads_per_worker"]),
        ),
    )


def _build_rno_material(
    config: dict[str, Any],
    fe2_material: RVEHomogenizedMaterial,
) -> RVERNOMaterialAdapter:
    """Load the fixed RNO artifact with the FE2 mass and CFL properties."""
    artifact = config["artifact"]
    model_config = RVERNOModelConfig(**artifact["model"])
    model = load_rve_rno_state_dict(
        model_config,
        Path(artifact["checkpoint"]),
        Path(artifact["metadata"]),
    )
    return RVERNOMaterialAdapter(
        model,
        "c48_direct_rve_rno_speed_candidate",
        fe2_material.density,
        fe2_material.maximum_wave_speed,
    )


def _run_repeated_dynamics(
    bundle: PreprocessBundle,
    timer: TimedMaterialAdapter,
    time_settings: Any,
    initial: NDArray[np.float64],
    dirichlet: Any,
    load_function: Any,
    update_settings: Any,
    warmup_repeats: int,
    measured_repeats: int,
) -> tuple[MaterialDynamicSolution, dict[str, list[float] | list[int]]]:
    """Warm up and repeat one identical explicit dynamic solve."""
    for _ in range(warmup_repeats):
        solve_material_explicit(
            bundle,
            time_settings,
            initial,
            initial,
            dirichlet,
            load_function,
            "three_dimensional",
            update_settings,
        )
    timer.reset_timings()

    solutions = []
    material_seconds = []
    solve_seconds = []
    material_calls = []
    point_updates = []
    for _ in range(measured_repeats):
        first_call = len(timer.update_seconds)
        first_point = len(timer.point_counts)
        solution = solve_material_explicit(
            bundle,
            time_settings,
            initial,
            initial,
            dirichlet,
            load_function,
            "three_dimensional",
            update_settings,
        )
        repeat_times = timer.update_seconds[first_call:]
        repeat_points = timer.point_counts[first_point:]
        solutions.append(solution)
        material_seconds.append(float(np.sum(repeat_times)))
        solve_seconds.append(float(solution.solve_time))
        material_calls.append(len(repeat_times))
        point_updates.append(int(np.sum(repeat_points)))
    return solutions[-1], {
        "material_update_seconds": material_seconds,
        "solve_seconds": solve_seconds,
        "material_update_calls": material_calls,
        "material_point_updates": point_updates,
    }


def _compute_response_histories(
    bundle: PreprocessBundle,
    fe2: MaterialDynamicSolution,
    rno: MaterialDynamicSolution,
) -> dict[str, NDArray[np.float64]]:
    """Extract matched right-face displacement and mean stress histories."""
    if not np.array_equal(fe2.times, rno.times):
        raise ValueError("FE2 and RNO dynamic time grids differ.")
    right_nodes = bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = bundle.mesh_info.dof_map[right_nodes, 0]
    return {
        "time": fe2.times,
        "fe2_right_displacement": np.mean(fe2.displacement[:, right_dofs], axis=1),
        "rno_right_displacement": np.mean(rno.displacement[:, right_dofs], axis=1),
        "fe2_mean_stress_xx": np.mean(fe2.stresses[:, :, 0], axis=1),
        "rno_mean_stress_xx": np.mean(rno.stresses[:, :, 0], axis=1),
    }


def _compute_metrics(
    response: dict[str, NDArray[np.float64]],
    fe2_timings: dict[str, list[float] | list[int]],
    rno_timings: dict[str, list[float] | list[int]],
    fe2: MaterialDynamicSolution,
    rno: MaterialDynamicSolution,
    verification: dict[str, Any],
) -> dict[str, float]:
    """Compute response errors and robust measured timing ratios."""
    fe2_displacement = response["fe2_right_displacement"]
    rno_displacement = response["rno_right_displacement"]
    fe2_axial_stress = response["fe2_mean_stress_xx"]
    rno_axial_stress = response["rno_mean_stress_xx"]
    fe2_stress = fe2.stresses
    rno_stress = rno.stresses
    displacement_norm = float(np.linalg.norm(fe2_displacement))
    axial_stress_norm = float(np.linalg.norm(fe2_axial_stress))
    stress_norm = float(np.linalg.norm(fe2_stress))
    peak_displacement = float(np.max(np.abs(fe2_displacement)))
    peak_axial_stress = float(np.max(np.abs(fe2_axial_stress)))
    if (
        displacement_norm == 0.0
        or axial_stress_norm == 0.0
        or stress_norm == 0.0
        or peak_displacement == 0.0
        or peak_axial_stress == 0.0
    ):
        raise ValueError("Dynamic speed comparison requires nonzero FE2 response scales.")
    fe2_material = np.asarray(fe2_timings["material_update_seconds"], dtype=np.float64)
    rno_material = np.asarray(rno_timings["material_update_seconds"], dtype=np.float64)
    fe2_solve = np.asarray(fe2_timings["solve_seconds"], dtype=np.float64)
    rno_solve = np.asarray(rno_timings["solve_seconds"], dtype=np.float64)
    point_updates = np.asarray(fe2_timings["material_point_updates"], dtype=np.float64)
    rno_point_updates = np.asarray(
        rno_timings["material_point_updates"], dtype=np.float64
    )
    if not np.array_equal(point_updates, rno_point_updates):
        raise ValueError("FE2 and RNO material-point update counts differ.")
    fe2_material_median = float(np.median(fe2_material))
    rno_material_median = float(np.median(rno_material))
    fe2_solve_median = float(np.median(fe2_solve))
    rno_solve_median = float(np.median(rno_solve))
    updates_per_solve = float(np.median(point_updates))
    return {
        "material_update_speedup": fe2_material_median / rno_material_median,
        "end_to_end_solve_speedup": fe2_solve_median / rno_solve_median,
        "fe2_material_update_seconds_median": fe2_material_median,
        "rno_material_update_seconds_median": rno_material_median,
        "fe2_solve_seconds_median": fe2_solve_median,
        "rno_solve_seconds_median": rno_solve_median,
        "fe2_seconds_per_material_point_update": fe2_material_median
        / updates_per_solve,
        "rno_seconds_per_material_point_update": rno_material_median
        / updates_per_solve,
        "material_point_updates_per_solve": updates_per_solve,
        "fe2_material_time_fraction": fe2_material_median / fe2_solve_median,
        "rno_material_time_fraction": rno_material_median / rno_solve_median,
        "displacement_history_relative_error": float(
            np.linalg.norm(rno_displacement - fe2_displacement) / displacement_norm
        ),
        "axial_stress_history_relative_error": float(
            np.linalg.norm(rno_axial_stress - fe2_axial_stress) / axial_stress_norm
        ),
        "full_stress_tensor_relative_error": float(
            np.linalg.norm(rno_stress - fe2_stress) / stress_norm
        ),
        "peak_displacement_relative_error": abs(
            float(np.max(np.abs(rno_displacement))) - peak_displacement
        )
        / peak_displacement,
        "peak_axial_stress_relative_error": abs(
            float(np.max(np.abs(rno_axial_stress))) - peak_axial_stress
        )
        / peak_axial_stress,
        "minimum_material_update_speedup": float(
            verification["minimum_material_update_speedup"]
        ),
    }


def _write_response_csv(
    output: Path,
    response: dict[str, NDArray[np.float64]],
) -> Path:
    """Write matched response histories used by the report figure."""
    path = output / "response.csv"
    names = tuple(response)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(response[name] for name in names), strict=True))
    return path


def _write_response_figure(
    output: Path,
    response: dict[str, NDArray[np.float64]],
    metrics: dict[str, float],
) -> Path:
    """Render response agreement and measured timing bars."""
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    axes[0].plot(
        response["time"], response["fe2_right_displacement"], label="Direct FE2"
    )
    axes[0].plot(
        response["time"],
        response["rno_right_displacement"],
        label="RNO-FEM",
        linestyle="--",
    )
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Right-face displacement")
    axes[0].set_title("Dynamic response")
    axes[0].legend()
    axes[1].plot(response["time"], response["fe2_mean_stress_xx"])
    axes[1].plot(
        response["time"], response["rno_mean_stress_xx"], linestyle="--"
    )
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Mean stress xx")
    axes[1].set_title("Constitutive response")
    timing = np.asarray(
        [
            [
                metrics["fe2_material_update_seconds_median"],
                metrics["rno_material_update_seconds_median"],
            ],
            [metrics["fe2_solve_seconds_median"], metrics["rno_solve_seconds_median"]],
        ]
    )
    x = np.arange(2)
    width = 0.34
    axes[2].bar(x - width / 2, timing[:, 0], width, label="Direct FE2")
    axes[2].bar(x + width / 2, timing[:, 1], width, label="RNO-FEM")
    axes[2].set_xticks(x, ("Material update", "Full solve"))
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Median wall time / s")
    axes[2].set_title(
        f"Material speedup = {metrics['material_update_speedup']:.1f}x"
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output / "rve-rno-dynamic-material-speedup.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path
