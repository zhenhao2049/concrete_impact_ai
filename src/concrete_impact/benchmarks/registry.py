"""Benchmark registry.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

import copy
import math
import platform
import subprocess
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from concrete_impact.benchmarks.beam_response import (
    run_dynamic_tension_benchmark,
    run_elastic_tension_benchmark,
    run_wave_pulse_benchmark,
)
from concrete_impact.benchmarks.dynamic_pfczm import run_dynamic_pfczm
from concrete_impact.benchmarks.dynamic_viscoplastic import (
    run_dynamic_j2_viscoplastic_benchmark,
)
from concrete_impact.benchmarks.inc_dynamics import run_velocity_verlet_inc_linear_rod
from concrete_impact.benchmarks.plasticity import run_plasticity_benchmark
from concrete_impact.benchmarks.quasi_static_pfczm import run_quasi_static_pfczm
from concrete_impact.benchmarks.report_fe2_plastic_impact import (
    run_report_fe2_plastic_impact,
)
from concrete_impact.benchmarks.rve import run_single_phase_rve_benchmark
from concrete_impact.benchmarks.rve_dynamics import run_single_phase_rve_dynamic_embedding
from concrete_impact.benchmarks.rve_fe2_consistency import (
    run_fe2_path_consistency_benchmark,
)
from concrete_impact.benchmarks.rve_heterogeneous import run_cylindrical_inclusion_rve_benchmark
from concrete_impact.benchmarks.rve_mesh_convergence import (
    run_rve_mesh_convergence_benchmark,
)
from concrete_impact.benchmarks.rve_multiphase_dynamics import (
    run_multiphase_rve_dynamic_embedding,
)
from concrete_impact.benchmarks.rve_plastic_impact import run_multiphase_rve_plastic_impact
from concrete_impact.benchmarks.rve_rno_dynamic_speedup import (
    run_rve_rno_dynamic_material_speedup,
)
from concrete_impact.benchmarks.static_viscoplastic import run_static_viscoplastic_benchmark
from concrete_impact.core.paths import project_root
from fem.cases import RunResult

BenchmarkRunner = Callable[[dict[str, Any]], RunResult]

BENCHMARKS: dict[str, BenchmarkRunner] = {
    "elastic_tension_2d_displacement": run_elastic_tension_benchmark,
    "elastic_tension_2d_force": run_elastic_tension_benchmark,
    "elastic_tension_3d_displacement": run_elastic_tension_benchmark,
    "elastic_tension_3d_force": run_elastic_tension_benchmark,
    "dynamic_tension_2d_explicit": run_dynamic_tension_benchmark,
    "dynamic_tension_2d_implicit": run_dynamic_tension_benchmark,
    "dynamic_tension_3d_explicit": run_dynamic_tension_benchmark,
    "dynamic_tension_3d_implicit": run_dynamic_tension_benchmark,
    "wave_pulse_2d_explicit": run_wave_pulse_benchmark,
    "wave_pulse_3d_explicit": run_wave_pulse_benchmark,
    "plastic_j2_pure_shear": run_plasticity_benchmark,
    "plastic_dp_pure_shear": run_plasticity_benchmark,
    "plastic_dp_cap_hydrostatic": run_plasticity_benchmark,
    "plastic_dp_cap_confined_triaxial": run_plasticity_benchmark,
    "plastic_j2_viscoplastic_material_point": run_plasticity_benchmark,
    "static_j2_viscoplastic_cube_3d": run_static_viscoplastic_benchmark,
    "dynamic_j2_viscoplastic_pressure_pulse_3d": run_dynamic_j2_viscoplastic_benchmark,
    "dynamic_j2_viscoplastic_performance_3d": run_dynamic_j2_viscoplastic_benchmark,
    "single_phase_j2_viscoplastic_rve": run_single_phase_rve_benchmark,
    "single_phase_rve_dynamic_embedding": run_single_phase_rve_dynamic_embedding,
    "cylindrical_inclusion_rve": run_cylindrical_inclusion_rve_benchmark,
    "multiphase_rve_dynamic_embedding": run_multiphase_rve_dynamic_embedding,
    "multiphase_rve_plastic_impact": run_multiphase_rve_plastic_impact,
    "report_fe2_plastic_impact_c48": run_report_fe2_plastic_impact,
    "rve_rno_dynamic_material_speedup": run_rve_rno_dynamic_material_speedup,
    "fe2_path_consistency": run_fe2_path_consistency_benchmark,
    "heterogeneous_rve_mesh_convergence": run_rve_mesh_convergence_benchmark,
    "rve_rno_fixed_cell_mesh_acceptance": run_rve_mesh_convergence_benchmark,
    "quasi_static_pfczm_notched_beam": run_quasi_static_pfczm,
    "dynamic_pfczm_plate_impact": run_dynamic_pfczm,
    "velocity_verlet_inc_linear_rod": run_velocity_verlet_inc_linear_rod,
}


def get_benchmark_runner(name: str) -> BenchmarkRunner:
    """Return a runner that enforces the common benchmark result contract."""
    runner = BENCHMARKS[name]

    def validated_runner(config: dict[str, Any]) -> RunResult:
        """Execute and validate one registered benchmark result."""
        start = perf_counter()
        result = runner(config)
        duration = perf_counter() - start
        if not isinstance(result, RunResult):
            raise TypeError(
                "Registered benchmark runner must return RunResult: "
                f"case={name}, returned={type(result).__name__}."
            )
        if not result.metrics:
            raise ValueError(f"Benchmark returned no acceptance metrics: case={name}.")
        nonfinite = {
            metric_name: value
            for metric_name, value in result.metrics.items()
            if not math.isfinite(float(value))
        }
        if nonfinite:
            raise ValueError(
                "Benchmark returned non-finite acceptance metrics: "
                f"case={name}, metrics={nonfinite}."
            )
        failure_reason = result.failure_reason
        if not result.passed and failure_reason is None:
            failure_reason = "benchmark_acceptance_criteria_not_met"
        metadata = dict(result.metadata)
        metadata.update(_build_execution_metadata(duration))
        return replace(
            result,
            resolved_config=copy.deepcopy(config),
            failure_reason=failure_reason,
            metadata=metadata,
        )

    return validated_runner


def _build_execution_metadata(duration: float) -> dict[str, Any]:
    """Build reproducible benchmark execution metadata."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "duration_seconds": float(duration),
        "git_commit": commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
