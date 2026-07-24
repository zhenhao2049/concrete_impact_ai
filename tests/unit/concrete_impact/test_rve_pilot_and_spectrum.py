"""Tests for pilot-path design, accepted spectra, and performance summaries.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

import numpy as np

from concrete_impact.experiments.dynamic_solver_performance import _summarize_rows
from concrete_impact.experiments.rve_execution_performance import _summarize
from fem.post.material_dynamic import compute_strain_rate_spectrum


def test_fixed_step_strain_rate_spectrum_recovers_exact_frequency_bin() -> None:
    """Verify tensor weighting and the 99-percent discrete spectral frequency."""
    point_count = 2
    sample_count = 64
    time_step = 0.01
    times = np.arange(sample_count + 1, dtype=np.float64) * time_step
    target_id = 4
    omega = 2.0 * np.pi * target_id / (sample_count * time_step)
    strain_rate = np.cos(omega * times[:-1])
    strains = np.zeros((times.size, point_count, 6), dtype=np.float64)
    strains[1:, :, 0] = np.cumsum(strain_rate)[:, None] * time_step
    strains[1:, :, 3] = 2.0 * np.cumsum(strain_rate)[:, None] * time_step

    spectrum = compute_strain_rate_spectrum(times, strains)

    np.testing.assert_allclose(spectrum.omega_99, omega, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        spectrum.point_energy[0], spectrum.point_energy[1], rtol=0.0, atol=1.0e-12
    )


def test_dynamic_performance_summary_separates_accuracy_and_speed() -> None:
    """Verify stable-speed classification without making it a correctness gate."""
    rows = []
    for _repetition, explicit_time, implicit_time in (
        (1, 10.0, 5.0),
        (2, 10.2, 5.1),
        (3, 9.8, 4.9),
    ):
        rows.append(
            {
                "explicit_solve_time": explicit_time,
                "implicit_solve_time": implicit_time,
                "displacement_history_scaled_error": 0.0,
                "peak_displacement_scaled_error": 0.0,
                "dissipated_energy_scaled_error": 0.0,
                "explicit_energy_residual": 0.0,
                "implicit_energy_residual": 0.0,
                "profile_material_time": 1.0,
                "profile_assembly_time": 1.0,
                "profile_linear_solve_time": 1.0,
                "profile_rve_time": 0.0,
                "profile_history_time": 1.0,
            }
        )
    config = {
        "verification": {
            "response_tolerance": 1.0e-2,
            "energy_tolerance": 1.0e-2,
        }
    }

    summary = _summarize_rows(rows, config)

    assert summary["passed"]
    assert summary["implicit_stably_faster"]
    assert summary["engineering_speed_target_met"]


def test_rve_performance_summary_keeps_profiles_outside_wall_timing() -> None:
    """Verify FE2 speed statistics use only unprofiled repeated wall times."""
    rows = [
        {"batch_size": 16, "repetition": repetition, "backend": backend, "elapsed_time": time}
        for repetition, serial_time, process_time in (
            (1, 4.0, 3.0),
            (2, 4.2, 3.1),
            (3, 3.8, 2.9),
        )
        for backend, time in (("serial", serial_time), ("process_pool", process_time))
    ]
    names = (
        "profile_micro_material_time",
        "profile_micro_assembly_time",
        "profile_micro_linear_time",
        "profile_rve_solver_time",
        "profile_parent_communication_time",
    )
    profiles = {
        (16, backend): {name: float(index) for index, name in enumerate(names)}
        for backend in ("serial", "process_pool")
    }
    config = {"performance": {"batch_sizes": [16], "repetitions": 3, "workers": 2}}

    summary = _summarize(rows, profiles, config)
    batch = summary["batches"]["16"]

    assert batch["serial_median_time"] == 4.0
    assert batch["process_pool_median_time"] == 3.0
    assert batch["process_pool_speedup"] == 4.0 / 3.0
    assert batch["serial_profile_micro_material_time"] == 0.0
