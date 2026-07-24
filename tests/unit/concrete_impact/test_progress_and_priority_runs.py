"""Tests for accepted progress, strict mesh selection, and preflight plans.

Contents:
    Progress, background monitoring, mesh selection, and Pilot coverage tests.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5py
import numpy as np
import pytest

from concrete_impact.benchmarks.rve_fe2_consistency import (
    _build_five_stage_run_specs,
    _write_stage_history,
)
from concrete_impact.benchmarks.rve_mesh_convergence import (
    _build_extremum_record,
    _select_lowest_nonreference_level,
)
from concrete_impact.benchmarks.rve_plastic_impact import (
    solve_multiphase_rve_plastic_impact_scheme,
)
from concrete_impact.cli.generate_rve_data import main as generate_rve_data_main
from concrete_impact.core.progress import (
    BackgroundMonitorSettings,
    JsonProgressRecorder,
    _build_detached_monitor_command,
    _command_config_sha256,
    _git_metadata,
    _parse_linux_process_stat,
    monitor_process_progress,
)
from concrete_impact.experiments.rve_data_execution import _summarize_tangent_checks
from concrete_impact.experiments.rve_data_plan import (
    build_rve_data_task_manifest,
    load_rve_data_plan,
)
from fem.solvers.progress import SolverProgressEvent


class _FakeProcess:
    """Expose a deterministic finite poll sequence to the monitor."""

    def __init__(self, states: list[int | None]) -> None:
        self.pid = 31415
        self.states = states
        self.index = 0

    def poll(self) -> int | None:
        """Return the current state and advance only after a sleeper call."""
        return self.states[min(self.index, len(self.states) - 1)]


def test_json_progress_records_only_supplied_accepted_event(tmp_path: Path) -> None:
    """Verify one accepted event is published atomically with context."""
    output = tmp_path / "progress.json"
    recorder = JsonProgressRecorder(output, {"benchmark_stage": "explicit_fine"})
    recorder(
        SolverProgressEvent(
            "material_dynamics",
            "explicit",
            3,
            10,
            0.3,
            1.0,
            0,
            0,
            2.5,
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["accepted_step"] == 3
    assert payload["benchmark_stage"] == "explicit_fine"
    assert not output.with_suffix(".json.partial").exists()


def test_summary_log_separates_step_and_completed_stage_policies(tmp_path: Path) -> None:
    """Verify data steps are logged while mesh steps may remain JSON-only."""
    summary = tmp_path / "summary.log"
    data_recorder = JsonProgressRecorder(
        tmp_path / "data_progress.json",
        {"task_id": "path_000001"},
        summary,
    )
    data_recorder(
        SolverProgressEvent(
            "rve_path",
            "quasi_static_microequilibrium",
            1,
            5,
            0.001,
            0.005,
            3,
            0,
            12.5,
            2.0e-12,
            1.0e-10,
        )
    )
    mesh_recorder = JsonProgressRecorder(
        tmp_path / "mesh_progress.json",
        {"mesh_level": "c48_r6_z6", "path_family": "hold"},
        summary,
        log_accepted_steps=False,
    )
    mesh_recorder(
        SolverProgressEvent(
            "rve_path",
            "quasi_static_microequilibrium",
            1,
            5,
            0.001,
            0.005,
            4,
            0,
            15.0,
        )
    )
    mesh_recorder.publish_stage("rve_mesh_convergence", 11, 18)

    lines = summary.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    assert 'task="path_000001"' in lines[1]
    assert "step=1/5" in lines[2]
    assert "residual=2e-12" in lines[3]
    assert 'mesh="c48_r6_z6"' in lines[5]
    assert "completed=11/18" in lines[6]


def test_detached_monitor_command_does_not_reenter_detach_mode(tmp_path: Path) -> None:
    """Verify detached submission starts one foreground monitor subprocess."""
    command = _build_detached_monitor_command(
        ("concrete-impact-run", "--config", "case.yaml"),
        tmp_path,
        BackgroundMonitorSettings(60.0, 1200.0),
    )

    assert "--detach" not in command
    assert command[-4:] == ("--", "concrete-impact-run", "--config", "case.yaml")


def test_background_command_records_launch_time_config_hash(tmp_path: Path) -> None:
    """Verify one monitored command retains immutable configuration provenance."""
    config = tmp_path / "case.yaml"
    config.write_text("case: fixed\n", encoding="utf-8")

    hashes = _command_config_sha256(("solver", "--config", str(config)))

    assert set(hashes) == {str(config)}
    assert len(hashes[str(config)]) == 64


def test_background_metadata_records_git_revision() -> None:
    """Verify monitored calculations retain the current repository revision."""
    metadata = _git_metadata()

    assert len(str(metadata["revision"])) == 40
    assert isinstance(metadata["dirty"], bool)


def test_monitor_terminates_after_twenty_minutes_without_progress(tmp_path: Path) -> None:
    """Verify inactivity is measured independently of total planned runtime."""
    process = _FakeProcess([None])
    current = [0.0]

    def sleeper(seconds: float) -> None:
        current[0] += seconds
        process.index += 1

    termination = monitor_process_progress(
        process,
        tmp_path / "missing.json",
        BackgroundMonitorSettings(60.0, 1200.0),
        process_group=31415,
        clock=lambda: current[0],
        sleeper=sleeper,
        cpu_reader=lambda _group: 0.0,
    )

    assert termination is not None
    assert termination["reason"] == "no_observable_progress"
    assert termination["inactivity_seconds"] == 1200.0


def test_cpu_progress_allows_runtime_longer_than_inactivity_limit(tmp_path: Path) -> None:
    """Verify observable CPU progress prevents a wall-clock-only termination."""
    process = _FakeProcess([None] * 25 + [0])
    current = [0.0]

    def sleeper(seconds: float) -> None:
        current[0] += seconds
        process.index += 1

    termination = monitor_process_progress(
        process,
        tmp_path / "missing.json",
        BackgroundMonitorSettings(60.0, 1200.0),
        process_group=31415,
        clock=lambda: current[0],
        sleeper=sleeper,
        cpu_reader=lambda _group: current[0],
    )

    assert termination is None
    assert current[0] > 1200.0


def test_linux_process_stat_parser_preserves_spaced_command_name() -> None:
    """Verify proc stat parsing does not split a command name containing spaces."""
    stat_record = (
        "123 (python worker (phase 1)) R 10 31415 31415 0 -1 0 "
        "1 2 3 4 17 19 0 0 0 0 0"
    )

    process_group, user_ticks, system_ticks = _parse_linux_process_stat(stat_record)

    assert process_group == 31415
    assert user_ticks == 17
    assert system_ticks == 19


def test_reference_grid_cannot_select_itself() -> None:
    """Verify a zero-error reference does not certify mesh convergence."""
    comparisons = [
        {"passed": False, "element_count": 32},
        {"passed": True, "element_count": 1280},
    ]
    levels = [{"reference": False}, {"reference": True}]

    assert _select_lowest_nonreference_level(comparisons, levels) is None


def test_micro_stress_extremum_maps_phase_element_point_and_coordinate() -> None:
    """Verify strict microscopic maxima retain complete assembly-location context."""
    model = SimpleNamespace(phase_names=("matrix", "inclusion"))
    cache = SimpleNamespace(
        assembly=SimpleNamespace(jacobian_weights=np.ones((2, 2))),
        point_phase_ids=np.asarray([0, 0, 1, 1]),
        quadrature_weights=np.asarray([1.0, 2.0, 1.0, 2.0]),
    )
    response = SimpleNamespace(
        state=SimpleNamespace(
            material_state=SimpleNamespace(
                variables={
                    "phase_matrix__equivalent_plastic_strain": np.asarray(
                        [[0.1], [0.2]]
                    )
                }
            )
        ),
        micro_diagnostics={
            "yield_activation_margin": np.asarray([1.0, 2.0, np.nan, np.nan]),
            "viscoplastic_active": np.asarray([1.0, 0.0, np.nan, np.nan]),
        },
        micro_stresses=np.arange(24, dtype=np.float64).reshape(4, 6),
        micro_strains=np.arange(24, dtype=np.float64).reshape(4, 6) / 100.0,
    )
    coordinates = np.asarray(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]
    )

    record = _build_extremum_record(
        model,
        cache,
        coordinates,
        0.25,
        response,
        np.asarray([1.0, 7.0, 3.0, 9.0]),
        3,
        "load_unload_reload",
        5,
    )

    assert record["phase"] == "inclusion"
    assert record["element_id"] == 1
    assert record["quadrature_id"] == 1
    assert record["point_id"] == 3
    assert record["coordinate"] == [0.4, 0.0, 0.0]
    assert record["signed_interface_distance"] == pytest.approx(0.15)


def test_fe2_correctness_task_list_contains_exactly_five_single_scheme_stages() -> None:
    """Verify the fixed correctness run order never invokes a dual-scheme stage."""
    coarse = {"level": "coarse"}
    fine = {"level": "fine"}
    refined = {"level": "refined"}

    stages = _build_five_stage_run_specs(coarse, fine, refined)

    assert [stage[0] for stage in stages] == [
        "explicit_coarse",
        "explicit_fine",
        "implicit_coarse",
        "implicit_fine",
        "explicit_refined",
    ]
    assert [stage[2] for stage in stages] == [
        "explicit",
        "explicit",
        "implicit",
        "implicit",
        "explicit",
    ]


def test_single_scheme_fe2_interface_rejects_undeclared_integrator() -> None:
    """Verify no automatic time-integrator substitution is possible."""
    with pytest.raises(ValueError, match="Unsupported FE2 time integration scheme"):
        solve_multiphase_rve_plastic_impact_scheme(
            {}, cast(Any, "adaptive_fallback")
        )


def test_fe2_stage_reuse_rejects_provenance_hash_mismatch(tmp_path: Path) -> None:
    """Verify stale stage files cannot be reused under a changed configuration."""
    path = tmp_path / "stage.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["config_sha256"] = "old_config"
        handle.attrs["rve_selection_sha256"] = "old_mesh"

    with pytest.raises(ValueError, match="provenance does not match"):
        _write_stage_history(path, None, "new_config", "new_mesh", "explicit_fine")


def test_direct_fe2_preflight_contains_exactly_eight_paths() -> None:
    """Verify the independent direct-FE2 preflight remains an eight-path task."""
    plan = load_rve_data_plan("configs/experiments/rve_rno_fe2_preflight.yaml")
    tasks = build_rve_data_task_manifest(plan)

    assert len(tasks) == 8
    assert {task.macro_path_origin for task in tasks} == {"direct_fe2_explicit"}
    assert plan.output_directory == Path("results/rve_rno_fe2_preflight")


def test_data_cli_rejects_existing_directory_from_different_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify data generation cannot mix artifacts from different YAML inputs."""
    output = tmp_path / "pilot"
    output.mkdir()
    (output / "original_config.yaml").write_text("different: true\n", encoding="utf-8")
    source = Path("configs/experiments/rve_rno_data_pilot.yaml").read_text(
        encoding="utf-8"
    )
    config = tmp_path / "pilot.yaml"
    config.write_text(
        source.replace(
            "output_directory: results/rve_rno_data_pilot_v2",
            f"output_directory: {output}",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-generate-rve-data", "--config", str(config), "--manifest-only"],
    )

    with pytest.raises(ValueError, match="different configuration"):
        generate_rve_data_main()


def test_pilot_tangent_audit_requires_every_source_regime_stratum() -> None:
    """Verify stored tangent checks cannot be absent from any data stratum."""
    complete = [
        {
            "source_name": "material_point",
            "regime": "elastic",
            "maximum_tangent_direction_error": 1.0e-8,
        },
        {
            "source_name": "material_point",
            "regime": "viscoplastic",
            "maximum_tangent_direction_error": 2.0e-8,
        },
    ]

    summary, maximum = _summarize_tangent_checks(complete, 1.0e-5)

    assert maximum == 2.0e-8
    assert all(bool(record["passed"]) for record in summary.values())
    incomplete = [dict(complete[0]), dict(complete[1])]
    incomplete[1]["maximum_tangent_direction_error"] = None
    with pytest.raises(ValueError, match="missing for response strata"):
        _summarize_tangent_checks(incomplete, 1.0e-5)
