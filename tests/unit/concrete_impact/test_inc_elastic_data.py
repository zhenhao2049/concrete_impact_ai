"""Contracts for memory-bounded linear-elastic INC data generation.

Author:
    Zhen Hao.
Created:
    2026-09-15.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from concrete_impact.cluster.inc_elastic_production import prepare_elastic_inc_production
from concrete_impact.experiments.inc_elastic_generation import (
    _build_bundle,
    build_elastic_inc_preflight,
    generate_elastic_inc_path,
    pulse_shape,
)
from concrete_impact.experiments.inc_elastic_plan import (
    build_elastic_inc_tasks,
    load_elastic_inc_plan,
)
from fem.assembly.load import assemble_boundary_traction, assemble_uniform_boundary_traction
from fem.dynamics.material import solve_material_explicit
from fem.materials.data import MaterialUpdateSettings
from fem.solvers.data import DirichletDofSet, TimeIntegrationSettings

CONFIG_ROOT = Path("configs/experiments")


def test_reviewed_inc_path_counts_and_splits() -> None:
    """Require the exact P0, P1 trial, and P1 official path designs."""
    p0 = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p0_7.yaml")
    pilot = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_test_12.yaml")
    official = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_official_144.yaml")
    p0_tasks = build_elastic_inc_tasks(p0)
    pilot_tasks = build_elastic_inc_tasks(pilot)
    official_tasks = build_elastic_inc_tasks(official)

    assert pilot.fine_ratio == 64
    assert pilot.reference_ratio == 128
    assert official.fine_ratio == pilot.fine_ratio
    assert official.reference_ratio == pilot.reference_ratio
    assert len(p0_tasks) == 7
    assert len(pilot_tasks) == 12
    assert len(official_tasks) == 144
    assert [task.split for task in p0_tasks].count("train") == 4
    assert [task.split for task in p0_tasks].count("validation") == 2
    assert [task.split for task in p0_tasks].count("test") == 1
    assert [task.split for task in official_tasks].count("train") == 96
    assert [task.split for task in official_tasks].count("validation") == 24
    assert [task.split for task in official_tasks].count("test") == 24
    for amplitude in official.pressure_amplitudes:
        assert [task.pressure_amplitude for task in official_tasks].count(amplitude) == 48
    for split in ("train", "validation", "test"):
        selected = [task for task in official_tasks if task.split == split]
        assert len({task.spatial_coefficients for task in selected}) == 8
        assert len({task.pulse_kind for task in selected}) == 3


def test_p1_trial_balances_pulse_families() -> None:
    """Require four pilot paths from each temporal pulse family."""
    plan = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_test_12.yaml")
    tasks = build_elastic_inc_tasks(plan)
    for pulse_kind in ("half_sine", "raised_cosine", "double_pulse"):
        assert [task.pulse_kind for task in tasks].count(pulse_kind) == 4


def test_p1_time_refinement_plan_preserves_trial_design() -> None:
    """Allow the reviewed P1 time-refinement levels without changing paths."""
    baseline = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_test_12.yaml")
    refined = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_convergence_8_16.yaml")
    refined_again = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_convergence_16_32.yaml")
    refined_final = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_convergence_32_64.yaml")
    refined_strict = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_convergence_64_128.yaml")

    assert refined.fine_ratio == 8
    assert refined.reference_ratio == 16
    assert refined_again.fine_ratio == 16
    assert refined_again.reference_ratio == 32
    assert refined_final.fine_ratio == 32
    assert refined_final.reference_ratio == 64
    assert refined_strict.fine_ratio == 64
    assert refined_strict.reference_ratio == 128
    assert build_elastic_inc_tasks(refined) == build_elastic_inc_tasks(baseline)
    assert build_elastic_inc_tasks(refined_again) == build_elastic_inc_tasks(baseline)
    assert build_elastic_inc_tasks(refined_final) == build_elastic_inc_tasks(baseline)
    assert build_elastic_inc_tasks(refined_strict) == build_elastic_inc_tasks(baseline)


def test_official_plan_records_completed_trial_approval() -> None:
    """Require the formal P1 configuration to retain its approved release state."""
    official = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_official_144.yaml")

    assert official.release_status == "approved"


def test_pilot_prepare_writes_one_twelve_path_bundle(tmp_path: Path) -> None:
    """Prepare the fixed P1 pilot manifest without running finite elements."""
    base = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p1_test_12.yaml")
    plan = base.model_copy(update={"output_directory": tmp_path / "pilot"})
    config = tmp_path / "pilot.json"
    config.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = prepare_elastic_inc_production(
        config,
        bundle_count=1,
        parallel_paths=12,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = json.loads((tmp_path / "pilot/bundles/bundle_000.json").read_text())
    assert manifest["task_count"] == 12
    assert manifest["parallel_paths"] == 12
    assert len(bundle["task_ids"]) == 12


def test_coordinate_dependent_line_traction_matches_exact_integrals() -> None:
    """Check line-traction resultants for constant and linear fields."""
    mesh_info = SimpleNamespace(
        nodes=np.asarray(((1.0, 0.0), (1.0, 1.0))),
        dof_map=np.asarray(((0, 1), (2, 3)), dtype=np.int64),
        boundary_groups={
            "right": SimpleNamespace(
                cell_type="line",
                cells=np.asarray(((0, 1),), dtype=np.int64),
            )
        },
    )
    constant = assemble_boundary_traction(
        mesh_info,
        "right",
        lambda point: np.asarray((2.0, 3.0)),
        1.0,
    )
    linear = assemble_boundary_traction(
        mesh_info,
        "right",
        lambda point: np.asarray((point[1], 0.0)),
        1.0,
    )

    assert np.allclose(constant, (1.0, 1.5, 1.0, 1.5))
    assert np.allclose(linear, (1.0 / 6.0, 0.0, 1.0 / 3.0, 0.0))


def test_pulse_shapes_have_compact_support() -> None:
    """Require every pulse to vanish outside its declared duration."""
    for kind in ("sine_squared", "half_sine", "raised_cosine", "double_pulse"):
        assert pulse_shape(kind, -0.1, 1.0) == 0.0
        assert pulse_shape(kind, 1.1, 1.0) == 0.0


def test_short_p0_path_writes_complete_bounded_schema(tmp_path: Path) -> None:
    """Generate one short nested path without allocating complete fine histories."""
    base = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p0_7.yaml")
    root = tmp_path / "p0"
    plan = base.model_copy(
        update={
            "output_directory": root,
            "num_steps": 4,
            "chunk_steps": 2,
            "progress_updates": 2,
        }
    )
    task = build_elastic_inc_tasks(plan)[0]
    report = build_elastic_inc_preflight(plan, root)
    metrics = generate_elastic_inc_path(plan, task, root)

    assert report["passed"] is True
    assert metrics["passed"] is True
    assert metrics["velocity_reference_error_pulse"] >= 0.0
    assert metrics["velocity_reference_error_post"] is None
    assert metrics["reference_velocity_l2_mass_time"] > 0.0
    with h5py.File(root / "tasks" / task.task_id / "response.h5", "r") as handle:
        assert handle.attrs["status"] == "complete"
        assert handle["state/displacement"].shape == (5, 80)
        assert handle["target/residual_force_start"].shape == (4, 80)


def test_matrix_path_matches_generic_linear_solver(tmp_path: Path) -> None:
    """Match the existing material solver on one short P0 fine trajectory."""
    base = load_elastic_inc_plan(CONFIG_ROOT / "inc_elastic_p0_7.yaml")
    root = tmp_path / "matrix"
    plan = base.model_copy(
        update={
            "output_directory": root,
            "num_steps": 4,
            "chunk_steps": 2,
            "progress_updates": 2,
        }
    )
    task = build_elastic_inc_tasks(plan)[0]
    build_elastic_inc_preflight(plan, root)
    generate_elastic_inc_path(plan, task, root)

    bundle = _build_bundle(plan, tmp_path / "generic")
    nodes = bundle.mesh_info.nodes
    dof_map = bundle.mesh_info.dof_map
    left_nodes = np.flatnonzero(np.isclose(nodes[:, 0], np.min(nodes[:, 0])))
    fixed = np.unique(np.concatenate((dof_map[:, 1], dof_map[left_nodes, 0])))
    dirichlet = DirichletDofSet(
        dofs=fixed,
        values=np.zeros(fixed.size, dtype=np.float64),
    )

    def load_function(time: float) -> np.ndarray:
        """Evaluate the same P0 pulse through the generic load interface."""
        traction = np.asarray(
            (
                -task.pressure_amplitude * pulse_shape(task.pulse_kind, time, task.pulse_duration),
                0.0,
            )
        )
        return assemble_uniform_boundary_traction(
            bundle.mesh_info,
            plan.load_boundary,
            traction,
            plan.thickness,
        )

    fine = solve_material_explicit(
        bundle,
        TimeIntegrationSettings(
            scheme="velocity_verlet",
            time_step=plan.time_step / plan.fine_ratio,
            num_steps=plan.num_steps * plan.fine_ratio,
            beta=0.0,
            gamma=0.5,
            cfl_safety_factor=plan.coarse_cfl_ratio,
        ),
        np.zeros(bundle.mesh_info.dof_map.size),
        np.zeros(bundle.mesh_info.dof_map.size),
        dirichlet,
        load_function,
        plan.plane_state,
        MaterialUpdateSettings(
            max_iterations=1,
            yield_relative_tolerance=1.0e-12,
            residual_absolute_tolerance=1.0e-12,
            residual_relative_tolerance=1.0e-12,
        ),
    )
    sample = np.arange(plan.num_steps + 1) * plan.fine_ratio
    free = np.setdiff1d(np.arange(bundle.mesh_info.dof_map.size), fixed)
    with h5py.File(root / "tasks" / task.task_id / "response.h5", "r") as handle:
        displacement = handle["state/displacement"][...]
        velocity = handle["state/velocity"][...]

    assert (
        np.linalg.norm(displacement - fine.displacement[sample][:, free])
        / np.linalg.norm(fine.displacement[sample][:, free])
        < 1.0e-12
    )
    assert (
        np.linalg.norm(velocity - fine.velocity[sample][:, free])
        / np.linalg.norm(fine.velocity[sample][:, free])
        < 1.0e-12
    )
