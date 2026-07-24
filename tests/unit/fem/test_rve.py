"""Tests for periodic single-phase Hex8 RVE homogenization.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from pathlib import Path

import h5py
import numpy as np

from fem.materials.data import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
)
from fem.materials.plasticity import J2ViscoplasticMaterial
from fem.mesh.structured import build_structured_hex8_box
from fem.rve import (
    RVEPathResult,
    RVERequest,
    StructuredHex8RVE,
    build_structured_hex8_periodic_constraint,
    initialize_rve_state,
    read_rve_path_arrays,
    solve_prescribed_rve_path,
    solve_rve_microequilibrium,
    validate_rve_data_shard,
    write_rve_path_hdf5,
)
from fem.solvers.data import ArmijoSettings, NonlinearNewtonSettings


def test_structured_periodic_constraint_merges_faces_edges_and_corners() -> None:
    """Verify exact periodic classes and rigid-translation elimination."""
    macro_strain = np.asarray([0.01, -0.003, 0.002, 0.004, -0.006, 0.008])
    for divisions in ((1, 1, 1), (2, 2, 2), (4, 4, 4)):
        nodes, _ = build_structured_hex8_box((1.0, 1.0, 1.0), divisions)
        constraint = build_structured_hex8_periodic_constraint(nodes, divisions)
        class_count = int(np.prod(divisions))

        assert np.unique(constraint.equivalence_class_ids).size == class_count
        assert constraint.transformation.shape == (
            3 * nodes.shape[0],
            3 * (class_count - 1),
        )
        assert np.linalg.matrix_rank(constraint.transformation.toarray()) == 3 * (
            class_count - 1
        )
        displacement = constraint.affine_matrix @ macro_strain
        for node_id, class_id in enumerate(constraint.equivalence_class_ids):
            representative = constraint.representative_nodes[class_id]
            fluctuation_difference = (
                displacement.reshape(-1, 3)[node_id]
                - displacement.reshape(-1, 3)[representative]
            )
            affine_difference = (
                constraint.affine_matrix.reshape(nodes.shape[0], 3, 6)[node_id]
                - constraint.affine_matrix.reshape(nodes.shape[0], 3, 6)[representative]
            ) @ macro_strain
            assert np.allclose(fluctuation_difference, affine_difference, atol=1.0e-14)


def test_single_phase_rve_recovers_material_point_stress_state_and_tangent() -> None:
    """Verify homogeneous periodic RVEs reduce to the local J2 update."""
    material = _material()
    update_settings = _update_settings()
    path = (
        np.asarray([0.002, -0.0005, -0.0005, 0.0, 0.0, 0.01]),
        np.asarray([0.004, -0.0010, -0.0010, 0.002, -0.003, 0.02]),
        np.asarray([0.003, -0.0008, -0.0008, 0.001, -0.002, 0.012]),
        np.asarray([0.0045, -0.0012, -0.0012, -0.001, 0.002, -0.01]),
    )

    for divisions in ((1, 1, 1), (2, 2, 2)):
        model = _model(material, divisions)
        rve_state = initialize_rve_state(model)
        point_state = material.initialize_state(1)
        previous_strain = np.zeros(6, dtype=np.float64)
        for macro_strain in path:
            time_step = 1.0e-3
            rve_response = solve_rve_microequilibrium(
                model,
                RVERequest(
                    macro_strain=macro_strain,
                    time_step=time_step,
                    material_update_settings=update_settings,
                    requirements=MaterialResponseRequirements(tangent=True),
                ),
                rve_state,
            )
            point_response = material.update(
                MaterialPointRequest(
                    strains=macro_strain.reshape(1, 6),
                    strain_rates=((macro_strain - previous_strain) / time_step).reshape(1, 6),
                    time_step=time_step,
                    kinematics="three_dimensional",
                    update_settings=update_settings,
                ),
                point_state,
                MaterialResponseRequirements(tangent=True, free_energy=True, dissipation=True),
            )

            assert np.allclose(rve_response.macro_stress, point_response.stresses[0], atol=1.0e-10)
            assert np.allclose(
                rve_response.effective_tangent,
                point_response.tangents[0],
                rtol=1.0e-10,
                atol=1.0e-9,
            )
            assert rve_response.diagnostics["periodic_fluctuation_error"] <= 1.0e-14
            assert rve_response.diagnostics["hill_mandel_error"] <= 1.0e-12
            assert rve_response.diagnostics["virtual_work_stress_error"] <= 1.0e-10
            for name, values in point_response.state.variables.items():
                expected = np.repeat(
                    values,
                    rve_response.state.material_state.variables[name].shape[0],
                    axis=0,
                )
                assert np.allclose(
                    rve_response.state.material_state.variables[name],
                    expected,
                    atol=1.0e-12,
                )

            rve_state = rve_response.state
            point_state = point_response.state
            previous_strain = macro_strain


def test_rve_hdf5_round_trip_preserves_complete_path(tmp_path: Path) -> None:
    """Verify versioned HDF5 output preserves macro, micro, and state histories."""
    material = _material()
    model = _model(material, (1, 1, 1))
    state = initialize_rve_state(model)
    responses = []
    times = np.asarray([1.0e-3, 2.0e-3])
    for shear in (0.01, 0.02):
        response = solve_rve_microequilibrium(
            model,
            RVERequest(
                macro_strain=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, shear]),
                time_step=1.0e-3,
                material_update_settings=_update_settings(),
            ),
            state,
        )
        responses.append(response)
        state = response.state

    output = write_rve_path_hdf5(
        tmp_path / "single_phase_rve.h5",
        model,
        "shear_path",
        RVEPathResult(times=times, responses=tuple(responses)),
        {"benchmark": "single_phase_rve"},
    )
    arrays = read_rve_path_arrays(output, "shear_path")
    counts = validate_rve_data_shard(output)

    assert np.array_equal(arrays["time"], times)
    assert arrays["macro_strain"].shape == (2, 6)
    assert arrays["micro_stress"].shape == (2, 8, 6)
    assert counts == {"model_count": 1, "path_count": 1}
    with h5py.File(output, "r") as handle:
        assert handle.attrs["schema_version"] == "3.0"
        assert "paths/shear_path/micro/state/plastic_strain" in handle
        assert "paths/shear_path/micro/diagnostics/yield_activation_margin" in handle
        assert "paths/shear_path/solver/hill_mandel_error" in handle
        assert "paths/shear_path/phases/domain/average_stress" in handle
        assert "models/default/mesh/element_phase_id" in handle
        assert "paths/shear_path/features/material_parameters" in handle


def test_prescribed_rve_progress_reports_only_accepted_steps() -> None:
    """Verify prescribed-path callbacks observe committed physical steps only."""
    model = _model(_material(), (1, 1, 1))
    events = []
    times = np.asarray([0.0, 1.0e-3, 2.0e-3])
    strains = np.zeros((3, 6), dtype=np.float64)
    strains[1:, 5] = [0.01, 0.02]

    result = solve_prescribed_rve_path(
        model,
        times,
        strains,
        _update_settings(),
        events.append,
    )

    assert len(result.responses) == 2
    assert [event.accepted_step for event in events] == [1, 2]
    assert [event.physical_time for event in events] == [1.0e-3, 2.0e-3]


def _material() -> J2ViscoplasticMaterial:
    """Build the homogeneous benchmark material."""
    return J2ViscoplasticMaterial(
        name="single_phase_rve_j2",
        density=1.0,
        young_modulus=3339.9160679991464,
        poisson_ratio=0.36994096308414537,
        yield_stress=10.0,
        hardening_modulus=80.0,
        time_scale=0.02,
        reference_stress=10.0,
        rate_exponent=2.0,
    )


def _update_settings() -> MaterialUpdateSettings:
    """Build strict local constitutive integration settings."""
    return MaterialUpdateSettings(
        max_iterations=50,
        yield_relative_tolerance=1.0e-10,
        residual_absolute_tolerance=1.0e-12,
        residual_relative_tolerance=1.0e-10,
    )


def _model(
    material: J2ViscoplasticMaterial,
    divisions: tuple[int, int, int],
) -> StructuredHex8RVE:
    """Build one structured periodic benchmark RVE."""
    nodes, elements = build_structured_hex8_box((1.0, 1.0, 1.0), divisions)
    constraint = build_structured_hex8_periodic_constraint(nodes, divisions)

    return StructuredHex8RVE(
        nodes=nodes,
        elements=elements,
        divisions=divisions,
        material=material,
        constraint=constraint,
        newton_settings=NonlinearNewtonSettings(
            max_iterations=20,
            residual_absolute_tolerance=1.0e-10,
            residual_relative_tolerance=1.0e-10,
            increment_absolute_tolerance=1.0e-12,
            increment_relative_tolerance=1.0e-10,
            armijo=ArmijoSettings(enabled=True),
        ),
        hill_mandel_power_scale=10.0,
    )
