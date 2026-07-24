"""Report-specific explicit FE2 plastic-impact benchmark and visualization.

Contents:
    Benchmark execution, event metrics, output records, and macro/RVE montages.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from concrete_impact.benchmarks.rve_multiphase_dynamics import _maximum_scaled
from concrete_impact.benchmarks.rve_plastic_impact import (
    _maximum_postpeak_drop,
    _mean_equivalent_stress,
    _minimum_increment,
    solve_multiphase_rve_plastic_impact_scheme,
)
from concrete_impact.core.progress import JsonProgressRecorder
from fem.assembly.nonlinear_solid import build_material_assembly_cache
from fem.cases import RunResult
from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.post.material_dynamic import (
    extract_quadrature_strain_history,
    write_material_dynamic_vtk_series,
)
from fem.quadrature.rules import make_quadrature_rule
from fem.rve import solve_prescribed_rve_path
from fem.rve.solver import build_rve_operator_cache


def run_report_fe2_plastic_impact(config: dict[str, Any]) -> RunResult:
    """Run the explicit c48 FE2 impact used by the dated research report."""
    output_root = Path(config["output"]["root"])
    recorder = JsonProgressRecorder(
        output_root / "progress.json",
        {"benchmark": str(config["case"]["name"]), "scheme": "explicit"},
        output_root / "summary.log",
    )
    simulation = solve_multiphase_rve_plastic_impact_scheme(
        config,
        "explicit",
        recorder,
    )
    metrics = compute_report_impact_metrics(
        config,
        simulation.bundle,
        simulation.solution,
    )
    passed = evaluate_report_impact_metrics(config["verification"], metrics)
    output_paths = write_report_impact_outputs(
        config,
        simulation.bundle,
        simulation.solution,
        simulation.rve_model,
        simulation.material_update_settings,
        recorder,
    )
    recorder.publish_summary(
        "report_fe2_plastic_impact_complete",
        {
            "passed": passed,
            "maximum_q": metrics["maximum_q"],
            "minimum_negative_work": metrics["minimum_negative_work"],
            "energy_residual": metrics["energy_residual"],
        },
    )
    return RunResult(
        name=str(config["case"]["name"]),
        metrics=metrics,
        output_paths=output_paths,
        passed=passed,
    )


def compute_report_impact_metrics(
    config: dict[str, Any],
    bundle: Any,
    solution: Any,
) -> dict[str, float]:
    """Compute strict physical checks for the explicit report-impact history."""
    verification = config["verification"]
    q = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
    equivalent_stress = _mean_equivalent_stress(solution.stresses)
    arrival_stress = _near_fixed_end_equivalent_stress(bundle, solution)
    right_nodes = bundle.mesh_info.boundary_groups["right"].nodes
    right_dofs = bundle.mesh_info.dof_map[right_nodes, 0]
    arrival_id = _wave_arrival_step(arrival_stress)
    event_steps = select_report_event_steps(
        solution.times,
        arrival_stress,
        np.max(q, axis=1),
        solution.viscoplastic_active_volume_fraction,
        solution.negative_incremental_work,
        float(verification["plastic_q_threshold"]),
    )
    return {
        "maximum_q": float(np.max(q)),
        "maximum_active_fraction": float(np.max(solution.viscoplastic_active_volume_fraction)),
        "minimum_negative_work": float(np.min(solution.negative_incremental_work)),
        "equivalent_stress_drop": _maximum_postpeak_drop(equivalent_stress),
        "minimum_q_increment": _minimum_increment(q),
        "minimum_dissipation_density": float(
            np.min(solution.material_diagnostics["dissipation_density"])
        ),
        "minimum_cumulative_dissipation": float(np.min(solution.dissipated_energy)),
        "residual_displacement": float(abs(np.mean(solution.displacement[-1, right_dofs]))),
        "energy_residual": _maximum_scaled(
            solution.energy_residual,
            float(verification["energy_scale"]),
        ),
        "wave_arrival_time": float(solution.times[arrival_id]),
        "first_yield_time": float(solution.times[event_steps["first_yield"]]),
        "peak_activity_time": float(solution.times[event_steps["peak_activity"]]),
        "maximum_unloading_time": float(solution.times[event_steps["maximum_negative_work"]]),
        "solve_time": float(solution.solve_time),
    }


def evaluate_report_impact_metrics(
    verification: dict[str, Any],
    metrics: dict[str, float],
) -> bool:
    """Evaluate the configured plastic-impact physical requirements."""
    return bool(
        metrics["maximum_q"] >= float(verification["plastic_q_threshold"])
        and metrics["maximum_active_fraction"] > 0.0
        and metrics["minimum_negative_work"] <= -float(verification["negative_work_threshold"])
        and metrics["equivalent_stress_drop"] >= float(verification["stress_drop_threshold"])
        and metrics["minimum_q_increment"] >= -float(verification["state_monotonicity_tolerance"])
        and metrics["minimum_dissipation_density"] >= -float(verification["dissipation_tolerance"])
        and metrics["minimum_cumulative_dissipation"]
        >= -float(verification["dissipation_tolerance"])
        and metrics["residual_displacement"]
        >= float(verification["residual_displacement_threshold"])
        and metrics["energy_residual"] <= float(verification["energy_tolerance"])
    )


def select_report_event_steps(
    times: NDArray[np.float64],
    arrival_stress: NDArray[np.float64],
    maximum_q: NDArray[np.float64],
    active_fraction: NDArray[np.float64],
    negative_work: NDArray[np.float64],
    plastic_threshold: float,
) -> dict[str, int]:
    """Select five distinct accepted states for the report montage."""
    yield_ids = np.flatnonzero(maximum_q >= plastic_threshold)
    if yield_ids.size == 0:
        raise ValueError("Report FE2 impact contains no accepted first-yield state.")
    steps = {
        "wave_arrival": _wave_arrival_step(arrival_stress),
        "first_yield": int(yield_ids[0]),
        "peak_activity": int(np.argmax(active_fraction)),
        "maximum_negative_work": int(np.argmin(negative_work)),
        "final_residual": int(times.size - 1),
    }
    if len(set(steps.values())) != len(steps):
        raise ValueError(f"Report FE2 impact event states are not distinct: {steps}.")
    return steps


def write_report_impact_outputs(
    config: dict[str, Any],
    bundle: Any,
    solution: Any,
    rve_model: Any,
    material_update_settings: Any,
    progress_callback: Any,
) -> dict[str, Path]:
    """Write accepted VTK histories, event metadata, and the report montage."""
    output_root = Path(config["output"]["root"])
    vtk_paths = write_material_dynamic_vtk_series(
        bundle,
        solution,
        output_root / "vtk" / "explicit",
        "explicit_c48_evolution",
        int(config["output"]["vtk_step_stride"]),
        bool(config["output"]["save_quadrature_vtk"]),
    )
    q = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"]
    arrival_stress = _near_fixed_end_equivalent_stress(bundle, solution)
    events = select_report_event_steps(
        solution.times,
        arrival_stress,
        np.max(q, axis=1),
        solution.viscoplastic_active_volume_fraction,
        solution.negative_incremental_work,
        float(config["verification"]["plastic_q_threshold"]),
    )
    event_path = output_root / "event_frames.json"
    event_path.write_text(
        json.dumps(
            {
                name: {"step": step_id, "time": float(solution.times[step_id])}
                for name, step_id in events.items()
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    montage_path = output_root / "figures" / "plastic_strain_evolution_1x5.png"
    write_plastic_strain_montage(bundle, solution, events, montage_path)
    strain_history = extract_quadrature_strain_history(bundle, solution.displacement)
    representative_point = int(np.argmax(np.max(q, axis=0)))
    replay = solve_prescribed_rve_path(
        rve_model,
        solution.times,
        strain_history[:, representative_point],
        material_update_settings,
        progress_callback,
    )
    rve_montage_path = output_root / "figures" / "rve_q_evolution_xz_1x5.png"
    write_rve_plastic_strain_montage(
        rve_model,
        replay,
        solution.times,
        events,
        rve_montage_path,
    )
    replay_path = output_root / "rve_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "macro_quadrature_point": representative_point,
                "criterion": "maximum_matrix_average_q_over_accepted_history",
                "macro_coordinate": _quadrature_coordinates(bundle)[
                    representative_point
                ].tolist(),
                "rve_mesh": {
                    "elements": int(rve_model.elements.shape[0]),
                    "nodes": int(rve_model.nodes.shape[0]),
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    metrics_path = output_root / "metrics.json"
    metrics = compute_report_impact_metrics(config, bundle, solution)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return {
        **vtk_paths,
        "event_frames": event_path,
        "plastic_strain_montage": montage_path,
        "rve_plastic_strain_montage": rve_montage_path,
        "rve_replay": replay_path,
        "metrics": metrics_path,
    }


def write_plastic_strain_montage(
    bundle: Any,
    solution: Any,
    event_steps: dict[str, int],
    output_path: str | Path,
) -> None:
    """Render a continuously projected macro matrix-plastic-strain montage."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    nodal_q = _project_matrix_q_to_macro_nodes(bundle, solution)
    nodes = bundle.mesh_info.nodes
    z_coordinates = np.unique(nodes[:, 2])
    section_z = float(z_coordinates[np.argmin(np.abs(z_coordinates - np.mean(z_coordinates)))])
    triangles = _coordinate_plane_triangles(
        nodes,
        bundle.mesh_info.elements,
        axis=2,
        coordinate=section_z,
    )
    ordered_events = sorted(event_steps.items(), key=lambda item: item[1])
    maximum = max(float(np.max(nodal_q[step_id])) for _, step_id in ordered_events)
    if maximum <= 0.0:
        raise ValueError("Report montage requires a strictly positive plastic-strain range.")
    normalization = Normalize(vmin=0.0, vmax=maximum)
    figure, axes = plt.subplots(1, 5, figsize=(15.0, 3.0), constrained_layout=True)
    image = None
    for axis, (event_name, step_id) in zip(axes, ordered_events, strict=True):
        image = axis.tripcolor(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
            nodal_q[step_id],
            shading="gouraud",
            cmap="jet",
            norm=normalization,
        )
        axis.set_xlim(float(np.min(nodes[:, 0])), float(np.max(nodes[:, 0])))
        axis.set_ylim(float(np.min(nodes[:, 1])), float(np.max(nodes[:, 1])))
        axis.set_aspect("equal")
        axis.set_title(
            f"{event_name.replace('_', ' ')}\nt={solution.times[step_id]:.3e}",
            fontsize=8,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    if image is None:
        raise ValueError("Report montage received no selected event states.")
    figure.colorbar(image, ax=axes, label="matrix volume-averaged q", shrink=0.82)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _project_matrix_q_to_macro_nodes(bundle: Any, solution: Any) -> NDArray[np.float64]:
    """Lump-project macro quadrature matrix-average q onto continuous nodes."""
    cache = build_material_assembly_cache(bundle)
    element_count, quadrature_count = cache.jacobian_weights.shape
    values = solution.material_diagnostics["phase_matrix__equivalent_plastic_strain"].reshape(
        solution.times.size, element_count, quadrature_count
    )
    return _lumped_project_hex8_scalar_to_nodes(
        bundle.mesh_info.elements,
        values,
        cache.jacobian_weights,
        bundle.mesh_info.nodes.shape[0],
        quadrature_order=2,
    )


def write_rve_plastic_strain_montage(
    model: Any,
    replay: Any,
    macro_times: NDArray[np.float64],
    event_steps: dict[str, int],
    output_path: str | Path,
) -> None:
    """Render phase-aware continuous matrix-q fields on the RVE x-z section."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cache = build_rve_operator_cache(model)
    element_count, quadrature_count = cache.assembly.jacobian_weights.shape
    point_count = element_count * quadrature_count
    matrix_point_mask = cache.point_phase_ids == 0
    matrix_element_mask = np.asarray(model.element_phase_ids == 0, dtype=np.bool_)
    inclusion_element_mask = np.asarray(model.element_phase_ids == 1, dtype=np.bool_)
    q_history = np.zeros((len(replay.responses), point_count), dtype=np.float64)
    for response_id, response in enumerate(replay.responses):
        matrix_q = response.state.material_state.variables[
            "phase_matrix__equivalent_plastic_strain"
        ]
        q_history[response_id, matrix_point_mask] = matrix_q
    matrix_nodal_q = _lumped_project_hex8_scalar_to_nodes(
        model.elements,
        q_history.reshape(len(replay.responses), element_count, quadrature_count),
        cache.assembly.jacobian_weights,
        model.nodes.shape[0],
        quadrature_order=model.quadrature_order,
        element_mask=matrix_element_mask,
    )
    matrix_triangles = _coordinate_plane_triangles(
        model.nodes,
        model.elements,
        axis=1,
        coordinate=0.0,
        element_mask=matrix_element_mask,
    )
    inclusion_triangles = _coordinate_plane_triangles(
        model.nodes,
        model.elements,
        axis=1,
        coordinate=0.0,
        element_mask=inclusion_element_mask,
    )
    matrix_node_ids, local_matrix_triangles = _localize_triangles(matrix_triangles)
    inclusion_node_ids, local_inclusion_triangles = _localize_triangles(inclusion_triangles)
    interface_edges = _shared_triangle_boundary_edges(matrix_triangles, inclusion_triangles)

    ordered_events = sorted(event_steps.items(), key=lambda item: item[1])
    response_ids = [step_id - 1 for _, step_id in ordered_events]
    if min(response_ids) < 0:
        raise ValueError("RVE report montage cannot select the unadvanced initial state.")
    maximum = max(float(np.max(matrix_nodal_q[index, matrix_node_ids])) for index in response_ids)
    if maximum <= 0.0:
        raise ValueError("RVE report montage requires a strictly positive matrix-q range.")
    normalization = Normalize(vmin=0.0, vmax=maximum)
    figure, axes = plt.subplots(1, 5, figsize=(15.0, 3.2), constrained_layout=True)
    image = None
    for axis, ((event_name, step_id), response_id) in zip(
        axes,
        zip(ordered_events, response_ids, strict=True),
        strict=True,
    ):
        axis.tripcolor(
            model.nodes[inclusion_node_ids, 0],
            model.nodes[inclusion_node_ids, 2],
            local_inclusion_triangles,
            np.zeros(inclusion_node_ids.size, dtype=np.float64),
            shading="gouraud",
            cmap="jet",
            norm=normalization,
        )
        image = axis.tripcolor(
            model.nodes[matrix_node_ids, 0],
            model.nodes[matrix_node_ids, 2],
            local_matrix_triangles,
            matrix_nodal_q[response_id, matrix_node_ids],
            shading="gouraud",
            cmap="jet",
            norm=normalization,
        )
        for node_a, node_b in interface_edges:
            axis.plot(
                model.nodes[[node_a, node_b], 0],
                model.nodes[[node_a, node_b], 2],
                color="black",
                linewidth=0.8,
            )
        axis.set_xlim(float(np.min(model.nodes[:, 0])), float(np.max(model.nodes[:, 0])))
        axis.set_ylim(float(np.min(model.nodes[:, 2])), float(np.max(model.nodes[:, 2])))
        axis.set_aspect("equal")
        axis.set_title(
            f"{event_name.replace('_', ' ')}\nt={macro_times[step_id]:.3e}",
            fontsize=8,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("z")
    if image is None:
        raise ValueError("RVE report montage received no selected event states.")
    figure.colorbar(image, ax=axes, label="phase-aware projected q", shrink=0.82)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def _lumped_project_hex8_scalar_to_nodes(
    elements: NDArray[np.int64],
    quadrature_values: NDArray[np.float64],
    jacobian_weights: NDArray[np.float64],
    node_count: int,
    quadrature_order: int,
    element_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Apply a positive lumped L2 projection from Hex8 quadrature points to nodes."""
    points, _ = make_quadrature_rule("hex8", quadrature_order)
    shape_values, _ = evaluate_lagrange_shape_functions("hex8", points)
    selected = (
        np.ones(elements.shape[0], dtype=np.bool_) if element_mask is None else element_mask
    )
    numerator = np.zeros((quadrature_values.shape[0], node_count), dtype=np.float64)
    denominator = np.zeros(node_count, dtype=np.float64)
    for element_id in np.flatnonzero(selected):
        connectivity = elements[element_id]
        weights = jacobian_weights[element_id]
        local_denominator = np.sum(shape_values * weights[:, None], axis=0)
        local_numerator = np.einsum(
            "tg,gi,g->ti",
            quadrature_values[:, element_id],
            shape_values,
            weights,
        )
        denominator[connectivity] += local_denominator
        numerator[:, connectivity] += local_numerator
    active_nodes = np.unique(elements[selected])
    if np.any(denominator[active_nodes] <= 0.0):
        raise ValueError("Lumped Hex8 projection has a non-positive nodal mass.")
    projected = np.zeros_like(numerator)
    projected[:, active_nodes] = numerator[:, active_nodes] / denominator[active_nodes]
    return projected


def _coordinate_plane_triangles(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int64],
    axis: int,
    coordinate: float,
    element_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """Triangulate unique Hex8 faces lying on one coordinate plane."""
    local_faces = (
        (0, 1, 3, 2),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 3, 7, 5),
    )
    selected = (
        np.ones(elements.shape[0], dtype=np.bool_) if element_mask is None else element_mask
    )
    faces: dict[tuple[int, ...], tuple[int, ...]] = {}
    for element in elements[selected]:
        for local_face in local_faces:
            face = tuple(int(element[index]) for index in local_face)
            if np.allclose(nodes[np.asarray(face), axis], coordinate, rtol=0.0, atol=1.0e-12):
                faces.setdefault(tuple(sorted(face)), face)
    if not faces:
        raise ValueError("Requested coordinate plane contains no complete Hex8 faces.")
    triangles: list[tuple[int, int, int]] = []
    for face in faces.values():
        triangles.extend(((face[0], face[1], face[2]), (face[0], face[2], face[3])))
    return np.asarray(triangles, dtype=np.int64)


def _localize_triangles(
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Map global section triangles to a compact phase-local node array."""
    node_ids = np.unique(triangles)
    local_ids = np.full(int(np.max(node_ids)) + 1, -1, dtype=np.int64)
    local_ids[node_ids] = np.arange(node_ids.size, dtype=np.int64)
    return node_ids, local_ids[triangles]


def _shared_triangle_boundary_edges(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
) -> tuple[tuple[int, int], ...]:
    """Return geometric section edges shared by two phase triangulations."""
    def edges(triangles: NDArray[np.int64]) -> set[tuple[int, int]]:
        return {
            (
                min(int(triangle[a]), int(triangle[b])),
                max(int(triangle[a]), int(triangle[b])),
            )
            for triangle in triangles
            for a, b in ((0, 1), (1, 2), (2, 0))
        }

    return tuple(sorted(edges(first) & edges(second)))


def _quadrature_coordinates(bundle: Any) -> NDArray[np.float64]:
    """Evaluate macro quadrature coordinates in accepted assembly ordering."""
    coordinates = []
    for connectivity in bundle.mesh_info.elements:
        element_coordinates = bundle.mesh_info.nodes[connectivity]
        coordinates.extend(bundle.shape_function_cache.shape_values @ element_coordinates)
    return np.asarray(coordinates, dtype=np.float64)


def _near_fixed_end_equivalent_stress(
    bundle: Any,
    solution: Any,
) -> NDArray[np.float64]:
    """Average equivalent stress over quadrature points nearest the fixed end."""
    coordinates = []
    for connectivity in bundle.mesh_info.elements:
        element_coordinates = bundle.mesh_info.nodes[connectivity]
        coordinates.extend(bundle.shape_function_cache.shape_values @ element_coordinates)
    quadrature_coordinates = np.asarray(coordinates, dtype=np.float64)
    axial_minimum = float(np.min(bundle.mesh_info.nodes[:, 0]))
    axial_maximum = float(np.max(bundle.mesh_info.nodes[:, 0]))
    monitor_limit = axial_minimum + 0.125 * (axial_maximum - axial_minimum)
    monitor_ids = np.flatnonzero(quadrature_coordinates[:, 0] <= monitor_limit)
    if monitor_ids.size == 0:
        raise ValueError("Report FE2 impact fixed-end monitor region contains no points.")
    stresses = solution.stresses[:, monitor_ids, :]
    mean = np.mean(stresses[..., :3], axis=2)
    deviator = stresses[..., :3] - mean[..., None]
    squared = np.sum(deviator**2, axis=2) + 2.0 * np.sum(
        stresses[..., 3:] ** 2,
        axis=2,
    )
    return np.mean(np.sqrt(1.5 * squared), axis=1)


def _wave_arrival_step(equivalent_stress: NDArray[np.float64]) -> int:
    """Return the first accepted step reaching ten percent of peak stress."""
    threshold = 0.1 * float(np.max(equivalent_stress))
    indices = np.flatnonzero(equivalent_stress >= threshold)
    if indices.size == 0:
        raise ValueError("Report FE2 impact has no measurable stress-wave arrival.")
    return int(indices[0])
