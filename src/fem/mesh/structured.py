"""Structured low-order solid mesh builders.

Contents:
    Structured Hex8 boxes, node indexing, and cylindrical O-grid generation.
Author:
    Zhen Hao.
Created:
    2026-07-09.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CylindricalOGridHex8Mesh:
    """Store a conforming matrix-cylinder O-grid and exact periodic topology."""

    nodes: NDArray[np.float64]
    elements: NDArray[np.int64]
    element_phase_ids: NDArray[np.int64]
    periodic_class_ids: NDArray[np.int64]
    opposite_face_nodes: tuple[
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ]
    analytic_inclusion_area: float
    discrete_inclusion_area: float


def build_structured_hex8_box(
    lengths: tuple[float, float, float],
    divisions: tuple[int, int, int],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a tensor-product Hex8 box mesh."""
    x_coordinates = np.linspace(0.0, lengths[0], divisions[0] + 1)
    y_coordinates = np.linspace(0.0, lengths[1], divisions[1] + 1)
    z_coordinates = np.linspace(0.0, lengths[2], divisions[2] + 1)
    nodes = np.zeros(
        ((divisions[0] + 1) * (divisions[1] + 1) * (divisions[2] + 1), 3),
        dtype=np.float64,
    )

    for k, z_value in enumerate(z_coordinates):
        for j, y_value in enumerate(y_coordinates):
            for i, x_value in enumerate(x_coordinates):
                nodes[_structured_node_id(i, j, k, divisions)] = (
                    x_value,
                    y_value,
                    z_value,
                )

    elements = []
    for k in range(divisions[2]):
        for j in range(divisions[1]):
            for i in range(divisions[0]):
                elements.append(
                    [
                        _structured_node_id(i, j, k, divisions),
                        _structured_node_id(i + 1, j, k, divisions),
                        _structured_node_id(i, j + 1, k, divisions),
                        _structured_node_id(i + 1, j + 1, k, divisions),
                        _structured_node_id(i, j, k + 1, divisions),
                        _structured_node_id(i + 1, j, k + 1, divisions),
                        _structured_node_id(i, j + 1, k + 1, divisions),
                        _structured_node_id(i + 1, j + 1, k + 1, divisions),
                    ]
                )

    return nodes, np.asarray(elements, dtype=np.int64)


def _structured_node_id(i: int, j: int, k: int, divisions: tuple[int, int, int]) -> int:
    """Compute the flattened structured-grid node id."""
    return i + (divisions[0] + 1) * (j + (divisions[1] + 1) * k)


def build_cylindrical_ogrid_hex8(
    lengths: tuple[float, float, float],
    inclusion_volume_fraction: float,
    circumferential_divisions: int,
    matrix_radial_divisions: int,
    axial_divisions: int,
) -> CylindricalOGridHex8Mesh:
    """Build a conforming centered z-cylinder O-grid with Hex8 extrusion."""
    if circumferential_divisions % 4 != 0:
        raise ValueError("O-grid circumferential divisions must be divisible by four.")
    if matrix_radial_divisions <= 0 or axial_divisions <= 0:
        raise ValueError("O-grid radial and axial divisions must be positive.")
    lx, ly, lz = lengths
    radius = np.sqrt(inclusion_volume_fraction * lx * ly / np.pi)
    if radius >= 0.5 * min(lx, ly):
        raise ValueError("Cylindrical inclusion intersects the periodic cell boundary.")

    core_divisions = circumferential_divisions // 4
    cross_nodes, cross_quads, cross_phases, outer_keys = _build_ogrid_cross_section(
        lx,
        ly,
        radius,
        core_divisions,
        matrix_radial_divisions,
    )
    cross_count = cross_nodes.shape[0]
    nodes = np.zeros(((axial_divisions + 1) * cross_count, 3), dtype=np.float64)
    for layer in range(axial_divisions + 1):
        start = layer * cross_count
        nodes[start : start + cross_count, :2] = cross_nodes
        nodes[start : start + cross_count, 2] = lz * layer / axial_divisions

    elements: list[list[int]] = []
    phase_ids: list[int] = []
    for layer in range(axial_divisions):
        lower = layer * cross_count
        upper = (layer + 1) * cross_count
        for quad, phase_id in zip(cross_quads, cross_phases, strict=True):
            elements.append(
                [
                    lower + int(quad[0]),
                    lower + int(quad[1]),
                    lower + int(quad[2]),
                    lower + int(quad[3]),
                    upper + int(quad[0]),
                    upper + int(quad[1]),
                    upper + int(quad[2]),
                    upper + int(quad[3]),
                ]
            )
            phase_ids.append(int(phase_id))

    class_ids, face_nodes = _build_ogrid_periodic_topology(
        cross_count,
        axial_divisions,
        outer_keys,
    )
    discrete_area = sum(
        _quad_area(cross_nodes[quad])
        for quad, phase_id in zip(cross_quads, cross_phases, strict=True)
        if phase_id == 1
    )

    return CylindricalOGridHex8Mesh(
        nodes=nodes,
        elements=np.asarray(elements, dtype=np.int64),
        element_phase_ids=np.asarray(phase_ids, dtype=np.int64),
        periodic_class_ids=class_ids,
        opposite_face_nodes=face_nodes,
        analytic_inclusion_area=float(np.pi * radius**2),
        discrete_inclusion_area=float(discrete_area),
    )


def _build_ogrid_cross_section(
    lx: float,
    ly: float,
    radius: float,
    core_divisions: int,
    matrix_radial_divisions: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    dict[int, tuple[int, int]],
]:
    """Build the mapped steel core and square-annulus matrix quadrilaterals."""
    parameter = np.linspace(-1.0, 1.0, core_divisions + 1)
    nodes: list[tuple[float, float]] = []
    core_node_ids = np.zeros((core_divisions + 1, core_divisions + 1), dtype=np.int64)
    for j, eta in enumerate(parameter):
        for i, xi in enumerate(parameter):
            x_value = radius * xi * np.sqrt(1.0 - 0.5 * eta**2)
            y_value = radius * eta * np.sqrt(1.0 - 0.5 * xi**2)
            core_node_ids[i, j] = len(nodes)
            nodes.append((float(x_value), float(y_value)))

    quads: list[list[int]] = []
    phases: list[int] = []
    for j in range(core_divisions):
        for i in range(core_divisions):
            quads.append(
                [
                    int(core_node_ids[i, j]),
                    int(core_node_ids[i + 1, j]),
                    int(core_node_ids[i, j + 1]),
                    int(core_node_ids[i + 1, j + 1]),
                ]
            )
            phases.append(1)

    boundary_keys = _square_boundary_keys(core_divisions)
    previous_ring = [int(core_node_ids[i, j]) for i, j in boundary_keys]
    outer_keys: dict[int, tuple[int, int]] = {}
    for radial_layer in range(1, matrix_radial_divisions + 1):
        fraction = radial_layer / matrix_radial_divisions
        current_ring: list[int] = []
        for key in boundary_keys:
            i, j = key
            xi = parameter[i]
            eta = parameter[j]
            inner = np.asarray(nodes[int(core_node_ids[i, j])])
            outer = np.asarray([0.5 * lx * xi, 0.5 * ly * eta])
            point = (1.0 - fraction) * inner + fraction * outer
            node_id = len(nodes)
            nodes.append((float(point[0]), float(point[1])))
            current_ring.append(node_id)
            if radial_layer == matrix_radial_divisions:
                outer_keys[node_id] = key
        ring_size = len(current_ring)
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            quads.append(
                [
                    previous_ring[next_index],
                    previous_ring[index],
                    current_ring[next_index],
                    current_ring[index],
                ]
            )
            phases.append(0)
        previous_ring = current_ring

    return (
        np.asarray(nodes, dtype=np.float64),
        np.asarray(quads, dtype=np.int64),
        np.asarray(phases, dtype=np.int64),
        outer_keys,
    )


def _square_boundary_keys(divisions: int) -> list[tuple[int, int]]:
    """Return counterclockwise nonduplicated integer keys on a square boundary."""
    keys = [(i, 0) for i in range(divisions)]
    keys.extend((divisions, j) for j in range(divisions))
    keys.extend((i, divisions) for i in range(divisions, 0, -1))
    keys.extend((0, j) for j in range(divisions, 0, -1))
    return keys


def _build_ogrid_periodic_topology(
    cross_count: int,
    axial_divisions: int,
    outer_keys: dict[int, tuple[int, int]],
) -> tuple[
    NDArray[np.int64],
    tuple[
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ],
]:
    """Build exact union-find classes and opposite-face node sets."""
    node_count = (axial_divisions + 1) * cross_count
    parent = np.arange(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    divisions = max(max(key) for key in outer_keys.values())
    x_minus_cross = {j: node for node, (i, j) in outer_keys.items() if i == 0}
    x_plus_cross = {j: node for node, (i, j) in outer_keys.items() if i == divisions}
    y_minus_cross = {i: node for node, (i, j) in outer_keys.items() if j == 0}
    y_plus_cross = {i: node for node, (i, j) in outer_keys.items() if j == divisions}
    x_minus: list[int] = []
    x_plus: list[int] = []
    y_minus: list[int] = []
    y_plus: list[int] = []
    for layer in range(axial_divisions + 1):
        offset = layer * cross_count
        for key in sorted(x_minus_cross):
            minus = offset + x_minus_cross[key]
            plus = offset + x_plus_cross[key]
            union(minus, plus)
            x_minus.append(minus)
            x_plus.append(plus)
        for key in sorted(y_minus_cross):
            minus = offset + y_minus_cross[key]
            plus = offset + y_plus_cross[key]
            union(minus, plus)
            y_minus.append(minus)
            y_plus.append(plus)
    for cross_node in range(cross_count):
        union(cross_node, axial_divisions * cross_count + cross_node)

    roots = np.asarray([find(node) for node in range(node_count)], dtype=np.int64)
    _, class_ids = np.unique(roots, return_inverse=True)
    z_minus = np.arange(cross_count, dtype=np.int64)
    z_plus = axial_divisions * cross_count + z_minus
    face_nodes = (
        (np.asarray(x_minus), np.asarray(x_plus)),
        (np.asarray(y_minus), np.asarray(y_plus)),
        (z_minus, z_plus),
    )
    return class_ids.astype(np.int64), face_nodes


def _quad_area(points: NDArray[np.float64]) -> float:
    """Compute one straight-sided quadrilateral area by two triangles."""
    first_vector = points[1] - points[0]
    second_vector = points[2] - points[0]
    third_vector = points[3] - points[1]
    fourth_vector = points[2] - points[1]
    first = 0.5 * abs(
        first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
    )
    second = 0.5 * abs(
        third_vector[0] * fourth_vector[1] - third_vector[1] * fourth_vector[0]
    )
    return float(first + second)
