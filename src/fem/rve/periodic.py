"""Exact periodic constraints for structured Hex8 boxes.

Contents:
    Periodic node pairing and constraint construction for structured RVEs.
Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix

from fem.rve.data import PeriodicConstraint


def build_structured_hex8_periodic_constraint(
    nodes: NDArray[np.float64],
    divisions: tuple[int, int, int],
) -> PeriodicConstraint:
    """Build exact periodic equivalence classes and affine strain mapping."""
    nx, ny, nz = divisions
    expected_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    if nodes.shape != (expected_nodes, 3):
        raise ValueError("Structured periodic constraint received an incompatible node array.")
    if min(divisions) <= 0:
        raise ValueError("Structured periodic constraint requires positive divisions.")

    class_ids = np.zeros(expected_nodes, dtype=np.int64)
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_id = i + (nx + 1) * (j + (ny + 1) * k)
                class_id = (i % nx) + nx * ((j % ny) + ny * (k % nz))
                class_ids[node_id] = class_id

    face_nodes = _structured_face_nodes(divisions)

    return build_periodic_constraint_from_classes(
        nodes,
        class_ids,
        divisions,
        face_nodes,
    )


def build_periodic_constraint_from_classes(
    nodes: NDArray[np.float64],
    class_ids: NDArray[np.int64],
    divisions: tuple[int, int, int],
    opposite_face_nodes: tuple[
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
        tuple[NDArray[np.int64], NDArray[np.int64]],
    ],
) -> PeriodicConstraint:
    """Build elimination operators from exact topology-provided periodic classes."""
    unique_classes, normalized_ids = np.unique(class_ids, return_inverse=True)
    if unique_classes.size == 0:
        raise ValueError("Periodic constraint requires at least one equivalence class.")
    class_ids = normalized_ids.astype(np.int64)
    class_count = unique_classes.size
    representatives = np.full(class_count, -1, dtype=np.int64)
    for node_id, class_id in enumerate(class_ids):
        if representatives[class_id] < 0:
            representatives[class_id] = node_id

    reference_class = int(class_ids[opposite_face_nodes[0][0][0]])
    reduced_class_ids = {
        class_id: reduced_id
        for reduced_id, class_id in enumerate(
            class_id for class_id in range(class_count) if class_id != reference_class
        )
    }
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for node_id, point_class_id in enumerate(class_ids):
        if point_class_id == reference_class:
            continue
        reduced_class = reduced_class_ids[int(point_class_id)]
        for component in range(3):
            rows.append(3 * node_id + component)
            columns.append(3 * reduced_class + component)
            values.append(1.0)
    transformation = coo_matrix(
        (values, (rows, columns)),
        shape=(3 * nodes.shape[0], 3 * (class_count - 1)),
    ).tocsr()

    affine_matrix = _build_affine_matrix(nodes)

    return PeriodicConstraint(
        transformation=transformation,
        affine_matrix=affine_matrix,
        equivalence_class_ids=class_ids,
        representative_nodes=representatives,
        reference_class=reference_class,
        divisions=divisions,
        opposite_face_nodes=opposite_face_nodes,
    )


def _structured_face_nodes(
    divisions: tuple[int, int, int],
) -> tuple[
    tuple[NDArray[np.int64], NDArray[np.int64]],
    tuple[NDArray[np.int64], NDArray[np.int64]],
    tuple[NDArray[np.int64], NDArray[np.int64]],
]:
    """Return exact opposite-face node sets for a structured box."""
    nx, ny, nz = divisions
    face_sets: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for axis, maximum in enumerate(divisions):
        minus: list[int] = []
        plus: list[int] = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    indices = (i, j, k)
                    node_id = i + (nx + 1) * (j + (ny + 1) * k)
                    if indices[axis] == 0:
                        minus.append(node_id)
                    if indices[axis] == maximum:
                        plus.append(node_id)
        face_sets.append(
            (np.asarray(minus, dtype=np.int64), np.asarray(plus, dtype=np.int64))
        )

    return face_sets[0], face_sets[1], face_sets[2]


def compute_periodic_fluctuation_error(
    displacement: NDArray[np.float64],
    macro_strain: NDArray[np.float64],
    constraint: PeriodicConstraint,
) -> float:
    """Compute the maximum mismatch within periodic fluctuation classes."""
    fluctuation = displacement - constraint.affine_matrix @ macro_strain
    nodal_fluctuation = fluctuation.reshape(-1, 3)
    maximum_error = 0.0
    for node_id, class_id in enumerate(constraint.equivalence_class_ids):
        representative = constraint.representative_nodes[class_id]
        mismatch = nodal_fluctuation[node_id] - nodal_fluctuation[representative]
        maximum_error = max(maximum_error, float(np.linalg.norm(mismatch)))

    return maximum_error


def _build_affine_matrix(nodes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the nodal map for engineering strain order xx, yy, zz, yz, xz, xy."""
    relative = nodes - np.min(nodes, axis=0)
    matrix = np.zeros((3 * nodes.shape[0], 6), dtype=np.float64)
    for node_id, (x_value, y_value, z_value) in enumerate(relative):
        row = 3 * node_id
        matrix[row, 0] = x_value
        matrix[row, 4] = 0.5 * z_value
        matrix[row, 5] = 0.5 * y_value
        matrix[row + 1, 1] = y_value
        matrix[row + 1, 3] = 0.5 * z_value
        matrix[row + 1, 5] = 0.5 * x_value
        matrix[row + 2, 2] = z_value
        matrix[row + 2, 3] = 0.5 * y_value
        matrix[row + 2, 4] = 0.5 * x_value

    return matrix
