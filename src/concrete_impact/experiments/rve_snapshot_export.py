"""Deterministic VTK export for compact RVE microscopic snapshots.

Contents:
    Quadrature coordinates, snapshot field extraction, and point-cloud VTK files.
Author:
    Zhen Hao.
Created:
    2026-07-16.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fem.elements.lagrange import evaluate_lagrange_shape_functions
from fem.post.vtk import write_named_point_cloud_vtk
from fem.quadrature.rules import make_quadrature_rule


def export_rve_snapshots(
    shard_path: str | Path,
    path_id: str,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Export every stored microscopic event snapshot in deterministic name order."""
    shard = Path(shard_path)
    output = Path(output_directory)
    with h5py.File(shard, "r") as handle:
        if handle.attrs["status"] != "complete":
            raise ValueError(f"RVE snapshot shard is incomplete: {shard}.")
        path = handle[f"paths/{path_id}"]
        if "micro_snapshots" not in path:
            raise ValueError(
                "RVE path has no stored snapshots and must be replayed from its frozen manifest: "
                f"{path_id}."
            )
        model_id = str(handle.attrs["model_id"])
        mesh = handle[f"models/{model_id}/mesh"]
        coordinates = _quadrature_coordinates(mesh["nodes"][...], mesh["elements"][...])
        exports = []
        for snapshot_name in sorted(path["micro_snapshots"]):
            snapshot = path[f"micro_snapshots/{snapshot_name}"]
            fields = {
                "strain": snapshot["strain"][...],
                "stress": snapshot["stress"][...],
            }
            for group_name in ("state", "diagnostics"):
                for field_name, dataset in snapshot[group_name].items():
                    values = dataset[...]
                    if values.ndim == 1 or (
                        values.ndim == 2 and values.shape[0] == coordinates.shape[0]
                    ):
                        fields[f"{group_name}__{field_name}"] = values
            export_path = output / path_id / f"{snapshot_name}.vtu"
            exports.append(
                write_named_point_cloud_vtk(export_path, coordinates, fields)
            )
    return tuple(exports)


def _quadrature_coordinates(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Compute physical coordinates of standard order-two Hex8 quadrature points."""
    quadrature_points, _ = make_quadrature_rule("hex8", 2)
    shape, _ = evaluate_lagrange_shape_functions("hex8", quadrature_points)
    return np.einsum("qi,eij->eqj", shape, nodes[elements]).reshape(-1, 3)
