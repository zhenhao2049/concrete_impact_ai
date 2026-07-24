"""VTK time-series animation output helpers.

Author:
    Zhen Hao.
Created:
    2026-07-03.
"""

from pathlib import Path


def write_pvd_collection(
    output_path: str | Path,
    vtk_paths: tuple[Path, ...],
    times: tuple[float, ...],
) -> Path:
    """Write a ParaView PVD collection for VTU time series."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]

    for vtk_path, time in zip(vtk_paths, times, strict=True):
        relative_path = vtk_path.relative_to(path.parent)
        lines.append(
            f'    <DataSet timestep="{time:.16e}" group="" part="0" file="{relative_path}"/>'
        )

    lines.extend(["  </Collection>", "</VTKFile>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path
