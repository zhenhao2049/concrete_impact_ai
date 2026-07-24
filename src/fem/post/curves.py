"""Response curve output helpers.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_response_csv(data: Mapping[str, object], output_path: str | Path) -> Path:
    """Write response curve data to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)

    return path


def write_response_plot(
    data: Mapping[str, object],
    output_path: str | Path,
    x_name: str,
    y_pairs: tuple[tuple[str, str], ...],
) -> Path:
    """Write numerical and analytical response curves to a PNG file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(len(y_pairs), 1, figsize=(7.0, 3.0 * len(y_pairs)))
    axes_tuple = (axes,) if len(y_pairs) == 1 else tuple(axes)

    for axis, pair in zip(axes_tuple, y_pairs, strict=True):
        numerical_name, exact_name = pair
        axis.plot(data[x_name], data[numerical_name], label=numerical_name)
        axis.plot(data[x_name], data[exact_name], "--", label=exact_name)
        axis.set_xlabel(x_name)
        axis.set_ylabel(numerical_name)
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

    return path
