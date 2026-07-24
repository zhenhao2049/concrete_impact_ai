"""Command-line validation for completed RVE HDF5 v3 shards.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fem.rve import validate_rve_data_shard


def main() -> int:
    """Validate every selected HDF5 v3 response shard."""
    parser = argparse.ArgumentParser(description="Validate RVE HDF5 response shards.")
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    path = Path(args.path)
    files = (path,) if path.is_file() else tuple(sorted(path.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No RVE HDF5 shard found at {path}.")
    for file_path in files:
        validate_rve_data_shard(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
