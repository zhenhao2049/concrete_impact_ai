"""Serialization and validation for surrogate artifact metadata.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fem.surrogates.data import (
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    TensorFieldSpec,
)
from fem.surrogates.errors import SurrogateConfigError


def load_surrogate_metadata(path: str | Path) -> SurrogateArtifactMetadata:
    """Load and validate versioned surrogate metadata from JSON."""
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    metadata = _build_metadata(payload)
    validate_surrogate_metadata(metadata)

    return metadata


def save_surrogate_metadata(metadata: SurrogateArtifactMetadata, path: str | Path) -> Path:
    """Validate and write versioned surrogate metadata to JSON."""
    validate_surrogate_metadata(metadata)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(asdict(metadata), stream, indent=2, sort_keys=True)

    return output_path


def validate_surrogate_metadata(metadata: SurrogateArtifactMetadata) -> None:
    """Validate semantic fields that cannot be inferred by a backend."""
    if metadata.schema_version != "1.0":
        raise SurrogateConfigError(
            f"Unsupported surrogate metadata schema: {metadata.schema_version}."
        )
    if metadata.dtype != "float64":
        raise SurrogateConfigError("Initial surrogate deployment requires float64 metadata.")
    input_names = tuple(field.name for field in metadata.input_fields)
    if len(input_names) != len(set(input_names)):
        raise SurrogateConfigError("Surrogate input field names must be unique.")
    if metadata.capabilities.time_representation == "continuous" and "time_step" in input_names:
        raise SurrogateConfigError("Continuous-time surrogate inputs must not contain time_step.")
    if metadata.capabilities.time_representation == "discrete" and "time_step" not in input_names:
        raise SurrogateConfigError("Discrete-time surrogate inputs must contain time_step.")
    if metadata.capabilities.uses_strain_rate != ("strain_rate" in input_names):
        raise SurrogateConfigError("uses_strain_rate conflicts with the declared input fields.")
    for field_spec in (*metadata.input_fields, *metadata.output_fields, *metadata.state_fields):
        if field_spec.width <= 0:
            raise SurrogateConfigError(
                f"Surrogate tensor field width must be positive: {field_spec.name}."
            )


def _build_metadata(payload: dict[str, Any]) -> SurrogateArtifactMetadata:
    """Build nested metadata dataclasses from one decoded JSON object."""
    data = dict(payload)
    data["input_fields"] = tuple(TensorFieldSpec(**item) for item in data["input_fields"])
    data["output_fields"] = tuple(TensorFieldSpec(**item) for item in data["output_fields"])
    data["state_fields"] = tuple(TensorFieldSpec(**item) for item in data["state_fields"])
    data["capabilities"] = SurrogateCapabilities(**data["capabilities"])
    data["kinematics"] = tuple(data["kinematics"])
    data["normalization"] = {
        field_name: {
            statistic_name: tuple(values)
            for statistic_name, values in statistics.items()
        }
        for field_name, statistics in data.get("normalization", {}).items()
    }

    return SurrogateArtifactMetadata(**data)
