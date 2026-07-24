"""PyTorch inference backends for material-point surrogate artifacts.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from fem.materials.data import (
    DEFAULT_RESPONSE_REQUIREMENTS,
    MaterialPointRequest,
    MaterialPointResponse,
    MaterialResponseRequirements,
    MaterialState,
)
from fem.surrogates.data import SurrogateArtifactMetadata
from fem.surrogates.errors import SurrogateCapabilityError, SurrogateConfigError
from fem.surrogates.metadata import load_surrogate_metadata
from fem.surrogates.registry import register_surrogate_backend


class TorchExportMaterialSurrogate:
    """Deploy a stress-and-state material surrogate exported by PyTorch."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        density: float,
    ) -> None:
        """Load a torch.export material surrogate on CPU with float64 inputs."""
        import torch

        self.metadata = load_surrogate_metadata(metadata_path)
        _validate_inference_metadata(self.metadata, "torch_export")
        self.name = self.metadata.model_name
        self.density = density
        self.model = torch.export.load(str(model_path)).module()

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize every declared surrogate state field to zero."""
        return _initialize_metadata_state(self.metadata, n_points)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Evaluate one stress-and-state inference batch."""
        return _evaluate_torch_model(self.model, self.metadata, request, state, requirements)


class TorchStateDictMaterialSurrogate:
    """Deploy a stress-and-state material surrogate from a PyTorch state dict."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        density: float,
        model_builder: Callable[[], Any],
    ) -> None:
        """Load a state-dict material surrogate on CPU with float64 parameters."""
        import torch

        self.metadata = load_surrogate_metadata(metadata_path)
        _validate_inference_metadata(self.metadata, "torch_state_dict")
        self.name = self.metadata.model_name
        self.density = density
        self.model = model_builder().to(dtype=torch.float64, device="cpu")
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def initialize_state(self, n_points: int) -> MaterialState:
        """Initialize every declared surrogate state field to zero."""
        return _initialize_metadata_state(self.metadata, n_points)

    def update(
        self,
        request: MaterialPointRequest,
        state: MaterialState,
        requirements: MaterialResponseRequirements = DEFAULT_RESPONSE_REQUIREMENTS,
    ) -> MaterialPointResponse:
        """Evaluate one stress-and-state inference batch."""
        return _evaluate_torch_model(self.model, self.metadata, request, state, requirements)


def export_torch_material_surrogate(
    model: Any,
    metadata: SurrogateArtifactMetadata,
    request: MaterialPointRequest,
    state: MaterialState,
    output_path: str | Path,
) -> Path:
    """Export a PyTorch model using the semantic metadata input contract."""
    import torch

    model_input = _build_model_input(metadata, request, state)
    exported_program = torch.export.export(model, (model_input,))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, path)

    return path


def _evaluate_torch_model(
    model: Any,
    metadata: SurrogateArtifactMetadata,
    request: MaterialPointRequest,
    state: MaterialState,
    requirements: MaterialResponseRequirements,
) -> MaterialPointResponse:
    """Evaluate a pure inference model after strict capability checks."""
    import torch

    _check_request(metadata, request, requirements)
    model_input = _build_model_input(metadata, request, state)
    with torch.no_grad():
        model_output = model(model_input)
    output = model_output.detach().cpu().numpy()

    return _unpack_model_output(metadata, output)


def _check_request(
    metadata: SurrogateArtifactMetadata,
    request: MaterialPointRequest,
    requirements: MaterialResponseRequirements,
) -> None:
    """Check kinematics and every requested optional output."""
    if request.kinematics not in metadata.kinematics:
        raise SurrogateConfigError(
            f"Surrogate {metadata.model_name} does not support {request.kinematics}."
        )
    capabilities = metadata.capabilities
    unsupported = []
    if requirements.tangent and not capabilities.provides_tangent:
        unsupported.append("tangent")
    if requirements.free_energy and not capabilities.provides_free_energy:
        unsupported.append("free_energy")
    if requirements.dissipation and not capabilities.provides_dissipation:
        unsupported.append("dissipation")
    if unsupported:
        raise SurrogateCapabilityError(
            f"Surrogate {metadata.model_name} cannot provide: {', '.join(unsupported)}."
        )


def _build_model_input(
    metadata: SurrogateArtifactMetadata,
    request: MaterialPointRequest,
    state: MaterialState,
):
    """Build a model tensor from explicitly ordered semantic fields."""
    import torch

    n_points = request.strains.shape[0]
    fields = []
    for field_spec in metadata.input_fields:
        if field_spec.name == "strain":
            values = request.strains
        elif field_spec.name == "strain_rate":
            if request.strain_rates is None:
                raise SurrogateConfigError("The surrogate requires strain_rate input.")
            values = request.strain_rates
        elif field_spec.name == "time_step":
            values = np.full((n_points, 1), request.time_step, dtype=np.float64)
        elif field_spec.name.startswith("state."):
            state_name = field_spec.name.removeprefix("state.")
            try:
                values = state.variables[state_name]
            except KeyError as error:
                raise SurrogateConfigError(
                    f"Missing surrogate state field: {state_name}."
                ) from error
            if values.ndim == 1:
                values = values[:, None]
        else:
            raise SurrogateConfigError(f"Unknown surrogate input field: {field_spec.name}.")
        if values.shape != (n_points, field_spec.width):
            raise SurrogateConfigError(
                f"Surrogate field {field_spec.name} expected shape "
                f"{(n_points, field_spec.width)}, received {values.shape}."
            )
        fields.append(torch.as_tensor(values, dtype=torch.float64, device="cpu"))

    return torch.cat(fields, dim=1)


def _unpack_model_output(
    metadata: SurrogateArtifactMetadata,
    output: np.ndarray,
) -> MaterialPointResponse:
    """Unpack stress and updated state fields without fabricating optional outputs."""
    expected_width = sum(field.width for field in metadata.output_fields) + sum(
        field.width for field in metadata.state_fields
    )
    if output.ndim != 2 or output.shape[1] != expected_width:
        raise SurrogateConfigError(
            f"Surrogate output expected width {expected_width}, received shape {output.shape}."
        )
    if len(metadata.output_fields) != 1 or metadata.output_fields[0].name != "stress":
        raise SurrogateConfigError("Initial material backend requires one stress output field.")
    stress_width = metadata.output_fields[0].width
    stresses = output[:, :stress_width]
    offset = stress_width
    variables: dict[str, np.ndarray] = {}
    for field_spec in metadata.state_fields:
        values = output[:, offset : offset + field_spec.width]
        variables[field_spec.name] = values[:, 0] if field_spec.width == 1 else values
        offset += field_spec.width

    return MaterialPointResponse(stresses=stresses, state=MaterialState(variables=variables))


def _initialize_metadata_state(
    metadata: SurrogateArtifactMetadata,
    n_points: int,
) -> MaterialState:
    """Initialize all metadata-declared state tensors."""
    variables = {}
    for field_spec in metadata.state_fields:
        shape = (n_points,) if field_spec.width == 1 else (n_points, field_spec.width)
        variables[field_spec.name] = np.zeros(shape, dtype=np.float64)

    return MaterialState(variables=variables)


def _validate_inference_metadata(
    metadata: SurrogateArtifactMetadata,
    artifact_format: str,
) -> None:
    """Require an inference-only material artifact contract."""
    if metadata.kind != "material_point":
        raise SurrogateConfigError("PyTorch material backend requires kind=material_point.")
    if metadata.backend != "pytorch" or metadata.artifact_format != artifact_format:
        raise SurrogateConfigError("Surrogate backend metadata does not match the loader.")
    capabilities = metadata.capabilities
    if (
        capabilities.provides_tangent
        or capabilities.provides_free_energy
        or capabilities.provides_dissipation
    ):
        raise SurrogateConfigError(
            "Initial PyTorch artifact wrapper supports stress-and-state inference only."
        )


register_surrogate_backend("pytorch_export", TorchExportMaterialSurrogate)
register_surrogate_backend("pytorch_state_dict", TorchStateDictMaterialSurrogate)
