"""Tests for PyTorch AD and typed surrogate interfaces.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from fem.ad import torch_jacobian
from fem.materials.data import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialState,
    MaterialUpdateSettings,
)
from fem.materials.linear_elastic import LinearElasticMaterial, build_elasticity_matrix
from fem.surrogates import (
    SurrogateArtifactMetadata,
    SurrogateCapabilities,
    SurrogateCapabilityError,
    SurrogateConfigError,
    TensorFieldSpec,
    build_surrogate_backend,
    export_torch_material_surrogate,
    save_surrogate_metadata,
    validate_surrogate_metadata,
)


def test_torch_jacobian_matches_linear_elastic_tangent() -> None:
    """Verify PyTorch forward AD against an analytic elastic tangent."""
    material = LinearElasticMaterial(
        name="concrete_linear",
        density=2400.0,
        young_modulus=3.0e10,
        poisson_ratio=0.2,
    )
    tangent = build_elasticity_matrix(material, 2, "plane_strain")
    tangent_torch = torch.as_tensor(tangent, dtype=torch.float64)
    strain = torch.tensor([1.0e-5, -2.0e-5, 0.5e-5], dtype=torch.float64)

    jacobian = torch_jacobian(lambda value: tangent_torch @ value, strain)

    assert np.allclose(jacobian.detach().cpu().numpy(), tangent)


def test_torch_export_material_surrogate_returns_only_declared_outputs(tmp_path: Path) -> None:
    """Verify torch.export returns stress and state without fabricated zeros."""

    class IdentitySurrogate(torch.nn.Module):
        """Return stress and updated-state slices from semantic inputs."""

        def forward(self, model_input):
            """Evaluate the identity test surrogate."""
            return model_input[:, :5]

    request, state = _build_test_material_data()
    metadata = _metadata("torch_export")
    metadata_path = save_surrogate_metadata(metadata, tmp_path / "metadata.json")
    model_path = export_torch_material_surrogate(
        IdentitySurrogate().to(dtype=torch.float64),
        metadata,
        request,
        state,
        tmp_path / "identity_surrogate.pt2",
    )
    surrogate = build_surrogate_backend(
        "pytorch_export",
        model_path,
        metadata_path,
        density=1.0,
    )
    response = surrogate.update(request, state)

    assert response.stresses.shape == (2, 3)
    assert response.state.variables["history"].shape == (2, 2)
    assert response.tangents is None
    assert response.free_energy is None
    assert response.dissipation is None


def test_torch_state_dict_material_surrogate_rejects_unsupported_tangent(
    tmp_path: Path,
) -> None:
    """Verify requested unsupported outputs raise rather than returning zero arrays."""

    class IdentitySurrogate(torch.nn.Module):
        """Return stress and updated-state slices from semantic inputs."""

        def forward(self, model_input):
            """Evaluate the identity test surrogate."""
            return model_input[:, :5]

    request, state = _build_test_material_data()
    metadata = _metadata("torch_state_dict")
    metadata_path = save_surrogate_metadata(metadata, tmp_path / "metadata.json")
    model_path = tmp_path / "identity_surrogate_state.pt"
    torch.save(IdentitySurrogate().state_dict(), model_path)
    surrogate = build_surrogate_backend(
        "pytorch_state_dict",
        model_path,
        metadata_path,
        density=1.0,
        model_builder=IdentitySurrogate,
    )

    with pytest.raises(SurrogateCapabilityError, match="tangent"):
        surrogate.update(request, state, MaterialResponseRequirements(tangent=True))


def test_continuous_surrogate_metadata_rejects_time_step_feature() -> None:
    """Verify continuous-time networks cannot silently consume the integration step."""
    metadata = _metadata("torch_state_dict")
    invalid = replace(
        metadata,
        input_fields=(*metadata.input_fields, TensorFieldSpec("time_step", 1, "s")),
    )

    with pytest.raises(SurrogateConfigError, match="must not contain time_step"):
        validate_surrogate_metadata(invalid)


def _build_test_material_data() -> tuple[MaterialPointRequest, MaterialState]:
    """Build a request and committed state for surrogate tests."""
    request = MaterialPointRequest(
        strains=np.ones((2, 3), dtype=np.float64),
        strain_rates=None,
        time_step=1.0e-6,
        kinematics="plane_strain",
        update_settings=MaterialUpdateSettings(
            max_iterations=50,
            yield_relative_tolerance=1.0e-10,
            residual_absolute_tolerance=1.0e-12,
            residual_relative_tolerance=1.0e-10,
        ),
    )
    state = MaterialState(variables={"history": np.zeros((2, 2), dtype=np.float64)})

    return request, state


def _metadata(artifact_format: str) -> SurrogateArtifactMetadata:
    """Build strict stress-and-state inference metadata."""
    return SurrogateArtifactMetadata(
        schema_version="1.0",
        model_name="identity_test",
        model_family="identity_test",
        kind="material_point",
        backend="pytorch",
        artifact_format=artifact_format,
        dtype="float64",
        device="cpu",
        kinematics=("plane_strain",),
        mandel_convention="engineering_voigt_11_22_12",
        input_fields=(
            TensorFieldSpec("strain", 3, "dimensionless"),
            TensorFieldSpec("state.history", 2, "dimensionless"),
        ),
        output_fields=(TensorFieldSpec("stress", 3, "Pa"),),
        state_fields=(TensorFieldSpec("history", 2, "dimensionless"),),
        capabilities=SurrogateCapabilities(
            provides_tangent=False,
            provides_free_energy=False,
            provides_dissipation=False,
            supports_autodiff_tangent=False,
            uses_strain_rate=False,
            time_representation="continuous",
        ),
    )
