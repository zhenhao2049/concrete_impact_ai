"""Tests for validated multi-model surrogate deployment configuration.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from concrete_impact.nn.config import (
    SurrogateConfigEnvelope,
    build_surrogate_settings,
)
from fem.surrogates import (
    ResidualCorrectionRequest,
    ResidualCorrectionResponse,
    SurrogateConfigError,
    build_surrogate_backend,
)


def test_surrogate_config_forbids_unknown_fields() -> None:
    """Verify deployment YAML cannot silently carry misspelled settings."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurrogateConfigEnvelope.model_validate(
            {"surrogate": {"enabled": False, "backend": "none"}}
        )


def test_enabled_surrogate_config_builds_complete_solver_settings(tmp_path: Path) -> None:
    """Verify a complete artifact selection reaches the solver data boundary."""
    model_path = tmp_path / "model.pt"
    metadata_path = tmp_path / "metadata.json"
    model_path.touch()
    metadata_path.touch()
    envelope = SurrogateConfigEnvelope.model_validate(
        {
            "surrogate": {
                "enabled": True,
                "kind": "material_point",
                "model": "j2_vp_rno",
                "backend": "pytorch",
                "artifact": {
                    "format": "state_dict",
                    "model_path": str(model_path),
                    "metadata_path": str(metadata_path),
                },
                "runtime": {"device": "cpu", "dtype": "float64"},
            }
        }
    )

    settings = build_surrogate_settings(envelope.surrogate)

    assert settings.enabled
    assert settings.kind == "material_point"
    assert settings.model == "j2_vp_rno"
    assert settings.artifact_format == "state_dict"


def test_unknown_surrogate_backend_raises_directly(tmp_path: Path) -> None:
    """Verify registries do not infer or substitute unknown artifact backends."""
    with pytest.raises(SurrogateConfigError, match="Unknown surrogate backend"):
        build_surrogate_backend("unknown", tmp_path / "a", tmp_path / "b")


def test_inc_residual_contract_is_distinct_from_material_state_contract() -> None:
    """Verify INC data represents additive solver-residual corrections."""
    request = ResidualCorrectionRequest(
        base_residual=np.ones(3),
        state_features=np.ones((3, 2)),
        control_parameters=np.ones(1),
    )
    response = ResidualCorrectionResponse(
        residual_correction=-request.base_residual,
        jacobian_correction=None,
    )

    assert np.array_equal(response.residual_correction, -request.base_residual)
    assert response.jacobian_correction is None
