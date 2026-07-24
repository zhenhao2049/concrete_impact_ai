"""Tests for fixed-cell energy-dissipation RVE-RNO training.

Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from concrete_impact.cli.smoke_rve_rno import _training_step_smoke
from concrete_impact.cli.train_rve_rno import main as train_rve_rno_main
from concrete_impact.nn.config import RVERNOModelConfig, RVERNOTrainingConfig
from concrete_impact.nn.datasets.rve import RNOPathBatch
from concrete_impact.nn.deployment import RVERNOMaterialAdapter, load_rve_rno_state_dict
from concrete_impact.nn.models.rve_rno import EnergyDissipationRVERNO, ScalarMLP
from concrete_impact.nn.training.rve_rno import (
    TrainingNumericalError,
    _batch_loss,
    _canonical_state_dict,
    _capture_training_warnings,
    _compile_model_networks,
    _load_canonical_state_dict,
    _require_finite_gradients,
    _require_finite_parameters,
    _synchronize_batch_scalars,
    benchmark_rve_rno_batching,
    train_rve_rno,
)
from fem.materials import (
    MaterialPointRequest,
    MaterialResponseRequirements,
    MaterialUpdateSettings,
)


@pytest.mark.parametrize("evolution", ["direct_rate", "mobility_gradient"])
def test_rve_rno_batch_ordering_anchor_gradient_and_tangent(evolution: str) -> None:
    """Verify zero anchoring, batch shapes, scalar equivalence, and parameter gradients."""
    model = EnergyDissipationRVERNO(_model_config(evolution))
    strain = torch.tensor(
        [
            [0.01, -0.002, 0.001, 0.003, 0.0, -0.001],
            [0.0, 0.004, -0.001, 0.0, 0.002, 0.0],
        ],
        dtype=torch.float64,
    )
    latent = model.initial_state_batch(2)
    anchors = model.prepare_energy_anchors()
    update = model.update_batch(
        strain,
        latent,
        torch.full((2,), 1.0e-3, dtype=torch.float64),
        anchors,
        compute_tangent=True,
    )
    scalar = model.update(
        strain[0],
        model.initial_state(),
        torch.tensor(1.0e-3, dtype=torch.float64),
        compute_tangent=True,
    )
    zero = model.update_batch(
        torch.zeros((2, 6), dtype=torch.float64),
        model.initial_state_batch(2),
        torch.full((2,), 1.0e-3, dtype=torch.float64),
        model.prepare_energy_anchors(),
    )
    assert torch.max(torch.abs(zero.stress)).item() < 1.0e-14
    assert torch.allclose(update.stress[0], scalar.stress, atol=1.0e-13, rtol=1.0e-13)
    assert torch.allclose(update.latent_state[0], scalar.latent_state, atol=1.0e-13, rtol=1.0e-13)
    expected_increment = (
        1.0e-3 / model.config.reference_time_scale * update.dimensionless_state_rate
    )
    assert torch.allclose(
        update.latent_state - latent,
        expected_increment,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    assert torch.allclose(
        update.state_rate,
        update.dimensionless_state_rate / model.config.reference_time_scale,
    )
    assert update.tangent is not None and update.tangent.shape == (2, 6, 6)
    assert torch.allclose(update.thermodynamic_violation, torch.relu(-update.dissipation))
    if evolution == "mobility_gradient":
        assert torch.all(update.dissipation >= 0.0)
        assert torch.count_nonzero(update.thermodynamic_violation) == 0
    loss = (
        update.stress.square().mean()
        + update.latent_state.square().mean()
        + update.dissipation.square().mean()
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_current_state_stress_uses_updated_latent_and_algorithmic_tangent() -> None:
    """Verify the Liu ordering and its full finite-step strain derivative."""
    model = EnergyDissipationRVERNO(
        _model_config("direct_rate", "current_state_stress_forward_euler")
    )
    strain = torch.tensor(
        [[0.012, -0.003, 0.001, 0.002, -0.001, 0.004]],
        dtype=torch.float64,
    )
    previous_latent = model.initial_state_batch(1)
    time_step = torch.tensor([2.0e-3], dtype=torch.float64)
    update = model.update_batch(
        strain,
        previous_latent,
        time_step,
        model.prepare_energy_anchors(),
        compute_tangent=True,
    )
    zero_increment = model.update_batch(
        strain,
        previous_latent,
        torch.zeros_like(time_step),
        model.prepare_energy_anchors(),
    )
    assert not torch.allclose(update.stress, zero_increment.stress, atol=1.0e-12, rtol=0.0)

    fixed_strain = strain.detach().requires_grad_(True)
    fixed_latent = update.latent_state.detach().requires_grad_(True)
    fixed_anchors = model.prepare_energy_anchors()
    fixed_energy = model._anchored_equilibrium(fixed_strain, fixed_anchors)
    fixed_energy = fixed_energy + model._anchored_nonequilibrium(
        torch.cat((fixed_strain, fixed_latent), dim=-1),
        fixed_strain,
        fixed_anchors,
    )
    fixed_stress = torch.autograd.grad(fixed_energy.sum(), fixed_strain)[0]
    assert torch.allclose(update.stress, fixed_stress, atol=1.0e-12, rtol=1.0e-12)

    perturbation = 1.0e-6
    columns = []
    for component in range(6):
        direction = torch.zeros_like(strain)
        direction[0, component] = perturbation
        plus = model.update_batch(
            strain + direction,
            previous_latent,
            time_step,
            model.prepare_energy_anchors(),
        )
        minus = model.update_batch(
            strain - direction,
            previous_latent,
            time_step,
            model.prepare_energy_anchors(),
        )
        columns.append((plus.stress[0] - minus.stress[0]) / (2.0 * perturbation))
    finite_difference = torch.stack(columns, dim=1)
    assert update.tangent is not None
    assert torch.allclose(update.tangent[0], finite_difference, atol=2.0e-8, rtol=2.0e-6)


def test_zero_preserving_strain_scale_returns_physical_algorithmic_tangent() -> None:
    """Verify scaled energy inputs retain zero stress and physical-strain derivatives."""
    config = _model_config(
        "direct_rate",
        "current_state_stress_forward_euler",
    ).model_copy(
        update={"strain_input_scale": (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)}
    )
    model = EnergyDissipationRVERNO(config)
    strain = torch.tensor(
        [[0.004, -0.003, 0.002, 0.001, -0.002, 0.003]],
        dtype=torch.float64,
    )
    previous_latent = model.initial_state_batch(1)
    time_step = torch.tensor([1.0e-3], dtype=torch.float64)
    zero = model.update_batch(
        torch.zeros_like(strain),
        model.initial_state_batch(1),
        time_step,
        model.prepare_energy_anchors(),
    )
    update = model.update_batch(
        strain,
        previous_latent,
        time_step,
        model.prepare_energy_anchors(),
        compute_tangent=True,
    )

    perturbation = 1.0e-7
    columns = []
    for component in range(6):
        direction = torch.zeros_like(strain)
        direction[0, component] = perturbation
        plus = model.update_batch(
            strain + direction,
            previous_latent,
            time_step,
            model.prepare_energy_anchors(),
        )
        minus = model.update_batch(
            strain - direction,
            previous_latent,
            time_step,
            model.prepare_energy_anchors(),
        )
        columns.append((plus.stress[0] - minus.stress[0]) / (2.0 * perturbation))
    finite_difference = torch.stack(columns, dim=1)

    assert torch.max(torch.abs(zero.stress)).item() < 1.0e-12
    assert update.tangent is not None
    assert torch.allclose(update.tangent[0], finite_difference, atol=2.0e-6, rtol=2.0e-6)


def test_component_balanced_stress_loss_weights_six_components_equally(
    tmp_path: Path,
) -> None:
    """Verify componentwise scale floors prevent normal stress from hiding shear error."""

    class FixedStressRollout(torch.nn.Module):
        """Return one prescribed stress history and zero auxiliary quantities."""

        def forward(
            self,
            strain: torch.Tensor,
            time_step: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            """Build the seven histories required by the RVE-RNO loss contract."""
            del time_step
            stress = torch.zeros_like(strain)
            stress[..., 0] = 10.0
            stress[..., 1] = 1.0
            scalar = torch.zeros(mask.shape, dtype=strain.dtype)
            return stress, scalar, scalar, scalar, scalar, scalar, scalar

    config = _write_tiny_training_case(tmp_path)
    config = config.model_copy(
        update={
            "loss": config.loss.model_copy(
                update={
                    "stress": "component_balanced_path_normalized_mse",
                    "thermodynamic_violation_weight": 0.0,
                    "dissipation_supervision_weight": 0.0,
                }
            )
        }
    )
    batch = RNOPathBatch(
        path_ids=("path_0",),
        time=torch.ones((1, 1), dtype=torch.float64),
        time_step=torch.ones((1, 1), dtype=torch.float64),
        network_inputs=torch.zeros((1, 1, 6), dtype=torch.float64),
        macro_stress=torch.zeros((1, 1, 6), dtype=torch.float64),
        dissipation_density=torch.zeros((1, 1), dtype=torch.float64),
        valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    normalization = {
        "macro_stress": {"rms_scale": [10.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        "dissipation_density": {"rms_scale": [1.0]},
    }
    loss, components, _, _ = _batch_loss(
        FixedStressRollout(),
        batch,
        config,
        normalization,
    )

    assert torch.allclose(components[0], torch.tensor(1.0 / 3.0, dtype=torch.float64))
    assert torch.allclose(loss, components[0])


def test_scalar_energy_explicit_input_gradient_matches_autograd() -> None:
    """Verify explicit SiLU chain differentiation and mixed parameter gradients."""
    torch.manual_seed(72000)
    energy = ScalarMLP(input_size=5, hidden_size=8, hidden_layers=3).to(torch.float64)
    values = torch.randn((4, 5), dtype=torch.float64, requires_grad=True)
    reference_value = energy(values)
    reference_gradient = torch.autograd.grad(reference_value.sum(), values, create_graph=True)[0]
    explicit_value, explicit_gradient = energy.value_and_input_gradient(values)

    assert torch.allclose(explicit_value, reference_value, atol=1.0e-14, rtol=1.0e-14)
    assert torch.allclose(explicit_gradient, reference_gradient, atol=2.0e-14, rtol=2.0e-13)
    explicit_gradient.square().mean().backward()
    assert all(parameter.grad is not None for parameter in energy.parameters())


def test_scalar_reference_and_path_batch_rollout_are_numerically_equivalent() -> None:
    """Compare batched and scalar execution of the identical explicit algorithm."""
    model = EnergyDissipationRVERNO(_model_config("mobility_gradient"))
    batch = RNOPathBatch(
        path_ids=("path_0", "path_1"),
        time=torch.arange(1, 9, dtype=torch.float64).repeat(2, 1) * 1.0e-3,
        time_step=torch.full((2, 8), 1.0e-3, dtype=torch.float64),
        network_inputs=torch.randn((2, 8, 6), dtype=torch.float64) * 1.0e-3,
        macro_stress=torch.zeros((2, 8, 6), dtype=torch.float64),
        dissipation_density=torch.zeros((2, 8), dtype=torch.float64),
        valid_mask=torch.ones((2, 8), dtype=torch.bool),
    )
    report = benchmark_rve_rno_batching(model, batch)
    assert report["accuracy_equivalent"] is True
    assert report["maximum_absolute_difference"] <= report["equivalence_tolerance"]


def test_training_step_smoke_uses_current_batch_loss_contract(tmp_path: Path) -> None:
    """Run one complete CPU smoke step through the four-value batch-loss interface."""
    config = _write_tiny_training_case(tmp_path)
    model = EnergyDissipationRVERNO(config.model)
    batch = RNOPathBatch(
        path_ids=("path_0", "path_1"),
        time=torch.arange(1, 9, dtype=torch.float64).repeat(2, 1) * 1.0e-3,
        time_step=torch.full((2, 8), 1.0e-3, dtype=torch.float64),
        network_inputs=torch.randn((2, 8, 6), dtype=torch.float64) * 1.0e-3,
        macro_stress=torch.zeros((2, 8, 6), dtype=torch.float64),
        dissipation_density=torch.zeros((2, 8), dtype=torch.float64),
        valid_mask=torch.ones((2, 8), dtype=torch.bool),
    )
    normalization = {
        "macro_stress": {"rms_scale": [1.0] * 6},
        "dissipation_density": {"rms_scale": [1.0]},
    }

    report = _training_step_smoke(model, batch, config, normalization, 1)

    assert report["all_gradients_finite"] is True
    assert np.isfinite(report["loss"])
    assert report["valid_material_steps"] == 16


def test_cpu_tiny_training_writes_logs_and_restorable_checkpoint(tmp_path: Path) -> None:
    """Run four eight-increment paths through two CPU training epochs."""
    config = _write_tiny_training_case(tmp_path)
    summary = train_rve_rno(config)
    output = config.output_directory
    assert summary["passed"] is True
    assert summary["selection_split"] == "validation"
    assert summary["frozen_test_evaluation_count"] == 1
    for relative in (
        "train.log",
        "metrics.jsonl",
        "state_diagnostics.jsonl",
        "warnings.jsonl",
        "progress.json",
        "summary.log",
        "resolved_config.yaml",
        "run_metadata.json",
        "normalization.json",
        "best_model.pt",
        "artifact_metadata.json",
        "performance_report.json",
        "training_summary.json",
        "checkpoints/epoch_00001.pt",
        "checkpoints/epoch_00002.pt",
    ):
        assert (output / relative).is_file()
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    restored = EnergyDissipationRVERNO(config.model)
    restored.load_state_dict(checkpoint["model_state_dict"])
    deployed = load_rve_rno_state_dict(
        config.model,
        output / "best_model.pt",
        output / "artifact_metadata.json",
    )
    assert deployed.training is False
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    artifact = json.loads((output / "artifact_metadata.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "rve_rno_training_complete"
    assert progress["completed"] == config.epochs
    assert progress["total"] == config.epochs
    assert artifact["acceptance_scope"] == "strict"
    summary_log = (output / "summary.log").read_text(encoding="utf-8")
    assert summary_log.count('stage="rve_rno_epoch_complete"') == config.epochs
    assert 'stage="rve_rno_training_complete"' in summary_log


def test_non_finite_training_values_fail_with_json_safe_context() -> None:
    """Reject non-finite losses, gradients, and parameters at their first check."""
    context = {
        "epoch": 3,
        "split": "train",
        "batch_index": 4,
        "path_ids": ["shard.h5:path_7"],
        "learning_rate": 1.0e-3,
    }
    with pytest.raises(TrainingNumericalError) as loss_error:
        _synchronize_batch_scalars(
            torch.tensor(float("nan")),
            tuple(torch.tensor(value) for value in (1.0, 2.0, 3.0, 4.0, 5.0)),
            tuple(torch.tensor(0.0) for _ in range(9)),
            context,
        )
    assert loss_error.value.reason == "non_finite_loss"
    assert loss_error.value.diagnostics["losses"]["total_loss"] == "nan"
    json.dumps(loss_error.value.diagnostics, allow_nan=False)

    model = EnergyDissipationRVERNO(_model_config())
    parameter_name, parameter = next(model.named_parameters())
    parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(TrainingNumericalError) as gradient_error:
        _require_finite_gradients(model, context)
    assert gradient_error.value.reason == "non_finite_gradient"
    assert gradient_error.value.diagnostics["parameter_name"] == parameter_name

    parameter.grad = None
    with torch.no_grad():
        parameter.flatten()[0] = float("-inf")
    with pytest.raises(TrainingNumericalError) as parameter_error:
        _require_finite_parameters(model, context)
    assert parameter_error.value.reason == "non_finite_parameter"
    assert parameter_error.value.diagnostics["parameter_name"] == parameter_name


def test_version_two_config_rejects_legacy_newton_fields() -> None:
    """Reject the removed implicit local-solver configuration instead of ignoring it."""
    payload = _model_config().model_dump(mode="json")
    payload["newton"] = {
        "max_iterations": 25,
        "residual_absolute_tolerance": 1.0e-12,
        "residual_relative_tolerance": 1.0e-10,
    }
    with pytest.raises(ValidationError, match="newton"):
        RVERNOModelConfig.model_validate(payload)


def test_model_config_requires_six_strain_input_scales() -> None:
    """Reject an incomplete physical-to-network strain scaling contract."""
    payload = _model_config().model_dump(mode="json")
    payload["strain_input_scale"] = [0.01] * 5
    with pytest.raises(ValidationError, match="six values"):
        RVERNOModelConfig.model_validate(payload)


def test_compiled_evolution_exports_canonical_eager_state_dict(tmp_path: Path) -> None:
    """Keep deployment checkpoint names independent of the compile wrapper."""
    config = _write_tiny_training_case(tmp_path)
    compile_config = config.execution.compile.model_copy(update={"enabled": True})
    execution = config.execution.model_copy(update={"compile": compile_config})
    config = config.model_copy(update={"execution": execution})
    model = EnergyDissipationRVERNO(config.model)
    _compile_model_networks(model, config)
    update = model.update_batch(
        torch.randn((2, 6), dtype=torch.float64) * 1.0e-3,
        model.initial_state_batch(2),
        torch.full((2,), 1.0e-3, dtype=torch.float64),
        model.prepare_energy_anchors(),
    )
    (
        update.stress.square().mean()
        + update.latent_state.square().mean()
        + update.dissipation.square().mean()
    ).backward()
    state = _canonical_state_dict(model)
    assert not any("_orig_mod" in name for name in state)
    _load_canonical_state_dict(model, state)
    eager = EnergyDissipationRVERNO(config.model)
    eager.load_state_dict(state, strict=True)


def test_training_warnings_are_structured_with_batch_context(tmp_path: Path) -> None:
    """Write one warning as JSON-lines data and a human-readable training entry."""
    warning_path = tmp_path / "warnings.jsonl"
    warning_path.touch()
    log_path = tmp_path / "train.log"
    logger = logging.getLogger(f"test_training_warning::{tmp_path}")
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logger.addHandler(handler)
    context = {
        "epoch": 2,
        "split": "validation",
        "batch_index": 1,
        "path_ids": ["shard.h5:path_3"],
        "learning_rate": 1.0e-3,
    }
    with _capture_training_warnings(warning_path, logger, context):
        warnings.warn("deterministic warning", RuntimeWarning, stacklevel=1)
    handler.close()
    logger.removeHandler(handler)

    record = json.loads(warning_path.read_text(encoding="utf-8"))
    assert record["category"] == "RuntimeWarning"
    assert record["message"] == "deterministic warning"
    assert record["epoch"] == 2
    assert record["path_ids"] == ["shard.h5:path_3"]
    assert "deterministic warning" in log_path.read_text(encoding="utf-8")


def test_cuda_request_fails_without_cpu_substitution(tmp_path: Path) -> None:
    """Require an explicit failure when CUDA is requested but unavailable."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host.")
    config = _write_tiny_training_case(tmp_path)
    cuda = config.model.model_copy(update={"device": "cuda"})
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        train_rve_rno(config.model_copy(update={"model": cuda}))


def test_training_cli_publishes_root_progress_and_timestamped_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the CLI links background progress to one independent run directory."""
    config = _write_tiny_training_case(tmp_path)
    config_path = tmp_path / "tiny_training.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-train-rve-rno", "--config", str(config_path)],
    )

    assert train_rve_rno_main() == 0

    output_root = config.output_directory
    run_record = json.loads((output_root / "active_run.json").read_text(encoding="utf-8"))
    progress = json.loads((output_root / "progress.json").read_text(encoding="utf-8"))
    run_directory = Path(run_record["run_directory"])
    assert set(run_record) == {"config", "run_directory"}
    assert progress["stage"] == "rve_rno_training_complete"
    assert run_directory.parent == output_root
    assert (run_directory / "training_summary.json").is_file()
    assert (run_directory / "best_model.pt").is_file()
    assert (output_root / "summary.log").is_file()


def test_training_cli_records_cuda_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify an unavailable GPU produces progress and a per-run failure record."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test host.")
    config = _write_tiny_training_case(tmp_path)
    cuda_model = config.model.model_copy(update={"device": "cuda"})
    config = config.model_copy(update={"model": cuda_model})
    config_path = tmp_path / "tiny_cuda_training.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-train-rve-rno", "--config", str(config_path)],
    )

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        train_rve_rno_main()

    output_root = config.output_directory
    run_record = json.loads((output_root / "active_run.json").read_text(encoding="utf-8"))
    progress = json.loads((output_root / "progress.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (Path(run_record["run_directory"]) / "failure.json").read_text(encoding="utf-8")
    )
    assert progress["stage"] == "rve_rno_training_failed"
    assert failure["error_type"] == "RuntimeError"


def test_training_cli_preserves_structured_numerical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist the exact reason and diagnostics from a strict training failure."""
    config = _write_tiny_training_case(tmp_path)
    config_path = tmp_path / "tiny_failure_training.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    diagnostics = {
        "epoch": 2,
        "split": "train",
        "batch_index": 1,
        "path_ids": ["tiny_paths.h5:path_0"],
        "losses": {"total_loss": "nan"},
    }

    def fail_training(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise TrainingNumericalError("non_finite_loss", diagnostics)

    monkeypatch.setattr("concrete_impact.cli.train_rve_rno.train_rve_rno", fail_training)
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-train-rve-rno", "--config", str(config_path)],
    )

    with pytest.raises(TrainingNumericalError, match="non_finite_loss"):
        train_rve_rno_main()

    run_record = json.loads(
        (config.output_directory / "active_run.json").read_text(encoding="utf-8")
    )
    failure = json.loads(
        (Path(run_record["run_directory"]) / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["reason"] == "non_finite_loss"
    assert failure["diagnostics"] == diagnostics


def test_rve_rno_material_adapter_returns_candidate_state_and_tangent() -> None:
    """Verify the deployed macro material interface and optional algorithmic tangent."""
    adapter = RVERNOMaterialAdapter(
        EnergyDissipationRVERNO(_model_config()),
        "tiny_rve_rno",
        density=2500.0,
        maximum_wave_speed=10.0,
    )
    state = adapter.initialize_state(2)
    response = adapter.update(
        MaterialPointRequest(
            strains=np.asarray(
                [
                    [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            strain_rates=None,
            time_step=1.0e-3,
            kinematics="three_dimensional",
            update_settings=MaterialUpdateSettings(1, 1.0e-10, 1.0e-12, 1.0e-10),
        ),
        state,
        MaterialResponseRequirements(tangent=True, dissipation=True),
    )
    assert response.stresses.shape == (2, 6)
    assert response.state.variables["latent_state"].shape == (2, 2)
    assert response.tangents is not None and response.tangents.shape == (2, 6, 6)
    assert response.dissipation is not None
    assert np.all(response.dissipation >= 0.0)


def _model_config(
    evolution: str = "mobility_gradient",
    integrator: str = "cs_semi_implicit_forward_euler",
) -> RVERNOModelConfig:
    """Build the prescribed tiny float64 architecture."""
    return RVERNOModelConfig.model_validate(
        {
            "family": "energy_rve_rno",
            "time_representation": "continuous",
            "integrator": integrator,
            "evolution": evolution,
            "reference_time_scale": 0.02,
            "latent_dimension": 2,
            "hidden_size": 16,
            "hidden_layers": 2,
            "activation": "silu",
            "dtype": "float64",
            "device": "cpu",
        }
    )


def _write_tiny_training_case(tmp_path: Path) -> RVERNOTrainingConfig:
    """Write four deterministic HDF5 paths and hash-consistent acceptance files."""
    shard = tmp_path / "tiny_paths.h5"
    path_ids = _write_tiny_shard(shard)
    manifest = tmp_path / "response_shards.json"
    manifest.write_text(json.dumps({"response_shards": [str(shard)]}), encoding="utf-8")
    mesh_report = tmp_path / "mesh_selection.json"
    mesh_report.write_text(
        json.dumps(
            {
                "passed": True,
                "reason": "tiny_pipeline_check_only",
                "selected_mesh": {
                    "circumferential_divisions": 8,
                    "matrix_radial_divisions": 1,
                    "axial_divisions": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    data_plan = tmp_path / "data_plan.yaml"
    data_plan.write_text(
        yaml.safe_dump(_tiny_data_plan(tmp_path, mesh_report), sort_keys=False),
        encoding="utf-8",
    )
    split = {
        "seed": 1,
        "paths": [
            {
                "path_id": path_id,
                "lineage_id": f"lineage_{index}",
                "source_name": "tiny_fixed_cell",
                "regime": "elastic" if index < 2 else "viscoplastic",
                "family": "monotonic",
                "macro_path_origin": "prescribed_macro_strain",
                "split": ("train", "train", "validation", "test")[index],
            }
            for index, path_id in enumerate(path_ids)
        ],
    }
    normalization = {
        "macro_strain": {"mean": [0.0] * 6, "rms_scale": [0.01] * 6},
        "macro_stress": {"mean": [0.0] * 6, "rms_scale": [1.0] * 6},
        "dissipation_density": {"mean": [0.0], "rms_scale": [0.01]},
    }
    (tmp_path / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    (tmp_path / "normalization.json").write_text(json.dumps(normalization), encoding="utf-8")
    acceptance = {
        "passed": True,
        "data_plan_sha256": _sha256_file(data_plan),
        "response_shard_manifest_sha256": _sha256_file(manifest),
        "shard_sha256": {str(shard): _sha256_file(shard)},
        "split_manifest_sha256": _sha256_payload(split),
        "normalization_sha256": _sha256_payload(normalization),
        "split_manifest": split,
        "normalization": normalization,
    }
    acceptance_path = tmp_path / "training_data_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    return RVERNOTrainingConfig(
        schema_version="2.0",
        model=_model_config(),
        loss={
            "stress": "path_normalized_mse",
            "thermodynamic_violation_weight": 1.0e-3,
            "dissipation_supervision_weight": 1.0e-2,
        },
        execution={
            "preload_to_memory": True,
            "pin_memory": False,
            "non_blocking_transfer": False,
            "num_workers": 0,
            "compile": {
                "enabled": False,
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
            },
        },
        data_plan=data_plan,
        response_shard_manifest=manifest,
        data_acceptance_report=acceptance_path,
        output_directory=tmp_path / "training",
        random_seed=71001,
        epochs=2,
        batch_size=2,
        learning_rate=1.0e-3,
        checkpoint_interval=1,
    )


def _write_tiny_shard(path: Path) -> tuple[str, ...]:
    """Write four eight-increment paths in the compact schema."""
    names = tuple(f"path_{index}" for index in range(4))
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "4.0"
        handle.attrs["status"] = "complete"
        for path_index, name in enumerate(names):
            group = handle.create_group(f"paths/{name}")
            features = group.create_group("features")
            for kind in ("material_parameters", "microstructure_features"):
                features.create_dataset(f"{kind}_names", data=np.asarray([], dtype="S"))
                features.create_dataset(kind, data=np.asarray([], dtype=np.float64))
            macro = group.create_group("macro")
            time = np.arange(1, 9, dtype=np.float64) * 1.0e-3
            direction = np.zeros(6)
            direction[path_index % 6] = 1.0
            strain = time[:, None] * (1.0 + path_index) * direction[None, :]
            stress = 100.0 * strain
            dissipation = 0.01 * np.sum(strain**2, axis=1)
            macro.create_dataset("time", data=time)
            macro.create_dataset("time_step", data=np.full(8, 1.0e-3))
            macro.create_dataset("strain", data=strain)
            macro.create_dataset("stress", data=stress)
            macro.create_dataset("dissipation_density", data=dissipation)
    return tuple(f"{path.name}:{name}" for name in names)


def _tiny_data_plan(tmp_path: Path, mesh_report: Path) -> dict[str, object]:
    """Build the smallest valid fixed-cell plan needed for artifact metadata."""
    return {
        "schema_version": "4.0",
        "purpose": "fixed_cell_rve_training",
        "storage_mode": "compact_streaming",
        "output_directory": str(tmp_path / "unused"),
        "mesh_selection_report": str(mesh_report),
        "parallel": {
            "workers": 1,
            "paths_per_task": 1,
            "shard_path_count": 1,
            "compression": "lzf",
            "thread_count_per_worker": 1,
            "logical_cpu_budget": 1,
        },
        "audit": {
            "q_tolerance": 1.0e-12,
            "plastic_q_threshold": 1.0e-8,
            "dissipation_tolerance": 1.0e-12,
            "positive_cumulative_dissipation_threshold": 1.0e-12,
            "work_tolerance": 1.0e-12,
            "stress_drop_threshold": 1.0e-10,
            "strain_increment_tolerance": 1.0e-14,
            "hold_strain_tolerance": 1.0e-14,
            "hold_stress_relaxation_threshold": 1.0e-10,
            "tangent_perturbation": 1.0e-7,
            "tangent_stress_scale": 10.0,
            "tangent_direction_tolerance": 1.0e-5,
        },
        "sources": [
            {
                "name": "tiny_fixed_cell",
                "kind": "heterogeneous_rve",
                "model": "tiny",
                "boundary": {"kind": "periodic_macro_strain", "description": "tiny"},
                "load_cases": [
                    {
                        "name": "tiny",
                        "path_count": 4,
                        "time_points": 9,
                        "duration": 0.008,
                        "excitation": "prescribed_six_component_strain",
                        "seed": 1,
                        "peak_scale": 0.01,
                        "path_families": ["monotonic"],
                        "expected_response": {
                            "regime": "viscoplastic",
                            "require_unloading": False,
                            "require_reverse_loading": False,
                        },
                    }
                ],
                "fields": {
                    "inputs": ["time", "time_step", "macro_strain"],
                    "labels": ["macro_stress", "dissipation_density"],
                    "state_fields": [],
                    "diagnostics": [],
                },
                "training_role": "homogenization",
                "macro_path_origin": "prescribed_macro_strain",
                "material_parameters": {},
                "microstructure_parameters": {
                    "circumferential_divisions": {"kind": "fixed", "value": 8.0},
                    "matrix_radial_divisions": {"kind": "fixed", "value": 1.0},
                    "axial_divisions": {"kind": "fixed", "value": 1.0},
                },
            }
        ],
    }


def _sha256_file(path: Path) -> str:
    """Hash one complete test artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_payload(payload: object) -> str:
    """Hash one canonical JSON test payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
