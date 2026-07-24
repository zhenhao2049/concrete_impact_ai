"""Strict RNO training-data acceptance and split generation.

Contents:
    Dataset hashes, state-space coverage, deterministic splits, and acceptance reports.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from concrete_impact.experiments.rve_data_execution import (
    audit_response_shards,
    audit_response_subset,
)
from concrete_impact.experiments.rve_data_plan import load_rve_data_plan
from fem.io.config import load_yaml_config

WEIGHTS = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float64)
COVERAGE_AXIS_NAMES = (
    "strain_norm",
    "strain_rate_norm",
    "equivalent_stress",
    "yield_activation_margin",
    "triaxiality",
    "lode_invariant",
    "turning_angle",
    "equivalent_plastic_strain",
)


class StrictAcceptanceModel(BaseModel):
    """Forbid undeclared training-data acceptance fields."""

    model_config = ConfigDict(extra="forbid")


class SplitSettings(StrictAcceptanceModel):
    """Store deterministic lineage-level split fractions."""

    seed: int = Field(ge=0)
    strategy: Literal["preserve_preassigned", "deterministic_design_stratum"] = (
        "preserve_preassigned"
    )
    train_fraction: PositiveFloat
    validation_fraction: PositiveFloat
    test_fraction: PositiveFloat

    @model_validator(mode="after")
    def require_unit_sum(self) -> SplitSettings:
        """Require fractions that sum to one without renormalization."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-14):
            raise ValueError("RNO split fractions must sum exactly to one within 1e-14.")
        return self


class CoverageBins(StrictAcceptanceModel):
    """Store fixed state-space bin edges and invariant tolerances."""

    strain_norm: tuple[float, ...]
    strain_rate_norm: tuple[float, ...]
    equivalent_stress: tuple[float, ...]
    yield_activation_margin: tuple[float, ...]
    triaxiality: tuple[float, ...]
    lode_invariant: tuple[float, ...]
    turning_angle: tuple[float, ...]
    equivalent_plastic_strain: tuple[float, ...]
    stress_floor: PositiveFloat
    increment_floor: PositiveFloat
    invariant_domain_tolerance: PositiveFloat

    @model_validator(mode="after")
    def require_strict_bin_edges(self) -> CoverageBins:
        """Require every configured coverage axis to have increasing edges."""
        for name in COVERAGE_AXIS_NAMES:
            edges = np.asarray(getattr(self, name), dtype=np.float64)
            if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
                raise ValueError(f"Coverage bin edges must be strictly increasing: {name}.")
        return self


class CoveragePolicy(StrictAcceptanceModel):
    """Declare strict or explicitly limited pilot coverage acceptance."""

    mode: Literal["strict", "pilot"] = "strict"
    allowed_missing_bin_indices: dict[str, tuple[int, ...]] = Field(default_factory=dict)
    limitation_note: str | None = None

    @model_validator(mode="after")
    def require_explicit_pilot_limitations(self) -> CoveragePolicy:
        """Require pilot waivers to remain explicit and forbid them in strict mode."""
        if self.mode == "strict" and self.allowed_missing_bin_indices:
            raise ValueError("Strict coverage policy cannot allow missing bins.")
        if self.mode == "strict" and self.limitation_note is not None:
            raise ValueError("Strict coverage policy cannot declare a pilot limitation note.")
        if self.mode == "pilot" and (
            self.limitation_note is None or not self.limitation_note.strip()
        ):
            raise ValueError("Pilot coverage policy requires a nonempty limitation note.")
        return self


class AcceptanceThresholds(StrictAcceptanceModel):
    """Store fixed acceptance thresholds for data coverage and tangents."""

    fe2_out_of_coverage_fraction: float = Field(ge=0.0, le=1.0)
    tangent_direction_error: PositiveFloat


class JointCoverageRequirements(StrictAcceptanceModel):
    """Define multivariate physical predicates for production RVE coverage."""

    elastic_amplitude_bands: tuple[str, ...] = ()
    transition_amplitude_bands: tuple[str, ...] = ()
    minimum_path_count: PositiveInt
    medium_rate_lower: PositiveFloat
    medium_rate_upper: PositiveFloat
    medium_high_stress_lower: PositiveFloat
    positive_yield_margin: float = Field(ge=0.0)
    positive_plastic_strain: PositiveFloat
    weak_plastic_strain_upper: PositiveFloat
    negative_incremental_work: float = Field(lt=0.0)
    positive_cumulative_dissipation: PositiveFloat
    turning_angle_lower: PositiveFloat
    turning_angle_upper: PositiveFloat
    triaxiality_sign_floor: PositiveFloat
    lode_bin_edges: tuple[float, ...]
    minimum_lode_bins_per_sign: PositiveInt
    elastic_plastic_strain_tolerance: PositiveFloat
    elastic_dissipation_tolerance: PositiveFloat

    @model_validator(mode="after")
    def require_ordered_joint_ranges(self) -> JointCoverageRequirements:
        """Require nonempty joint-coverage intervals and Lode bins."""
        elastic_bands = set(self.elastic_amplitude_bands)
        transition_bands = set(self.transition_amplitude_bands)
        if bool(elastic_bands) != bool(transition_bands):
            raise ValueError(
                "Joint amplitude-band regimes require both elastic and transition bands."
            )
        if elastic_bands & transition_bands:
            raise ValueError("Joint elastic and transition amplitude bands must be disjoint.")
        if self.medium_rate_upper <= self.medium_rate_lower:
            raise ValueError("Joint medium-rate interval requires upper > lower.")
        if self.turning_angle_upper <= self.turning_angle_lower:
            raise ValueError("Joint turning-angle interval requires upper > lower.")
        if self.weak_plastic_strain_upper <= self.positive_plastic_strain:
            raise ValueError("Weak-plastic interval requires upper > activation threshold.")
        edges = np.asarray(self.lode_bin_edges, dtype=np.float64)
        if edges.size < 3 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("Joint Lode bin edges must be strictly increasing.")
        if self.minimum_lode_bins_per_sign > edges.size - 1:
            raise ValueError("Requested joint Lode occupancy exceeds configured bins.")
        return self


class ExternalChecks(StrictAcceptanceModel):
    """Reference independent FEM reports required before RNO training."""

    single_phase_degeneracy_report: Path | None = None
    rve_mesh_selection_report: Path | None = None


class TrainingDataAcceptanceConfig(StrictAcceptanceModel):
    """Store the complete deterministic training-data acceptance request."""

    schema_version: Literal["1.0"]
    data_plan: Path
    response_shard_manifest: Path
    subset_manifest: Path | None = None
    output_directory: Path
    split: SplitSettings
    coverage: CoverageBins
    coverage_policy: CoveragePolicy = Field(default_factory=CoveragePolicy)
    joint_coverage: JointCoverageRequirements | None = None
    thresholds: AcceptanceThresholds
    external_checks: ExternalChecks

    @model_validator(mode="after")
    def require_valid_missing_bin_indices(self) -> TrainingDataAcceptanceConfig:
        """Require every pilot waiver to identify a valid configured bin exactly once."""
        if self.subset_manifest is not None and self.coverage_policy.mode != "pilot":
            raise ValueError("Explicit response subsets are accepted only in pilot mode.")
        for name, indices in self.coverage_policy.allowed_missing_bin_indices.items():
            if name not in COVERAGE_AXIS_NAMES:
                raise ValueError(f"Unknown coverage axis in pilot waiver: {name}.")
            if tuple(sorted(set(indices))) != indices:
                raise ValueError(
                    f"Pilot missing-bin indices must be unique and increasing: {name}."
                )
            bin_count = len(getattr(self.coverage, name)) - 1
            if any(index < 0 or index >= bin_count for index in indices):
                raise ValueError(f"Pilot missing-bin index left configured bins: {name}={indices}.")
        return self


@dataclass(frozen=True)
class PathRecord:
    """Store one accepted HDF5 path and immutable provenance metadata."""

    file_path: Path
    path_name: str
    source_name: str
    source_kind: str
    regime: str
    family: str
    lineage_id: str
    macro_path_origin: str
    preassigned_split: str | None = None
    design_stratum: str | None = None
    amplitude_band: str | None = None
    rate_band: str | None = None
    tensor_direction: str | None = None


class TrainingDataAcceptanceError(RuntimeError):
    """Report training data that do not satisfy configured acceptance criteria."""


def load_training_data_acceptance_config(path: str | Path) -> TrainingDataAcceptanceConfig:
    """Load one strict RNO training-data acceptance configuration."""
    return TrainingDataAcceptanceConfig.model_validate(load_yaml_config(path))


def validate_training_data_acceptance(
    config: TrainingDataAcceptanceConfig,
) -> dict[str, Any]:
    """Run all data, split, normalization, and independent-reference checks."""
    plan = load_rve_data_plan(config.data_plan)
    shard_paths = _load_shard_manifest(config.response_shard_manifest)
    subset_task_ids = _validate_subset_binding(config, shard_paths)
    records = _index_paths(shard_paths)
    checks: dict[str, dict[str, Any]] = {}

    try:
        physical = (
            audit_response_shards(plan, shard_paths)
            if subset_task_ids is None
            else audit_response_subset(plan, shard_paths, subset_task_ids)
        )
        checks["physical_audit"] = {"passed": True, "summary": physical}
    except ValueError as error:
        checks["physical_audit"] = {"passed": False, "failure": str(error)}

    coverage, coverage_passed = _compute_coverage(
        records,
        config.coverage,
        config.coverage_policy,
    )
    checks["state_space_coverage"] = {
        "passed": coverage_passed,
        "acceptance_scope": config.coverage_policy.mode,
        "waived_missing_bin_indices": coverage["waived_missing_bin_indices"],
        "summary": coverage,
    }
    if config.joint_coverage is not None:
        joint_coverage, joint_passed = _compute_joint_coverage(
            records,
            config.coverage,
            config.joint_coverage,
        )
        checks["joint_state_space_coverage"] = {
            "passed": joint_passed,
            "summary": joint_coverage,
        }
    split_manifest, split_passed, split_failures = _build_split_manifest(
        records,
        config.split,
    )
    checks["lineage_split"] = {
        "passed": split_passed,
        "failures": split_failures,
    }
    normalization, normalization_passed, normalization_failures = _compute_normalization(
        records,
        split_manifest,
        {source.name: set(source.fields.inputs) for source in plan.sources},
    )
    checks["training_normalization"] = {
        "passed": normalization_passed,
        "failures": normalization_failures,
    }
    if plan.purpose == "fixed_cell_rve_training":
        fe2_comparison, fe2_passed = ({"status": "not_applicable_to_fixed_cell_data"}, True)
    else:
        fe2_comparison, fe2_passed = _compare_fe2_coverage(
            records,
            config.thresholds.fe2_out_of_coverage_fraction,
        )
    checks["j2_fe2_coverage"] = {"passed": fe2_passed, "summary": fe2_comparison}
    tangent_summary, tangent_passed = _audit_tangent_checks(
        records,
        config.thresholds.tangent_direction_error,
    )
    checks["effective_tangent"] = {
        "passed": tangent_passed,
        "summary": tangent_summary,
    }
    if config.external_checks.single_phase_degeneracy_report is not None:
        checks["single_phase_degeneracy"] = _load_external_pass_report(
            config.external_checks.single_phase_degeneracy_report
        )
    if config.external_checks.rve_mesh_selection_report is not None:
        checks["rve_mesh_selection"] = _load_external_pass_report(
            config.external_checks.rve_mesh_selection_report
        )

    shard_hashes = {str(path): _sha256_file(path) for path in shard_paths}
    split_hash = _sha256_payload(split_manifest)
    normalization_hash = _sha256_payload(normalization)
    passed = all(bool(check["passed"]) for check in checks.values())
    report = {
        "schema_version": config.schema_version,
        "passed": passed,
        "acceptance_scope": config.coverage_policy.mode,
        "coverage_policy": config.coverage_policy.model_dump(mode="json"),
        "checks": checks,
        "data_plan": str(config.data_plan),
        "data_plan_sha256": _sha256_file(config.data_plan),
        "response_shard_manifest": str(config.response_shard_manifest),
        "response_shard_manifest_sha256": _sha256_file(config.response_shard_manifest),
        "shard_sha256": shard_hashes,
        "split_manifest_sha256": split_hash,
        "normalization_sha256": normalization_hash,
        "split_manifest": split_manifest,
        "normalization": normalization,
        "coverage": coverage,
        "fe2_coverage": fe2_comparison,
    }
    if config.subset_manifest is not None:
        report["subset_manifest"] = str(config.subset_manifest)
        report["subset_manifest_sha256"] = _sha256_file(config.subset_manifest)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (config.output_directory / "normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (config.output_directory / "training_data_acceptance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def compare_j2_fe2_preflight_coverage(
    j2_shard_manifest: str | Path,
    fe2_shard_manifest: str | Path,
    coverage: CoverageBins,
    threshold: float,
    output_path: str | Path,
) -> dict[str, Any]:
    """Compare pilot J2 and direct-FE2 impact states without changing either dataset."""
    j2_records = _index_paths(_load_shard_manifest(Path(j2_shard_manifest)))
    fe2_records = _index_paths(_load_shard_manifest(Path(fe2_shard_manifest)))
    records = tuple(j2_records) + tuple(fe2_records)
    comparison, range_passed = _compare_fe2_coverage(records, threshold)
    dimensions = tuple(comparison)
    j2_axes = _collect_selected_axes(
        [record for record in records if record.macro_path_origin == "direct_j2_explicit"],
        dimensions,
    )
    fe2_axes = _collect_selected_axes(
        [record for record in records if record.macro_path_origin == "direct_fe2_explicit"],
        dimensions,
    )
    bin_names = {
        "strain_norm": "strain_norm",
        "strain_rate_norm": "strain_rate_norm",
        "triaxiality": "triaxiality",
        "lode_invariant": "lode_invariant",
        "equivalent_stress": "equivalent_stress",
        "equivalent_plastic_strain": "equivalent_plastic_strain",
    }
    occupancy = {}
    fixed_bin_passed = True
    for dimension, bin_name in bin_names.items():
        edges = np.asarray(getattr(coverage, bin_name), dtype=np.float64)
        record, dimension_passed = _fixed_bin_coverage(
            j2_axes[dimension], fe2_axes[dimension], edges, threshold
        )
        occupancy[dimension] = record
        fixed_bin_passed = fixed_bin_passed and dimension_passed
    report = {
        "passed": range_passed and fixed_bin_passed,
        "threshold": threshold,
        "range_comparison": comparison,
        "fixed_bin_occupancy": occupancy,
        "j2_shard_manifest": str(j2_shard_manifest),
        "fe2_shard_manifest": str(fe2_shard_manifest),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def _fixed_bin_coverage(
    j2_values: np.ndarray,
    fe2_values: np.ndarray,
    edges: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], bool]:
    """Measure FE2 samples occupying fixed state bins absent from the J2 pilot."""
    if j2_values.size == 0 or fe2_values.size == 0:
        raise TrainingDataAcceptanceError(
            "Fixed-bin FE2 coverage requires nonempty J2 and FE2 state coordinates."
        )
    j2_counts, _ = np.histogram(j2_values, bins=edges)
    fe2_counts, _ = np.histogram(fe2_values, bins=edges)
    empty_j2_bins = j2_counts == 0
    outside_fixed_bins = int(fe2_values.size - np.sum(fe2_counts))
    uncovered_count = int(np.sum(fe2_counts[empty_j2_bins])) + outside_fixed_bins
    uncovered_fraction = uncovered_count / int(fe2_values.size)
    record = {
        "edges": edges.tolist(),
        "j2_counts": j2_counts.tolist(),
        "fe2_counts": fe2_counts.tolist(),
        "fe2_occupied_j2_empty_bins": np.flatnonzero((fe2_counts > 0) & empty_j2_bins).tolist(),
        "fe2_outside_fixed_bins": outside_fixed_bins,
        "fe2_uncovered_fraction": uncovered_fraction,
    }
    return record, bool(uncovered_fraction <= threshold)


def require_current_training_data_acceptance(
    acceptance_path: str | Path,
    data_plan: str | Path,
    response_shard_manifest: str | Path,
) -> dict[str, Any]:
    """Require a passing acceptance report whose source hashes still match."""
    report_path = Path(acceptance_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    accepted = report.get("passed", report.get("ready"))
    if not bool(accepted):
        raise TrainingDataAcceptanceError("RNO training data did not pass acceptance.")
    if report["data_plan_sha256"] != _sha256_file(Path(data_plan)):
        raise TrainingDataAcceptanceError("RNO data-plan hash changed after data acceptance.")
    if report["response_shard_manifest_sha256"] != _sha256_file(Path(response_shard_manifest)):
        raise TrainingDataAcceptanceError(
            "RNO response-shard manifest changed after data acceptance."
        )
    response_payload = json.loads(Path(response_shard_manifest).read_text(encoding="utf-8"))
    accepted_subset = report.get("subset_manifest")
    if accepted_subset is None and "subset_manifest" in response_payload:
        raise TrainingDataAcceptanceError(
            "Subset-bound RNO responses lack an accepted subset manifest."
        )
    if accepted_subset is not None:
        subset_path = Path(accepted_subset)
        subset_hash = _sha256_file(subset_path)
        if report["subset_manifest_sha256"] != subset_hash:
            raise TrainingDataAcceptanceError("RNO subset manifest changed after data acceptance.")
        if Path(response_payload["subset_manifest"]) != subset_path:
            raise TrainingDataAcceptanceError(
                "RNO response manifest references a different subset manifest."
            )
        if response_payload["subset_manifest_sha256"] != subset_hash:
            raise TrainingDataAcceptanceError(
                "RNO response manifest has a stale subset-manifest hash."
            )
    shard_paths = _load_shard_manifest(Path(response_shard_manifest))
    current_hashes = {str(path): _sha256_file(path) for path in shard_paths}
    if report["shard_sha256"] != current_hashes:
        raise TrainingDataAcceptanceError("RNO HDF5 shard hashes changed after data acceptance.")
    if report["split_manifest_sha256"] != _sha256_payload(report["split_manifest"]):
        raise TrainingDataAcceptanceError("RNO embedded split manifest hash is inconsistent.")
    if report["normalization_sha256"] != _sha256_payload(report["normalization"]):
        raise TrainingDataAcceptanceError("RNO embedded normalization hash is inconsistent.")
    split_path = report_path.parent / "split_manifest.json"
    normalization_path = report_path.parent / "normalization.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    if _sha256_payload(split_payload) != report["split_manifest_sha256"]:
        raise TrainingDataAcceptanceError("RNO split manifest changed after data acceptance.")
    if _sha256_payload(normalization_payload) != report["normalization_sha256"]:
        raise TrainingDataAcceptanceError("RNO normalization changed after data acceptance.")
    return report


def _load_shard_manifest(path: Path) -> tuple[Path, ...]:
    """Load explicitly recorded legacy or portable response shards."""
    from concrete_impact.core.response_manifest import load_response_shard_paths

    paths = load_response_shard_paths(path)
    if not paths:
        raise ValueError("RNO response-shard manifest is empty.")
    for shard_path in paths:
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
    return paths


def _validate_subset_binding(
    config: TrainingDataAcceptanceConfig,
    shard_paths: tuple[Path, ...],
) -> tuple[str, ...] | None:
    """Validate the immutable binding among Pilot plan, subset, and responses."""
    response_payload = json.loads(config.response_shard_manifest.read_text(encoding="utf-8"))
    response_declares_subset = "subset_manifest" in response_payload
    if config.subset_manifest is None:
        if response_declares_subset:
            raise ValueError("Subset-bound response manifest requires an explicit subset_manifest.")
        return None
    if not response_declares_subset:
        raise ValueError("Pilot subset configuration references an unbound response manifest.")
    if Path(response_payload["subset_manifest"]) != config.subset_manifest:
        raise ValueError("Response manifest references a different Pilot subset manifest.")

    subset_hash = _sha256_file(config.subset_manifest)
    if response_payload["subset_manifest_sha256"] != subset_hash:
        raise ValueError("Response manifest has a stale Pilot subset-manifest hash.")
    subset = json.loads(config.subset_manifest.read_text(encoding="utf-8"))
    if subset["schema_version"] != "1.0":
        raise ValueError("Unsupported Pilot subset-manifest schema.")
    if subset["dataset_type"] != "c48_rve_rno_pilot_subset":
        raise ValueError("Training-data subset has the wrong dataset type.")
    if subset["acceptance_scope"] != "pilot":
        raise ValueError("Training-data subset does not declare pilot acceptance scope.")
    if Path(subset["frozen_data_plan"]) != config.data_plan:
        raise ValueError("Pilot acceptance uses a different frozen RVE data plan.")
    if subset["frozen_data_plan_sha256"] != _sha256_file(config.data_plan):
        raise ValueError("Pilot frozen RVE data-plan hash changed after subset creation.")

    tasks = tuple(subset["tasks"])
    if int(subset["path_count"]) != len(tasks) or not tasks:
        raise ValueError("Pilot subset task count is empty or inconsistent.")
    task_ids = tuple(str(task["task_id"]) for task in tasks)
    lineage_ids = tuple(str(task["lineage_id"]) for task in tasks)
    recorded_paths = tuple(Path(task["response_path"]) for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Pilot subset manifest contains duplicate task ids.")
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError("Pilot subset manifest contains duplicate lineage ids.")
    if len(recorded_paths) != len(set(recorded_paths)):
        raise ValueError("Pilot subset manifest contains duplicate HDF5 paths.")
    if recorded_paths != shard_paths:
        raise ValueError("Pilot subset and response manifests list different HDF5 paths.")
    return task_ids


def _index_paths(shard_paths: tuple[Path, ...]) -> tuple[PathRecord, ...]:
    """Index complete accepted paths and require immutable provenance fields."""
    records = []
    for file_path in shard_paths:
        with h5py.File(file_path, "r") as handle:
            if handle.attrs["schema_version"] not in {"3.0", "4.0"}:
                raise ValueError(f"Unsupported training HDF5 schema: {file_path}.")
            if handle.attrs["status"] != "complete":
                raise ValueError(f"Training shard is incomplete: {file_path}.")
            for path_name, group in handle["paths"].items():
                metadata = json.loads(group.attrs["metadata_json"])
                records.append(
                    PathRecord(
                        file_path=file_path,
                        path_name=path_name,
                        source_name=str(metadata["source_name"]),
                        source_kind=str(metadata["dataset_kind"]),
                        regime=str(metadata["expected_response"]["regime"]),
                        family=str(metadata["family"]),
                        lineage_id=str(metadata["lineage_id"]),
                        macro_path_origin=str(metadata["macro_path_origin"]),
                        preassigned_split=metadata.get("preassigned_split"),
                        design_stratum=metadata.get("design_stratum"),
                        amplitude_band=metadata.get("amplitude_band"),
                        rate_band=metadata.get("rate_band"),
                        tensor_direction=metadata.get("tensor_direction"),
                    )
                )
    if not records:
        raise ValueError("Training shard manifest contains no accepted paths.")
    return tuple(records)


def _compute_coverage(records, settings, policy: CoveragePolicy | None = None):
    """Compute fixed-bin state-space coverage without changing bin boundaries."""
    active_policy = CoveragePolicy() if policy is None else policy
    values_by_axis: dict[str, list[np.ndarray]] = {name: [] for name in COVERAGE_AXIS_NAMES}
    matrix: dict[str, dict[str, int]] = {}
    undefined_triaxiality = 0
    for record in records:
        axes, undefined = _path_state_axes(record, settings)
        undefined_triaxiality += undefined
        matrix_key = f"{record.source_name}|{record.regime}|{record.family}"
        matrix[matrix_key] = {name: int(values.size) for name, values in axes.items()}
        for name, values in axes.items():
            values_by_axis[name].append(values)
    histograms = {}
    missing = {}
    waived_missing = {}
    unexpected_missing = {}
    axis_diagnostics = {}
    passed = True
    for name, chunks in values_by_axis.items():
        values = np.concatenate(chunks)
        edges = np.asarray(getattr(settings, name), dtype=np.float64)
        finite = np.isfinite(values)
        finite_values = values[finite]
        below_count = int(np.count_nonzero(finite_values < edges[0]))
        above_count = int(np.count_nonzero(finite_values > edges[-1]))
        nonfinite_count = int(np.count_nonzero(~finite))
        if below_count or above_count or nonfinite_count:
            passed = False
        counts, _ = np.histogram(finite_values, bins=edges)
        missing_ids = np.flatnonzero(counts == 0).tolist()
        allowed_ids = set(active_policy.allowed_missing_bin_indices.get(name, ()))
        waived_ids = [index for index in missing_ids if index in allowed_ids]
        unexpected_ids = [index for index in missing_ids if index not in allowed_ids]
        if unexpected_ids:
            passed = False
        histograms[name] = {"edges": edges.tolist(), "counts": counts.tolist()}
        missing[name] = missing_ids
        waived_missing[name] = waived_ids
        unexpected_missing[name] = unexpected_ids
        axis_diagnostics[name] = {
            "sample_count": int(values.size),
            "finite_count": int(finite_values.size),
            "nonfinite_count": nonfinite_count,
            "minimum": float(np.min(finite_values)) if finite_values.size else None,
            "maximum": float(np.max(finite_values)) if finite_values.size else None,
            "below_range_count": below_count,
            "above_range_count": above_count,
        }
    return {
        "coverage_policy": active_policy.model_dump(mode="json"),
        "histograms": histograms,
        "missing_bin_indices": missing,
        "waived_missing_bin_indices": waived_missing,
        "unexpected_missing_bin_indices": unexpected_missing,
        "axis_diagnostics": axis_diagnostics,
        "source_regime_family_matrix": matrix,
        "undefined_triaxiality_count": undefined_triaxiality,
    }, passed


def _compute_joint_coverage(
    records: tuple[PathRecord, ...],
    coverage: CoverageBins,
    requirements: JointCoverageRequirements,
) -> tuple[dict[str, Any], bool]:
    """Evaluate coupled rate, stress, plasticity, turning, and invariant predicates."""
    counts = {
        "medium_rate_high_stress_positive_yield": 0,
        "unloading_plastic_negative_work": 0,
        "nonproportional_turning_positive_dissipation": 0,
        "elastic_zero_plasticity_and_dissipation": 0,
    }
    elastic_path_count = 0
    elastic_violations: list[str] = []
    transition_strata: dict[str, dict[str, bool]] = {}
    signed_lode: dict[str, list[np.ndarray]] = {"positive": [], "negative": []}
    elastic_bands = set(requirements.elastic_amplitude_bands)
    transition_bands = set(requirements.transition_amplitude_bands)
    for record in records:
        with h5py.File(record.file_path, "r") as handle:
            group = handle[f"paths/{record.path_name}"]
            strain = group["macro/strain"][...]
            stress = group["macro/stress"][...]
            time_step = group["macro/time_step"][...]
            dissipation = group["macro/dissipation_density"][...]
            q = _read_q(group).reshape(-1)
            margin = _read_maximum_yield_activation_margin(group)
        increments = np.diff(strain, axis=0)
        rates = _tensor_norm(increments / time_step[1:, None])
        equivalent, mean_stress, lode = _stress_invariants(
            stress,
            coverage.stress_floor,
            coverage.invariant_domain_tolerance,
        )
        defined = equivalent >= coverage.stress_floor
        triaxiality = mean_stress[defined] / equivalent[defined]
        defined_lode = lode[defined]
        signed_lode["positive"].append(
            defined_lode[triaxiality >= requirements.triaxiality_sign_floor]
        )
        signed_lode["negative"].append(
            defined_lode[triaxiality <= -requirements.triaxiality_sign_floor]
        )
        cumulative_dissipation = float(np.sum(dissipation * time_step))
        maximum_rate = float(np.max(rates)) if rates.size else 0.0
        if (
            requirements.medium_rate_lower <= maximum_rate <= requirements.medium_rate_upper
            and float(np.max(equivalent)) >= requirements.medium_high_stress_lower
            and float(np.max(margin)) > requirements.positive_yield_margin
        ):
            counts["medium_rate_high_stress_positive_yield"] += 1
        trapezoidal_stress = 0.5 * (stress[1:] + stress[:-1])
        work = np.sum(
            trapezoidal_stress * increments * WEIGHTS,
            axis=1,
        )
        if (
            record.family in {"load_unload_reload", "reverse"}
            and float(np.max(q)) > requirements.positive_plastic_strain
            and work.size
            and float(np.min(work)) < requirements.negative_incremental_work
        ):
            counts["unloading_plastic_negative_work"] += 1
        turning = _turning_angles(
            increments,
            coverage.increment_floor,
            coverage.invariant_domain_tolerance,
        )
        if (
            record.family == "nonproportional"
            and np.any(
                (turning > requirements.turning_angle_lower)
                & (turning < requirements.turning_angle_upper)
            )
            and cumulative_dissipation > requirements.positive_cumulative_dissipation
        ):
            counts["nonproportional_turning_positive_dissipation"] += 1
        is_elastic = (
            record.amplitude_band in elastic_bands if elastic_bands else record.regime == "elastic"
        )
        is_transition = (
            record.amplitude_band in transition_bands
            if transition_bands
            else record.regime == "transition"
        )
        if is_elastic:
            elastic_path_count += 1
            elastic_passed = (
                float(np.max(q)) <= requirements.elastic_plastic_strain_tolerance
                and abs(cumulative_dissipation) <= requirements.elastic_dissipation_tolerance
            )
            if elastic_passed:
                counts["elastic_zero_plasticity_and_dissipation"] += 1
            else:
                elastic_violations.append(record.path_name)
        if is_transition:
            if record.design_stratum is None:
                raise ValueError(f"Transition path lacks design stratum: {record.path_name}.")
            transition = transition_strata.setdefault(
                record.design_stratum,
                {"preyield": False, "yield_activation": False, "weak_plasticity": False},
            )
            transition["preyield"] = transition["preyield"] or bool(
                np.any(margin <= requirements.positive_yield_margin)
            )
            transition["yield_activation"] = transition["yield_activation"] or bool(
                np.any(margin > requirements.positive_yield_margin)
            )
            transition["weak_plasticity"] = transition["weak_plasticity"] or bool(
                np.any(
                    (q > requirements.positive_plastic_strain)
                    & (q <= requirements.weak_plastic_strain_upper)
                )
            )
    lode_occupancy = {}
    lode_passed = True
    edges = np.asarray(requirements.lode_bin_edges, dtype=np.float64)
    for sign, chunks in signed_lode.items():
        values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        bin_counts, _ = np.histogram(values, bins=edges)
        occupied = int(np.count_nonzero(bin_counts))
        lode_occupancy[sign] = {
            "sample_count": int(values.size),
            "bin_counts": bin_counts.tolist(),
            "occupied_bin_count": occupied,
        }
        lode_passed = lode_passed and (occupied >= requirements.minimum_lode_bins_per_sign)
    predicate_passed = all(
        count >= requirements.minimum_path_count
        for name, count in counts.items()
        if name != "elastic_zero_plasticity_and_dissipation"
    )
    elastic_passed = (
        elastic_path_count > 0
        and counts["elastic_zero_plasticity_and_dissipation"] == elastic_path_count
    )
    failed_transition_strata = [
        stratum
        for stratum, predicates in sorted(transition_strata.items())
        if not all(predicates.values())
    ]
    transition_passed = bool(transition_strata) and not failed_transition_strata
    summary = {
        "requirements": requirements.model_dump(mode="json"),
        "predicate_path_counts": counts,
        "elastic_path_count": elastic_path_count,
        "elastic_violation_paths": elastic_violations,
        "signed_lode_occupancy": lode_occupancy,
        "transition_strata": transition_strata,
        "failed_transition_strata": failed_transition_strata,
    }
    return (
        summary,
        predicate_passed and elastic_passed and transition_passed and lode_passed,
    )


def _path_state_axes(record: PathRecord, settings):
    """Compute invariant and path-turning coordinates for one accepted path."""
    with h5py.File(record.file_path, "r") as handle:
        group = handle[f"paths/{record.path_name}"]
        strain = group["macro/strain"][...]
        stress = group["macro/stress"][...]
        time_step = group["macro/time_step"][...]
        strain_norm = _tensor_norm(strain)
        strain_rate_norm = _tensor_norm(np.diff(strain, axis=0) / time_step[1:, None])
        equivalent, mean_stress, lode = _stress_invariants(
            stress,
            settings.stress_floor,
            settings.invariant_domain_tolerance,
        )
        defined = equivalent >= settings.stress_floor
        triaxiality = mean_stress[defined] / equivalent[defined]
        turning = _turning_angles(
            np.diff(strain, axis=0),
            settings.increment_floor,
            settings.invariant_domain_tolerance,
        )
        margin = _read_maximum_yield_activation_margin(group)
        q = _read_q(group).reshape(-1)
    return {
        "strain_norm": strain_norm,
        "strain_rate_norm": strain_rate_norm,
        "equivalent_stress": equivalent,
        "yield_activation_margin": margin,
        "triaxiality": triaxiality,
        "lode_invariant": lode[defined],
        "turning_angle": turning,
        "equivalent_plastic_strain": q,
    }, int(np.count_nonzero(~defined))


def _stress_invariants(stress, stress_floor, domain_tolerance):
    """Compute J2 equivalent stress, mean stress, and normalized third invariant."""
    mean = np.mean(stress[:, :3], axis=1)
    deviator = stress.copy()
    deviator[:, :3] -= mean[:, None]
    equivalent = np.sqrt(
        1.5 * (np.sum(deviator[:, :3] ** 2, axis=1) + 2.0 * np.sum(deviator[:, 3:] ** 2, axis=1))
    )
    tensor = np.zeros((stress.shape[0], 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = deviator[:, 0]
    tensor[:, 1, 1] = deviator[:, 1]
    tensor[:, 2, 2] = deviator[:, 2]
    tensor[:, 1, 2] = tensor[:, 2, 1] = deviator[:, 3]
    tensor[:, 0, 2] = tensor[:, 2, 0] = deviator[:, 4]
    tensor[:, 0, 1] = tensor[:, 1, 0] = deviator[:, 5]
    lode = np.zeros(stress.shape[0], dtype=np.float64)
    defined = equivalent >= stress_floor
    lode[defined] = 27.0 * np.linalg.det(tensor[defined]) / (2.0 * equivalent[defined] ** 3)
    if np.any(np.abs(lode[defined]) > 1.0 + domain_tolerance):
        raise ValueError("Lode invariant left its mathematical domain; data were not clipped.")
    return equivalent, mean, lode


def _turning_angles(increments, increment_floor, domain_tolerance):
    """Compute consecutive engineering-tensor increment angles with a strict domain check."""
    norms = _tensor_norm(increments)
    valid = (norms[:-1] >= increment_floor) & (norms[1:] >= increment_floor)
    products = np.sum(increments[1:] * increments[:-1] * WEIGHTS, axis=1)
    cosine = products[valid] / (norms[1:][valid] * norms[:-1][valid])
    if np.any(np.abs(cosine) > 1.0 + domain_tolerance):
        raise ValueError("Strain-increment turning cosine left [-1, 1]; data were not clipped.")
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)


def _build_split_manifest(records, settings):
    """Build a deterministic stratified split over complete lineages."""
    if settings.strategy == "deterministic_design_stratum":
        return _build_deterministic_design_split_manifest(records, settings)
    assigned = [record for record in records if record.preassigned_split is not None]
    if assigned:
        if len(assigned) != len(records):
            return (
                {"seed": settings.seed, "paths": []},
                False,
                ["preassigned split metadata are present for only part of the dataset"],
            )
        return _build_preassigned_split_manifest(records, settings)
    strata: dict[tuple[str, str], set[str]] = {}
    for record in records:
        strata.setdefault((record.source_name, record.regime), set()).add(record.lineage_id)
    lineage_split: dict[str, str] = {}
    failures = []
    for stratum, lineages in sorted(strata.items()):
        ordered = sorted(
            lineages,
            key=lambda lineage: hashlib.sha256(
                f"{settings.seed}|{stratum}|{lineage}".encode()
            ).hexdigest(),
        )
        if len(ordered) < 3:
            failures.append(f"stratum={stratum} has {len(ordered)} independent lineages")
            continue
        stratum_records = [
            record for record in records if (record.source_name, record.regime) == stratum
        ]
        direct_fe2 = all(
            record.macro_path_origin.startswith("direct_fe2") for record in stratum_records
        )
        if direct_fe2:
            train_count = 0
            validation_count = max(1, len(ordered) // 2)
        else:
            train_count = max(1, int(np.floor(len(ordered) * settings.train_fraction)))
            validation_count = max(1, int(np.floor(len(ordered) * settings.validation_fraction)))
            if train_count + validation_count >= len(ordered):
                train_count = len(ordered) - 2
                validation_count = 1
        for index, lineage in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            if lineage in lineage_split and lineage_split[lineage] != split:
                raise ValueError(f"Lineage received conflicting dataset splits: {lineage}.")
            lineage_split[lineage] = split
    paths = [
        {
            "path_id": f"{record.file_path.name}:{record.path_name}",
            "lineage_id": record.lineage_id,
            "source_name": record.source_name,
            "regime": record.regime,
            "family": record.family,
            "macro_path_origin": record.macro_path_origin,
            "split": lineage_split.get(record.lineage_id, "unassigned"),
        }
        for record in records
    ]
    return {"seed": settings.seed, "paths": paths}, not failures, failures


def _build_deterministic_design_split_manifest(
    records: tuple[PathRecord, ...],
    settings: SplitSettings,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Build exact deterministic splits inside every complete design stratum.

    Inputs:
        records: Accepted complete-path records with design-stratum identifiers.
        settings: Seed, split strategy, and exact split fractions.
    Outputs:
        Split manifest, pass flag, and explicit failure records.
    Author:
        Zhen Hao.
    Created:
        2026-07-19.
    """
    strata: dict[str, list[PathRecord]] = {}
    failures: list[str] = []
    for record in records:
        if record.design_stratum is None:
            failures.append(f"production path lacks design_stratum: {record.path_name}")
            continue
        strata.setdefault(record.design_stratum, []).append(record)
    lineage_split: dict[str, str] = {}
    for stratum, stratum_records in sorted(strata.items()):
        lineages = {record.lineage_id for record in stratum_records}
        if len(lineages) != len(stratum_records):
            failures.append(f"design stratum contains repeated lineages: {stratum}")
            continue
        ordered = sorted(
            lineages,
            key=lambda lineage: hashlib.sha256(
                f"{settings.seed}|{stratum}|{lineage}".encode()
            ).hexdigest(),
        )
        raw_counts = {
            "train": len(ordered) * settings.train_fraction,
            "validation": len(ordered) * settings.validation_fraction,
            "test": len(ordered) * settings.test_fraction,
        }
        counts = {name: int(round(value)) for name, value in raw_counts.items()}
        if any(
            not np.isclose(raw_counts[name], counts[name], rtol=0.0, atol=1.0e-14)
            for name in raw_counts
        ):
            failures.append(
                f"design stratum cannot realize exact split fractions: {stratum}, "
                f"lineages={len(ordered)}, fractions={settings.model_dump(mode='json')}"
            )
            continue
        boundaries = (counts["train"], counts["train"] + counts["validation"])
        for index, lineage in enumerate(ordered):
            split = (
                "train"
                if index < boundaries[0]
                else "validation"
                if index < boundaries[1]
                else "test"
            )
            previous = lineage_split.setdefault(lineage, split)
            if previous != split:
                raise ValueError(f"Lineage received conflicting dataset splits: {lineage}.")
    paths = [
        {
            "path_id": f"{record.file_path.name}:{record.path_name}",
            "lineage_id": record.lineage_id,
            "source_name": record.source_name,
            "regime": record.regime,
            "family": record.family,
            "macro_path_origin": record.macro_path_origin,
            "design_stratum": record.design_stratum,
            "amplitude_band": record.amplitude_band,
            "rate_band": record.rate_band,
            "tensor_direction": record.tensor_direction,
            "split": lineage_split.get(record.lineage_id, "unassigned"),
        }
        for record in records
    ]
    return (
        {
            "seed": settings.seed,
            "strategy": settings.strategy,
            "paths": paths,
        },
        not failures,
        failures,
    )


def _build_preassigned_split_manifest(records, settings):
    """Validate and publish the frozen exact per-stratum production split."""
    valid_splits = {"train", "validation", "test"}
    lineage_split: dict[str, str] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for record in records:
        split = str(record.preassigned_split)
        if split not in valid_splits:
            failures.append(f"invalid preassigned split: {record.path_name}={split}")
            continue
        previous = lineage_split.setdefault(record.lineage_id, split)
        if previous != split:
            failures.append(f"lineage received conflicting splits: {record.lineage_id}")
        if record.design_stratum is None:
            failures.append(f"production path lacks design_stratum: {record.path_name}")
            continue
        counts = stratum_counts.setdefault(
            record.design_stratum,
            {"train": 0, "validation": 0, "test": 0},
        )
        counts[split] += 1
    for stratum, counts in sorted(stratum_counts.items()):
        total = sum(counts.values())
        expected = {
            "train": int(round(total * settings.train_fraction)),
            "validation": int(round(total * settings.validation_fraction)),
            "test": int(round(total * settings.test_fraction)),
        }
        if counts != expected:
            failures.append(
                f"production stratum split mismatch: {stratum}, "
                f"observed={counts}, expected={expected}"
            )
    paths = [
        {
            "path_id": f"{record.file_path.name}:{record.path_name}",
            "lineage_id": record.lineage_id,
            "source_name": record.source_name,
            "regime": record.regime,
            "family": record.family,
            "macro_path_origin": record.macro_path_origin,
            "design_stratum": record.design_stratum,
            "amplitude_band": record.amplitude_band,
            "rate_band": record.rate_band,
            "tensor_direction": record.tensor_direction,
            "split": record.preassigned_split,
        }
        for record in records
    ]
    return {"seed": settings.seed, "paths": paths}, not failures, failures


def _compute_normalization(records, split_manifest, conditioning_inputs):
    """Compute per-component statistics from training lineages only."""
    split_by_id = {item["path_id"]: item["split"] for item in split_manifest["paths"]}
    chunks: dict[str, list[np.ndarray]] = {}
    for record in records:
        path_id = f"{record.file_path.name}:{record.path_name}"
        if split_by_id[path_id] != "train":
            continue
        with h5py.File(record.file_path, "r") as handle:
            group = handle[f"paths/{record.path_name}"]
            _append_rows(chunks, "macro_strain", group["macro/strain"][...])
            _append_rows(chunks, "macro_stress", group["macro/stress"][...])
            if "macro/dissipation_density" in group:
                _append_rows(
                    chunks,
                    "dissipation_density",
                    group["macro/dissipation_density"][...],
                )
            if "micro/state" in group:
                for name, dataset in group["micro/state"].items():
                    values = dataset[...]
                    if name.endswith("plastic_strain") and not name.endswith(
                        "equivalent_plastic_strain"
                    ):
                        _append_rows(chunks, f"state::{name}", values)
                    else:
                        chunks.setdefault(f"state::{name}", []).append(
                            np.asarray(values, dtype=np.float64).reshape(-1, 1)
                        )
            feature_group = group["features"]
            for kind in ("material_parameters", "microstructure_features"):
                if kind not in conditioning_inputs[record.source_name]:
                    continue
                names = [value.decode() for value in feature_group[f"{kind}_names"][...]]
                values = feature_group[kind][...]
                for name, value in zip(names, values, strict=True):
                    _append_rows(chunks, f"{kind}::{name}", np.asarray([[value]]))
    failures = []
    normalization = {}
    for name, arrays in sorted(chunks.items()):
        values = np.concatenate(arrays, axis=0)
        mean = np.mean(values, axis=0)
        scale = np.sqrt(np.mean((values - mean) ** 2, axis=0))
        if np.any(scale == 0.0):
            failures.append(f"zero training variance: {name}")
        normalization[name] = {"mean": mean.tolist(), "rms_scale": scale.tolist()}
    if not chunks:
        failures.append("no training paths were assigned")
    return normalization, not failures, failures


def _compare_fe2_coverage(records, threshold):
    """Measure direct FE2 samples outside the traditional-J2 impact ranges."""
    j2_records = [record for record in records if record.macro_path_origin == "direct_j2_explicit"]
    fe2_records = [
        record for record in records if record.macro_path_origin == "direct_fe2_explicit"
    ]
    if not j2_records or not fe2_records:
        return {"failure": "both J2 and direct FE2 impact paths are required"}, False
    dimensions = (
        "strain_norm",
        "strain_rate_norm",
        "triaxiality",
        "lode_invariant",
        "equivalent_stress",
        "equivalent_plastic_strain",
    )
    j2_axes = _collect_selected_axes(j2_records, dimensions)
    fe2_axes = _collect_selected_axes(fe2_records, dimensions)
    fractions = {}
    passed = True
    for name in dimensions:
        lower = float(np.min(j2_axes[name]))
        upper = float(np.max(j2_axes[name]))
        values = fe2_axes[name]
        fraction = float(np.mean((values < lower) | (values > upper)))
        fractions[name] = {
            "j2_minimum": lower,
            "j2_maximum": upper,
            "fe2_out_of_coverage_fraction": fraction,
        }
        passed = passed and fraction <= threshold
    return fractions, passed


def _collect_selected_axes(records, dimensions):
    """Collect coverage coordinates with fixed invariant settings for source comparison."""
    result = {name: [] for name in dimensions}
    for record in records:
        with h5py.File(record.file_path, "r") as handle:
            group = handle[f"paths/{record.path_name}"]
            strain = group["macro/strain"][...]
            stress = group["macro/stress"][...]
            dt = group["macro/time_step"][...]
            equivalent, mean, lode = _stress_invariants(stress, 1.0e-12, 1.0e-10)
            defined = equivalent >= 1.0e-12
            axes = {
                "strain_norm": _tensor_norm(strain),
                "strain_rate_norm": _tensor_norm(np.diff(strain, axis=0) / dt[1:, None]),
                "triaxiality": mean[defined] / equivalent[defined],
                "lode_invariant": lode[defined],
                "equivalent_stress": equivalent,
                "equivalent_plastic_strain": _read_q(group).reshape(-1),
            }
            for name in dimensions:
                result[name].append(axes[name])
    return {name: np.concatenate(values) for name, values in result.items()}


def _audit_tangent_checks(records, tolerance):
    """Require stored central-difference tangent checks per source and regime."""
    strata: dict[str, list[float]] = {}
    for record in records:
        with h5py.File(record.file_path, "r") as handle:
            path = f"paths/{record.path_name}/validation/tangent_direction_error"
            if path in handle:
                key = f"{record.source_name}|{record.regime}"
                strata.setdefault(key, []).extend(handle[path][...].tolist())
    required = {f"{record.source_name}|{record.regime}" for record in records}
    summary = {}
    passed = True
    for key in sorted(required):
        values = np.asarray(strata.get(key, []), dtype=np.float64)
        if values.size < 2 or not np.all(np.isfinite(values)):
            summary[key] = {"passed": False, "failure": "requires at least two states"}
            passed = False
            continue
        maximum = float(np.max(values))
        summary[key] = {"passed": maximum <= tolerance, "maximum_error": maximum}
        passed = passed and maximum <= tolerance
    return summary, passed


def _load_external_pass_report(path: Path) -> dict[str, Any]:
    """Load an independent benchmark report and require an explicit passed field."""
    if not path.is_file():
        return {"passed": False, "failure": f"missing external report: {path}"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "passed": bool(payload["passed"]),
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _read_available_diagnostic(group: h5py.Group, name: str) -> np.ndarray:
    """Read finite available diagnostic values from point or solver storage."""
    for path in (f"micro/diagnostics/{name}", f"solver/{name}"):
        if path in group:
            values = group[path][...]
            availability_path = f"{path}__available"
            if availability_path in group:
                values = values[group[availability_path][...].astype(bool)]
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"Diagnostic has no finite available values: {group.name}/{name}.")
            return values.reshape(-1)
    raise ValueError(f"Required diagnostic is absent: {group.name}/{name}.")


def _read_maximum_yield_activation_margin(group: h5py.Group) -> np.ndarray:
    """Read the pathwise local-onset diagnostic without using the RVE minimum.

    Inputs:
        group: One accepted HDF5 response-path group.
    Outputs:
        Finite maximum microscopic yield-activation margins at stored time states.
    Author:
        Zhen Hao.
    Created:
        2026-07-19.
    """
    path = "solver/maximum_yield_activation_margin"
    if path in group:
        values = group[path][...]
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Diagnostic has no finite available values: {group.name}/{path}.")
        return values.reshape(-1)
    return _read_available_diagnostic(group, "yield_activation_margin")


def _read_q(group: h5py.Group) -> np.ndarray:
    """Read canonical homogeneous or matrix equivalent plastic strain."""
    if "micro/state" in group:
        state = group["micro/state"]
        for name in ("phase_matrix__equivalent_plastic_strain", "equivalent_plastic_strain"):
            if name in state:
                return state[name][...]
    if "macro/matrix_maximum_q" in group:
        return group["macro/matrix_maximum_q"][...]
    raise ValueError(f"Path has no equivalent plastic strain: {group.name}.")


def _tensor_norm(values: np.ndarray) -> np.ndarray:
    """Compute the engineering-tensor weighted Euclidean norm."""
    return np.sqrt(np.sum(values**2 * WEIGHTS, axis=-1))


def _append_rows(chunks, name, values):
    """Append samples while preserving the final physical component axis."""
    array = np.asarray(values, dtype=np.float64)
    width = 1 if array.ndim == 1 else array.shape[-1]
    chunks.setdefault(name, []).append(array.reshape(-1, width))


def _sha256_file(path: Path) -> str:
    """Compute a complete file SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    """Hash one canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
