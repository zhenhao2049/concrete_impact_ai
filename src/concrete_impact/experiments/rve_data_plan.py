"""Validated planning interfaces for the four RVE-RNO data sources.

Contents:
    Plan schemas, plan loading, mesh resolution, and parallel-resource validation.
Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from fem.io.config import load_yaml_config

DataSourceKind = Literal[
    "material_point",
    "single_phase_rve",
    "heterogeneous_rve",
    "impact_path",
]
MacroPathOrigin = Literal[
    "prescribed_macro_strain",
    "direct_j2_explicit",
    "direct_fe2_explicit",
    "direct_fe2_implicit",
]


class StrictPlanModel(BaseModel):
    """Forbid undeclared fields in data-generation plans."""

    model_config = ConfigDict(extra="forbid")


class BoundaryPlan(StrictPlanModel):
    """Describe the spatial boundary condition used by one data source."""

    kind: Literal["none", "periodic_macro_strain", "macro_impact_structure"]
    description: str


class ExpectedResponsePlan(StrictPlanModel):
    """Declare physical path coverage that generated labels must satisfy."""

    regime: Literal["elastic", "transition", "viscoplastic"]
    require_unloading: bool
    require_reverse_loading: bool


class ProductionAmplitudeBand(StrictPlanModel):
    """Define one fixed-cell strain-amplitude stratum."""

    name: str
    lower: PositiveFloat
    upper: PositiveFloat
    expected_response: Literal["elastic", "transition", "viscoplastic"]

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> ProductionAmplitudeBand:
        """Require a nonempty amplitude interval."""
        if self.upper <= self.lower:
            raise ValueError("Production amplitude bands require upper > lower.")
        return self


class ProductionRateBand(StrictPlanModel):
    """Define one fixed maximum-strain-rate stratum."""

    name: str
    lower: PositiveFloat
    upper: PositiveFloat

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> ProductionRateBand:
        """Require a nonempty positive rate interval."""
        if self.upper <= self.lower:
            raise ValueError("Production rate bands require upper > lower.")
        return self


ProductionTensorDirection = Literal[
    "deviatoric_uniaxial",
    "deviatoric_biaxial",
    "shear_dominant",
    "positive_triaxial",
    "negative_triaxial",
]


class FixedCellProductionDesign(StrictPlanModel):
    """Define the deterministic 2000-path fixed-cell factorial design."""

    amplitude_bands: tuple[ProductionAmplitudeBand, ...]
    rate_bands: tuple[ProductionRateBand, ...]
    tensor_directions: tuple[ProductionTensorDirection, ...]
    accepted_increment_counts: tuple[PositiveInt, ...]
    nonproportional_turning_angles: tuple[PositiveFloat, ...]
    paths_per_stratum: PositiveInt
    train_per_stratum: PositiveInt
    validation_per_stratum: PositiveInt
    test_per_stratum: PositiveInt
    snapshots_per_stratum: PositiveInt
    tangent_checks_per_stratum: PositiveInt
    direction_perturbation: float = Field(ge=0.0, lt=0.25)

    @model_validator(mode="after")
    def require_canonical_design(self) -> FixedCellProductionDesign:
        """Require the exact balanced design used by the production dataset."""
        if len(self.amplitude_bands) != 5 or len(self.rate_bands) != 4:
            raise ValueError("Fixed-cell production requires five amplitude and four rate bands.")
        if len(self.tensor_directions) != 5 or len(set(self.tensor_directions)) != 5:
            raise ValueError("Fixed-cell production requires five unique tensor directions.")
        if len(self.accepted_increment_counts) != 4:
            raise ValueError("Fixed-cell production requires four time resolutions.")
        if len(self.nonproportional_turning_angles) != 4:
            raise ValueError("Fixed-cell production requires four turning-angle levels.")
        if self.paths_per_stratum != (
            len(self.tensor_directions) * len(self.accepted_increment_counts)
        ):
            raise ValueError(
                "Production paths_per_stratum must equal directions times resolutions."
            )
        split_total = (
            self.train_per_stratum
            + self.validation_per_stratum
            + self.test_per_stratum
        )
        if split_total != self.paths_per_stratum:
            raise ValueError("Production per-stratum split counts must cover every path.")
        if self.snapshots_per_stratum > self.paths_per_stratum:
            raise ValueError("Production snapshot count exceeds paths_per_stratum.")
        if self.tangent_checks_per_stratum > self.paths_per_stratum:
            raise ValueError("Production tangent-check count exceeds paths_per_stratum.")
        _require_nonoverlapping_bands(self.amplitude_bands, "amplitude")
        _require_nonoverlapping_bands(self.rate_bands, "rate")
        return self


class LoadCasePlan(StrictPlanModel):
    """Describe one deterministic path family without executing it."""

    name: str
    path_count: PositiveInt
    time_points: PositiveInt
    duration: float = Field(gt=0.0)
    excitation: Literal[
        "prescribed_six_component_strain",
        "cyclic_nonproportional_strain",
        "half_sine_pressure_pulse",
    ]
    seed: int = Field(ge=0)
    peak_scale: PositiveFloat
    pulse_duration_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    axial_region_count: PositiveInt | None = None
    path_families: tuple[
        Literal["monotonic", "load_unload_reload", "reverse", "hold", "nonproportional"],
        ...,
    ]
    expected_response: ExpectedResponsePlan

    @model_validator(mode="after")
    def require_path_family(self) -> LoadCasePlan:
        """Reject a load case without an explicitly selected path family."""
        if not self.path_families:
            raise ValueError("RVE data load case requires at least one path family.")
        if self.excitation == "half_sine_pressure_pulse" and self.pulse_duration_fraction is None:
            raise ValueError("Impact data load case requires pulse_duration_fraction.")
        if self.excitation == "half_sine_pressure_pulse" and self.axial_region_count is None:
            raise ValueError("Impact data load case requires axial_region_count.")
        if (
            self.excitation != "half_sine_pressure_pulse"
            and self.pulse_duration_fraction is not None
        ):
            raise ValueError("Only impact data may define pulse_duration_fraction.")
        if self.excitation != "half_sine_pressure_pulse" and self.axial_region_count is not None:
            raise ValueError("Only impact data may define axial_region_count.")
        if self.axial_region_count is not None and self.path_count % self.axial_region_count != 0:
            raise ValueError("Impact path_count must be divisible by axial_region_count.")
        return self


class FixedParameterSpec(StrictPlanModel):
    """Store one fixed production parameter while preserving sampling metadata."""

    kind: Literal["fixed"]
    value: float


class GridParameterSpec(StrictPlanModel):
    """Store an explicitly enabled finite parameter grid."""

    kind: Literal["grid"]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def require_nonempty_grid(self) -> GridParameterSpec:
        """Reject a parameter grid without sampling values."""
        if not self.values:
            raise ValueError("Grid parameter sampling requires at least one value.")
        return self


class UniformParameterSpec(StrictPlanModel):
    """Store an explicitly enabled uniform parameter distribution."""

    kind: Literal["uniform"]
    lower: float
    upper: float

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> UniformParameterSpec:
        """Reject a non-positive uniform sampling interval."""
        if self.upper <= self.lower:
            raise ValueError("Uniform parameter sampling requires upper > lower.")
        return self


class LogUniformParameterSpec(StrictPlanModel):
    """Store an explicitly enabled positive log-uniform distribution."""

    kind: Literal["log_uniform"]
    lower: PositiveFloat
    upper: PositiveFloat

    @model_validator(mode="after")
    def require_ordered_bounds(self) -> LogUniformParameterSpec:
        """Reject a non-positive log-uniform sampling interval."""
        if self.upper <= self.lower:
            raise ValueError("Log-uniform parameter sampling requires upper > lower.")
        return self


ParameterSpec = Annotated[
    FixedParameterSpec | GridParameterSpec | UniformParameterSpec | LogUniformParameterSpec,
    Field(discriminator="kind"),
]


class FieldPlan(StrictPlanModel):
    """Declare stored inputs, labels, state, and diagnostics."""

    inputs: tuple[str, ...]
    labels: tuple[str, ...]
    state_fields: tuple[str, ...]
    diagnostics: tuple[str, ...]


class DataSourcePlan(StrictPlanModel):
    """Define geometry, boundary data, load paths, and fields for one source."""

    name: str
    kind: DataSourceKind
    model: str
    boundary: BoundaryPlan
    load_cases: tuple[LoadCasePlan, ...]
    fields: FieldPlan
    training_role: Literal["pretraining", "homogenization", "coverage", "validation"]
    macro_path_origin: MacroPathOrigin
    material_parameters: dict[str, ParameterSpec]
    microstructure_parameters: dict[str, ParameterSpec]
    production_design: FixedCellProductionDesign | None = None

    @model_validator(mode="after")
    def require_compatible_macro_path_origin(self) -> DataSourcePlan:
        """Require explicit provenance consistent with the data-source kind."""
        if self.kind == "impact_path" and self.macro_path_origin == "prescribed_macro_strain":
            raise ValueError("Impact paths require a direct macro-dynamics origin.")
        if self.kind != "impact_path" and self.macro_path_origin != "prescribed_macro_strain":
            raise ValueError("Non-impact sources require prescribed_macro_strain origin.")
        if self.production_design is not None:
            if self.kind != "heterogeneous_rve":
                raise ValueError("Fixed-cell production design requires a heterogeneous RVE.")
            if len(self.load_cases) != 1:
                raise ValueError("Fixed-cell production design requires exactly one load case.")
            load_case = self.load_cases[0]
            expected = (
                len(load_case.path_families)
                * len(self.production_design.amplitude_bands)
                * len(self.production_design.rate_bands)
                * self.production_design.paths_per_stratum
            )
            if load_case.path_count != expected:
                raise ValueError(
                    "Fixed-cell production path_count does not match its factorial design: "
                    f"configured={load_case.path_count}, expected={expected}."
                )
        return self


class AuditThresholds(StrictPlanModel):
    """Store fixed physical audit scales without runtime adjustment."""

    q_tolerance: PositiveFloat
    plastic_q_threshold: PositiveFloat
    dissipation_tolerance: PositiveFloat
    positive_cumulative_dissipation_threshold: PositiveFloat
    work_tolerance: PositiveFloat
    stress_drop_threshold: PositiveFloat
    strain_increment_tolerance: PositiveFloat
    hold_strain_tolerance: PositiveFloat
    hold_stress_relaxation_threshold: PositiveFloat
    tangent_perturbation: PositiveFloat
    tangent_stress_scale: PositiveFloat
    tangent_direction_tolerance: PositiveFloat


class ParallelPlan(StrictPlanModel):
    """Define explicit process and shard sizes without automatic adjustment."""

    workers: PositiveInt
    paths_per_task: PositiveInt
    shard_path_count: PositiveInt
    compression: Literal["lzf", "gzip"]
    thread_count_per_worker: PositiveInt
    logical_cpu_budget: PositiveInt = 24


class RVEDataPlan(StrictPlanModel):
    """Store the complete four-source RVE-RNO data-generation plan."""

    schema_version: Literal["3.0", "4.0"]
    purpose: Literal[
        "complete_four_source",
        "direct_fe2_preflight",
        "fixed_cell_rve_training",
    ] = (
        "complete_four_source"
    )
    storage_mode: Literal["full_history", "compact_streaming"] = "full_history"
    output_directory: Path
    mesh_selection_report: Path | None = None
    sources: tuple[DataSourcePlan, ...]
    parallel: ParallelPlan
    audit: AuditThresholds

    @model_validator(mode="after")
    def require_all_source_kinds(self) -> RVEDataPlan:
        """Require all canonical kinds while allowing additional impact calibration sources."""
        kinds = [source.kind for source in self.sources]
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("RVE-RNO data source names must be unique.")
        if self.purpose == "direct_fe2_preflight":
            if len(self.sources) != 1 or kinds != ["impact_path"]:
                raise ValueError(
                    "Direct FE2 preflight requires exactly one impact_path source."
                )
            if self.sources[0].macro_path_origin != "direct_fe2_explicit":
                raise ValueError(
                    "Direct FE2 preflight requires macro_path_origin=direct_fe2_explicit."
                )
            return self
        if self.purpose == "fixed_cell_rve_training":
            if self.schema_version != "4.0" or self.storage_mode != "compact_streaming":
                raise ValueError(
                    "Fixed-cell RVE training requires schema_version=4.0 and "
                    "storage_mode=compact_streaming."
                )
            if len(self.sources) != 1 or kinds != ["heterogeneous_rve"]:
                raise ValueError(
                    "Fixed-cell RVE training requires exactly one heterogeneous_rve source."
                )
            required_mesh = {
                "circumferential_divisions",
                "matrix_radial_divisions",
                "axial_divisions",
            }
            missing_mesh = required_mesh - set(self.sources[0].microstructure_parameters)
            if missing_mesh:
                raise ValueError(
                    "Fixed-cell RVE training requires explicit mesh parameters: "
                    f"missing={sorted(missing_mesh)}."
                )
            if self.mesh_selection_report is None:
                raise ValueError(
                    "Fixed-cell RVE training requires an explicit mesh_selection_report."
                )
            return self
        required = {
            "material_point",
            "single_phase_rve",
            "heterogeneous_rve",
            "impact_path",
        }
        if not required.issubset(set(kinds)):
            raise ValueError(
                "RVE-RNO data plan requires every canonical source kind."
            )
        if kinds.count("material_point") != 1:
            raise ValueError("RVE-RNO data plan requires exactly one material_point source.")
        if kinds.count("single_phase_rve") != 1:
            raise ValueError("RVE-RNO data plan requires exactly one single_phase_rve source.")
        if kinds.count("heterogeneous_rve") != 1:
            raise ValueError("RVE-RNO data plan requires exactly one heterogeneous_rve source.")
        return self


class RVEDataTask(StrictPlanModel):
    """Represent one deterministic, independently executable path task."""

    task_id: str
    source_name: str
    source_kind: DataSourceKind
    load_case_name: str
    path_index: int
    seed: int
    shard_index: int
    lineage_id: str
    macro_path_origin: MacroPathOrigin
    family: str | None = None
    amplitude_band: str | None = None
    rate_band: str | None = None
    tensor_direction: ProductionTensorDirection | None = None
    repeat_index: int | None = Field(default=None, ge=0)
    amplitude: float | None = Field(default=None, gt=0.0)
    target_strain_rate: float | None = Field(default=None, gt=0.0)
    accepted_increment_count: int | None = Field(default=None, gt=0)
    turning_angle: float | None = Field(default=None, gt=0.0)
    design_stratum: str | None = None
    preassigned_split: Literal["train", "validation", "test"] | None = None
    expected_response_regime: Literal["elastic", "transition", "viscoplastic"] | None = None
    snapshot_required: bool = False
    tangent_check_required: bool = False


def load_rve_data_plan(path: str | Path) -> RVEDataPlan:
    """Load and strictly validate a four-source data plan."""
    return RVEDataPlan.model_validate(load_yaml_config(path))


def resolve_fixed_cell_mesh(plan: RVEDataPlan) -> RVEDataPlan:
    """Resolve the accepted fixed-cell mesh without modifying the source YAML."""
    if plan.purpose != "fixed_cell_rve_training":
        return plan
    report_path = plan.mesh_selection_report
    if report_path is None or not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report["passed"]):
        raise ValueError("Fixed-cell mesh selection report did not pass acceptance.")
    selected = report["selected_mesh"]
    source = plan.sources[0]
    micro = dict(source.microstructure_parameters)
    for name in (
        "circumferential_divisions",
        "matrix_radial_divisions",
        "axial_divisions",
    ):
        micro[name] = FixedParameterSpec(kind="fixed", value=float(selected[name]))
    resolved_source = source.model_copy(update={"microstructure_parameters": micro})
    return plan.model_copy(update={"sources": (resolved_source,)})


def validate_parallel_resources(plan: RVEDataPlan, available_cpus: int) -> None:
    """Reject a path-process plan exceeding its explicit logical-CPU budget."""
    requested = plan.parallel.workers * plan.parallel.thread_count_per_worker
    allowed = min(
        available_cpus // 2,
        plan.parallel.logical_cpu_budget,
    )
    if requested > allowed:
        raise ValueError(
            "Configured RVE path processes exceed half of available CPUs or the "
            "explicit logical CPU budget: "
            f"workers={plan.parallel.workers}, "
            f"threads_per_worker={plan.parallel.thread_count_per_worker}, "
            f"requested={requested}, allowed={allowed}."
        )


def build_rve_data_task_manifest(plan: RVEDataPlan) -> tuple[RVEDataTask, ...]:
    """Build deterministic task identifiers, seeds, and shard assignments."""
    tasks: list[RVEDataTask] = []
    global_path_index = 0
    for source in plan.sources:
        for load_case in source.load_cases:
            if source.production_design is not None:
                production_tasks = _build_fixed_cell_production_tasks(
                    source,
                    load_case,
                    global_path_index,
                    plan.parallel.shard_path_count,
                )
                tasks.extend(production_tasks)
                global_path_index += len(production_tasks)
                continue
            for path_index in range(load_case.path_count):
                tasks.append(
                    RVEDataTask(
                        task_id=(
                            f"{source.name}__{load_case.name}__path_{path_index:06d}"
                        ),
                        source_name=source.name,
                        source_kind=source.kind,
                        load_case_name=load_case.name,
                        path_index=path_index,
                        seed=load_case.seed + path_index,
                        shard_index=global_path_index // plan.parallel.shard_path_count,
                        lineage_id=(
                            f"{source.name}__{load_case.name}__macro_run"
                            if source.kind == "impact_path"
                            else f"{source.name}__{load_case.name}__path_{path_index:06d}"
                        ),
                        macro_path_origin=source.macro_path_origin,
                    )
                )
                global_path_index += 1
    return tuple(tasks)


def build_rve_supplemental_task_manifest(
    plan: RVEDataPlan,
) -> tuple[RVEDataTask, ...]:
    """Build the optional 1000-path extension with an exact 700/150/150 split."""
    source = plan.sources[0]
    design = source.production_design
    if plan.purpose != "fixed_cell_rve_training" or design is None:
        raise ValueError("RVE supplemental paths require the c48 production design.")
    load_case = source.load_cases[0]
    tasks: list[RVEDataTask] = []
    local_index = 0
    base_count = len(build_rve_data_task_manifest(plan))
    for family in load_case.path_families:
        for amplitude_band in design.amplitude_bands:
            for rate_band in design.rate_bands:
                stratum = f"{family}|{amplitude_band.name}|{rate_band.name}"
                for supplement_index in range(10):
                    direction = design.tensor_directions[supplement_index % 5]
                    repeat_index = supplement_index % 4
                    amplitude_fraction = (supplement_index + 1.0) / 11.0
                    rate_fraction = ((3 * supplement_index + 1) % 10 + 1.0) / 11.0
                    task_id = (
                        f"{source.name}__supplement__{family}__{amplitude_band.name}__"
                        f"{rate_band.name}__sample_{supplement_index:02d}"
                    )
                    tasks.append(
                        RVEDataTask(
                            task_id=task_id,
                            source_name=source.name,
                            source_kind=source.kind,
                            load_case_name=load_case.name,
                            path_index=base_count + local_index,
                            seed=load_case.seed + base_count + local_index,
                            shard_index=(base_count + local_index)
                            // plan.parallel.shard_path_count,
                            lineage_id=task_id,
                            macro_path_origin=source.macro_path_origin,
                            family=family,
                            amplitude_band=amplitude_band.name,
                            rate_band=rate_band.name,
                            tensor_direction=direction,
                            repeat_index=repeat_index,
                            amplitude=amplitude_band.lower
                            + amplitude_fraction
                            * (amplitude_band.upper - amplitude_band.lower),
                            target_strain_rate=rate_band.lower
                            + rate_fraction * (rate_band.upper - rate_band.lower),
                            accepted_increment_count=design.accepted_increment_counts[
                                repeat_index
                            ],
                            turning_angle=(
                                design.nonproportional_turning_angles[repeat_index]
                                if family == "nonproportional"
                                else None
                            ),
                            design_stratum=f"supplement|{stratum}",
                            expected_response_regime=amplitude_band.expected_response,
                            snapshot_required=False,
                            tangent_check_required=False,
                        )
                    )
                    local_index += 1
    ordered = sorted(
        range(len(tasks)),
        key=lambda index: hashlib.sha256(tasks[index].task_id.encode("utf-8")).hexdigest(),
    )
    split_by_index = {
        task_index: (
            "train" if rank < 700 else "validation" if rank < 850 else "test"
        )
        for rank, task_index in enumerate(ordered)
    }
    return tuple(
        task.model_copy(update={"preassigned_split": split_by_index[index]})
        for index, task in enumerate(tasks)
    )


def _build_fixed_cell_production_tasks(
    source: DataSourcePlan,
    load_case: LoadCasePlan,
    global_path_offset: int,
    shard_path_count: int,
) -> tuple[RVEDataTask, ...]:
    """Expand one exact factorial production design into immutable path tasks."""
    design = source.production_design
    if design is None:
        raise ValueError("Fixed-cell production task expansion requires a design.")
    tasks: list[RVEDataTask] = []
    local_index = 0
    for family in load_case.path_families:
        for amplitude_band in design.amplitude_bands:
            for rate_band in design.rate_bands:
                stratum = f"{family}|{amplitude_band.name}|{rate_band.name}"
                split_ranks = _production_split_ranks(stratum, design)
                for direction_index, direction in enumerate(design.tensor_directions):
                    for repeat_index, increments in enumerate(
                        design.accepted_increment_counts
                    ):
                        within_stratum = direction_index * len(
                            design.accepted_increment_counts
                        ) + repeat_index
                        amplitude_fraction = (within_stratum + 0.5) / design.paths_per_stratum
                        rate_permutation = (7 * within_stratum + 3) % design.paths_per_stratum
                        rate_fraction = (rate_permutation + 0.5) / design.paths_per_stratum
                        amplitude = amplitude_band.lower + amplitude_fraction * (
                            amplitude_band.upper - amplitude_band.lower
                        )
                        target_rate = rate_band.lower + rate_fraction * (
                            rate_band.upper - rate_band.lower
                        )
                        split_rank = split_ranks[within_stratum]
                        split = _production_split(split_rank, design)
                        task_id = (
                            f"{source.name}__{family}__{amplitude_band.name}__"
                            f"{rate_band.name}__{direction}__rep_{repeat_index:02d}"
                        )
                        global_index = global_path_offset + local_index
                        tasks.append(
                            RVEDataTask(
                                task_id=task_id,
                                source_name=source.name,
                                source_kind=source.kind,
                                load_case_name=load_case.name,
                                path_index=local_index,
                                seed=load_case.seed + local_index,
                                shard_index=global_index // shard_path_count,
                                lineage_id=task_id,
                                macro_path_origin=source.macro_path_origin,
                                family=family,
                                amplitude_band=amplitude_band.name,
                                rate_band=rate_band.name,
                                tensor_direction=direction,
                                repeat_index=repeat_index,
                                amplitude=amplitude,
                                target_strain_rate=target_rate,
                                accepted_increment_count=increments,
                                turning_angle=(
                                    design.nonproportional_turning_angles[repeat_index]
                                    if family == "nonproportional"
                                    else None
                                ),
                                design_stratum=stratum,
                                preassigned_split=split,
                                expected_response_regime=amplitude_band.expected_response,
                                snapshot_required=_select_ranked_diagnostic(
                                    split_rank,
                                    design.snapshots_per_stratum,
                                    design.paths_per_stratum,
                                ),
                                tangent_check_required=(
                                    split_rank < design.tangent_checks_per_stratum
                                ),
                            )
                        )
                        local_index += 1
    return tuple(tasks)


def _production_split(
    split_rank: int,
    design: FixedCellProductionDesign,
) -> Literal["train", "validation", "test"]:
    """Assign one exact 14/3/3-style split inside every design stratum."""
    if split_rank < design.train_per_stratum:
        return "train"
    if split_rank < design.train_per_stratum + design.validation_per_stratum:
        return "validation"
    return "test"


def _production_split_ranks(
    stratum: str,
    design: FixedCellProductionDesign,
) -> tuple[int, ...]:
    """Return stable hash ranks so tensor directions are not split by list order."""
    candidates = tuple(range(design.paths_per_stratum))
    ordered = sorted(
        candidates,
        key=lambda index: hashlib.sha256(
            f"{stratum}|candidate_{index:02d}".encode()
        ).hexdigest(),
    )
    ranks = [0] * design.paths_per_stratum
    for rank, candidate in enumerate(ordered):
        ranks[candidate] = rank
    return tuple(ranks)


def _select_ranked_diagnostic(rank: int, count: int, total: int) -> bool:
    """Select deterministic diagnostics across both ends of each stratum."""
    if count == 1:
        return rank == 0
    selected = {
        round(index * (total - 1) / (count - 1))
        for index in range(count)
    }
    return rank in selected


def _require_nonoverlapping_bands(
    bands: tuple[ProductionAmplitudeBand, ...] | tuple[ProductionRateBand, ...],
    label: str,
) -> None:
    """Require ordered adjacent production intervals without overlap."""
    names = tuple(band.name for band in bands)
    if len(names) != len(set(names)):
        raise ValueError(f"Production {label} band names must be unique.")
    for previous, current in zip(bands[:-1], bands[1:], strict=True):
        if current.lower < previous.upper:
            raise ValueError(f"Production {label} bands must not overlap.")
