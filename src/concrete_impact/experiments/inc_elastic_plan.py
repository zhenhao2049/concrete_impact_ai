"""Validated plans for fixed-system linear-elastic INC data.

Author:
    Zhen Hao.
Created:
    2026-09-15.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from fem.io.config import load_yaml_config
from fem.materials.data import compute_isotropic_pressure_wave_speed

PulseKind = Literal["sine_squared", "half_sine", "raised_cosine", "double_pulse"]
SplitKind = Literal["pilot", "train", "validation", "test"]


class StrictElasticINCModel(BaseModel):
    """Forbid undeclared elastic INC plan fields."""

    model_config = ConfigDict(extra="forbid")


class ElasticINCReferenceThresholds(StrictElasticINCModel):
    """Store fine-reference and label-reconstruction tolerances."""

    displacement: PositiveFloat
    velocity: PositiveFloat
    energy: PositiveFloat
    label_reconstruction: PositiveFloat
    minimum_response: PositiveFloat


class ElasticINCPlan(StrictElasticINCModel):
    """Define one fixed-system linear-elastic INC dataset."""

    schema_version: Literal["2.0"]
    dataset_type: Literal["linear_elastic_velocity_verlet_inc"]
    design: Literal["p0_7", "p1_pilot_12", "p1_official_144"]
    release_status: Literal["pilot", "draft", "approved"]
    output_directory: Path
    model: dict[str, Any]
    plane_state: Literal["plane_stress"]
    thickness: PositiveFloat
    fixed_boundary: str
    load_boundary: str
    time_step: PositiveFloat
    num_steps: PositiveInt
    coarse_cfl_ratio: float = Field(gt=0.0, le=1.0)
    fine_ratio: PositiveInt
    reference_ratio: PositiveInt
    pressure_amplitudes: tuple[PositiveFloat, ...]
    duration_ratios: tuple[PositiveFloat, ...]
    pilot_pressure_amplitude: PositiveFloat
    compression: Literal["gzip"]
    compression_level: int = Field(ge=0, le=9)
    chunk_steps: PositiveInt
    progress_updates: PositiveInt
    thresholds: ElasticINCReferenceThresholds

    @model_validator(mode="after")
    def require_fixed_design(self) -> ElasticINCPlan:
        """Require the reviewed P0 or P1 fixed-system design."""
        if self.fixed_boundary != "left" or self.load_boundary != "right":
            raise ValueError("Elastic INC requires a fixed left end and right-end loading.")
        if self.model["geometry"]["size"] != [1.0, 0.05]:
            raise ValueError("Elastic INC requires the reviewed 1.0 x 0.05 geometry.")
        if self.reference_ratio <= self.fine_ratio:
            raise ValueError("INC reference ratio must exceed the fine ratio.")
        if self.reference_ratio % self.fine_ratio != 0:
            raise ValueError("INC fine and reference ratios must be nested.")
        if self.design == "p0_7":
            if self.release_status != "pilot":
                raise ValueError("P0 data must retain pilot release status.")
            if self.fine_ratio != 8 or self.reference_ratio != 16:
                raise ValueError("P0 requires coarse:fine:reference ratios 1:8:16.")
            if self.model["mesh"]["divisions"] != [40, 1]:
                raise ValueError("P0 requires the reviewed 40 x 1 mesh.")
        else:
            allowed_ratios = ((4, 8), (8, 16), (16, 32), (32, 64), (64, 128))
            if (self.fine_ratio, self.reference_ratio) not in allowed_ratios:
                raise ValueError(
                    "P1 requires coarse:fine:reference ratios 1:4:8, 1:8:16, "
                    "1:16:32, 1:32:64, or 1:64:128."
                )
            if self.duration_ratios != (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                raise ValueError("P1 requires the six reviewed duration ratios.")
            if self.model["mesh"]["divisions"] != [40, 4]:
                raise ValueError("P1 requires the reviewed 40 x 4 mesh.")
        if self.design == "p1_pilot_12" and self.release_status != "pilot":
            raise ValueError("P1 trial data must retain pilot release status.")
        if self.design == "p1_official_144" and self.pressure_amplitudes != (
            7.5e5,
            1.0e6,
            1.25e6,
        ):
            raise ValueError("P1 official data requires the three reviewed amplitudes.")
        return self


class ElasticINCTask(StrictElasticINCModel):
    """Store one complete elastic INC loading path."""

    task_id: str
    split: SplitKind
    pressure_amplitude: PositiveFloat
    pulse_duration: PositiveFloat
    pulse_kind: PulseKind
    spatial_coefficients: tuple[float, float, float, float]


PURE_SPATIAL_COEFFICIENTS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
MIXED_SPATIAL_COEFFICIENTS = (
    tuple(np.asarray((1.0, 1.0, 0.0, 0.0)) / np.sqrt(2.0)),
    tuple(np.asarray((1.0, 0.0, 1.0, 0.0)) / np.sqrt(2.0)),
    tuple(np.asarray((0.0, 1.0, 0.0, 1.0)) / np.sqrt(2.0)),
    (0.5, 0.5, 0.5, 0.5),
)
P1_PULSE_KINDS: tuple[PulseKind, ...] = (
    "half_sine",
    "raised_cosine",
    "double_pulse",
)


def load_elastic_inc_plan(path: str | Path) -> ElasticINCPlan:
    """Load one strict linear-elastic INC plan."""
    return ElasticINCPlan.model_validate(load_yaml_config(path))


def build_elastic_inc_tasks(plan: ElasticINCPlan) -> tuple[ElasticINCTask, ...]:
    """Expand one reviewed P0 or P1 path design."""
    builders = {
        "p0_7": _build_p0_tasks,
        "p1_pilot_12": _build_p1_pilot_tasks,
        "p1_official_144": _build_p1_official_tasks,
    }
    tasks = builders[plan.design](plan)
    expected = {"p0_7": 7, "p1_pilot_12": 12, "p1_official_144": 144}[plan.design]
    if len(tasks) != expected:
        raise ValueError(f"INC path count differs from the reviewed design: {len(tasks)}.")
    task_ids = tuple(task.task_id for task in tasks)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("INC task identifiers must be unique.")
    return tasks


def build_control_parameters(plan: ElasticINCPlan, task: ElasticINCTask) -> np.ndarray:
    """Build one model-ready path-control vector."""
    if plan.design == "p0_7":
        return np.asarray(
            (task.pressure_amplitude, task.pulse_duration),
            dtype=np.float64,
        )
    pulse_one_hot = tuple(float(task.pulse_kind == kind) for kind in P1_PULSE_KINDS)
    return np.asarray(
        (
            task.pressure_amplitude,
            task.pulse_duration,
            *pulse_one_hot,
            *task.spatial_coefficients,
        ),
        dtype=np.float64,
    )


def control_parameter_order(plan: ElasticINCPlan) -> tuple[str, ...]:
    """Return the fixed control-vector field order."""
    if plan.design == "p0_7":
        return ("pressure_amplitude", "pulse_duration")
    return (
        "pressure_amplitude",
        "pulse_duration",
        "pulse_half_sine",
        "pulse_raised_cosine",
        "pulse_double_pulse",
        "load_coefficient_1",
        "load_coefficient_2",
        "load_coefficient_3",
        "load_coefficient_4",
    )


def _build_p0_tasks(plan: ElasticINCPlan) -> tuple[ElasticINCTask, ...]:
    """Build the fixed seven-path minimal dataset."""
    values = (
        ("train_a075_t060", "train", 7.5e5, 6.0e-5),
        ("train_a075_t080", "train", 7.5e5, 8.0e-5),
        ("train_a100_t060", "train", 1.0e6, 6.0e-5),
        ("train_a100_t080", "train", 1.0e6, 8.0e-5),
        ("validation_a0875_t065", "validation", 8.75e5, 6.5e-5),
        ("validation_a0875_t075", "validation", 8.75e5, 7.5e-5),
        ("test_a0875_t070", "test", 8.75e5, 7.0e-5),
    )
    return tuple(
        ElasticINCTask(
            task_id=task_id,
            split=cast(SplitKind, split),
            pressure_amplitude=amplitude,
            pulse_duration=duration,
            pulse_kind="sine_squared",
            spatial_coefficients=PURE_SPATIAL_COEFFICIENTS[0],
        )
        for task_id, split, amplitude, duration in values
    )


def _build_p1_pilot_tasks(plan: ElasticINCPlan) -> tuple[ElasticINCTask, ...]:
    """Build the twelve representative P1 trial paths."""
    duration = _p1_durations(plan)
    specifications = (
        ("pure1_short", PURE_SPATIAL_COEFFICIENTS[0], 0, "half_sine"),
        ("pure1_long", PURE_SPATIAL_COEFFICIENTS[0], 5, "raised_cosine"),
        ("pure2_short", PURE_SPATIAL_COEFFICIENTS[1], 0, "raised_cosine"),
        ("pure2_long", PURE_SPATIAL_COEFFICIENTS[1], 5, "double_pulse"),
        ("pure3_short", PURE_SPATIAL_COEFFICIENTS[2], 0, "double_pulse"),
        ("pure3_long", PURE_SPATIAL_COEFFICIENTS[2], 5, "half_sine"),
        ("pure4_short", PURE_SPATIAL_COEFFICIENTS[3], 0, "half_sine"),
        ("pure4_long", PURE_SPATIAL_COEFFICIENTS[3], 5, "double_pulse"),
        ("mixed1_mid", MIXED_SPATIAL_COEFFICIENTS[0], 2, "raised_cosine"),
        ("mixed2_mid", MIXED_SPATIAL_COEFFICIENTS[1], 3, "double_pulse"),
        ("mixed3_mid", MIXED_SPATIAL_COEFFICIENTS[2], 2, "half_sine"),
        ("mixed4_mid", MIXED_SPATIAL_COEFFICIENTS[3], 3, "raised_cosine"),
    )
    return tuple(
        ElasticINCTask(
            task_id=f"pilot_{name}_{pulse_kind}",
            split="pilot",
            pressure_amplitude=plan.pilot_pressure_amplitude,
            pulse_duration=duration[duration_id],
            pulse_kind=cast(PulseKind, pulse_kind),
            spatial_coefficients=coefficients,
        )
        for name, coefficients, duration_id, pulse_kind in specifications
    )


def _build_p1_official_tasks(plan: ElasticINCPlan) -> tuple[ElasticINCTask, ...]:
    """Build the balanced 144-path P1 dataset."""
    durations = _p1_durations(plan)
    coefficients = PURE_SPATIAL_COEFFICIENTS + MIXED_SPATIAL_COEFFICIENTS
    split_by_duration = ("train", "validation", "train", "train", "test", "train")
    tasks = []
    for spatial_id, spatial_values in enumerate(coefficients):
        for pulse_id, pulse_kind in enumerate(P1_PULSE_KINDS):
            for duration_id, pulse_duration in enumerate(durations):
                amplitude_id = (spatial_id + pulse_id + duration_id) % 3
                tasks.append(
                    ElasticINCTask(
                        task_id=(
                            f"official_s{spatial_id + 1}_w{pulse_id + 1}_"
                            f"d{duration_id + 1}_a{amplitude_id + 1}"
                        ),
                        split=cast(SplitKind, split_by_duration[duration_id]),
                        pressure_amplitude=plan.pressure_amplitudes[amplitude_id],
                        pulse_duration=pulse_duration,
                        pulse_kind=pulse_kind,
                        spatial_coefficients=spatial_values,
                    )
                )
    return tuple(tasks)


def _p1_durations(plan: ElasticINCPlan) -> tuple[float, ...]:
    """Convert transit-time ratios into physical pulse durations."""
    geometry = plan.model["geometry"]
    material = plan.model["material"]
    length = float(geometry["size"][0])
    wave_speed = compute_isotropic_pressure_wave_speed(
        float(material["young_modulus"]),
        float(material["poisson_ratio"]),
        float(material["density"]),
    )
    transit_time = length / wave_speed
    return tuple(transit_time * ratio for ratio in plan.duration_ratios)
