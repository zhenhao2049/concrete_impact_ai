"""Periodic representative-volume-element solvers.

Author:
    Zhen Hao.
Created:
    2026-07-11.
"""

from fem.rve.builders import build_cylindrical_two_phase_rve
from fem.rve.data import (
    PeriodicConstraint,
    RVEExecutionSettings,
    RVEOperatorCache,
    RVEPathResult,
    RVERequest,
    RVEResponse,
    RVEState,
    StructuredHex8RVE,
)
from fem.rve.dynamics import (
    RVEDynamicScreening,
    assemble_rve_consistent_mass,
    compute_linear_rve_dynamic_tangent,
    compute_rve_dynamic_screening,
)
from fem.rve.io import (
    RVE_COMPACT_DATA_SCHEMA_VERSION,
    RVE_DATA_SCHEMA_VERSION,
    RVECompactDataShardWriter,
    RVEDataShardWriter,
    read_rve_path_arrays,
    validate_rve_data_shard,
    write_rve_path_hdf5,
)
from fem.rve.material import RVEHomogenizedMaterial, decode_rve_point_state
from fem.rve.path import (
    AcceptedRVEPathStep,
    iterate_prescribed_rve_path,
    solve_prescribed_rve_path,
)
from fem.rve.periodic import (
    build_structured_hex8_periodic_constraint,
    compute_periodic_fluctuation_error,
)
from fem.rve.solver import (
    RVEEquilibriumError,
    build_rve_operator_cache,
    initialize_rve_state,
    solve_rve_microequilibrium,
)

__all__ = [
    "AcceptedRVEPathStep",
    "PeriodicConstraint",
    "RVEEquilibriumError",
    "RVE_DATA_SCHEMA_VERSION",
    "RVE_COMPACT_DATA_SCHEMA_VERSION",
    "RVECompactDataShardWriter",
    "RVEDataShardWriter",
    "RVEDynamicScreening",
    "RVEPathResult",
    "RVERequest",
    "RVEHomogenizedMaterial",
    "RVEExecutionSettings",
    "RVEOperatorCache",
    "RVEResponse",
    "RVEState",
    "StructuredHex8RVE",
    "build_structured_hex8_periodic_constraint",
    "assemble_rve_consistent_mass",
    "build_cylindrical_two_phase_rve",
    "build_rve_operator_cache",
    "compute_periodic_fluctuation_error",
    "compute_linear_rve_dynamic_tangent",
    "compute_rve_dynamic_screening",
    "decode_rve_point_state",
    "initialize_rve_state",
    "iterate_prescribed_rve_path",
    "read_rve_path_arrays",
    "validate_rve_data_shard",
    "solve_rve_microequilibrium",
    "solve_prescribed_rve_path",
    "write_rve_path_hdf5",
]
