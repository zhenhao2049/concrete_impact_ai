"""Generic FEM case definitions and runners.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""

from fem.cases.data import CaseDef, OutputDef, RunResult
from fem.cases.records import write_run_records

__all__ = [
    "CaseDef",
    "OutputDef",
    "RunResult",
    "write_run_records",
]
