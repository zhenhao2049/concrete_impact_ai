"""PETSc backend interface.

Author:
    Zhen Hao.
Created:
    2026-07-02.
"""


def require_petsc() -> None:
    """Import PETSc backend modules."""
    from petsc4py import PETSc  # noqa: F401


def create_petsc_solver() -> None:
    """Create the project PETSc solver wrapper."""
    raise NotImplementedError("PETSc solver wrapper is not implemented yet.")

