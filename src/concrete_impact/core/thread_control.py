"""Process-local numerical-library thread control.

Contents:
    Context-managed BLAS and OpenMP thread limits for numerical workers.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


def numerical_thread_limit(thread_count: int) -> AbstractContextManager[Any]:
    """Create a process-local BLAS/OpenMP thread limit context."""
    if thread_count <= 0:
        raise ValueError("Numerical-library thread count must be positive.")
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=thread_count)
