"""Surrogate configuration and capability exceptions.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""


class SurrogateConfigError(ValueError):
    """Report an invalid surrogate configuration or artifact contract."""


class SurrogateCapabilityError(RuntimeError):
    """Report a requested output that a surrogate cannot provide."""
