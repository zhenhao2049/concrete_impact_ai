"""Environment inspection command with explicit dependency profiles.

Author:
    Zhen Hao.
Created:
    2026-07-10.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec

PROFILE_MODULES = {
    "base": ("numpy", "scipy", "threadpoolctl", "yaml", "pydantic", "meshio"),
    "ml": ("numpy", "scipy", "threadpoolctl", "yaml", "pydantic", "meshio", "torch"),
    "gpu": ("numpy", "scipy", "threadpoolctl", "yaml", "pydantic", "meshio", "torch"),
    "petsc": (
        "numpy",
        "scipy",
        "threadpoolctl",
        "yaml",
        "pydantic",
        "meshio",
        "mpi4py",
        "petsc4py",
    ),
    "full": (
        "numpy",
        "scipy",
        "threadpoolctl",
        "yaml",
        "pydantic",
        "meshio",
        "torch",
        "mpi4py",
        "petsc4py",
        "slepc4py",
        "gmsh",
        "pyvista",
        "h5py",
    ),
}


def module_available(module_name: str) -> bool:
    """Check whether a Python module is import-discoverable."""
    return find_spec(module_name) is not None


def inspect_profile(profile: str) -> list[str]:
    """Inspect one declared dependency profile and return missing capabilities."""
    modules = PROFILE_MODULES[profile]
    missing = []
    print(f"Concrete Impact AI environment profile: {profile}")
    for module_name in modules:
        available = module_available(module_name)
        print(f"  {module_name}: {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(module_name)
    if "torch" in modules and module_available("torch"):
        missing.extend(_inspect_torch_capabilities())
    if profile == "gpu" and module_available("torch"):
        missing.extend(_inspect_cuda_capabilities())

    return missing


def _inspect_torch_capabilities() -> list[str]:
    """Require PyTorch transforms used by RNO integration and deployment."""
    import torch

    capabilities = {
        "torch.func.jacfwd": hasattr(torch.func, "jacfwd"),
        "torch.func.vmap": hasattr(torch.func, "vmap"),
        "torch.export.export": hasattr(torch.export, "export"),
        "torch.export.save": hasattr(torch.export, "save"),
        "torch.export.load": hasattr(torch.export, "load"),
    }
    missing = []
    for capability, available in capabilities.items():
        print(f"  {capability}: {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(capability)

    return missing


def _inspect_cuda_capabilities() -> list[str]:
    """Require the CUDA runtime and one visible GPU for training."""
    import torch

    capabilities = {
        "torch.version.cuda": torch.version.cuda is not None,
        "torch.cuda.is_available": torch.cuda.is_available(),
    }
    missing = []
    for capability, available in capabilities.items():
        print(f"  {capability}: {'OK' if available else 'MISSING'}")
        if not available:
            missing.append(capability)
    if torch.cuda.is_available():
        print(f"  torch.cuda.device: {torch.cuda.get_device_name(0)}")

    return missing


def main() -> int:
    """Inspect the selected project dependency profile."""
    parser = argparse.ArgumentParser(description="Inspect concrete-impact dependencies.")
    parser.add_argument("--profile", choices=tuple(PROFILE_MODULES), default="base")
    arguments = parser.parse_args()
    missing = inspect_profile(arguments.profile)

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
