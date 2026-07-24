"""Run one material-point update with the committed RVE-RNO artifact.

Author:
    Zhen Hao.
Created:
    2026-07-24.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from concrete_impact.nn.config import RVERNOModelConfig
from concrete_impact.nn.deployment import load_rve_rno_state_dict


def main() -> int:
    """Load the public artifact and print one stress/state update.

    Inputs:
        Command-line paths to the checkpoint and its metadata contract.
    Outputs:
        A JSON record containing stress, latent state, dissipation, and free energy.
    """
    parser = argparse.ArgumentParser(description="Run one RVE-RNO material update.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/rve_rno/c48_v2_liu_direct"),
    )
    args = parser.parse_args()
    config = RVERNOModelConfig(
        family="energy_rve_rno",
        time_representation="continuous",
        integrator="current_state_stress_forward_euler",
        evolution="direct_rate",
        reference_time_scale=0.02,
        latent_dimension=6,
        hidden_size=128,
        hidden_layers=4,
        activation="silu",
        dtype="float32",
        device="cpu",
    )
    model = load_rve_rno_state_dict(
        config,
        args.artifact_root / "best_model.pt",
        args.artifact_root / "artifact_metadata.json",
    )
    strain = torch.tensor(
        [[1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=model.dtype,
        device=model.device,
    )
    response = model.update_batch(
        strain,
        model.initial_state_batch(batch_size=1),
        torch.tensor([1.0e-4], dtype=model.dtype, device=model.device),
        model.prepare_energy_anchors(),
    )
    print(
        json.dumps(
            {
                "stress": response.stress.detach().cpu().tolist(),
                "latent_state": response.latent_state.detach().cpu().tolist(),
                "dissipation": response.dissipation.detach().cpu().tolist(),
                "free_energy": response.free_energy.detach().cpu().tolist(),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
