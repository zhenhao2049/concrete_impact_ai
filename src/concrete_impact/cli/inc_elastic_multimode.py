"""Command-line workflow for the elastic multimode INC delivery model.

Author:
    Zhen Hao.
Created:
    2026-09-16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from concrete_impact.experiments.inc_elastic_generation import load_elastic_inc_system
from concrete_impact.nn.config_inc_elastic import load_inc_elastic_multimode_config
from concrete_impact.nn.datasets.inc_elastic_multimode import (
    prepare_inc_elastic_training_cache,
    validate_inc_elastic_sources,
)
from concrete_impact.nn.evaluation.inc_elastic_benchmark import (
    benchmark_inc_elastic_delivery,
)
from concrete_impact.nn.evaluation.inc_elastic_multimode import (
    evaluate_inc_elastic_paths,
    write_inc_elastic_evaluation,
)
from concrete_impact.nn.training.inc_elastic_multimode import (
    INCElasticTrainingError,
    inspect_inc_elastic_cuda,
    load_inc_elastic_artifact,
    select_inc_elastic_kinematic_checkpoint,
    train_inc_elastic_multimode,
)


def main() -> int:
    """Dispatch strict preflight, cache, training, evaluation, or timing stages."""
    parser = argparse.ArgumentParser(description="Run elastic multimode INC workflow.")
    parser.add_argument(
        "command",
        choices=("preflight", "prepare", "run", "select-response", "evaluate", "benchmark"),
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_inc_elastic_multimode_config(args.config)

    if args.command == "preflight":
        validate_inc_elastic_sources(config)
        cuda = inspect_inc_elastic_cuda(config)
        if config.output_directory.exists():
            raise FileExistsError(config.output_directory)
        print(json.dumps(cuda, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        prepare_inc_elastic_training_cache(config)
        return 0
    if args.command == "run":
        _run_training(config, Path(args.config))
        return 0
    if args.command == "select-response":
        result = select_inc_elastic_kinematic_checkpoint(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        _run_evaluation(config)
        return 0
    if args.command == "benchmark":
        _run_benchmark(config)
        return 0
    raise AssertionError("Unreachable elastic INC command dispatch.")


def _run_training(config, config_path: Path) -> None:
    """Run training and preserve structured diagnostics after output creation."""
    try:
        train_inc_elastic_multimode(config)
    except Exception as error:
        if config.output_directory.is_dir():
            failure_path = config.output_directory / "failure.json"
            if not failure_path.exists():
                failure: dict[str, object] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "config": str(config_path),
                }
                if isinstance(error, INCElasticTrainingError):
                    failure["reason"] = error.reason
                    failure["diagnostics"] = error.diagnostics
                failure_path.write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        raise


def _run_evaluation(config) -> None:
    """Open the locked test paths after kinematic validation selects the response model."""
    summary_path = config.output_directory / "response_candidate_summary.json"
    response_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if response_summary["kinematic_acceptance_passed"] is not True:
        raise ValueError("Elastic INC kinematic validation did not pass.")
    if response_summary["test_paths_opened"] != []:
        raise ValueError("Elastic INC independent test paths were already opened.")
    model, _, normalization, frequencies, final_time = load_inc_elastic_artifact(
        config,
        checkpoint_name="response_model.pt",
    )
    system = load_elastic_inc_system(config.data.system_path)
    response_summary["test_paths_opened"] = list(config.data.test_paths)
    summary_path.write_text(
        json.dumps(response_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    records, trajectories = evaluate_inc_elastic_paths(
        config,
        model,
        system,
        normalization,
        frequencies,
        final_time,
        config.data.test_paths,
        keep_trajectories=True,
    )
    summary = write_inc_elastic_evaluation(
        config,
        records,
        trajectories,
        config.output_directory / "evaluation",
    )
    if summary["delivery_acceptance_passed"] is not True:
        raise INCElasticTrainingError("independent_test_kinematic_acceptance_failed", summary)


def _run_benchmark(config) -> None:
    """Run repeated fine, coarse, CPU-INC, and CUDA-INC timings."""
    evaluation_path = config.output_directory / "evaluation" / "evaluation_summary.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation["delivery_acceptance_passed"] is not True:
        raise ValueError("Elastic INC independent kinematic test did not pass.")
    model, _, normalization, frequencies, final_time = load_inc_elastic_artifact(
        config,
        checkpoint_name="response_model.pt",
    )
    system = load_elastic_inc_system(config.data.system_path)
    benchmark_inc_elastic_delivery(
        config,
        model,
        system,
        normalization,
        frequencies,
        final_time,
        config.output_directory / "benchmark",
    )


if __name__ == "__main__":
    raise SystemExit(main())
