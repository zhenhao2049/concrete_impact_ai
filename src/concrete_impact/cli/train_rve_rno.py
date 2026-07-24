"""Fixed-cell RVE-RNO training command-line entrypoint.

Contents:
    Timestamped run setup, signal handling, training dispatch, and failure records.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import argparse
import signal
from datetime import datetime
from pathlib import Path
from time import time
from types import FrameType

from concrete_impact.core.progress import JsonProgressRecorder, write_json_atomic
from concrete_impact.nn.config import load_rve_rno_training_config
from concrete_impact.nn.training.rve_rno import TrainingNumericalError, train_rve_rno


class TrainingInterruptedError(RuntimeError):
    """Represent an explicit operating-system interruption of RVE-RNO training."""


def _raise_training_interrupted(signum: int, frame: FrameType | None) -> None:
    """Convert SIGTERM into a recordable training interruption."""
    del frame
    raise TrainingInterruptedError(f"RVE-RNO training received signal {signum}.")


def main() -> int:
    """Run one explicitly configured RVE-RNO training job with failure logging."""
    parser = argparse.ArgumentParser(description="Train an explicit fixed-cell RVE-RNO.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_rve_rno_training_config(args.config)
    output_root = config.output_directory
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    config = config.model_copy(update={"output_directory": output_root / run_name})
    recorder = JsonProgressRecorder(
        output_root / "progress.json",
        {
            "calculation": "rve_rno_training",
            "run_directory": str(config.output_directory),
        },
        output_root / "summary.log",
    )
    started = time()
    signal.signal(signal.SIGTERM, _raise_training_interrupted)
    write_json_atomic(
        output_root / "active_run.json",
        {
            "config": str(args.config),
            "run_directory": str(config.output_directory),
        },
    )
    try:
        summary = train_rve_rno(config, recorder)
    except (KeyboardInterrupt, TrainingInterruptedError) as error:
        output = Path(config.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        interrupted = {
            "status": "interrupted",
            "error_type": type(error).__name__,
            "message": str(error),
            "config": str(args.config),
            "run_directory": str(output),
            "started_unix_time": started,
            "interrupted_unix_time": time(),
        }
        write_json_atomic(output / "interruption.json", interrupted)
        recorder.publish_summary(
            "rve_rno_training_interrupted",
            {
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    except Exception as error:
        output = Path(config.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        failed = time()
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
            "config": str(args.config),
            "run_directory": str(output),
            "started_unix_time": started,
            "failed_unix_time": failed,
        }
        if isinstance(error, TrainingNumericalError):
            failure["reason"] = error.reason
            failure["diagnostics"] = error.diagnostics
        write_json_atomic(output / "failure.json", failure)
        recorder.publish_summary(
            "rve_rno_training_failed",
            {
                "passed": False,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    del summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
