"""Tests for the strict concrete-impact benchmark result contract.

Author:
    Zhen Hao.
Created:
    2026-07-12.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from concrete_impact.benchmarks.errors import BenchmarkAcceptanceError
from concrete_impact.benchmarks.registry import BENCHMARKS, get_benchmark_runner
from concrete_impact.cli.run_benchmark import main
from fem.cases import RunResult
from fem.dynamics.material import NonlinearStepConvergenceError


def test_registered_benchmark_result_is_enriched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolved configuration and execution metadata are mandatory outputs."""
    monkeypatch.setitem(
        BENCHMARKS,
        "contract_pass",
        lambda _config: RunResult("contract_pass", {"error": 0.0}, {}, True),
    )
    config = {"case": {"name": "contract_pass"}}

    result = get_benchmark_runner("contract_pass")(config)

    assert result.resolved_config == config
    assert result.failure_reason is None
    assert len(result.metadata["git_commit"]) == 40
    assert result.metadata["duration_seconds"] >= 0.0


@pytest.mark.parametrize("returned", [None, "invalid"])
def test_registered_benchmark_rejects_non_result(
    monkeypatch: pytest.MonkeyPatch,
    returned,
) -> None:
    """Verify registry runners cannot omit the RunResult contract."""
    monkeypatch.setitem(BENCHMARKS, "contract_invalid", lambda _config: returned)

    with pytest.raises(TypeError, match="must return RunResult"):
        get_benchmark_runner("contract_invalid")({"case": {"name": "contract_invalid"}})


def test_registered_benchmark_rejects_nonfinite_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-finite acceptance metrics terminate benchmark execution."""
    monkeypatch.setitem(
        BENCHMARKS,
        "contract_nonfinite",
        lambda _config: RunResult(
            "contract_nonfinite",
            {"error": np.inf},
            {},
            False,
        ),
    )

    with pytest.raises(ValueError, match="non-finite"):
        get_benchmark_runner("contract_nonfinite")(
            {"case": {"name": "contract_nonfinite"}}
        )


def test_benchmark_cli_raises_on_failed_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a completed failed benchmark cannot return CLI success."""
    config_path = tmp_path / "failure.yaml"
    config_path.write_text(
        f"case:\n  name: contract_failure\noutput:\n  root: {tmp_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        BENCHMARKS,
        "contract_failure",
        lambda _config: RunResult("contract_failure", {"error": 1.0}, {}, False),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-run", "--config", str(config_path)],
    )

    with pytest.raises(BenchmarkAcceptanceError) as caught:
        main()

    assert caught.value.reason == "benchmark_acceptance_criteria_not_met"
    assert (tmp_path / "benchmark_result.json").is_file()
    assert (tmp_path / "acceptance_failure.json").is_file()


def test_benchmark_cli_records_structured_numerical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a nonlinear solver exception preserves its complete diagnostics."""
    config_path = tmp_path / "numerical_failure.yaml"
    config_path.write_text(
        f"case:\n  name: contract_numerical_failure\noutput:\n  root: {tmp_path}\n",
        encoding="utf-8",
    )

    def fail(_config):
        raise NonlinearStepConvergenceError(
            "armijo_exhausted",
            {"iteration": 7, "increment": np.asarray([1.0, -2.0])},
        )

    monkeypatch.setitem(BENCHMARKS, "contract_numerical_failure", fail)
    monkeypatch.setattr(
        "sys.argv",
        ["concrete-impact-run", "--config", str(config_path)],
    )

    with pytest.raises(NonlinearStepConvergenceError):
        main()

    report = json.loads((tmp_path / "numerical_failure.json").read_text())
    assert report["failure_reason"] == "armijo_exhausted"
    assert report["diagnostics"]["iteration"] == 7
    assert report["diagnostics"]["increment"] == [1.0, -2.0]
