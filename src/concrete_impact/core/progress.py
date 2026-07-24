"""Progress recording and inactivity monitoring for project calculations.

Contents:
    Atomic progress logs, background submission, monitoring, and process diagnostics.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from fem.solvers.progress import SolverProgressEvent


@dataclass(frozen=True)
class BackgroundMonitorSettings:
    """Configure polling and the maximum interval without observable progress."""

    poll_seconds: float = 60.0
    inactivity_seconds: float = 1200.0


class ProcessHandle(Protocol):
    """Describe the subprocess operations required by the monitor."""

    pid: int

    def poll(self) -> int | None:
        """Return the process status without blocking."""


class JsonProgressRecorder:
    """Atomically publish the most recent accepted solver step."""

    def __init__(
        self,
        output_path: str | Path,
        context: dict[str, Any] | None = None,
        summary_log_path: str | Path | None = None,
        log_accepted_steps: bool = True,
    ) -> None:
        """Initialize one recorder for a single project calculation."""
        self.output_path = Path(output_path)
        self.context = {} if context is None else dict(context)
        self.summary_log_path = None if summary_log_path is None else Path(summary_log_path)
        self.log_accepted_steps = log_accepted_steps

    def __call__(self, event: SolverProgressEvent) -> None:
        """Write one accepted-step event without exposing trial states."""
        payload = {**self.context, **asdict(event), "updated_unix_time": time.time()}
        write_json_atomic(self.output_path, payload)
        if self.summary_log_path is not None and self.log_accepted_steps:
            append_summary_log(self.summary_log_path, payload)

    def publish_stage(self, stage: str, completed: int, total: int) -> None:
        """Publish completion of a benchmark, mesh, or data-generation stage."""
        payload = {
            **self.context,
            "stage": stage,
            "completed": completed,
            "total": total,
            "updated_unix_time": time.time(),
        }
        write_json_atomic(self.output_path, payload)
        if self.summary_log_path is not None:
            append_summary_log(self.summary_log_path, payload)

    def publish_summary(self, stage: str, fields: dict[str, Any]) -> None:
        """Publish one completed path or benchmark summary."""
        payload = {
            **self.context,
            "stage": stage,
            **fields,
            "updated_unix_time": time.time(),
        }
        write_json_atomic(self.output_path, payload)
        if self.summary_log_path is not None:
            append_summary_log(self.summary_log_path, payload)


def append_summary_log(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one concise timestamped progress record with one write call."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_format_summary_log_block(payload) + "\n").encode("utf-8")
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _format_summary_log_block(payload: dict[str, Any]) -> str:
    """Format one stable progress block using no more than five lines."""
    timestamp = datetime.fromtimestamp(float(payload["updated_unix_time"]), UTC).isoformat()
    lines = [
        f"[{timestamp}] stage={_format_summary_value(payload['stage'])}",
    ]
    object_fields = (
        (("task_id", "task"),)
        if "task_id" in payload
        else (
            ("case_name", "case"),
            ("calculation", "calculation"),
            ("source_name", "source"),
            ("run_directory", "run"),
        )
    )
    _append_summary_group(
        lines,
        "object",
        payload,
        object_fields,
    )
    context_fields = (
        (
            ("mesh_level", "mesh"),
            ("path_family", "path_family"),
            ("family", "family"),
            ("regime", "regime"),
        )
        if "task_id" in payload
        else (
            ("mesh_level", "mesh"),
            ("path_family", "path_family"),
            ("family", "family"),
            ("load_case", "load_case"),
            ("regime", "regime"),
            ("scheme", "scheme"),
            ("acceptance_scope", "acceptance"),
        )
    )
    _append_summary_group(
        lines,
        "context",
        payload,
        context_fields,
    )
    progress_fields = _summary_progress_fields(payload)
    if progress_fields:
        lines.append(f"  progress: {' '.join(progress_fields)}")
    _append_summary_group(
        lines,
        "solve",
        payload,
        (
            ("newton_iterations", "Newton"),
            ("residual_norm", "residual"),
            ("residual_tolerance", "tolerance"),
            ("armijo_backtracks", "Armijo"),
            ("elapsed_seconds", "elapsed_s"),
            ("path_solve_seconds", "path_solve_s"),
            ("path_write_seconds", "path_write_s"),
            ("train_loss", "train_loss"),
            ("validation_loss", "validation_loss"),
            ("best_validation_loss", "best_validation_loss"),
            ("stress_loss", "stress_loss"),
            ("thermodynamic_violation_loss", "thermo_loss"),
            ("dissipation_loss", "dissipation_loss"),
            ("steps_per_second", "steps_per_s"),
            ("test_loss", "test_loss"),
            ("maximum_path_stress_error", "max_path_stress_error"),
            ("maximum_newton_iterations", "Newton_max"),
            ("maximum_newton_residual", "residual_max"),
            ("peak_cuda_memory_bytes", "peak_cuda_bytes"),
        ),
    )
    _append_summary_group(
        lines,
        "failure",
        payload,
        (
            ("error_type", "type"),
            ("error_message", "message"),
        ),
    )
    if len(lines) > 5:
        raise ValueError(f"Summary log record exceeds five lines: stage={payload['stage']}.")
    return "\n".join(lines)


def _append_summary_group(
    lines: list[str],
    label: str,
    payload: dict[str, Any],
    fields: tuple[tuple[str, str], ...],
) -> None:
    """Append one nonempty labeled group to a concise summary block."""
    formatted = [
        f"{display_name}={_format_summary_value(payload[payload_name])}"
        for payload_name, display_name in fields
        if payload_name in payload and payload[payload_name] is not None
    ]
    if formatted:
        lines.append(f"  {label}: {' '.join(formatted)}")


def _summary_progress_fields(payload: dict[str, Any]) -> list[str]:
    """Collect step, completion, and acceptance fields into one short line."""
    fields = []
    if "accepted_step" in payload:
        fields.append(
            "step="
            f"{_format_summary_value(payload['accepted_step'])}/"
            f"{_format_summary_value(payload['nominal_total_steps'])}"
        )
        fields.append(f"time={_format_summary_value(payload['physical_time'])}")
        fields.append(f"final_time={_format_summary_value(payload['final_time'])}")
    if "completed" in payload:
        fields.append(
            "completed="
            f"{_format_summary_value(payload['completed'])}/"
            f"{_format_summary_value(payload['total'])}"
        )
    for name in (
        "accepted_steps",
        "final_time",
        "passed",
        "best_epoch",
        "selected_level",
        "selection_quality",
    ):
        if name in payload and not (name == "final_time" and "accepted_step" in payload):
            display_name = {
                "selected_level": "selected_mesh",
                "selection_quality": "quality",
            }.get(name, name)
            fields.append(f"{display_name}={_format_summary_value(payload[name])}")
    return fields


def _format_summary_value(value: Any) -> str:
    """Format one scalar log value without locale-dependent output."""
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Replace one JSON record atomically inside its result directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def run_monitored_command(
    command: Sequence[str],
    output_directory: str | Path,
    settings: BackgroundMonitorSettings | None = None,
) -> int:
    """Run one project command and terminate only its process group after inactivity."""
    monitor_settings = BackgroundMonitorSettings() if settings is None else settings
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "stdout.log"
    stderr_path = output_root / "stderr.log"
    status_path = output_root / "run_status.json"
    progress_path = output_root / "progress.json"
    started = time.time()
    config_sha256 = _command_config_sha256(command)
    git_metadata = _git_metadata()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_stream,
        stderr_path.open("w", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.Popen(
            tuple(command),
            cwd=Path.cwd(),
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
            text=True,
        )
        write_json_atomic(
            status_path,
            {
                "status": "running",
                "command": list(command),
                "pid": process.pid,
                "process_group": process.pid,
                "started_unix_time": started,
                "config_sha256": config_sha256,
                "git": git_metadata,
            },
        )
        termination = monitor_process_progress(
            process,
            progress_path,
            monitor_settings,
            process_group=process.pid,
        )
        if termination is not None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            write_json_atomic(
                output_root / "termination.json",
                {
                    **termination,
                    "command": list(command),
                    "config_sha256": config_sha256,
                    "git": git_metadata,
                },
            )
        return_code = process.wait()
    write_json_atomic(
        status_path,
        {
            "status": "completed" if return_code == 0 else "failed",
            "command": list(command),
            "pid": process.pid,
            "process_group": process.pid,
            "started_unix_time": started,
            "finished_unix_time": time.time(),
            "return_code": return_code,
            "config_sha256": config_sha256,
            "git": git_metadata,
        },
    )
    return return_code


def submit_monitored_command(
    command: Sequence[str],
    output_directory: str | Path,
    settings: BackgroundMonitorSettings | None = None,
) -> int:
    """Submit one detached monitor process and return without waiting."""
    monitor_settings = BackgroundMonitorSettings() if settings is None else settings
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    monitor_command = _build_detached_monitor_command(
        command,
        output_root,
        monitor_settings,
    )
    monitor_stdout_path = output_root / "monitor_stdout.log"
    monitor_stderr_path = output_root / "monitor_stderr.log"
    with (
        monitor_stdout_path.open("a", encoding="utf-8") as stdout_stream,
        monitor_stderr_path.open("a", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.Popen(
            monitor_command,
            cwd=Path.cwd(),
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
            close_fds=True,
            text=True,
        )
    write_json_atomic(
        output_root / "submission.json",
        {
            "status": "submitted",
            "monitor_pid": process.pid,
            "command": list(command),
            "monitor_command": list(monitor_command),
            "submitted_unix_time": time.time(),
            "config_sha256": _command_config_sha256(command),
            "git": _git_metadata(),
        },
    )
    return process.pid


def _build_detached_monitor_command(
    command: Sequence[str],
    output_directory: Path,
    settings: BackgroundMonitorSettings,
) -> tuple[str, ...]:
    """Build the non-recursive monitor command used by detached submission."""
    return (
        sys.executable,
        "-m",
        "concrete_impact.cli.run_background",
        "--output",
        str(output_directory),
        "--poll-seconds",
        str(settings.poll_seconds),
        "--inactivity-seconds",
        str(settings.inactivity_seconds),
        "--",
        *command,
    )


def monitor_process_progress(
    process: ProcessHandle,
    progress_path: str | Path,
    settings: BackgroundMonitorSettings,
    process_group: int,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    cpu_reader: Callable[[int], float] | None = None,
) -> dict[str, Any] | None:
    """Return termination diagnostics after a continuous interval without progress."""
    read_cpu = read_process_group_cpu_seconds if cpu_reader is None else cpu_reader
    last_token = _read_progress_token(Path(progress_path))
    last_cpu = read_cpu(process_group)
    last_progress_time = clock()
    while process.poll() is None:
        sleeper(settings.poll_seconds)
        token = _read_progress_token(Path(progress_path))
        cpu_seconds = read_cpu(process_group)
        now = clock()
        if token != last_token or cpu_seconds > last_cpu:
            last_token = token
            last_cpu = cpu_seconds
            last_progress_time = now
        if now - last_progress_time >= settings.inactivity_seconds:
            return {
                "reason": "no_observable_progress",
                "pid": process.pid,
                "process_group": process_group,
                "inactivity_seconds": now - last_progress_time,
                "last_progress_token": list(last_token),
                "last_cpu_seconds": last_cpu,
            }
    return None


def read_process_group_cpu_seconds(process_group: int) -> float:
    """Sum Linux process CPU time for one explicitly launched process group."""
    clock_ticks = os.sysconf("SC_CLK_TCK")
    total_ticks = 0
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat_record = stat_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        record_process_group, user_ticks, system_ticks = _parse_linux_process_stat(stat_record)
        if record_process_group == process_group:
            total_ticks += user_ticks + system_ticks
    return total_ticks / clock_ticks


def _parse_linux_process_stat(stat_record: str) -> tuple[int, int, int]:
    """Parse process-group and CPU ticks from one Linux proc stat record."""
    command_end = stat_record.rfind(")")
    command_start = stat_record.find("(")
    if command_start < 0 or command_end <= command_start:
        raise ValueError(f"Invalid Linux process stat command field: {stat_record!r}.")
    fields = stat_record[command_end + 1 :].split()
    if len(fields) < 13:
        raise ValueError(f"Incomplete Linux process stat record: {stat_record!r}.")
    return int(fields[2]), int(fields[11]), int(fields[12])


def _read_progress_token(path: Path) -> tuple[str, int, int]:
    """Read the stage and accepted-count token from one atomic progress record."""
    if not path.exists():
        return ("", -1, -1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = int(payload.get("accepted_step", payload.get("completed", -1)))
    return (str(payload["stage"]), completed, path.stat().st_mtime_ns)


def _command_config_sha256(command: Sequence[str]) -> dict[str, str]:
    """Hash every explicit command-line configuration file before launch."""
    config_paths = [
        Path(command[index + 1]) for index, value in enumerate(command) if value == "--config"
    ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in config_paths}


def _git_metadata() -> dict[str, str | bool]:
    """Read the current repository revision and worktree modification state."""
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1"),
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status)}
