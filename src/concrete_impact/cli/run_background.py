"""Command-line launcher for monitored project calculations.

Contents:
    Foreground and detached monitor command dispatch.
Author:
    Zhen Hao.
Created:
    2026-07-14.
"""

from __future__ import annotations

import argparse

from concrete_impact.core.progress import (
    BackgroundMonitorSettings,
    run_monitored_command,
    submit_monitored_command,
)


def main() -> int:
    """Launch one command with accepted-progress and CPU inactivity monitoring."""
    parser = argparse.ArgumentParser(description="Run one monitored project command.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--inactivity-seconds", type=float, default=1200.0)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        raise ValueError("Background monitor requires a project command.")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        raise ValueError("Background monitor command delimiter has no command.")
    settings = BackgroundMonitorSettings(args.poll_seconds, args.inactivity_seconds)
    if args.detach:
        monitor_pid = submit_monitored_command(command, args.output, settings)
        print(f"Submitted monitored project calculation: monitor_pid={monitor_pid}")
        return 0
    return run_monitored_command(command, args.output, settings)


if __name__ == "__main__":
    raise SystemExit(main())
