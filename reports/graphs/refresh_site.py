#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = [
    ["python3", "build_experiment_manifest.py"],
    ["python3", "build_code_callouts.py"],
    ["python3", "build_dashboard.py"],
    ["python3", "build_comparison_media.py"],
    ["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
    ["python3", "build_static_site.py"],
    ["python3", "build_static_site.py", "--check"],
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    for cmd in COMMANDS:
        run(cmd)
