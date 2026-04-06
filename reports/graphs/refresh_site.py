#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run(["python3", "build_experiment_manifest.py"])
    run(["python3", "build_code_callouts.py"])
    run(["python3", "build_dashboard.py"])
    run(["python3", "build_comparison_media.py"])
    run(["python3", "build_static_site.py"])
