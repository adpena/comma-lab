"""Regression: Apogee intN Pareto output must remain forensic-only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _pareto_output(*extra_args: str) -> str:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "apogee_intN_pareto.py"),
            *extra_args,
        ],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


def test_pareto_tool_runs_clean():
    """Smoke: the Pareto tool runs on the live manifests."""
    out = _pareto_output("--emit-forensic-one-liners")
    assert "FORENSIC BYTE-ONLY ONE-LINERS" in out
    assert "ready_for_exact_eval_dispatch=false" in out


def test_pareto_default_blocks_dispatch_one_liners():
    """Default Pareto output must not encourage Apogee score dispatch."""
    out = _pareto_output()
    assert "ready_for_exact_eval_dispatch=false" in out
    assert "No dispatch one-liners emitted" in out
    assert "launch_lane_on_vastai.py full" not in out


def test_pareto_json_marks_apogee_not_dispatch_ready():
    """Machine-readable matrix must fail closed for score-lane dispatch."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "apogee_intN_pareto.py"), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert "missing_contest_faithful_distortion_model" in payload["dispatch_blockers"]
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in payload["rows"])


def test_pareto_forensic_flag_still_emits_no_executable_launch_command():
    """Compatibility forensic mode must never look copy-paste launchable."""
    out = _pareto_output("--emit-forensic-one-liners")
    assert "withheld" in out
    assert "launch_lane_on_vastai.py full" not in out
    assert "APOGEE_INTN_BITS=" not in out
    assert "--predicted-band" not in out
    assert "--lane-script" not in out
