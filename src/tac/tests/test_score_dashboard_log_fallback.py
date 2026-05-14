# SPDX-License-Identifier: MIT
"""Regression test for tools/score_dashboard.py auth_eval.log fallback path.

After commit dbb0032d, the dashboard scans `auth_eval.log` files in dirs not
already covered by canonical `contest_auth_eval*.json`, extracting embedded
RESULT_JSON for visibility into Lightning/Vast.ai dispatches whose JSON
didn't sync back. This added 293 scores (5.6×) to the dashboard view.

These tests pin the new behavior so a future refactor doesn't silently
re-introduce the blind spot:

  test_dashboard_picks_up_log_files
    Asserts at least 1 score row's path ends in `auth_eval.log` —
    confirms the fallback parser is reachable from `scan()`.

  test_dashboard_pr106_frontier_surfaces
    Asserts the canonical PR106 score (0.20945673) appears with the
    expected archive bytes (186,239) — the public frontier at top of
    `operator_briefing` Phase 2.

  test_dashboard_canonical_json_takes_precedence
    Asserts dirs with BOTH .json AND .log only produce ONE row (the
    JSON one), preserving the precedence ordering documented in the
    scan() docstring.

  test_dashboard_log_parser_extracts_components
    Asserts the log-parsed rows have populated seg_dist_avg /
    pose_dist_avg / device fields (the columns the operator briefing
    relies on for in-band/out-of-band detection).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DASHBOARD = REPO / "tools" / "score_dashboard.py"


def _run_json(*extra_args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(DASHBOARD), "--json", *extra_args],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def test_dashboard_picks_up_log_files():
    """At least one row should come from an auth_eval.log file."""
    out = _run_json()
    log_paths = [r["path"] for r in out["rows"] if r["path"].endswith("auth_eval.log")]
    assert log_paths, (
        "Dashboard scanned but found ZERO auth_eval.log files. "
        "The fallback parser path may have been silently disabled. "
        "If JSONs were synced back from remote, update this assertion's expectation."
    )


def test_dashboard_pr106_frontier_surfaces():
    """The canonical PR106 frontier score (0.20945673) should appear at the top."""
    out = _run_json("--filter", "pr106")
    if out["n_total"] == 0:
        pytest.skip("PR106 intake not on disk")
    # Find the adapter variant (the one that's the documented frontier in reports/latest.md)
    pr106_adapter = [
        r for r in out["rows"]
        if "pr106_belt_and_suspenders_adapter" in r["path"]
        and r["path"].endswith("auth_eval.log")
    ]
    if not pr106_adapter:
        pytest.skip("PR106 adapter dispatch not on disk")
    row = pr106_adapter[0]
    assert row["score"] == pytest.approx(0.20945673, abs=1e-7), (
        f"PR106 adapter canonical score is {row['score']}; expected 0.20945673 "
        f"per reports/latest.md frontier section."
    )
    assert row["archive_bytes"] == 186239, (
        f"PR106 adapter archive_bytes is {row['archive_bytes']}; expected 186,239."
    )


def test_dashboard_canonical_json_takes_precedence():
    """Dirs with BOTH a contest_auth_eval.json AND an auth_eval.log should produce ONE row.

    The scan() pass-1 (JSON) wins; pass-2 (log) skips dirs already in seen_dirs.
    """
    base = REPO / "experiments" / "results"
    if not base.is_dir():
        pytest.skip("experiments/results not on disk")
    # Find a dir that has both files
    both_dirs: list[Path] = []
    for log_path in base.rglob("auth_eval.log"):
        for json_pattern in ("contest_auth_eval.json", "contest_auth_eval.adjudicated.json"):
            if (log_path.parent / json_pattern).exists():
                both_dirs.append(log_path.parent)
                break
    if not both_dirs:
        pytest.skip("no dir contains both auth_eval.log and contest_auth_eval*.json")
    out = _run_json()
    sample_dir = str(both_dirs[0].relative_to(REPO))
    rows_in_dir = [r for r in out["rows"] if sample_dir in r["path"]]
    log_rows = [r for r in rows_in_dir if r["path"].endswith("auth_eval.log")]
    assert not log_rows, (
        f"Dir {sample_dir} has both .json and .log but produced log row(s): "
        f"{[r['path'] for r in log_rows]}. "
        "Pass-2 should have skipped this dir per the seen_dirs precedence rule."
    )


def test_dashboard_log_parser_extracts_components():
    """Log-parsed rows must populate seg/pose/rate/samples/device fields."""
    out = _run_json()
    log_rows = [r for r in out["rows"] if r["path"].endswith("auth_eval.log")]
    if not log_rows:
        pytest.skip("no log-parsed rows")
    # Pick a row with a known-good schema (PR106 family, recent dispatches)
    pr_rows = [r for r in log_rows if "exact_eval_public_pr1" in r["path"]]
    if not pr_rows:
        pytest.skip("no PR-family log rows")
    sample = pr_rows[0]
    assert sample["score"] is not None, "log row missing score"
    assert sample["archive_bytes"] is not None, "log row missing archive_bytes"
    assert sample["seg_dist_avg"] is not None, "log row missing seg_dist_avg"
    assert sample["pose_dist_avg"] is not None, "log row missing pose_dist_avg"
    assert sample["samples"] is not None, "log row missing samples"
    assert sample["device"] in ("cuda", "cpu", "mps", "?"), (
        f"log row device field is {sample['device']!r}; expected cuda/cpu/mps/?"
    )
