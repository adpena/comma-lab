"""Tests for experiments/canonical_local_auth_eval_smoke.py.

The smoke tool is the deliverable that closes the static-vs-pipeline
gap (Lane RM-d 0.mkv post-mortem). Bugs here regress Check 64.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SMOKE_TOOL = REPO / "experiments/canonical_local_auth_eval_smoke.py"
DEFAULT_FIXTURE = REPO / "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip"


def _run_smoke(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    """Invoke the smoke tool as a subprocess (mirrors operator usage)."""
    return subprocess.run(
        [sys.executable, str(SMOKE_TOOL), *args],
        capture_output=capture, text=True, timeout=120,
    )


def test_smoke_tool_exists() -> None:
    """The canonical smoke tool is committed at the expected path."""
    assert SMOKE_TOOL.exists(), f"smoke tool missing: {SMOKE_TOOL}"


def test_smoke_default_fixture_exists() -> None:
    """The default fixture (Lane G v3 archive) is committed."""
    assert DEFAULT_FIXTURE.exists(), (
        f"default fixture missing: {DEFAULT_FIXTURE}. The smoke tool "
        f"is canonical-only when the fixture is present."
    )


def test_smoke_passes_on_lane_g_v3() -> None:
    """The Lane G v3 fixture (1.05 [contest-CUDA]) must pass smoke."""
    res = _run_smoke("--lane", "lane_g_v3_corrected_kl_weight", "--quiet")
    assert res.returncode == 0, (
        f"smoke FAILED on canonical fixture (this means the pipeline is "
        f"broken at the codebase level):\nstdout:\n{res.stdout}\n"
        f"stderr:\n{res.stderr}"
    )
    assert "[canonical-e2e-smoke] PASS" in res.stdout


def test_smoke_writes_proof_atomic() -> None:
    """Smoke writes a proof entry to .omx/state/lane_e2e_smoke_proofs.json."""
    proofs = REPO / ".omx/state/lane_e2e_smoke_proofs.json"
    res = _run_smoke("--lane", "lane_g_v3_corrected_kl_weight", "--quiet")
    assert res.returncode == 0
    assert proofs.exists()
    data = json.loads(proofs.read_text())
    assert "remote_lane_g_v3_corrected_kl_weight" in data
    proof = data["remote_lane_g_v3_corrected_kl_weight"]
    # Required fields enforced by Check 64
    for field in ("timestamp_utc", "archive_sha256", "stages_passed",
                  "tool_version"):
        assert field in proof, f"proof missing required field: {field}"
    # Schema invariants
    assert proof["tool_version"] == 1
    assert len(proof["archive_sha256"]) == 64  # full SHA256 hex
    assert isinstance(proof["stages_passed"], list)
    assert "extract" in proof["stages_passed"]
    assert "config_env" in proof["stages_passed"]
    assert "inflate_dispatch_path" in proof["stages_passed"]


def test_smoke_under_60_seconds_budget() -> None:
    """The full smoke must complete in < 60 seconds per the design budget.

    If this test starts failing, an expensive new stage was added to the
    smoke tool. Either justify the new stage or move it out of smoke.
    """
    import time
    t0 = time.monotonic()
    res = _run_smoke("--lane", "lane_g_v3_corrected_kl_weight", "--quiet")
    elapsed = time.monotonic() - t0
    assert res.returncode == 0
    assert elapsed < 60.0, (
        f"smoke exceeded 60s budget: {elapsed:.1f}s. The smoke is meant "
        f"to be fast (designed for local MPS/CPU, sub-60s). Move expensive "
        f"checks to auth-eval-on-CUDA, NOT into smoke."
    )


def test_smoke_pass_sentinel_format() -> None:
    """The PASS sentinel format must include lane= and stages= and elapsed=
    fields so log scrapers can parse smoke runs."""
    res = _run_smoke("--lane", "lane_g_v3_corrected_kl_weight", "--quiet")
    assert res.returncode == 0
    out = res.stdout
    # The sentinel format is: [canonical-e2e-smoke] PASS lane=X score=N/A
    # stages=N elapsed=Xs sha256=PREFIX
    assert "[canonical-e2e-smoke] PASS lane=" in out
    assert "stages=" in out
    assert "elapsed=" in out
    assert "sha256=" in out


def test_smoke_fail_sentinel_on_missing_fixture(tmp_path: Path) -> None:
    """Missing fixture archive produces the canonical FAIL sentinel +
    nonzero exit so CI / operators detect the failure."""
    res = _run_smoke(
        "--lane", "lane_g_v3_corrected_kl_weight",
        "--fixture-archive", str(tmp_path / "does_not_exist.zip"),
        "--quiet",
    )
    assert res.returncode != 0
    assert "[canonical-e2e-smoke] FAIL" in res.stderr


def test_smoke_lane_glob_smokes_multiple() -> None:
    """--lane-glob smokes every matching lane and emits one sentinel per lane."""
    res = _run_smoke("--lane-glob", "remote_lane_g_v3_*", "--quiet")
    assert res.returncode == 0
    # Lane G v3 has multiple variants — at least 2 PASS sentinels expected
    assert res.stdout.count("[canonical-e2e-smoke] PASS") >= 2


def test_smoke_backfill_all_smokes_every_lane() -> None:
    """--backfill-all smokes every scripts/remote_lane_*.sh script."""
    res = _run_smoke("--backfill-all", "--quiet")
    assert res.returncode == 0
    # Confirm summary line lands and reports nonzero count
    assert "[canonical-e2e-smoke] SUMMARY:" in res.stdout
    # No failures on canonical fixture
    assert " 0 failed" in res.stdout


def test_smoke_lane_name_accepts_short_form() -> None:
    """`--lane g_v3_corrected_kl_weight` (without remote_lane_ prefix) works."""
    res = _run_smoke("--lane", "g_v3_corrected_kl_weight", "--quiet")
    # If short form is accepted, exit 0; otherwise the lane name lookup fails.
    assert res.returncode == 0
    assert "[canonical-e2e-smoke] PASS lane=remote_lane_g_v3_corrected_kl_weight" in res.stdout


def test_smoke_whitelist_parity_with_contest_auth_eval() -> None:
    """The smoke tool's archive whitelist MUST stay in parity with
    contest_auth_eval._KNOWN_ARCHIVE_SUFFIXES. Drift = false-positive /
    false-negative archive validation.
    """
    cae = (REPO / "experiments/contest_auth_eval.py").read_text()
    smoke = SMOKE_TOOL.read_text()
    # Both must list these canonical suffixes.
    for sfx in (".bin", ".bin.br", ".mkv", ".pt", ".json", ".npz"):
        assert sfx in cae, f"contest_auth_eval missing whitelist suffix {sfx}"
        assert sfx in smoke, f"smoke tool missing whitelist suffix {sfx}"
