"""Smoke + schema test for tools/predicted_vs_actual_reconciler.py.

The reconciler joins repack_metadata.json (predicted_band) with
contest_auth_eval.json (actual final_score) per apogee_intN bits config.
This test verifies the join logic + the JSON output schema, so future
manifest-format changes break at CI time.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _run_json() -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "predicted_vs_actual_reconciler.py"), "--json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def test_reconciler_runs_clean():
    out = _run_json()
    assert "pr106_baseline_score" in out
    assert "n_configs" in out
    assert "n_landed" in out
    assert "n_in_band" in out
    assert "n_beats_pr106" in out
    assert "rows" in out


def test_reconciler_consumes_all_existing_manifests():
    """Every apogee_int{N}_repack_*/repack_metadata.json must appear as a row."""
    out = _run_json()
    manifests = list((REPO / "experiments" / "results").glob(
        "apogee_int*_repack_*/repack_metadata.json"
    ))
    assert out["n_configs"] == len(manifests), (
        f"Reconciler produced {out['n_configs']} rows but {len(manifests)} manifests exist on disk. "
        "Manifest discovery glob is wrong or schema drift broke the parser."
    )


def test_reconciler_row_schema_complete():
    out = _run_json()
    if not out["rows"]:
        pytest.skip("no manifests yet")
    required_keys = {
        "bits", "predicted_low", "predicted_high", "archive_size_bytes",
        "distortion_risk", "rate_score_delta", "prediction_status",
        "ready_for_exact_eval_dispatch", "dispatch_blockers",
        "actual_score", "actual_path", "in_band", "beats_pr106",
        "device", "samples",
    }
    for r in out["rows"]:
        missing = required_keys - set(r.keys())
        assert not missing, f"row missing keys {missing}: {r}"
        assert r["ready_for_exact_eval_dispatch"] is False
        assert "missing_contest_faithful_distortion_model" in r["dispatch_blockers"]


def test_reconciler_band_parsed_correctly():
    """predicted_low must be < predicted_high (band parser sanity)."""
    out = _run_json()
    for r in out["rows"]:
        assert r["predicted_low"] < r["predicted_high"], (
            f"int{r['bits']} band [{r['predicted_low']}, {r['predicted_high']}] "
            f"is malformed (low must be < high)"
        )


def test_reconciler_baseline_constant_matches_pr106_score():
    """The hard-coded PR106 baseline must match the well-known public 0.20945673."""
    out = _run_json()
    assert abs(out["pr106_baseline_score"] - 0.20945673) < 1e-9, (
        "PR106 baseline drifted from canonical 0.20945673 — update reconciler if a "
        "newer public-frontier baseline is confirmed."
    )
