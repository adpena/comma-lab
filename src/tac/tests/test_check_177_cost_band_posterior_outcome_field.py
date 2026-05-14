# SPDX-License-Identifier: MIT
"""Tests for Catalog #177 — check_cost_band_posterior_rows_have_outcome_field.

FIX-WAVE-2 R2-3 (2026-05-13). Read-side defense-in-depth sister of
Catalog #175 (write side). Refuses stored rows in
``.omx/state/cost_band_posterior.jsonl`` that are missing the
``outcome`` field.

Memory: feedback_fix_wave_2_r2_findings_LANDED_20260513.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cost_band_posterior_rows_have_outcome_field,
)


def _write_posterior(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_check_177_live_count_zero():
    """Live posterior should have 0 violations (all rows have outcome)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert violations == [], (
        f"Live posterior violations should be 0, got: {violations}"
    )


def test_check_177_returns_empty_when_no_posterior_file(tmp_path):
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_detects_row_missing_outcome(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    _write_posterior(
        posterior,
        [
            {
                "schema": "cost_band_posterior_v1",
                "logged_at_utc": "2026-05-13T00:00:00+00:00",
                "dispatch_label": "test-1",
                "trainer": "experiments/train_x.py",
                "platform": "modal",
                "gpu": "T4",
                "epochs": 3000,
                "batch_size": 16,
                "all_flags_on": True,
                "actual_wall_clock_sec": 100.0,
                "actual_cost_usd": 0.5,
                # NOTE: missing 'outcome'
            }
        ],
    )
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("missing required `outcome` field" in v for v in violations)
    assert any("test-1" in v for v in violations)


def test_check_177_accepts_row_with_outcome(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    _write_posterior(
        posterior,
        [
            {
                "schema": "cost_band_posterior_v1",
                "logged_at_utc": "2026-05-13T00:00:00+00:00",
                "dispatch_label": "test-2",
                "trainer": "experiments/train_x.py",
                "platform": "modal",
                "gpu": "T4",
                "epochs": 3000,
                "batch_size": 16,
                "all_flags_on": True,
                "actual_wall_clock_sec": 100.0,
                "actual_cost_usd": 0.5,
                "outcome": "successful_dispatch",
            }
        ],
    )
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_strict_raises_preflight_error(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    _write_posterior(
        posterior,
        [
            {
                "schema": "cost_band_posterior_v1",
                "dispatch_label": "bad-row",
                # missing outcome
            }
        ],
    )
    with pytest.raises(PreflightError):
        check_cost_band_posterior_rows_have_outcome_field(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_177_skips_malformed_json(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    posterior.parent.mkdir(parents=True, exist_ok=True)
    posterior.write_text("this is not json\n{garbage\n", encoding="utf-8")
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_skips_schema_mismatch_rows(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    _write_posterior(
        posterior,
        [
            {"schema": "some_other_schema", "dispatch_label": "off-schema"},
        ],
    )
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_handles_mixed_rows(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    _write_posterior(
        posterior,
        [
            {
                "schema": "cost_band_posterior_v1",
                "dispatch_label": "good-1",
                "outcome": "successful_dispatch",
            },
            {
                "schema": "cost_band_posterior_v1",
                "dispatch_label": "bad-1",
                # missing outcome
            },
            {
                "schema": "cost_band_posterior_v1",
                "dispatch_label": "good-2",
                "outcome": "failed_dispatch",
            },
        ],
    )
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "bad-1" in violations[0]


def test_check_177_handles_empty_file(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    posterior.parent.mkdir(parents=True, exist_ok=True)
    posterior.write_text("", encoding="utf-8")
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_handles_blank_lines(tmp_path):
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    posterior.parent.mkdir(parents=True, exist_ok=True)
    posterior.write_text(
        "\n\n"
        '{"schema": "cost_band_posterior_v1", "dispatch_label": "x", "outcome": "successful_dispatch"}\n'
        "\n",
        encoding="utf-8",
    )
    violations = check_cost_band_posterior_rows_have_outcome_field(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_177_load_anchors_now_tags_legacy_pre_nv7():
    """Companion fix: load_anchors() now tags missing-outcome rows with
    LEGACY_PRE_NV7 explicit tag rather than silently coercing to
    SUCCESSFUL_DISPATCH.
    """
    from tac.cost_band_calibration import (
        LEGACY_PRE_NV7,
        SUCCESSFUL_DISPATCH,
        VALID_OUTCOMES,
    )
    # LEGACY_PRE_NV7 must be a member of VALID_OUTCOMES so load_anchors
    # accepts rows tagged with it.
    assert LEGACY_PRE_NV7 in VALID_OUTCOMES
    assert LEGACY_PRE_NV7 != SUCCESSFUL_DISPATCH
    # The string value is the explicit canonical token.
    assert LEGACY_PRE_NV7 == "legacy_pre_nv7"


def test_check_177_legacy_pre_nv7_excluded_from_predict_by_default(tmp_path):
    """LEGACY_PRE_NV7 anchors should not contribute to predict()'s
    percentile band by default (they are NOT SUCCESSFUL_DISPATCH).
    """
    from tac.cost_band_calibration import (
        LEGACY_PRE_NV7,
        load_anchors,
    )
    posterior = tmp_path / ".omx" / "state" / "cost_band_posterior.jsonl"
    # Pre-NV7 anchor with missing outcome (will be tagged LEGACY_PRE_NV7)
    _write_posterior(
        posterior,
        [
            {
                "schema": "cost_band_posterior_v1",
                "logged_at_utc": "2026-05-13T00:00:00+00:00",
                "dispatch_label": "legacy-row",
                "trainer": "experiments/train_x.py",
                "platform": "modal",
                "gpu": "T4",
                "epochs": 3000,
                "batch_size": 16,
                "all_flags_on": True,
                "actual_wall_clock_sec": 100.0,
                "actual_cost_usd": 0.5,
                # NO outcome field
            }
        ],
    )
    anchors = load_anchors(posterior)
    assert len(anchors) == 1
    assert anchors[0].outcome == LEGACY_PRE_NV7
