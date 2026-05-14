# SPDX-License-Identifier: MIT
"""Tests for autopilot consumption of macOS-CPU advisory manifests.

Per operator routing 2026-05-13: the autopilot dispatch ranker is wired to
RANK candidates derived from macOS-CPU advisory manifests so cheap pre-GPU
ordering happens for free. These tests pin:
  1) the loader rejects promoted manifests / rows (defense-in-depth)
  2) loaded CandidateRow rows carry the ranking-only notes prefix
  3) the proxy_evidence tagger annotates halt events
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tac.optimization.macos_cpu_advisory_signal import (
    EVIDENCE_GRADE,
    build_macos_cpu_advisory_signal_manifest,
    json_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from cathedral_autopilot_autonomous_loop import (  # noqa: E402
    MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG,
    CandidateRow,
    EventClass,
    HaltEvent,
    load_candidates_from_macos_cpu_advisory_manifest,
    tag_halt_events_with_proxy_evidence,
)


def _write_manifest(tmp_path: Path) -> Path:
    obs = [
        {
            "family": "pr106_hnerv_cluster",
            "variant_id": "r2",
            "archive_bytes": 186_822,
            "score": 0.1966,
            "d_seg": 0.067,
            "d_pose": 0.000034,
            "archive_sha256": "a" * 64,
        },
        {
            "family": "pr101_lossy_coarsening",
            "variant_id": "blocks4_7bit",
            "archive_bytes": 177_903,
            "score": 0.2024,
            "d_seg": 0.070,
            "d_pose": 0.000040,
            "archive_sha256": "b" * 64,
        },
    ]
    manifest = build_macos_cpu_advisory_signal_manifest(
        obs, source="fixture", run_id="auto_loader_test"
    )
    p = tmp_path / "macos_cpu_manifest.json"
    p.write_text(json_text(manifest))
    return p


def test_load_returns_candidate_rows_with_ranking_only_notes(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path)
    rows = load_candidates_from_macos_cpu_advisory_manifest(p)
    assert len(rows) == 2
    for r in rows:
        assert isinstance(r, CandidateRow)
        assert "[macOS-CPU advisory; ranking-only]" in r.notes
        assert f"proxy_evidence: {MACOS_CPU_ADVISORY_PROXY_EVIDENCE_TAG}" in r.notes
        assert "promotion_blocked" in r.notes
        assert r.candidate_id.startswith("macos_cpu_advisory__")
        # Per CLAUDE.md: macOS-CPU rows participate in ranking but NEVER promote.
        # The blockers list propagates the manifest's dispatch_blockers so the
        # operator sees every reason at halt-event time.
        assert len(r.blockers) >= 4
        assert any("macos_cpu_advisory_not_score_evidence" in b for b in r.blockers)


def test_load_predicted_score_delta_uses_projected_p50_when_available(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path)
    rows = load_candidates_from_macos_cpu_advisory_manifest(p)
    # The projected_contest_cpu_score_p50 is the score itself for PR107
    # placeholder (no offset). Both rows should have non-zero predicted_score_delta.
    scores = sorted(r.predicted_score_delta for r in rows)
    assert scores[0] == pytest.approx(0.1966)
    assert scores[1] == pytest.approx(0.2024)


def test_load_refuses_promoted_manifest(tmp_path: Path) -> None:
    # Hand-craft a "manifest" with promoted top-level flag.
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps({
        "schema": "macos_cpu_advisory_signal_manifest.v1",
        "score_claim": True,  # FORBIDDEN
        "rows": [],
        "ranking_atoms": [],
    }))
    with pytest.raises(ValueError, match="score_claim=True"):
        load_candidates_from_macos_cpu_advisory_manifest(bad)


def test_load_refuses_promoted_row(tmp_path: Path) -> None:
    bad = tmp_path / "bad_row.json"
    bad.write_text(json.dumps({
        "schema": "macos_cpu_advisory_signal_manifest.v1",
        "score_claim": False,
        "rows": [
            {
                "family": "f",
                "variant_id": "v",
                "evidence_grade": EVIDENCE_GRADE,
                "score_macos_cpu": 0.2,
                "archive_bytes": 1234,
                "promotion_eligible": True,  # FORBIDDEN
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "ranking_atoms": [],
    }))
    with pytest.raises(ValueError, match="promotion_eligible"):
        load_candidates_from_macos_cpu_advisory_manifest(bad)


def test_load_refuses_unknown_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad_schema.json"
    bad.write_text(json.dumps({
        "schema": "some_other_schema.v1",
        "rows": [],
        "ranking_atoms": [],
    }))
    with pytest.raises(ValueError, match="unexpected schema"):
        load_candidates_from_macos_cpu_advisory_manifest(bad)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_candidates_from_macos_cpu_advisory_manifest(tmp_path / "nonexistent.json")


def test_load_skips_rows_with_no_score(tmp_path: Path) -> None:
    bad = tmp_path / "no_score.json"
    bad.write_text(json.dumps({
        "schema": "macos_cpu_advisory_signal_manifest.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rows": [
            {
                "family": "f",
                "variant_id": "v",
                "archive_bytes": 1234,
                # no score_macos_cpu, no projected_contest_cpu_score_p50
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
    }))
    rows = load_candidates_from_macos_cpu_advisory_manifest(bad)
    assert rows == []


def test_tag_halt_events_with_proxy_evidence(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path)
    candidates = load_candidates_from_macos_cpu_advisory_manifest(p)
    # Construct fake halt events for the candidates plus one unrelated.
    halt_events = [
        HaltEvent(
            event_class=EventClass.DISPATCH,
            candidate_id=candidates[0].candidate_id,
            reason="test",
            predicted_score_delta=candidates[0].predicted_score_delta,
            estimated_cost_usd=0.0,
            requires_approval=True,
        ),
        HaltEvent(
            event_class=EventClass.DISPATCH,
            candidate_id="unrelated_candidate",
            reason="test",
            predicted_score_delta=-0.01,
            estimated_cost_usd=0.0,
            requires_approval=True,
        ),
    ]
    tag_halt_events_with_proxy_evidence(halt_events, candidates=candidates)
    # First halt event should be tagged.
    assert "proxy_evidence=macos_cpu_advisory" in halt_events[0].decision_notes
    # Second halt event should NOT be tagged.
    assert "proxy_evidence" not in halt_events[1].decision_notes


def test_tag_halt_events_is_idempotent(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path)
    candidates = load_candidates_from_macos_cpu_advisory_manifest(p)
    halt = HaltEvent(
        event_class=EventClass.DISPATCH,
        candidate_id=candidates[0].candidate_id,
        reason="t",
        predicted_score_delta=0.0,
        estimated_cost_usd=0.0,
        requires_approval=True,
    )
    tag_halt_events_with_proxy_evidence([halt], candidates=candidates)
    notes_after_first = halt.decision_notes
    tag_halt_events_with_proxy_evidence([halt], candidates=candidates)
    # Re-running must not duplicate the marker.
    assert halt.decision_notes == notes_after_first
    assert halt.decision_notes.count("proxy_evidence=macos_cpu_advisory") == 1


def test_load_default_cost_and_eig_are_zero(tmp_path: Path) -> None:
    """The autopilot loader does NOT invent a dispatch cost. Default 0.0 makes
    these rows participate in ranking-only ordering; operator authorization
    cannot fire them via the le-$5 envelope because the cost defaults to 0.0
    which is explicitly refused by OperatorAuthorizedModeConfig.can_authorize."""
    p = _write_manifest(tmp_path)
    rows = load_candidates_from_macos_cpu_advisory_manifest(p)
    for r in rows:
        assert r.estimated_dispatch_cost_usd == 0.0
        assert r.expected_information_gain == 0.0


def test_load_cost_and_eig_overrides_threaded_through(tmp_path: Path) -> None:
    p = _write_manifest(tmp_path)
    rows = load_candidates_from_macos_cpu_advisory_manifest(
        p,
        default_estimated_dispatch_cost_usd=2.50,
        default_expected_information_gain=0.005,
    )
    for r in rows:
        assert r.estimated_dispatch_cost_usd == 2.50
        assert r.expected_information_gain == 0.005
