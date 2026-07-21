"""Tests for tac.artifact_quarantine (operator 2026-07-21 quarantine directive)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.artifact_quarantine import (
    QuarantineManifestError,
    has_waiver,
    load_manifest,
    refuse_message,
    scan_text,
)

WAIVER = "QUARANTINE-WAIVER: HARVEST-SIGNAL-ONLY — consuming measured receipts only, no bytes"


def _mk_root(tmp_path: Path, rows: list[dict] | None = None, raw: str | None = None) -> Path:
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    p = state / "artifact_quarantine.json"
    if raw is not None:
        p.write_text(raw, encoding="utf-8")
    else:
        p.write_text(json.dumps({"quarantined": rows or []}), encoding="utf-8")
    return tmp_path


def test_hit_without_waiver_is_flagged(tmp_path):
    root = _mk_root(tmp_path, [{"id": "ep725", "kind": "token", "what": "w", "reason": "r"}])
    hits = scan_text("anchor on the ep725 checkpoint", repo_root=root)
    assert "ep725" in [h.identifier for h in hits]
    assert "ep725" in refuse_message(hits)


def test_clean_prompt_passes(tmp_path):
    root = _mk_root(tmp_path, [{"id": "ep725", "kind": "token"}])
    assert scan_text("from-scratch inverse solve, gt_n600 only", repo_root=root) == []


def test_manifest_extends_defaults(tmp_path):
    root = _mk_root(tmp_path, [{"id": "customXYZ", "kind": "token", "what": "w", "reason": "r"}])
    hits = scan_text("uses customXYZ and ep725", repo_root=root)
    assert {h.identifier for h in hits} == {"customXYZ", "ep725"}


def test_waiver_suppresses_hits(tmp_path):
    root = _mk_root(tmp_path, [{"id": "ep725", "kind": "token"}])
    assert scan_text(f"cites ep725 receipts. {WAIVER}", repo_root=root) == []


def test_waiver_requires_real_rationale():
    assert not has_waiver("QUARANTINE-WAIVER: HARVEST-SIGNAL-ONLY — x")
    assert has_waiver(WAIVER)


def test_sha_prefix_is_case_sensitive_token_ci_is_not(tmp_path):
    rows = [
        {"id": "149fefd097c1fa85", "kind": "archive_sha_prefix"},
        {"id": "r1 dxi", "kind": "token_ci"},
    ]
    root = _mk_root(tmp_path, rows)
    assert len(scan_text("uses 149fefd097c1fa85... bytes", repo_root=root)) == 1
    assert len(scan_text("transplant the R1 DXI section", repo_root=root)) == 1


def test_missing_manifest_falls_back_to_tracked_defaults(tmp_path):
    (tmp_path / ".omx").mkdir()
    assert load_manifest(repo_root=tmp_path) is None
    hits = scan_text("warm-start from ep725", repo_root=tmp_path)
    assert any(h.identifier == "ep725" for h in hits)
    assert scan_text("clean v10 seed compose", repo_root=tmp_path) == []


def test_corrupt_manifest_fails_closed(tmp_path):
    root = _mk_root(tmp_path, raw="{not json")
    with pytest.raises(QuarantineManifestError):
        scan_text("anything", repo_root=root)


def test_manifest_missing_list_fails_closed(tmp_path):
    root = _mk_root(tmp_path, raw=json.dumps({"policy": "x"}))
    with pytest.raises(QuarantineManifestError):
        scan_text("anything", repo_root=root)


def test_live_repo_manifest_flags_known_poison():
    repo = Path(__file__).resolve().parents[3]
    if not (repo / ".omx/state/artifact_quarantine.json").is_file():
        pytest.skip("live manifest absent")
    hits = scan_text("warm-start from ep725 and splice 149fefd097c1fa85", repo_root=repo)
    assert {h.identifier for h in hits} >= {"ep725", "149fefd097c1fa85"}
    assert scan_text("pure v10 seed compose from gt_n600", repo_root=repo) == []
