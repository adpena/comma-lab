# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.materialize_g2_a2_composition_receipt as materializer
from tac.witness_dsl.dynamic_frontier_target import DynamicFrontierTargetSnapshot
from tac.witness_dsl.taskspace_pair_fragment_receiver import parse_taskspace_pair_fragment_receipt


def _frontier() -> DynamicFrontierTargetSnapshot:
    return DynamicFrontierTargetSnapshot(
        pointer_path="/synthetic/test-only/canonical_frontier_pointer.json",
        pointer_bytes=123,
        pointer_sha256="9" * 64,
        pointer_device=1,
        pointer_inode=2,
        pointer_mtime_ns=3,
        last_refreshed_utc="2030-01-01T00:00:00+00:00",
        source_snapshot_at_utc="2030-01-01T00:00:00+00:00",
        target_score=0.172,
        selected_axis="official_leaderboard",
        selected_source="upstream_official_leaderboard",
        selected_source_kind="external_public_leaderboard_target",
        selected_score_precision="official_display",
        selected_custody="external target only",
        selected_evidence_grade="[synthetic-test-only]",
        selected_archive_sha256=None,
        selected_lane_id=None,
        selected_hardware_substrate=None,
        selection_rule="synthetic test fixture",
    )


def test_materializer_replays_pair_and_binds_matched_a_only_causality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materializer,
        "verify_dynamic_frontier_target_snapshot",
        lambda snapshot: snapshot,
    )
    payload = materializer.build_receipt(repo_root=materializer.REPO, frontier=_frontier())
    envelope = json.loads(payload.decode("ascii"))
    body = envelope["body"]

    assert envelope["schema"] == materializer.SCHEMA
    assert envelope["body_sha256"] == hashlib.sha256(materializer._canonical_json(body)).hexdigest()
    assert all(body["matched_causal_proof"].values())
    assert body["truth"]["bounded_pair_composition_complete"] is True
    assert body["truth"]["source_custodied_counted_p_present"] is False
    assert body["truth"]["score_claim"] is False
    assert body["truth"]["candidate_archive_eligible"] is False
    baseline = parse_taskspace_pair_fragment_receipt(materializer._canonical_json(body["baseline"]))
    counterfactual = parse_taskspace_pair_fragment_receipt(materializer._canonical_json(body["a_only_counterfactual"]))
    assert baseline.sections[0] == counterfactual.sections[0]
    assert baseline.sections[1] != counterfactual.sections[1]
    assert baseline.scorer_y1_sha256 == counterfactual.scorer_y1_sha256
    assert baseline.scorer_y0_sha256 != counterfactual.scorer_y0_sha256


def test_receipt_install_is_write_once_or_equal(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    materializer._write_once_or_equal(output, b"first")
    materializer._write_once_or_equal(output, b"first")

    assert output.read_bytes() == b"first"
    with pytest.raises(materializer.CompositionReceiptError, match="different or unsafe"):
        materializer._write_once_or_equal(output, b"second")
