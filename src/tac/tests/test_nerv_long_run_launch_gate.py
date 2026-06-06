# SPDX-License-Identifier: MIT
"""Fail-closed behavior of the NeRV long-run launch gate.

All evidence files here are synthetic fixtures (labelled, tmp-dir only) used
to verify the gate's refusal logic; they are not empirical anchors and grant
no score authority.  The gate must approve ONLY on a complete, consistent
ladder, and every missing/mismatched row must be NAMED in the verdict.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac.analysis.nerv_long_run_launch_gate import (
    BIRTH_HYSTERESIS_SCHEMA,
    BIRTH_RECEIPT_SCHEMA,
    BIRTH_SURVIVAL_SCHEMA,
    REPRESENTATIVE_COVERAGE_SCHEMA,
    SNERV_SOURCE_FORWARD_SCHEMA,
    NervLongRunLaunchGateError,
    evaluate_nerv_long_run_launch_gate,
)

NOW = datetime(2026, 6, 6, 21, 0, 0, tzinfo=UTC)
ACTION = "a" * 64


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pointer(tmp_path: Path, *, age_hours: float = 1.0) -> Path:
    path = tmp_path / "canonical_frontier_pointer.json"
    refreshed = NOW - timedelta(hours=age_hours)
    _write(path, {"last_refreshed_utc": refreshed.isoformat()})
    return path


def _live_birth_receipt(
    *,
    action_id: str = ACTION,
    pose_trusted: bool = True,
    hard_won: int = 7932,
    net_support: int = 7932,
) -> dict:
    return {
        "schema": BIRTH_RECEIPT_SCHEMA,
        "fixture_not_real": True,
        "surface": "live_mlx",
        "action_id": action_id,
        "accepted_step_count": 1,
        "runtime_sidecar_bytes": 0,
        "argmax_transitions": {
            "target_hard_won_count": hard_won,
            "target_hard_lost_count": max(0, hard_won - net_support),
            "net_target_support_delta": net_support,
        },
        "pose_guard": {
            "available": pose_trusted,
            "pose_input_contest_resolution": pose_trusted,
        },
        "exact_nonrate": {
            "pose_term_available": pose_trusted,
            "delta_score_nonrate": -0.012 if pose_trusted else None,
        },
    }


def _survival(
    surface: str,
    *,
    action_id: str = ACTION,
    survived: bool = True,
    include_support: bool = True,
    hard_won: int = 2048,
    net_support: int = 2048,
) -> dict:
    argmax_transitions = (
        {
            "target_hard_won_count": hard_won,
            "target_hard_lost_count": max(0, hard_won - net_support),
            "net_target_support_delta": net_support,
        }
        if include_support
        else None
    )
    return {
        "schema": BIRTH_SURVIVAL_SCHEMA,
        "fixture_not_real": True,
        "surface": surface,
        "action_id": action_id,
        "survived": survived,
        "argmax_transitions": argmax_transitions,
    }


def _full_hi_nerv_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(root / "fakequant.json", _survival("fakequant_mlx"))
    _write(root / "parseback.json", _survival("parseback_mlx"))
    _write(root / "inflate.json", _survival("inflated_torch_cpu"))
    _write(
        root / "hysteresis.json",
        {
            "schema": BIRTH_HYSTERESIS_SCHEMA,
            "fixture_not_real": True,
            "action_id": ACTION,
            "passed": True,
        },
    )
    _write(
        root / "coverage.json",
        {
            "schema": REPRESENTATIVE_COVERAGE_SCHEMA,
            "fixture_not_real": True,
            "region_classes_covered": 3,
        },
    )
    return root


def test_unknown_family_and_missing_root_fail_loud(tmp_path: Path) -> None:
    with pytest.raises(NervLongRunLaunchGateError, match="family"):
        evaluate_nerv_long_run_launch_gate(family="nope", run_root=tmp_path, now_utc=NOW)
    with pytest.raises(NervLongRunLaunchGateError, match="run_root"):
        evaluate_nerv_long_run_launch_gate(family="hi_nerv", run_root=tmp_path / "missing", now_utc=NOW)


def test_empty_root_blocks_everything(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert verdict["highest_level"] == "none"
    assert "real_video_birth_receipt_missing" in verdict["blocking_evidence"]
    # The gate itself is planning-only and never a score authority.
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False


def test_live_birth_without_pose_trust_is_l2(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt(pose_trusted=False))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["highest_level"] == "L2"
    assert "pose_trusted_birth_receipt_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_zero_net_support_is_not_a_birth(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt(hard_won=1, net_support=0))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["highest_level"] == "none"
    assert "real_video_birth_receipt_missing" in verdict["blocking_evidence"]
    assert "live_birth_target_support_not_positive" in verdict["blocking_evidence"]


def test_survival_action_id_mismatch_is_named(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(
        root / "fakequant.json",
        _survival("fakequant_mlx", action_id="b" * 64),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "l4_survival_action_id_mismatch:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["highest_level"] == "L3"


def test_not_survived_row_blocks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(root / "parseback.json", _survival("parseback_mlx", survived=False))
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert "birth_not_survived:parseback_mlx" in verdict["blocking_evidence"]
    assert verdict["approved"] is False


def test_pose_compensation_must_survive_even_when_target_support_survives(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    row = _survival("fakequant_mlx")
    row["pose_compensation_required"] = True
    row["pose_compensation_survived"] = False
    _write(root / "fakequant.json", row)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "birth_survival_pose_compensation_not_survived:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["highest_level"] == "L3"
    assert verdict["approved"] is False


def test_survived_row_without_target_support_blocks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(root / "birth.json", _live_birth_receipt())
    _write(
        root / "fakequant.json",
        _survival("fakequant_mlx", include_support=False),
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "birth_survival_target_support_missing:fakequant_mlx" in blocking
    assert "birth_survival_receipt_missing:fakequant_mlx" in blocking
    assert verdict["approved"] is False


def test_full_ladder_with_fresh_pointer_approves(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["blocking_evidence"] == []
    assert verdict["highest_level"] == "L5"
    assert verdict["approved"] is True


def test_hinerv_family_alias_is_canonicalized(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hinerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["family"] == "hinerv"
    assert verdict["approved"] is True


def test_stale_pointer_blocks_even_complete_ladder(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path, age_hours=48.0),
        now_utc=NOW,
    )
    assert verdict["approved"] is False
    assert "frontier_pointer_stale" in verdict["blocking_evidence"]


def test_missing_pointer_blocks(tmp_path: Path) -> None:
    root = _full_hi_nerv_root(tmp_path)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=tmp_path / "nope.json",
        now_utc=NOW,
    )
    assert "frontier_pointer_missing" in verdict["blocking_evidence"]


def test_truthy_authority_evidence_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    receipt = _live_birth_receipt()
    receipt["score_claim"] = True  # forged authority on an evidence row
    _write(root / "birth.json", receipt)
    verdict = evaluate_nerv_long_run_launch_gate(
        family="hi_nerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert any(item.startswith("evidence_truthy_authority:") for item in verdict["blocking_evidence"])
    assert verdict["highest_level"] == "none"


def test_snerv_requires_proof_and_bitflip(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    blocking = verdict["blocking_evidence"]
    assert "snerv_full_source_forward_parity_missing" in blocking
    assert "snerv_payload_bitflip_falsification_missing" in blocking

    _write(
        root / "proof.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "full_tub_source_forward_parity_proven": True,
        },
    )
    _write(
        root / "bitflip.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "bitflip_section": "TUB",
            "proof_passed": False,
            "first_failed_tensor": "TUB_out",
        },
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert verdict["blocking_evidence"] == []
    assert verdict["highest_level"] == "L4"
    assert verdict["approved"] is True


def test_snerv_bitflip_that_passes_proof_is_not_falsification(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write(
        root / "proof.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "full_tub_source_forward_parity_proven": True,
        },
    )
    # A bit flip that does NOT fail the proof means the proof is metadata-only.
    _write(
        root / "bitflip.json",
        {
            "schema": SNERV_SOURCE_FORWARD_SCHEMA,
            "fixture_not_real": True,
            "bitflip_section": "TUB",
            "proof_passed": True,
            "first_failed_tensor": None,
        },
    )
    verdict = evaluate_nerv_long_run_launch_gate(
        family="snerv",
        run_root=root,
        frontier_pointer=_pointer(tmp_path),
        now_utc=NOW,
    )
    assert "snerv_payload_bitflip_falsification_missing" in verdict["blocking_evidence"]
    assert verdict["approved"] is False
