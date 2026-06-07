# SPDX-License-Identifier: MIT
"""Tests for the minimal evaluator-action lowering race."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.analysis.action_effect import ActionEffect
from tac.analysis.evaluator_action_lowering_race import (
    BYTE_ACCOUNTING_MISSING,
    COMPOSITE_NOT_MEASURED,
    FAKEQUANT_FAILED,
    INFLATE_FAILED,
    INVALID_LOWERING_TARGET,
    LOWERING_TARGETS,
    PARSEBACK_FAILED,
    SEMANTIC_PRIMITIVE_MISSING,
    SUPPORT_IDENTITY_MISMATCH,
    SUPPORT_IDENTITY_MISSING,
    build_lowering_race_report,
)


def _effect(
    *,
    action_id: str = "action-1",
    action_kind: str = "frame1_seg_margin_frontier_path",
    support_encoding: str = "rle",
    delta_good: bool = False,
    parseback: bool = False,
    inflate: bool = False,
    fakequant: bool = True,
    support_sha256: str | None = "a" * 64,
    payload_sections: tuple[str, ...] = ("support_codec=rle", "action_payload_bytes=0", "metadata_bytes=0"),
) -> ActionEffect:
    old_d_seg, new_d_seg = (0.2, 0.19) if delta_good else (0.2, 0.2)
    return ActionEffect.build(
        action_id=action_id,
        family="hinerv",
        action_kind=action_kind,
        inverse_source="path_tube_segnet_margin_frontier",
        frame_index=1,
        frame_incidence="seg_pose_joint",
        candidate_status="selected",
        authority="batch_local_path_support",
        normalization_scope="batch_local",
        producer="support_codec_router",
        consumer="inverse_evaluate_candidate_queue",
        pair_ids=[7],
        region_ids=["b0/c4/r1"],
        payload_sections=payload_sections,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=0.3,
        new_d_pose=0.3,
        old_bytes=1000,
        new_bytes=1100,
        receiver_surface={"uint8_changed_pixels": 1, "seg_argmax_changed_pixels": 1},
        exact_score_decision="reject",
        parseback_survived=parseback,
        inflate_survived=inflate,
        fakequant_survived=fakequant,
        wrong_to_target=1,
        support_source="fixture",
        support_cardinality=10,
        support_sha256=support_sha256,
        support_encoding=support_encoding,
        support_encoded_bytes=100,
        support_research_only=False,
    )


def test_lowering_race_blocks_sidecar_without_parseback_inflate_and_names_missing_targets() -> None:
    report = build_lowering_race_report(action_id="action-1", action_effects=[_effect()])
    verdict = report["verdict"]
    sidecar = report["lowering_candidates"][0]

    assert verdict["best_lowering"] == "discard"
    assert verdict["next_blocker"] == PARSEBACK_FAILED
    assert verdict["sidecar_status"] == PARSEBACK_FAILED
    assert verdict["composite_status"] == COMPOSITE_NOT_MEASURED
    assert verdict["semantic_pose_status"] == SEMANTIC_PRIMITIVE_MISSING
    assert report["target_accounting"]["expected_targets"] == list(LOWERING_TARGETS)
    assert "pair_local_latent_action" in report["target_accounting"]["missing_targets"]
    assert "frame0_pose_compensation" in report["target_accounting"]["missing_targets"]
    assert "byte_entropy_rewrite" in report["target_accounting"]["missing_targets"]
    assert report["target_accounting"]["all_targets_accounted"] is False
    assert report["commutator_ledger"]["schema"] == "tac.action_commutator_ledger.v1"
    assert sidecar["support_encoded_bytes"] == 100
    assert sidecar["action_payload_bytes"] == 0
    assert sidecar["metadata_bytes"] == 0
    assert sidecar["promotion_eligible"] is False


def test_lowering_race_selects_viable_lowest_delta_candidate() -> None:
    sidecar = _effect(delta_good=True, parseback=True, inflate=True)
    composite = _effect(
        action_kind="frame1_seg_then_frame0_pose_composite",
        delta_good=True,
        parseback=True,
        inflate=True,
    )
    report = build_lowering_race_report(action_id="action-1", action_effects=[sidecar, composite])

    assert report["verdict"]["best_lowering"] in {"byte_priced_sidecar", "pose_compensated_composite"}
    assert report["verdict"]["first_failing_surface"] == "none"
    assert report["verdict"]["delta_score_total"] < 0.0


def test_lowering_race_requires_fakequant_survival_before_acceptance() -> None:
    row = _effect(delta_good=True, parseback=True, inflate=True, fakequant=False)

    report = build_lowering_race_report(action_id="action-1", action_effects=[row])

    assert report["verdict"]["best_lowering"] == "discard"
    assert report["verdict"]["first_failing_surface"] == FAKEQUANT_FAILED
    assert report["lowering_candidates"][0]["viable"] is False


def test_lowering_race_requires_same_support_for_same_action() -> None:
    sidecar = _effect(delta_good=True, parseback=True, inflate=True, support_sha256="a" * 64)
    composite = _effect(
        action_kind="frame1_seg_then_frame0_pose_composite",
        delta_good=True,
        parseback=True,
        inflate=True,
        support_sha256="b" * 64,
    )

    report = build_lowering_race_report(action_id="action-1", action_effects=[sidecar, composite])

    assert report["verdict"]["best_lowering"] == "discard"
    assert report["verdict"]["first_failing_surface"] == SUPPORT_IDENTITY_MISMATCH
    assert report["support_identity"]["all_candidates_same_support"] is False
    assert "lowering_race_candidate_support_sha256_mismatch" in report["support_identity"]["blockers"]
    assert all(row["viable"] is False for row in report["lowering_candidates"])


def test_lowering_race_expected_support_hash_is_authoritative() -> None:
    sidecar = _effect(delta_good=True, parseback=True, inflate=True, support_sha256="b" * 64)

    report = build_lowering_race_report(
        action_id="action-1",
        action_effects=[sidecar],
        expected_support_sha256="a" * 64,
    )

    assert report["verdict"]["best_lowering"] == "discard"
    assert report["verdict"]["first_failing_surface"] == SUPPORT_IDENTITY_MISMATCH
    assert "lowering_race_expected_support_sha256_mismatch" in report["support_identity"]["blockers"]


def test_lowering_race_uses_explicit_target_before_text_inference() -> None:
    effect = _effect(
        action_kind="generic_backend_sounding_name",
        delta_good=True,
        parseback=True,
        inflate=True,
        payload_sections=(
            "lowering_target=pose_compensated_composite",
            "support_codec=rle",
            "action_payload_bytes=0",
            "metadata_bytes=0",
        ),
    )

    report = build_lowering_race_report(action_id="action-1", action_effects=[effect])
    candidate = report["lowering_candidates"][0]

    assert candidate["lowering_target"] == "pose_compensated_composite"
    assert candidate["lowering_target_source"] == "explicit"
    assert report["verdict"]["best_lowering"] == "pose_compensated_composite"


def test_lowering_race_invalid_explicit_target_fails_closed() -> None:
    effect = _effect(
        delta_good=True,
        parseback=True,
        inflate=True,
        payload_sections=(
            "lowering_target=definitely_not_a_target",
            "support_codec=rle",
            "action_payload_bytes=0",
            "metadata_bytes=0",
        ),
    )

    report = build_lowering_race_report(action_id="action-1", action_effects=[effect])
    candidate = report["lowering_candidates"][0]

    assert candidate["viable"] is False
    assert candidate["first_failing_surface"] == INVALID_LOWERING_TARGET
    assert report["verdict"]["best_lowering"] == "discard"


def test_lowering_race_rejects_missing_byte_accounting() -> None:
    bad = _effect(payload_sections=())
    payload = bad.as_dict()
    payload["support_encoded_bytes"] = None
    bad = ActionEffect.from_dict(payload)
    report = build_lowering_race_report(action_id="action-1", action_effects=[bad])

    assert report["lowering_candidates"][0]["first_failing_surface"] == BYTE_ACCOUNTING_MISSING


def test_lowering_race_rejects_missing_action_payload_or_metadata_bytes() -> None:
    missing_action_bytes = _effect(
        payload_sections=("support_codec=rle", "metadata_bytes=0"),
        delta_good=True,
        parseback=True,
        inflate=True,
    )
    missing_metadata_bytes = _effect(
        payload_sections=("support_codec=rle", "action_payload_bytes=0"),
        delta_good=True,
        parseback=True,
        inflate=True,
    )

    report = build_lowering_race_report(
        action_id="action-1",
        action_effects=[missing_action_bytes, missing_metadata_bytes],
    )

    assert [row["first_failing_surface"] for row in report["lowering_candidates"]] == [
        BYTE_ACCOUNTING_MISSING,
        BYTE_ACCOUNTING_MISSING,
    ]


def test_lowering_race_rejects_blocked_negative_delta_row() -> None:
    good_but_blocked = _effect(delta_good=True, parseback=True, inflate=True)
    payload = good_but_blocked.as_dict()
    payload["blockers"] = ["support_identity_mismatch"]
    good_but_blocked = ActionEffect.from_dict(payload)

    report = build_lowering_race_report(action_id="action-1", action_effects=[good_but_blocked])

    assert report["verdict"]["best_lowering"] == "discard"
    assert report["lowering_candidates"][0]["viable"] is False
    assert report["lowering_candidates"][0]["first_failing_surface"] == "support_identity_mismatch"


def test_lowering_race_consumes_support_codec_report_and_cli_writes_verdict(tmp_path: Path) -> None:
    effect = _effect()
    support_codec_report = {
        "schema": "tac.support_codec_router.v1",
        "reports": [
            {
                "action_id": effect.action_id,
                "selected_support_encoding": "rle",
                "selected_total_cost_bytes": 100,
                "selected_action_effect": effect.as_dict(),
            }
        ],
    }
    report_path = tmp_path / "support_codec_report.json"
    report_path.write_text(json.dumps(support_codec_report), encoding="utf-8")
    out_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[3]

    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "run_evaluator_action_lowering_race.py"),
            "--action-id",
            effect.action_id,
            "--support-codec-report",
            str(report_path),
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["best_lowering"] == "discard"
    assert summary["first_failing_surface"] in {PARSEBACK_FAILED, INFLATE_FAILED}
    verdict = json.loads((out_dir / "lowering_verdict.json").read_text(encoding="utf-8"))
    assert verdict["action_id"] == effect.action_id


def test_lowering_race_reports_exact_bytes_score_terms_for_winner() -> None:
    sidecar = _effect(delta_good=True, parseback=True, inflate=True)

    report = build_lowering_race_report(action_id="action-1", action_effects=[sidecar])
    verdict = report["verdict"]

    assert verdict["best_lowering"] == "byte_priced_sidecar"
    assert verdict["sidecar_lowering_complete"] is True
    assert verdict["backend_realization_complete"] is False
    assert report["sidecar_lowering_complete"] is True
    assert report["backend_realization_complete"] is False
    assert verdict["bytes_by_section"] == {
        "action_payload_bytes": 0,
        "delta_bytes": 100,
        "metadata_bytes": 0,
        "support_encoded_bytes": 100,
    }
    assert verdict["exact_score_terms"]["delta_score_total"] == verdict["delta_score_total"]
    assert verdict["exact_score_terms"]["value_per_byte"] == verdict["value_per_byte"]


def test_lowering_race_missing_support_branch_does_not_poison_supported_sidecar() -> None:
    sidecar = _effect(
        delta_good=True,
        parseback=True,
        inflate=True,
        support_sha256="a" * 64,
        payload_sections=(
            "lowering_target=byte_priced_sidecar",
            "support_codec=rle",
            "action_payload_bytes=0",
            "metadata_bytes=0",
        ),
    )
    backend = _effect(
        action_kind="backend_target_region_fit",
        support_encoding=None,
        delta_good=True,
        parseback=True,
        inflate=True,
        support_sha256=None,
        payload_sections=(
            "lowering_target=backend_realization",
            "action_payload_bytes=0",
            "metadata_bytes=0",
        ),
    )

    report = build_lowering_race_report(
        action_id="action-1",
        action_effects=[sidecar, backend],
        expected_support_sha256="a" * 64,
    )
    candidates = {
        str(row["lowering_target"]): row for row in report["lowering_candidates"]
    }

    assert report["verdict"]["best_lowering"] == "byte_priced_sidecar"
    assert report["verdict"]["sidecar_lowering_complete"] is True
    assert report["verdict"]["backend_realization_complete"] is False
    assert report["verdict"]["first_failing_surface"] == "none"
    assert report["support_identity"]["failure"] is None
    assert "lowering_race_support_sha256_missing" in report["support_identity"]["blockers"]
    assert candidates["byte_priced_sidecar"]["viable"] is True
    assert candidates["byte_priced_sidecar"]["first_failing_surface"] == "none"
    assert candidates["backend_realization"]["viable"] is False
    assert candidates["backend_realization"]["first_failing_surface"] == SUPPORT_IDENTITY_MISSING
