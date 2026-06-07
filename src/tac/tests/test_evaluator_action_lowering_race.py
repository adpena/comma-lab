# SPDX-License-Identifier: MIT
"""Tests for the minimal evaluator-action lowering race."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.analysis.action_effect import ActionEffect
from tac.analysis.evaluator_action_lowering_race import (
    COMPOSITE_NOT_MEASURED,
    INFLATE_FAILED,
    PARSEBACK_FAILED,
    SEMANTIC_PRIMITIVE_MISSING,
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
        wrong_to_target=1,
        support_source="fixture",
        support_cardinality=10,
        support_sha256="a" * 64,
        support_encoding=support_encoding,
        support_encoded_bytes=100,
        support_research_only=False,
    )


def test_lowering_race_blocks_sidecar_without_parseback_inflate_and_names_missing_targets() -> None:
    report = build_lowering_race_report(action_id="action-1", action_effects=[_effect()])
    verdict = report["verdict"]
    sidecar = report["lowering_candidates"][0]

    assert verdict["best_lowering"] == "none"
    assert verdict["sidecar_status"] == PARSEBACK_FAILED
    assert verdict["composite_status"] == COMPOSITE_NOT_MEASURED
    assert verdict["semantic_pose_status"] == SEMANTIC_PRIMITIVE_MISSING
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


def test_lowering_race_rejects_missing_byte_accounting() -> None:
    bad = _effect(payload_sections=())
    payload = bad.as_dict()
    payload["support_encoded_bytes"] = None
    bad = ActionEffect.from_dict(payload)
    report = build_lowering_race_report(action_id="action-1", action_effects=[bad])

    assert report["lowering_candidates"][0]["first_failing_surface"] == "BYTE_ACCOUNTING_MISSING"


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
    assert summary["best_lowering"] == "none"
    assert summary["first_failing_surface"] in {PARSEBACK_FAILED, INFLATE_FAILED}
    verdict = json.loads((out_dir / "lowering_verdict.json").read_text(encoding="utf-8"))
    assert verdict["action_id"] == effect.action_id
