from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "audit_hnerv_frontier_scorecard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_hnerv_frontier_scorecard_under_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _row_score(archive_bytes: int, seg: float, pose: float) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / 37_545_489


def _score_fields(archive_bytes: int, seg: float, pose: float) -> dict:
    return {
        "archive_bytes": archive_bytes,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "score": _row_score(archive_bytes, seg, pose),
        "score_seg_contribution": 100.0 * seg,
        "score_pose_contribution": math.sqrt(10.0 * pose),
        "score_rate_contribution": 25.0 * archive_bytes / 37_545_489,
        "runtime_tree_sha256": "f" * 64,
        "eval_artifact": "tools/audit_hnerv_frontier_scorecard.py",
    }


def _valid_scorecard() -> dict:
    return {
        "schema_version": 1,
        "score_truth": "exact_cuda_auth_eval_json",
        "rows": [
            {
                "label": "PR106",
                "evidence_grade": "A++",
                "canonical_frontier_eligible": True,
                "profile_match_key": "archive_sha256",
                "archive_sha256": "a" * 64,
                "payload_sha256": "c" * 64,
                **_score_fields(186_239, 0.00067142, 3.351e-5),
            },
            {
                "label": "PR106x",
                "evidence_grade": "A++",
                "canonical_frontier_eligible": True,
                "profile_match_key": "archive_sha256",
                "archive_sha256": "b" * 64,
                "payload_sha256": "c" * 64,
                **_score_fields(186_231, 0.00067142, 3.351e-5),
            },
        ],
        "payload_equivalence_groups": [
            {
                "labels": ["PR106", "PR106x"],
                "same_seg_contribution": True,
                "same_pose_contribution": True,
                "readiness": "byte-identical payload pair; use as repack custody/control only",
            }
        ],
        "followup_targets": [
            {
                "suggested_action": "decoder self-compression or weight-stream recoding fixture",
            },
            {
                "suggested_action": "latent/sidecar arithmetic-coding parity fixture",
            },
        ],
        "payload_section_manifests": [
            {
                "label": "PR106",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": [
                    {
                        "name": "decoder_packed_brotli",
                        "bytes": 8,
                        "sha256": "d" * 64,
                        "optimization_role": "decoder_weight_stream",
                    },
                    {
                        "name": "latents_and_sidecar_brotli",
                        "bytes": 16,
                        "sha256": "e" * 64,
                        "optimization_role": "latent_stream",
                    },
                ],
            },
            {
                "label": "PR106x",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": [
                    {
                        "name": "decoder_packed_brotli",
                        "bytes": 8,
                        "sha256": "d" * 64,
                        "optimization_role": "decoder_weight_stream",
                    }
                ],
            },
        ],
    }


def test_audit_scorecard_accepts_current_routing_contract() -> None:
    blockers, summary = audit.audit_scorecard(_valid_scorecard())

    assert blockers == []
    assert summary["row_count"] == 2
    assert summary["payload_equivalence_group_count"] == 1
    assert summary["followup_target_count"] == 2
    assert summary["payload_section_manifest_count"] == 2
    assert summary["canonical_labels"] == ["PR106", "PR106x"]


def test_audit_scorecard_blocks_missing_pr106x_and_followups() -> None:
    payload = _valid_scorecard()
    payload["rows"] = [payload["rows"][0]]
    payload["followup_targets"] = [
        {"suggested_action": "decoder self-compression or weight-stream recoding fixture"}
    ]
    payload["payload_section_manifests"] = []

    blockers, _summary = audit.audit_scorecard(payload)

    assert "missing_PR106x_row" in blockers
    assert "missing_PR106_PR106x_payload_control_group" in blockers
    assert "missing_latent_sidecar_arithmetic_followup" in blockers
    assert "missing_payload_section_manifests" in blockers


def test_audit_scorecard_blocks_formula_and_runtime_custody_drift() -> None:
    payload = _valid_scorecard()
    payload["rows"][0]["score"] += 0.01
    payload["rows"][1]["runtime_tree_sha256"] = "not-a-sha"

    blockers, _summary = audit.audit_scorecard(payload)

    assert "PR106_score_formula_mismatch" in blockers
    assert "PR106x_invalid_runtime_tree_sha256" in blockers


def test_cli_json_reports_no_score_claim(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    scorecard.write_text(json.dumps(_valid_scorecard()), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--scorecard", str(scorecard), "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["ready_for_hidden_gem_routing"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
