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
        "runtime_content_tree_sha256": "1" * 64,
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


def test_audit_scorecard_accepts_internal_score_lowering_frontier() -> None:
    payload = _valid_scorecard()
    row = {
        "label": "PR106-R2-lowlevel",
        "evidence_grade": "A++",
        "canonical_frontier_eligible": False,
        "canonicality_blockers": ["promotion_ineligible"],
        "profile_match_key": "archive_sha256",
        "archive_sha256": "9" * 64,
        "payload_sha256": "8" * 64,
        **_score_fields(186_629, 0.0006426, 0.00003236),
    }
    payload["rows"].append(row)
    payload["payload_section_manifests"].append(
        {
            "label": "PR106-R2-lowlevel",
            "score_claim": False,
            "dispatch_attempted": False,
            "sections": [
                {
                    "name": "decoder_compact_brotli_streams",
                    "bytes": 162_164,
                    "sha256": "7" * 64,
                    "optimization_role": "decoder_weight_stream",
                }
            ],
        }
    )
    payload["score_lowering_frontier"] = {
        "label": "PR106-R2-lowlevel",
        "score": row["score"],
        "archive_bytes": row["archive_bytes"],
        "archive_sha256": row["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "evidence_grade": "A++",
        "eval_artifact": row["eval_artifact"],
        "promotion_authority": False,
        "canonical_frontier_eligible": False,
        "canonicality_blockers": ["promotion_ineligible"],
    }
    payload["next_score_lowering_exact_evaluable_target"] = {
        "frontier_label": "PR106-R2-lowlevel",
        "label": "PR106-R2-lowlevel",
        "section": "decoder_compact_brotli_streams",
    }
    payload["score_lowering_hidden_gem_byte_mass_ranking"] = [
        {
            "frontier_label": "PR106-R2-lowlevel",
            "label": "PR106-R2-lowlevel",
            "section": "decoder_compact_brotli_streams",
        }
    ]

    blockers, summary = audit.audit_scorecard(payload)

    assert blockers == []
    assert summary["score_lowering_frontier_label"] == "PR106-R2-lowlevel"
    assert summary["next_score_lowering_target"] == {
        "label": "PR106-R2-lowlevel",
        "section": "decoder_compact_brotli_streams",
    }


def test_required_eval_must_be_visible_and_selected_when_lower(tmp_path: Path) -> None:
    payload = _valid_scorecard()
    eval_path = tmp_path / "contest_auth_eval.adjudicated.json"
    archive_sha = "9" * 64
    score_fields = _score_fields(186_423, 0.0006426, 0.00003236)
    eval_path.write_text(
        json.dumps(
            {
                "archive_size_bytes": 186_423,
                "avg_posenet_dist": 0.00003236,
                "avg_segnet_dist": 0.0006426,
                "score_pose_contribution": score_fields["score_pose_contribution"],
                "score_rate_contribution": score_fields["score_rate_contribution"],
                "score_seg_contribution": score_fields["score_seg_contribution"],
                "n_samples": 600,
                "provenance": {
                    "archive_sha256": archive_sha,
                    "device": "cuda",
                    "gpu_t4_match": True,
                    "inflate_runtime_manifest": {
                        "runtime_tree_sha256": "f" * 64,
                        "runtime_content_tree_sha256": "1" * 64,
                    },
                },
                "score_recomputed_from_components": score_fields["score"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    required = (f"PR106-R2-HDM4-HLM1={eval_path}",)

    blockers, _summary = audit.audit_scorecard(payload, required_evals=required)

    assert "PR106-R2-HDM4-HLM1_required_eval_missing_from_scorecard_rows" in blockers

    row = {
        "label": "PR106-R2-HDM4-HLM1",
        "evidence_grade": "A++",
        "canonical_frontier_eligible": True,
        "canonicality_blockers": [],
        "profile_match_key": "archive_sha256",
        "archive_sha256": archive_sha,
        "payload_sha256": "8" * 64,
        **score_fields,
    }
    row["eval_artifact"] = str(eval_path)
    payload["rows"].append(row)
    payload["score_lowering_frontier"] = {
        "label": "PR106",
        "score": payload["rows"][0]["score"],
        "archive_bytes": payload["rows"][0]["archive_bytes"],
        "archive_sha256": payload["rows"][0]["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "evidence_grade": "A++",
        "eval_artifact": payload["rows"][0]["eval_artifact"],
        "promotion_authority": False,
        "canonical_frontier_eligible": True,
        "canonicality_blockers": [],
    }

    blockers, _summary = audit.audit_scorecard(payload, required_evals=required)

    assert (
        "PR106-R2-HDM4-HLM1_required_eval_lower_than_score_lowering_frontier_but_not_selected"
        in blockers
    )

    payload["score_lowering_frontier"] = {
        "label": "PR106-R2-HDM4-HLM1",
        "score": row["score"],
        "archive_bytes": row["archive_bytes"],
        "archive_sha256": row["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "evidence_grade": "A++",
        "eval_artifact": row["eval_artifact"],
        "promotion_authority": False,
        "canonical_frontier_eligible": True,
        "canonicality_blockers": [],
    }
    blockers, summary = audit.audit_scorecard(payload, required_evals=required)

    assert blockers == []
    assert summary["required_eval_count"] == 1


def test_audit_scorecard_blocks_internal_frontier_on_regression_row() -> None:
    payload = _valid_scorecard()
    row = {
        "label": "bad-lower-score",
        "evidence_grade": "A++",
        "canonical_frontier_eligible": False,
        "canonicality_blockers": ["promotion_ineligible", "regression_triggered"],
        "profile_match_key": "archive_sha256",
        "archive_sha256": "9" * 64,
        "payload_sha256": "8" * 64,
        **_score_fields(180_000, 0.0001, 0.00001),
    }
    payload["rows"].append(row)
    payload["score_lowering_frontier"] = {
        "label": "bad-lower-score",
        "score": row["score"],
        "archive_bytes": row["archive_bytes"],
        "archive_sha256": row["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "promotion_authority": False,
    }

    blockers, _summary = audit.audit_scorecard(payload)

    assert "bad-lower-score_score_lowering_frontier_uses_severe_blocked_row" in blockers


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
