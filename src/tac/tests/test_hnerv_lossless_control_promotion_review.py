from __future__ import annotations

import math
import zipfile
from pathlib import Path

from tac.hnerv_lossless_control_promotion_review import (
    PROMOTABLE_VERDICT,
    build_lossless_control_promotion_review,
    inspect_single_member_archive,
    render_markdown,
)

CONTEST_ORIGINAL_BYTES = 37_545_489
TARGET_LABEL = "PR106x-lowlevel-brotli"
SOURCE_LABEL = "PR106x"
ARCHIVE_SHA = "a" * 64
SOURCE_ARCHIVE_SHA = "b" * 64
PAYLOAD_SHA = "c" * 64
RUNTIME_SHA = "d" * 64
PREFLIGHT_RUNTIME_SHA = "e" * 64
SEG = 0.00067142
POSE = 0.00003351
TARGET_BYTES = 186_080
SOURCE_BYTES = 186_231


def test_lossless_control_review_marks_existing_exact_artifact_promotable() -> None:
    review = build_lossless_control_promotion_review(**_review_inputs())

    assert review["review_verdict"] == PROMOTABLE_VERDICT
    assert review["existing_exact_control_promotable"] is True
    assert review["score_claim"] is False
    assert review["dispatch_attempted"] is False
    assert review["lane_claim_required_for_this_review"] is False
    assert review["blockers"] == []
    assert review["missing_evidence"] == []
    assert all(check["passed"] for check in review["checks"])
    assert {warning["id"] for warning in review["warnings"]} == {
        "scorecard_current_frontier_field_missing_or_stale",
        "public_preflight_runtime_tree_differs_from_exact_eval",
    }

    markdown = render_markdown(review)
    assert "promotable_existing_exact_control" in markdown
    assert "- none" in markdown
    assert "score_claim: `false`" in markdown


def test_lossless_control_review_blocks_missing_adjudication_promotion() -> None:
    inputs = _review_inputs()
    adjudication = dict(inputs["adjudication"])
    adjudication["promotion_eligible"] = False
    inputs["adjudication"] = adjudication

    review = build_lossless_control_promotion_review(**inputs)

    assert review["existing_exact_control_promotable"] is False
    assert "adjudication_marks_existing_exact_control_promotable" in review["blockers"]
    failed = {check["id"]: check for check in review["checks"] if not check["passed"]}
    assert failed["adjudication_marks_existing_exact_control_promotable"]["failure_class"] == "inconsistent_evidence"


def test_inspect_single_member_archive_records_payload_identity(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    payload = b"packed-hnerv-payload"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload)

    inspection = inspect_single_member_archive(archive, repo_root=tmp_path)

    assert inspection["path"] == "archive.zip"
    assert inspection["member_name"] == "x"
    assert inspection["member_bytes"] == len(payload)
    assert inspection["single_member_zip"] is True
    assert inspection["blockers"] == []


def _review_inputs() -> dict:
    return {
        "target_label": TARGET_LABEL,
        "scorecard": _scorecard(),
        "entropy_ranking": _entropy_ranking(),
        "candidate_manifest": _candidate_manifest(),
        "exact_eval": _exact_eval(),
        "adjudication": _adjudication(),
        "public_preflight": _public_preflight(),
        "candidate_archive": _archive_inspection(),
        "exact_eval_archive": _archive_inspection(),
        "input_paths": {
            "scorecard": "scorecard.json",
            "entropy_ranking": "ranking.json",
        },
    }


def _score(archive_bytes: int) -> float:
    return 100.0 * SEG + math.sqrt(10.0 * POSE) + 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES


def _scorecard() -> dict:
    return {
        "schema_version": 1,
        "score_truth": "exact_cuda_auth_eval_json",
        "current_frontier": None,
        "rows": [
            {
                "label": SOURCE_LABEL,
                "canonical_frontier_eligible": True,
                "canonicality_blockers": [],
                "evidence_grade": "A++",
                "frontier_scope": "exact_local_cuda_custody",
                "score": _score(SOURCE_BYTES),
                "archive_bytes": SOURCE_BYTES,
                "archive_sha256": SOURCE_ARCHIVE_SHA,
                "payload_sha256": "f" * 64,
                "runtime_tree_sha256": RUNTIME_SHA,
                "avg_posenet_dist": POSE,
                "avg_segnet_dist": SEG,
                "score_rate_contribution": 25.0 * SOURCE_BYTES / CONTEST_ORIGINAL_BYTES,
                "score_pose_contribution": math.sqrt(10.0 * POSE),
                "score_seg_contribution": 100.0 * SEG,
                "eval_artifact": "source_eval.json",
                "payload_sections": [],
            },
            {
                "label": TARGET_LABEL,
                "canonical_frontier_eligible": True,
                "canonicality_blockers": [],
                "evidence_grade": "A++",
                "frontier_scope": "exact_local_cuda_custody_lossless_repack_control",
                "score": _score(TARGET_BYTES),
                "archive_bytes": TARGET_BYTES,
                "archive_sha256": ARCHIVE_SHA,
                "payload_sha256": PAYLOAD_SHA,
                "runtime_tree_sha256": RUNTIME_SHA,
                "avg_posenet_dist": POSE,
                "avg_segnet_dist": SEG,
                "score_rate_contribution": 25.0 * TARGET_BYTES / CONTEST_ORIGINAL_BYTES,
                "score_pose_contribution": math.sqrt(10.0 * POSE),
                "score_seg_contribution": 100.0 * SEG,
                "eval_artifact": "target_eval.json",
                "candidate_source_label": SOURCE_LABEL,
                "candidate_source_archive_sha256": SOURCE_ARCHIVE_SHA,
                "candidate_source_archive_bytes": SOURCE_BYTES,
                "payload_sections": [],
            },
        ],
    }


def _entropy_ranking() -> dict:
    return {
        "schema_version": 1,
        "score_claim": False,
        "dispatch_attempted": False,
        "next_rate_only_action": {
            "action_id": "review_current_exact_lossless_brotli_control_before_promotion",
            "target_label": TARGET_LABEL,
            "score_claim": False,
            "dispatch_attempted": False,
        },
        "exact_lossless_control_actions": [
            {
                "target_label": TARGET_LABEL,
                "review_status": "ready_for_promotion_review_existing_exact_custody",
                "raw_equivalence_closed": True,
                "blockers": [],
            }
        ],
    }


def _candidate_manifest() -> dict:
    return {
        "source_label": SOURCE_LABEL,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA,
        "source_archive_bytes": SOURCE_BYTES,
        "candidate_archive_sha256": ARCHIVE_SHA,
        "candidate_archive_bytes": TARGET_BYTES,
        "candidate_payload_sha256": PAYLOAD_SHA,
        "candidate_diff_audit": {
            "blockers": [],
            "total_byte_delta": TARGET_BYTES - SOURCE_BYTES,
        },
        "brotli_raw_equivalence": [
            {
                "section_name": "decoder_packed_brotli",
                "raw_equal": True,
                "raw_bytes": 229_070,
            },
            {
                "section_name": "latents_and_sidecar_brotli",
                "raw_equal": True,
                "raw_bytes": 33_712,
            },
        ],
    }


def _exact_eval() -> dict:
    score = _score(TARGET_BYTES)
    return {
        "archive_size_bytes": TARGET_BYTES,
        "avg_posenet_dist": POSE,
        "avg_segnet_dist": SEG,
        "n_samples": 600,
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "provenance": {
            "archive_sha256": ARCHIVE_SHA,
            "archive_size_bytes": TARGET_BYTES,
            "device": "cuda",
            "cuda_available": True,
            "gpu_t4_match": True,
            "gpu_model": "Tesla T4",
            "inflate_runtime_manifest": {
                "runtime_tree_sha256": RUNTIME_SHA,
            },
        },
    }


def _adjudication() -> dict:
    return {
        "allowed_use": ["promotion_review", "rank_frontier_candidate"],
        "archive_delta_bytes": TARGET_BYTES - SOURCE_BYTES,
        "baseline_archive_bytes": SOURCE_BYTES,
        "component_gate_triggered": False,
        "component_gate_violations": [],
        "component_gates": [
            {
                "component": "posenet",
                "observed": POSE,
                "reference": POSE,
                "relative_to_reference": 1.0,
                "passed": True,
            },
            {
                "component": "segnet",
                "observed": SEG,
                "reference": SEG,
                "relative_to_reference": 1.0,
                "passed": True,
            },
        ],
        "contest_cuda_archive_bytes": TARGET_BYTES,
        "contest_cuda_archive_sha256": ARCHIVE_SHA,
        "contest_equivalent_hardware": True,
        "promotion_eligible": True,
        "scientific_score_eligible": True,
        "score_delta_vs_baseline": _score(TARGET_BYTES) - _score(SOURCE_BYTES),
        "regression_triggered": False,
    }


def _public_preflight() -> dict:
    return {
        "archive": {
            "status": "passed",
            "duplicate_member_names": [],
        },
        "blockers": [],
        "promotion_eligible": False,
        "runtime": {
            "status": "passed",
            "runtime_tree_sha256": PREFLIGHT_RUNTIME_SHA,
        },
    }


def _archive_inspection() -> dict:
    return {
        "path": "archive.zip",
        "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": TARGET_BYTES,
        "member_name": "x",
        "member_sha256": PAYLOAD_SHA,
        "blockers": [],
    }
