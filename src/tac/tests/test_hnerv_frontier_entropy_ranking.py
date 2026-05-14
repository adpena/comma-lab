# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.hnerv_frontier_entropy_ranking import (
    build_frontier_entropy_gap_ranking,
    render_markdown,
)
from tac.repo_io import json_text

REPO = Path(__file__).resolve().parents[3]


def test_frontier_entropy_ranking_maps_hdc2_to_current_lowlevel_section() -> None:
    manifest = build_frontier_entropy_gap_ranking(
        _scorecard(),
        entropy_audits=[_entropy_audit()],
        candidate_manifests=[_candidate_manifest()],
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["current_frontier"]["label"] == "PR106x-lowlevel-brotli"

    next_action = manifest["next_rate_only_action"]
    assert next_action["action_id"] == "review_current_exact_lossless_brotli_control_before_promotion"
    assert next_action["dispatch_allowed"] is False

    rows = manifest["entropy_gap_ranking"]
    assert rows[0]["target_kind"] == "known_model_overhead"
    assert rows[0]["frontier_section"] == "decoder_packed_brotli"
    assert rows[0]["frontier_anchor_source"] == "candidate_diff_source_to_current_section_sha256"
    assert rows[0]["net_byte_delta_vs_source_section"] == 40
    assert rows[0]["net_byte_delta_vs_current_frontier_section"] == 50
    assert rows[0]["net_byte_delta_after_target_vs_current_frontier_section"] == 10
    assert rows[0]["minimum_additional_reduction_after_target_to_beat_current_bytes"] == 11
    assert rows[0]["readiness_stage"] == "target_alone_insufficient_for_rate_positive_archive"

    combined = manifest["combined_entropy_gap_groups"][0]
    assert combined["known_target_bytes"] == 60
    assert combined["net_byte_delta_vs_current_frontier_section"] == 50
    assert combined["net_byte_delta_after_known_targets_vs_current_frontier_section"] == -10
    assert combined["verdict"] == "combined_targets_can_cross_rate_positive_if_byte_equivalent"

    entropy_action = manifest["next_entropy_research_action"]
    assert entropy_action["action_id"] == "build_combined_entropy_overhead_reduction_manifest"
    assert entropy_action["minimum_section_bytes_to_beat"] == 89
    assert "HNeRV Frontier Entropy Gap Ranking" in render_markdown(manifest)
    assert "Combined Entropy Gap Groups" in render_markdown(manifest)


def test_frontier_entropy_ranking_can_target_internal_score_lowering_frontier() -> None:
    scorecard = _scorecard()
    internal_row = {
        "label": "PR106-R2-lowlevel",
        "canonical_frontier_eligible": False,
        "canonicality_blockers": ["promotion_ineligible"],
        "score": 0.19,
        "archive_bytes": 210,
        "archive_sha256": "9" * 64,
        "payload_sha256": "8" * 64,
        "runtime_tree_sha256": "7" * 64,
        "evidence_grade": "A++",
        "frontier_scope": "exact_local_cuda_custody",
        "eval_artifact": "internal.json",
        "payload_sections": [
            _section("decoder_compact_brotli_streams", 162, "6" * 64),
            _section("sidecar_dim_delta_huffman_enum", 9, "5" * 64),
        ],
    }
    scorecard["rows"].append(internal_row)
    scorecard["score_lowering_frontier"] = {
        "label": "PR106-R2-lowlevel",
        "score": internal_row["score"],
        "archive_bytes": internal_row["archive_bytes"],
        "archive_sha256": internal_row["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "promotion_authority": False,
    }

    manifest = build_frontier_entropy_gap_ranking(
        scorecard,
        candidate_manifests=[_hdm_candidate_manifest(internal_row)],
        frontier_mode="score_lowering",
    )

    assert manifest["frontier_mode"] == "score_lowering"
    assert manifest["current_frontier"]["label"] == "PR106-R2-lowlevel"
    assert manifest["next_rate_only_action"]["action_id"] == (
        "review_current_exact_lossless_brotli_control_before_promotion"
    )
    assert manifest["next_rate_only_action"]["dispatch_allowed"] is False
    assert manifest["exact_lossless_control_actions"][0]["review_status"] == (
        "ready_for_promotion_review_existing_exact_custody"
    )
    assert manifest["exact_lossless_control_actions"][0]["review_contract"] == (
        "hnerv_hdm_decoder_recode_manifest"
    )
    assert manifest["exact_lossless_control_actions"][0]["total_byte_delta"] == -10
    assert manifest["exact_lossless_control_actions"][0]["rate_score_delta_if_components_equal"] < 0
    assert "frontier_mode: `score_lowering`" in render_markdown(manifest)


def test_frontier_entropy_ranking_accepts_hdm_runtime_equivalence_without_payload_identity() -> None:
    scorecard = _scorecard()
    internal_row = {
        "label": "PR106-R2-HDM7",
        "canonical_frontier_eligible": False,
        "canonicality_blockers": ["promotion_ineligible"],
        "score": 0.19,
        "archive_bytes": 207,
        "archive_sha256": "7" * 64,
        "payload_sha256": "8" * 64,
        "runtime_tree_sha256": "9" * 64,
        "evidence_grade": "A++",
        "frontier_scope": "exact_local_cuda_custody_candidate_manifest",
        "eval_artifact": "hdm7.json",
        "payload_sections": [
            _section("inner_decoder_packed_brotli", 159, "6" * 64),
            _section("inner_latents_and_sidecar_brotli", 12, "5" * 64),
        ],
    }
    scorecard["rows"].append(internal_row)
    scorecard["score_lowering_frontier"] = {
        "label": internal_row["label"],
        "score": internal_row["score"],
        "archive_bytes": internal_row["archive_bytes"],
        "archive_sha256": internal_row["archive_sha256"],
        "frontier_scope": "internal_exact_cuda_score_lowering",
        "promotion_authority": False,
    }
    candidate_manifest = _hdm_candidate_manifest(internal_row)
    candidate_manifest["runtime_adapter_payload_identity"] = {
        "payload_identity_proven": False,
        "restored_payload_matches_source": False,
        "lossless_decoder_equivalence_proven": True,
        "latents_and_sidecar_match_source": True,
        "submission_runtime_candidate_parse_claim": True,
        "submission_runtime_equivalence_claim": True,
    }

    ranking = build_frontier_entropy_gap_ranking(
        scorecard,
        candidate_manifests=[candidate_manifest],
        frontier_mode="score_lowering",
    )

    control = ranking["exact_lossless_control_actions"][0]
    assert control["target_label"] == "PR106-R2-HDM7"
    assert control["raw_equivalence_closed"] is True
    assert control["review_status"] == "ready_for_promotion_review_existing_exact_custody"
    assert control["total_byte_delta"] == -10


def test_rank_hnerv_frontier_entropy_gaps_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    entropy = tmp_path / "entropy.json"
    candidate = tmp_path / "candidate.json"
    json_out = tmp_path / "ranking.json"
    md_out = tmp_path / "ranking.md"
    scorecard.write_text(json_text(_scorecard()), encoding="utf-8")
    entropy.write_text(json_text(_entropy_audit()), encoding="utf-8")
    candidate.write_text(json_text(_candidate_manifest()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "rank_hnerv_frontier_entropy_gaps.py"),
            "--scorecard",
            str(scorecard),
            "--entropy-audit",
            str(entropy),
            "--candidate-manifest",
            str(candidate),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["tool"] == "tac.hnerv_frontier_entropy_ranking"
    assert payload["tool_run_manifest"]["tool"] == "tools/rank_hnerv_frontier_entropy_gaps.py"
    assert payload["next_entropy_research_action"]["dispatch_allowed"] is False
    assert "Next Entropy Research Action" in md_out.read_text(encoding="utf-8")


def _scorecard() -> dict:
    return {
        "schema_version": 1,
        "score_truth": "exact_cuda_auth_eval_json",
        "rows": [
            {
                "label": "PR106x",
                "canonical_frontier_eligible": True,
                "score": 0.21,
                "archive_bytes": 200,
                "archive_sha256": "1" * 64,
                "payload_sha256": "2" * 64,
                "runtime_tree_sha256": "3" * 64,
                "evidence_grade": "A++",
                "frontier_scope": "exact_local_cuda_custody",
                "eval_artifact": "old.json",
                "payload_sections": [
                    _section("decoder_packed_brotli", 100, "a" * 64),
                    _section("latents_and_sidecar_brotli", 12, "b" * 64),
                ],
            },
            {
                "label": "PR106x-lowlevel-brotli",
                "canonical_frontier_eligible": True,
                "score": 0.2,
                "archive_bytes": 190,
                "archive_sha256": "c" * 64,
                "payload_sha256": "d" * 64,
                "runtime_tree_sha256": "e" * 64,
                "evidence_grade": "A++",
                "frontier_scope": "exact_local_cuda_custody_lossless_repack_control",
                "eval_artifact": "frontier.json",
                "payload_sections": [
                    _section("decoder_packed_brotli", 90, "f" * 64),
                    _section("latents_and_sidecar_brotli", 12, "b" * 64),
                ],
            },
        ],
    }


def _entropy_audit() -> dict:
    return {
        "schema_version": 1,
        "tool": "fixture_entropy_audit",
        "source_label": "PR106x",
        "source_archive_sha256": "1" * 64,
        "source_decoder_section_sha256": "a" * 64,
        "score_claim": False,
        "dispatch_attempted": False,
        "streams": [
            {
                "label": "PR106x:hdc2_global_prev_symbol_contexts",
                "actual_bytes": 140,
                "source_decoder_section_bytes": 100,
                "source_decoder_section_sha256": "a" * 64,
            }
        ],
        "entropy_overhead_target_ranking": [
            {
                "rank": 1,
                "label": "PR106x:hdc2_global_prev_symbol_contexts",
                "target_kind": "known_model_overhead",
                "target_action": "reduce_or_share_static_model_context_metadata",
                "target_bytes": 40,
                "target_bytes_field": "hdc2.header_bytes",
                "required_next_artifact": "byte_accounted_model_overhead_reduction_manifest",
                "actual_bytes": 140,
                "source_decoder_section_sha256": "a" * 64,
                "byte_equivalence_blockers": ["missing_candidate_archive_manifest"],
            },
            {
                "rank": 2,
                "label": "PR106x:hdc2_global_prev_symbol_contexts",
                "target_kind": "known_payload_entropy_gap",
                "target_action": "prototype_byte_equivalent_entropy_coder_for_encoded_payload",
                "target_bytes": 20,
                "target_bytes_field": "hdc2.payload_gap",
                "required_next_artifact": "roundtrip_payload_recode_manifest",
                "actual_bytes": 140,
                "source_decoder_section_sha256": "a" * 64,
                "byte_equivalence_blockers": ["missing_candidate_archive_manifest"],
            },
        ],
    }


def _candidate_manifest() -> dict:
    return {
        "candidate_archive_sha256": "c" * 64,
        "candidate_archive_bytes": 190,
        "source_archive_sha256": "1" * 64,
        "source_archive_bytes": 200,
        "source_label": "PR106x",
        "brotli_raw_equivalence": [
            {
                "section_name": "decoder_packed_brotli",
                "raw_equal": True,
            }
        ],
        "candidate_diff_audit": {
            "blockers": [],
            "total_byte_delta": -10,
            "rate_score_delta_if_components_equal": -0.000006,
            "sections": [
                {
                    "section_name": "decoder_packed_brotli",
                    "source_section_sha256": "a" * 64,
                    "candidate_section_sha256": "f" * 64,
                    "source_bytes": 100,
                    "candidate_bytes": 90,
                    "byte_delta": -10,
                }
            ],
        },
    }


def _hdm_candidate_manifest(row: dict) -> dict:
    candidate_bytes = int(row["archive_bytes"])
    return {
        "candidate_archive_sha256": row["archive_sha256"],
        "candidate_archive_bytes": candidate_bytes,
        "source_archive_sha256": "1" * 64,
        "source_archive_bytes": candidate_bytes + 10,
        "source_label": "PR106-R2-lowlevel-source",
        "candidate_rate_score_delta_if_runtime_supported_and_components_equal": (
            -10 * 25 / 37_545_489
        ),
        "decoder_raw_equivalence": {
            "raw_equal": True,
            "q_roundtrip_equal": True,
            "scale_roundtrip_equal": True,
        },
        "runtime_adapter_payload_identity": {
            "payload_identity_proven": True,
            "restored_payload_matches_source": True,
        },
    }


def _section(name: str, size: int, sha: str) -> dict:
    return {
        "name": name,
        "bytes": size,
        "start": 0,
        "end": size,
        "sha256": sha,
        "entropy_bits_per_byte": 7.9,
    }
