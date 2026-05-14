# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.hnerv_section_repack import (
    HnervSectionPlanError,
    audit_candidate_section_diff,
    build_section_repack_plan,
    candidate_diff_from_scorecard_manifests,
    detector_cost_atoms_from_section_plan,
    render_markdown,
)
from tac.uniward_delta import build_detector_cost_manifest

REPO = Path(__file__).resolve().parents[3]


def test_build_section_repack_plan_prioritizes_decoder_then_latents() -> None:
    plan = build_section_repack_plan(_scorecard(), labels=["PR106x"])

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatch_blockers"] == [
        "planning_only_section_targets",
        "requires_byte_different_archive",
        "requires_old_new_section_sha256_proof",
        "requires_exact_cuda_auth_eval",
    ]
    assert plan["selected_labels"] == ["PR106x"]
    assert plan["role_counts"] == {
        "control_or_metadata": 1,
        "decoder_weight_stream": 1,
        "latent_stream": 1,
    }
    assert [row["optimization_role"] for row in plan["rows"]] == [
        "decoder_weight_stream",
        "latent_stream",
        "control_or_metadata",
    ]
    assert plan["rows"][0]["recommended_next_action"] == (
        "build decoder self-compression or weight-stream recoding fixture"
    )
    assert plan["rows"][0]["rate_score_gain_if_save_5pct"] > plan["rows"][0]["rate_score_gain_if_save_1pct"]
    assert plan["rows"][0]["dispatchable"] is False
    assert "old_new_section_sha256" in "_".join(plan["dispatch_blockers"])
    assert "HNeRV Section Repack Plan" in render_markdown(plan)


def test_section_repack_plan_rejects_missing_manifest_and_bad_sha() -> None:
    with pytest.raises(HnervSectionPlanError, match="missing payload_section_manifests"):
        build_section_repack_plan({})

    bad = _scorecard()
    bad["payload_section_manifests"][0]["sections"][0]["sha256"] = "short"
    with pytest.raises(HnervSectionPlanError, match="64-char sha256"):
        build_section_repack_plan(bad)


def test_plan_hnerv_section_repack_cli(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"
    scorecard.write_text(json.dumps(_scorecard()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_hnerv_section_repack.py"),
            "--scorecard",
            str(scorecard),
            "--label",
            "PR106x",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["selected_labels"] == ["PR106x"]
    assert payload["score_claim"] is False
    assert "decoder_weight_stream" in md_out.read_text()


def test_candidate_section_diff_audit_blocks_noops_and_accepts_byte_changes() -> None:
    plan = build_section_repack_plan(_scorecard(), labels=["PR106x"])
    source = plan["rows"][0]

    accepted = audit_candidate_section_diff(
        plan,
        {
            "source_archive_sha256": "a" * 64,
            "candidate_archive_sha256": "f" * 64,
            "sections": [
                {
                    "label": source["label"],
                    "section_name": source["section_name"],
                    "source_section_sha256": source["section_sha256"],
                    "candidate_section_sha256": "1" * 64,
                    "source_bytes": source["section_bytes"],
                    "candidate_bytes": source["section_bytes"] - 7,
                }
            ],
        },
    )

    assert accepted["score_claim"] is False
    assert accepted["ready_for_archive_preflight"] is True
    assert accepted["ready_for_exact_eval_dispatch"] is False
    assert accepted["changed_section_count"] == 1
    assert accepted["total_byte_delta"] == -7
    assert accepted["rate_score_delta_if_components_equal"] < 0

    raw_blocked = audit_candidate_section_diff(
        plan,
        {
            "source_archive_sha256": "a" * 64,
            "candidate_archive_sha256": "f" * 64,
            "sections": [
                {
                    "label": source["label"],
                    "section_name": source["section_name"],
                    "source_section_sha256": source["section_sha256"],
                    "candidate_section_sha256": "1" * 64,
                    "source_bytes": source["section_bytes"],
                    "candidate_bytes": source["section_bytes"] - 7,
                }
            ],
        },
        require_raw_equivalence=True,
    )

    assert raw_blocked["ready_for_archive_preflight"] is False
    assert "brotli_raw_equivalence_missing" in raw_blocked["blockers"]

    noop = audit_candidate_section_diff(
        plan,
        {
            "candidate_archive_sha256": "f" * 64,
            "sections": [
                {
                    "label": source["label"],
                    "section_name": source["section_name"],
                    "source_section_sha256": source["section_sha256"],
                    "candidate_section_sha256": source["section_sha256"],
                    "source_bytes": source["section_bytes"],
                    "candidate_bytes": source["section_bytes"],
                }
            ],
        },
    )

    assert noop["ready_for_archive_preflight"] is False
    assert "candidate_diff_has_no_changed_sections" in noop["blockers"]
    assert any(str(item).startswith("candidate_section_noop") for item in noop["blockers"])


def test_candidate_diff_from_scorecard_manifests_proves_repack_noop() -> None:
    scorecard = _scorecard_pair()
    plan = build_section_repack_plan(scorecard, labels=["PR106"])
    diff = candidate_diff_from_scorecard_manifests(
        scorecard,
        source_label="PR106",
        candidate_label="PR106x",
    )
    audit = audit_candidate_section_diff(plan, diff)

    assert diff["source_archive_sha256"] == "a" * 64
    assert diff["candidate_archive_sha256"] == "f" * 64
    assert diff["source_payload_sha256"] == diff["candidate_payload_sha256"]
    assert audit["ready_for_archive_preflight"] is False
    assert audit["changed_section_count"] == 0
    assert "candidate_diff_has_no_changed_sections" in audit["blockers"]


def test_section_plan_feeds_fridrich_detector_cost_manifest() -> None:
    plan = build_section_repack_plan(_scorecard(), labels=["PR106x"])
    atoms = detector_cost_atoms_from_section_plan(plan)
    manifest = build_detector_cost_manifest(atoms, source_label="PR106x")

    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "requires_charged_archive_consumption" in manifest["dispatch_blockers"]
    assert manifest["atom_count"] == 3
    rows = manifest["rows"]
    assert rows[0]["atom_id"] == "hnerv:PR106x:latents_and_sidecar_brotli"
    assert all(row["detector_capacity_source"] == "entropy_bits_per_byte/8" for row in rows)
    assert all(row["scorer_sensitivity_source"] == "rate_score_gain_if_save_1pct" for row in rows)
    assert rows[0]["allocation_priority"] > rows[-1]["allocation_priority"]
    assert all(row["promotion_eligible"] is False for row in rows)


def test_audit_hnerv_section_candidate_diff_cli_blocks_noop_repack(tmp_path: Path) -> None:
    scorecard = tmp_path / "scorecard.json"
    audit_out = tmp_path / "audit.json"
    scorecard.write_text(json.dumps(_scorecard_pair()), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_hnerv_section_candidate_diff.py"),
            "--scorecard",
            str(scorecard),
            "--source-label",
            "PR106",
            "--candidate-label",
            "PR106x",
            "--json-out",
            str(audit_out),
            "--fail-if-blocked",
        ],
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(audit_out.read_text())
    assert payload["ready_for_archive_preflight"] is False
    assert "candidate_diff_has_no_changed_sections" in payload["blockers"]


def test_inner_decoder_candidate_guard_accepts_bound_raw_and_full_frame_parity() -> None:
    plan = build_section_repack_plan(_inner_scorecard(), labels=["PR106-R2"])
    source = next(row for row in plan["rows"] if row["section_name"] == "inner_decoder_packed_brotli")

    audit = audit_candidate_section_diff(
        plan,
        _inner_decoder_candidate_diff(source),
        require_raw_equivalence=True,
        require_byte_reduction=True,
        require_same_runtime_full_frame_parity=True,
    )

    assert audit["ready_for_archive_preflight"] is True
    assert audit["ready_for_exact_eval_dispatch"] is False
    assert audit["blockers"] == []
    assert audit["changed_section_count"] == 1
    assert audit["total_byte_delta"] == -7
    assert audit["same_runtime_full_frame_parity"]["valid"] is True
    assert audit["same_runtime_full_frame_parity"]["device_axis_label"] == "local-cpu-streaming-runtime"


def test_inner_decoder_candidate_guard_fails_closed_without_full_frame_parity() -> None:
    plan = build_section_repack_plan(_inner_scorecard(), labels=["PR106-R2"])
    source = next(row for row in plan["rows"] if row["section_name"] == "inner_decoder_packed_brotli")
    diff = _inner_decoder_candidate_diff(source)
    diff.pop("same_runtime_full_frame_parity")

    audit = audit_candidate_section_diff(
        plan,
        diff,
        require_raw_equivalence=True,
        require_byte_reduction=True,
        require_same_runtime_full_frame_parity=True,
    )

    assert audit["ready_for_archive_preflight"] is False
    assert "same_runtime_full_frame_parity_missing" in audit["blockers"]


def test_decoder_candidate_byte_reduction_guard_rejects_growth() -> None:
    plan = build_section_repack_plan(_inner_scorecard(), labels=["PR106-R2"])
    source = next(row for row in plan["rows"] if row["section_name"] == "inner_decoder_packed_brotli")
    diff = _inner_decoder_candidate_diff(source)
    diff["sections"][0]["candidate_bytes"] = source["section_bytes"] + 1

    audit = audit_candidate_section_diff(
        plan,
        diff,
        require_raw_equivalence=True,
        require_byte_reduction=True,
        require_same_runtime_full_frame_parity=True,
    )

    assert audit["ready_for_archive_preflight"] is False
    assert "candidate_section_not_smaller:PR106-R2:inner_decoder_packed_brotli" in audit["blockers"]


def test_audit_hnerv_section_candidate_diff_cli_threads_strict_decoder_flags(
    tmp_path: Path,
) -> None:
    scorecard = tmp_path / "scorecard.json"
    diff_path = tmp_path / "diff.json"
    audit_out = tmp_path / "audit.json"
    scorecard_payload = _inner_scorecard()
    source = scorecard_payload["payload_section_manifests"][0]["sections"][2]
    scorecard.write_text(json.dumps(scorecard_payload), encoding="utf-8")
    diff_path.write_text(
        json.dumps(
            {
                "source_archive_sha256": "a" * 64,
                "candidate_archive_sha256": "f" * 64,
                "sections": [
                    {
                        "label": "PR106-R2",
                        "section_name": "inner_decoder_packed_brotli",
                        "source_section_sha256": source["sha256"],
                        "candidate_section_sha256": "1" * 64,
                        "source_bytes": source["bytes"],
                        "candidate_bytes": source["bytes"] - 7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_hnerv_section_candidate_diff.py"),
            "--scorecard",
            str(scorecard),
            "--source-label",
            "PR106-R2",
            "--candidate-diff-json",
            str(diff_path),
            "--json-out",
            str(audit_out),
            "--require-raw-equivalence",
            "--require-byte-reduction",
            "--require-same-runtime-full-frame-parity",
            "--fail-if-blocked",
        ],
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(audit_out.read_text())
    assert payload["ready_for_archive_preflight"] is False
    assert "brotli_raw_equivalence_missing" in payload["blockers"]
    assert "same_runtime_full_frame_parity_missing" in payload["blockers"]


def _scorecard() -> dict:
    return {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "exact_cuda_auth_eval_json",
        "payload_section_manifests": [
            {
                "label": "PR106x",
                "archive_sha256": "a" * 64,
                "archive_bytes": 123,
                "zip_member": "x",
                "payload_sha256": "b" * 64,
                "member_bytes": 100,
                "profile_match_key": "member_sha256",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": [
                    {
                        "index": 0,
                        "name": "packed_header_ff_len24",
                        "start": 0,
                        "end": 4,
                        "bytes": 4,
                        "sha256": "c" * 64,
                        "entropy_bits_per_byte": 2.0,
                        "optimization_role": "control_or_metadata",
                    },
                    {
                        "index": 1,
                        "name": "decoder_packed_brotli",
                        "start": 4,
                        "end": 94,
                        "bytes": 90,
                        "sha256": "d" * 64,
                        "entropy_bits_per_byte": 7.5,
                        "optimization_role": "decoder_weight_stream",
                    },
                    {
                        "index": 2,
                        "name": "latents_and_sidecar_brotli",
                        "start": 94,
                        "end": 100,
                        "bytes": 6,
                        "sha256": "e" * 64,
                        "entropy_bits_per_byte": 6.1,
                        "optimization_role": "latent_stream",
                    },
                ],
            }
        ],
    }


def _scorecard_pair() -> dict:
    payload = _scorecard()
    source = payload["payload_section_manifests"][0]
    source["label"] = "PR106"
    source["archive_sha256"] = "a" * 64
    candidate = json.loads(json.dumps(source))
    candidate["label"] = "PR106x"
    candidate["archive_sha256"] = "f" * 64
    payload["payload_section_manifests"] = [source, candidate]
    return payload


def _inner_scorecard() -> dict:
    return {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "exact_cuda_auth_eval_json",
        "payload_section_manifests": [
            {
                "label": "PR106-R2",
                "archive_sha256": "a" * 64,
                "archive_bytes": 123,
                "zip_member": "x",
                "payload_sha256": "b" * 64,
                "member_bytes": 100,
                "profile_match_key": "member_sha256",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": [
                    {
                        "index": 0,
                        "name": "pr106_sidecar_header_fe_fmt_len_u32",
                        "start": 0,
                        "end": 6,
                        "bytes": 6,
                        "sha256": "c" * 64,
                        "entropy_bits_per_byte": 2.0,
                        "optimization_role": "control_or_metadata",
                    },
                    {
                        "index": 1,
                        "name": "inner_packed_header_ff_len24",
                        "start": 6,
                        "end": 10,
                        "bytes": 4,
                        "sha256": "b" * 64,
                        "entropy_bits_per_byte": 2.0,
                        "optimization_role": "control_or_metadata",
                    },
                    {
                        "index": 2,
                        "name": "inner_decoder_packed_brotli",
                        "start": 10,
                        "end": 100,
                        "bytes": 90,
                        "sha256": "d" * 64,
                        "entropy_bits_per_byte": 7.5,
                        "optimization_role": "decoder_weight_stream",
                    },
                    {
                        "index": 3,
                        "name": "inner_latents_and_sidecar_brotli",
                        "start": 100,
                        "end": 106,
                        "bytes": 6,
                        "sha256": "e" * 64,
                        "entropy_bits_per_byte": 6.1,
                        "optimization_role": "latent_stream",
                    },
                ],
            }
        ],
    }


def _inner_decoder_candidate_diff(source: dict) -> dict:
    return {
        "source_archive_sha256": "a" * 64,
        "candidate_archive_sha256": "f" * 64,
        "sections": [
            {
                "label": "PR106-R2",
                "section_name": "inner_decoder_packed_brotli",
                "source_section_sha256": source["section_sha256"],
                "candidate_section_sha256": "1" * 64,
                "source_bytes": source["section_bytes"],
                "candidate_bytes": source["section_bytes"] - 7,
            }
        ],
        "brotli_raw_equivalence": [
            {
                "section_name": "inner_decoder_packed_brotli",
                "raw_equal": True,
                "source_raw_sha256": "9" * 64,
                "candidate_raw_sha256": "9" * 64,
                "raw_bytes": 1234,
            }
        ],
        "same_runtime_full_frame_parity": _same_runtime_full_frame_parity(
            source_archive_sha256="a" * 64,
            candidate_archive_sha256="f" * 64,
        ),
    }


def _same_runtime_full_frame_parity(
    *,
    source_archive_sha256: str,
    candidate_archive_sha256: str,
) -> dict:
    digest = "8" * 64
    stream = {
        "schema": "pr106_runtime_full_frame_streaming_digest_v1",
        "score_claim": False,
        "full_frame_digest": True,
        "streaming_raw_sha256": digest,
        "total_bytes": 3662409600,
        "n_pairs_hashed": 600,
        "n_pairs_total": 600,
    }
    return {
        "schema": "pr106_same_runtime_streaming_frame_parity_v1",
        "proof_scope": "same_runtime_streaming_full_frame_hash",
        "device_axis_label": "local-cpu-streaming-runtime",
        "score_claim": False,
        "contest_axis_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "full_frame_inflate_output_parity_claim": True,
        "streaming_output_sha256_equal": True,
        "streaming_output_total_bytes_equal": True,
        "source_archive": {"sha256": source_archive_sha256},
        "candidate_archive": {"sha256": candidate_archive_sha256},
        "source": dict(stream),
        "candidate": dict(stream),
    }
