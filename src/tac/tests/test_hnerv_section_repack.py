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
    render_markdown,
)

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
